"""Offline backtesting utilities for the scalping system."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, List, Optional

from ..config.base import BacktestConfig, FeatureConfig, MLConfig, RiskLimits
from ..data.models import OrderBook, OrderBookLevel
from ..features.pipeline import FeaturePipeline
from ..ml.engine import MLEngine
from ..risk.engine import RiskEngine
from ..data.engine import MarketIndicators


@dataclass(slots=True)
class BacktestResult:
    """Aggregated statistics for a backtest run."""

    trades: int
    pnl: float
    win_rate: float
    max_drawdown: float
    equity_curve: List[float]

    def to_dict(self) -> Dict[str, float | int | List[float]]:
        return {
            "trades": self.trades,
            "pnl": self.pnl,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "equity_curve": self.equity_curve,
        }


class BacktestEngine:
    """Replays historical order books to validate the trading stack."""

    def __init__(
        self,
        config: BacktestConfig,
        feature_config: FeatureConfig,
        ml_config: MLConfig,
        risk_limits: RiskLimits,
    ) -> None:
        self._config = config
        self._feature_pipeline = FeaturePipeline(feature_config)
        self._ml_engine = MLEngine(ml_config)
        self._risk_engine = RiskEngine(risk_limits)
        self._windows: Dict[str, Deque[OrderBook]] = defaultdict(
            lambda: deque(maxlen=feature_config.rolling_window)
        )

    def run(self) -> BacktestResult:
        orders_executed = 0
        trades_closed = 0
        winning_trades = 0
        equity = self._config.initial_capital
        equity_curve: List[float] = []
        peak_equity = equity
        max_drawdown = 0.0
        positions: Dict[str, int] = defaultdict(int)
        entry_prices: Dict[str, float] = {}
        last_prices: Dict[str, float] = {}

        for book, signal_override in self._load_books():
            window = self._windows[book.figi]
            window.append(book)
            last_prices[book.figi] = book.mid_price()
            if len(window) < 2:
                equity_curve.append(self._current_equity(equity, positions, entry_prices, last_prices))
                peak_equity = max(peak_equity, equity_curve[-1])
                max_drawdown = max(max_drawdown, peak_equity - equity_curve[-1])
                continue

            market = self._market_indicators(window)
            feature_vector = self._feature_pipeline.transform(window, market=market)
            prediction = self._ml_engine.infer([feature_vector])[0]
            direction = int(signal_override) if signal_override is not None else prediction.direction

            if direction == 0:
                equity_curve.append(self._current_equity(equity, positions, entry_prices, last_prices))
                peak_equity = max(peak_equity, equity_curve[-1])
                max_drawdown = max(max_drawdown, peak_equity - equity_curve[-1])
                continue

            price = book.mid_price()
            prediction.direction = direction
            prediction.figi = book.figi
            if not self._risk_engine.evaluate_signal(prediction, price):
                equity_curve.append(self._current_equity(equity, positions, entry_prices, last_prices))
                peak_equity = max(peak_equity, equity_curve[-1])
                max_drawdown = max(max_drawdown, peak_equity - equity_curve[-1])
                continue

            position = positions[book.figi]
            if position and position == -direction:
                self._risk_engine.register_order()
                fee_multiplier = (
                    self._config.transaction_cost_bps + self._config.slippage_bps
                ) / 10000
                equity -= abs(direction) * price * fee_multiplier
                trade_pnl = (price - entry_prices[book.figi]) * position
                equity += trade_pnl
                trades_closed += 1
                if trade_pnl > 0:
                    winning_trades += 1
                positions[book.figi] = 0
                entry_prices.pop(book.figi, None)
                self._risk_engine.update_position(book.figi, -position, price)
                orders_executed += 1
            elif position == 0:
                self._risk_engine.register_order()
                fee_multiplier = (
                    self._config.transaction_cost_bps + self._config.slippage_bps
                ) / 10000
                equity -= abs(direction) * price * fee_multiplier
                positions[book.figi] = direction
                entry_prices[book.figi] = price
                self._risk_engine.update_position(book.figi, direction, price)
                orders_executed += 1
            else:
                # same side signal, ignore to avoid runaway leverage
                pass

            current_equity = self._current_equity(equity, positions, entry_prices, last_prices)
            equity_curve.append(current_equity)
            peak_equity = max(peak_equity, current_equity)
            max_drawdown = max(max_drawdown, peak_equity - current_equity)

            if orders_executed >= self._config.max_orders:
                break

        # Mark-to-market remaining positions at final prices
        final_equity = self._current_equity(equity, positions, entry_prices, last_prices)
        if not equity_curve:
            equity_curve.append(final_equity)
        pnl = final_equity - self._config.initial_capital
        win_rate = winning_trades / trades_closed if trades_closed else 0.0
        return BacktestResult(
            trades=trades_closed,
            pnl=pnl,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            equity_curve=equity_curve,
        )

    def _load_books(self) -> Iterator[tuple[OrderBook, Optional[int]]]:
        path = Path(self._config.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Backtest dataset not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                yield self._parse_book(payload), payload.get("signal")

    def _parse_book(self, payload: Dict[str, object]) -> OrderBook:
        timestamp = datetime.fromisoformat(str(payload["timestamp"]))
        bids = tuple(
            OrderBookLevel(price=float(price), quantity=float(qty))
            for price, qty in payload.get("bids", [])
        )
        asks = tuple(
            OrderBookLevel(price=float(price), quantity=float(qty))
            for price, qty in payload.get("asks", [])
        )
        return OrderBook(
            figi=str(payload["figi"]),
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            depth=min(len(bids), len(asks)),
        )

    def _current_equity(
        self,
        base_equity: float,
        positions: Dict[str, int],
        entry_prices: Dict[str, float],
        last_prices: Dict[str, float],
    ) -> float:
        unrealised = 0.0
        for figi, qty in positions.items():
            if qty == 0:
                continue
            last_price = last_prices.get(figi, entry_prices.get(figi, 0.0))
            entry = entry_prices.get(figi, last_price)
            unrealised += (last_price - entry) * qty
        return base_equity + unrealised

    def _market_indicators(self, window: Iterable[OrderBook]) -> MarketIndicators:
        books = list(window)
        mids = [book.mid_price() for book in books]
        if len(mids) < 2:
            volatility = 0.0
        else:
            volatility = statistics.pstdev(mids) / (mids[-1] or 1.0)
        liquidity = sum(level.quantity for book in books for level in book.bids[:5])
        liquidity += sum(level.quantity for book in books for level in book.asks[:5])
        market_index = mids[-1] if mids else 0.0
        return MarketIndicators(
            market_volatility=volatility,
            market_liquidity=liquidity,
            market_index=market_index,
        )


__all__ = ["BacktestEngine", "BacktestResult"]
