"""Data management utilities for streaming order book processing."""
from __future__ import annotations

import gc
import os
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Sequence

from .models import Candle, OrderBook, Trade


@dataclass(slots=True)
class MarketIndicators:
    """Aggregated market level indicators updated at 1 Hz."""

    market_volatility: float
    market_liquidity: float
    market_index: float


@dataclass(slots=True)
class InstrumentStats:
    """Rolling statistics for instrument health checks."""

    spread: float = 0.0
    liquidity: float = 0.0
    volatility: float = 0.0
    samples: int = 0

    def update(self, spread: float, liquidity: float, volatility: float) -> None:
        self.samples += 1
        alpha = 0.2
        if self.samples == 1:
            self.spread = spread
            self.liquidity = liquidity
            self.volatility = volatility
        else:
            self.spread = (1 - alpha) * self.spread + alpha * spread
            self.liquidity = (1 - alpha) * self.liquidity + alpha * liquidity
            self.volatility = (1 - alpha) * self.volatility + alpha * volatility


class _TTLCache:
    """Simple TTL cache for current order books."""

    def __init__(self, maxsize: int, ttl: float) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._data: OrderedDict[str, tuple[float, OrderBook]] = OrderedDict()

    def get(self, key: str) -> Optional[OrderBook]:
        self._purge()
        item = self._data.get(key)
        if not item:
            return None
        timestamp, value = item
        if time.monotonic() - timestamp > self._ttl:
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    def put(self, key: str, value: OrderBook) -> None:
        self._purge()
        self._data[key] = (time.monotonic(), value)
        self._data.move_to_end(key)
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def _purge(self) -> None:
        now = time.monotonic()
        expired = [key for key, (ts, _) in self._data.items() if now - ts > self._ttl]
        for key in expired:
            self._data.pop(key, None)


class DataEngine:
    """Maintains rolling history and health metrics for streamed order books."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 60.0,
        max_instruments: int = 50,
        history_size: int = 30,
        monitored_instruments: Optional[Iterable[str]] = None,
        soft_rss_limit_mb: int = 400,
        hard_rss_limit_mb: int = 700,
        max_active_instruments: int = 15,
        monitor_pool_size: int = 30,
        trade_history_size: int = 200,
        candle_history_size: int = 240,
    ) -> None:
        self._current = _TTLCache(max_instruments, ttl_seconds)
        self._history: Dict[str, Deque[OrderBook]] = {}
        self._history_size = history_size
        self._stats: Dict[str, InstrumentStats] = {}
        self._active_instruments: List[str] = []
        self._monitored_pool: List[str] = []
        self._soft_rss_limit = soft_rss_limit_mb * 1024 * 1024
        self._hard_rss_limit = hard_rss_limit_mb * 1024 * 1024
        self._max_active = max_active_instruments
        self._monitor_pool_size = max(monitor_pool_size, max_active_instruments)
        self._trade_history_size = max(1, trade_history_size)
        self._candle_history_size = max(1, candle_history_size)
        self._trades: Dict[str, Deque[Trade]] = {}
        self._candles: Dict[str, Deque[Candle]] = {}
        if monitored_instruments:
            self.update_monitored_pool(monitored_instruments)

    # ------------------------------------------------------------------
    # Active & monitored instruments
    # ------------------------------------------------------------------
    def update_active_instruments(self, instruments: Iterable[str]) -> None:
        requested = list(dict.fromkeys(instruments))
        if len(requested) < self._max_active:
            supplement = [figi for figi in self._monitored_pool if figi not in requested]
            requested.extend(supplement)
        self._active_instruments = requested[: self._max_active]

    def update_monitored_pool(self, instruments: Iterable[str]) -> None:
        pool = list(dict.fromkeys(instruments))
        for figi in self._active_instruments:
            if figi not in pool:
                pool.insert(0, figi)
        self._monitored_pool = pool[: self._monitor_pool_size]

    def active_instruments(self) -> tuple[str, ...]:
        return tuple(self._active_instruments)

    def monitored_instruments(self) -> tuple[str, ...]:
        return tuple(self._monitored_pool)

    def max_active(self) -> int:
        return self._max_active

    def seed_instruments(self, active: Sequence[str], monitored: Sequence[str]) -> None:
        active_unique = list(dict.fromkeys(active))
        monitored_unique = list(dict.fromkeys(monitored))
        if len(active_unique) < self._max_active:
            for figi in monitored_unique:
                if figi not in active_unique:
                    active_unique.append(figi)
                    if len(active_unique) >= self._max_active:
                        break
        self._active_instruments = active_unique[: self._max_active]
        pool: List[str] = []
        for figi in list(active_unique) + monitored_unique:
            if figi not in pool:
                pool.append(figi)
        self._monitored_pool = pool[: self._monitor_pool_size]

    def snapshot(self) -> dict:
        return {
            "active_instruments": list(self._active_instruments),
            "monitored_instruments": list(self._monitored_pool),
        }

    def restore(self, payload: dict) -> None:
        active = payload.get("active_instruments")
        if isinstance(active, list):
            self.update_active_instruments(active)
        monitored = payload.get("monitored_instruments")
        if isinstance(monitored, list):
            self.update_monitored_pool(monitored)

    def rotate_instruments(self) -> List[tuple[str, str]]:
        """Replace instruments that fail liquidity/volatility checks."""

        replacements: List[tuple[str, str]] = []
        avg_liquidity = self._average_liquidity() or 1.0
        for instrument in list(self._active_instruments):
            stats = self._stats.get(instrument)
            if not stats or stats.samples < 5:
                continue
            if (
                stats.spread > 0.1
                or stats.liquidity < 0.7 * avg_liquidity
                or stats.volatility < 0.05
            ):
                replacement = self._select_from_monitored(exclude=self._active_instruments)
                if replacement is None:
                    continue
                self._active_instruments = [
                    replacement if figi == instrument else figi for figi in self._active_instruments
                ]
                replacements.append((instrument, replacement))
        if replacements:
            self.update_active_instruments(self._active_instruments)
        return replacements

    def _select_from_monitored(self, *, exclude: Iterable[str]) -> Optional[str]:
        exclude_set = set(exclude)
        for candidate in self._monitored_pool:
            if candidate not in exclude_set:
                return candidate
        return None

    # ------------------------------------------------------------------
    # Order book ingestion and retrieval
    # ------------------------------------------------------------------
    def ingest_order_book(self, order_book: OrderBook) -> None:
        self._current.put(order_book.figi, order_book)
        history = self._history.setdefault(order_book.figi, deque(maxlen=self._history_size))
        history.append(order_book)
        stats = self._stats.setdefault(order_book.figi, InstrumentStats())
        spread = order_book.spread()
        liquidity = self._top_liquidity(order_book)
        volatility = self._estimate_volatility(history)
        stats.update(spread=spread, liquidity=liquidity, volatility=volatility)
        self._enforce_memory_limits()

    def get_order_book(self, figi: str, offset: int = 0) -> Optional[OrderBook]:
        """Return order book snapshot for offset; negative offset looks back."""

        if offset == 0:
            return self._current.get(figi)
        history = self._history.get(figi)
        if not history:
            return None
        index = len(history) + offset if offset < 0 else offset
        if index < 0 or index >= len(history):
            return None
        return list(history)[index]

    def history(self, figi: str) -> tuple[OrderBook, ...]:
        history = self._history.get(figi)
        if not history:
            return ()
        return tuple(history)

    def ingest_trades(self, figi: str, trades: Iterable[Trade]) -> None:
        trades = list(trades)
        if not trades:
            return
        history = self._trades.setdefault(figi, deque(maxlen=self._trade_history_size))
        for trade in trades:
            history.append(trade)

    def recent_trades(self, figi: str) -> tuple[Trade, ...]:
        history = self._trades.get(figi)
        if not history:
            return ()
        return tuple(history)

    def ingest_candles(self, figi: str, candles: Iterable[Candle]) -> None:
        candles = list(candles)
        if not candles:
            return
        history = self._candles.setdefault(figi, deque(maxlen=self._candle_history_size))
        for candle in candles:
            history.append(candle)

    def recent_candles(self, figi: str) -> tuple[Candle, ...]:
        history = self._candles.get(figi)
        if not history:
            return ()
        return tuple(history)

    def atr(self, figi: str, periods: int = 5) -> float:
        """Compute a simple ATR proxy from the recent order book history."""

        history = self._history.get(figi)
        if not history or len(history) < 2:
            return 0.0
        window = list(history)[-max(periods + 1, 2) :]
        if len(window) < 2:
            return 0.0
        true_ranges: List[float] = []
        prev_close = window[0].mid_price()
        for book in window[1:]:
            high = book.asks[0].price if book.asks else prev_close
            low = book.bids[0].price if book.bids else prev_close
            tr = max(high, prev_close) - min(low, prev_close)
            true_ranges.append(tr)
            prev_close = book.mid_price()
        if not true_ranges:
            return 0.0
        return sum(true_ranges) / len(true_ranges)

    def resynchronise(self) -> None:
        """Flush cached order books so the engine refetches fresh data."""

        self._current.clear()
        for history in self._history.values():
            history.clear()
        self._stats.clear()
        self._trades.clear()
        self._candles.clear()

    def last_window(self, figi: str, length: Optional[int] = None) -> tuple[OrderBook, ...]:
        history = self._history.get(figi)
        if not history:
            return ()
        if length is None or length >= len(history):
            return tuple(history)
        it: Iterator[OrderBook] = iter(history)
        return tuple(list(history)[-length:])

    # ------------------------------------------------------------------
    # Market level indicators
    # ------------------------------------------------------------------
    def market_indicators(self) -> MarketIndicators:
        volatility = 0.0
        liquidity = 0.0
        spread = 0.0
        count = 0
        for figi in self._active_instruments:
            stats = self._stats.get(figi)
            if not stats:
                continue
            volatility += stats.volatility
            liquidity += stats.liquidity
            spread += stats.spread
            count += 1
        if count:
            volatility /= count
            liquidity /= count
            spread /= count
        market_index = max(0.0, 1.0 - spread)
        return MarketIndicators(
            market_volatility=volatility,
            market_liquidity=liquidity,
            market_index=market_index,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _average_liquidity(self) -> float:
        values = [self._stats[figi].liquidity for figi in self._active_instruments if figi in self._stats]
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _top_liquidity(order_book: OrderBook) -> float:
        return sum(level.quantity for level in order_book.bids[:10]) + sum(
            level.quantity for level in order_book.asks[:10]
        )

    @staticmethod
    def _estimate_volatility(history: Deque[OrderBook]) -> float:
        if len(history) < 2:
            return 0.0
        mids: List[float] = [book.mid_price() for book in history]
        if len(mids) < 2:
            return 0.0
        returns = []
        for prev, curr in zip(mids[:-1], mids[1:]):
            if prev <= 0 or curr <= 0:
                continue
            returns.append(abs(curr - prev) / prev)
        if not returns:
            return 0.0
        return sum(returns) / len(returns)

    def _enforce_memory_limits(self) -> None:
        rss = self._current_rss()
        if rss is None:
            return
        if rss > self._hard_rss_limit:
            self._current.clear()
            self._history.clear()
            self._stats.clear()
            gc.collect()
        elif rss > self._soft_rss_limit:
            for figi, history in list(self._history.items()):
                while len(history) > self._history_size // 2:
                    history.popleft()

    @staticmethod
    def _current_rss() -> Optional[int]:
        try:
            import psutil  # type: ignore

            process = psutil.Process(os.getpid())
            return int(process.memory_info().rss)
        except Exception:
            try:
                import resource

                usage = resource.getrusage(resource.RUSAGE_SELF)
                return int(usage.ru_maxrss * 1024)
            except Exception:
                return None


__all__ = ["DataEngine", "MarketIndicators", "InstrumentStats"]
