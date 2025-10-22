import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scalp_system.config.base import BacktestConfig, FeatureConfig, MLConfig, RiskLimits
from scalp_system.simulation.backtest import BacktestEngine


def _order_book_line(timestamp: datetime, price: float, signal: int | None = None) -> str:
    payload = {
        "timestamp": timestamp.isoformat(),
        "figi": "TEST_FIGI",
        "bids": [[price - 1, 2.0], [price - 2, 1.0]],
        "asks": [[price + 1, 2.0], [price + 2, 1.0]],
    }
    if signal is not None:
        payload["signal"] = signal
    return json.dumps(payload)


def test_backtest_engine_generates_profit(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    start = datetime(2024, 1, 1, 0, 0, 0)
    lines = [
        _order_book_line(start, 100.0, signal=0),
        _order_book_line(start + timedelta(seconds=1), 100.0, signal=1),
        _order_book_line(start + timedelta(seconds=2), 102.0, signal=-1),
    ]
    dataset.write_text("\n".join(lines), encoding="utf-8")

    engine = BacktestEngine(
        BacktestConfig(
            dataset_path=dataset,
            initial_capital=10_000.0,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
            max_orders=10,
        ),
        FeatureConfig(lob_levels=2, rolling_window=2),
        MLConfig(),
        RiskLimits(max_position=1, max_order_rate_per_minute=100),
    )

    result = engine.run()

    assert result.trades == 1
    assert result.pnl == pytest.approx(2.0, rel=1e-6)
    assert result.win_rate == pytest.approx(1.0)
    assert result.equity_curve[-1] == pytest.approx(10_002.0, rel=1e-6)


def test_backtest_engine_respects_missing_dataset(tmp_path: Path):
    engine = BacktestEngine(
        BacktestConfig(dataset_path=tmp_path / "missing.jsonl"),
        FeatureConfig(),
        MLConfig(),
        RiskLimits(),
    )
    with pytest.raises(FileNotFoundError):
        engine.run()
