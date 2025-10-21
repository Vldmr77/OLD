"""CLI entry point for running offline backtests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from ..config.loader import load_config
from ..simulation.backtest import BacktestEngine


def run_backtest(config_path: str | Path, dataset: Optional[str] = None, output: Optional[str] = None):
    """Execute a backtest using the provided configuration file."""

    config = load_config(config_path)
    if dataset:
        config.backtest.dataset_path = Path(dataset)
        config.backtest.ensure()
    engine = BacktestEngine(
        config.backtest,
        config.features,
        config.ml,
        config.risk,
    )
    result = engine.run()
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline scalping strategy backtests")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--dataset", help="Override dataset path", default=None)
    parser.add_argument("--output", help="Optional JSON report path", default=None)
    args = parser.parse_args(argv)
    result = run_backtest(args.config, dataset=args.dataset, output=args.output)
    print(
        "Backtest completed: trades=%d pnl=%.2f win_rate=%.2f%% max_drawdown=%.2f"
        % (result.trades, result.pnl, result.win_rate * 100, result.max_drawdown)
    )
    return 0


__all__ = ["run_backtest", "main"]
