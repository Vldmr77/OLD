"""Command-line utilities for the scalping system."""
from __future__ import annotations

from .health_check import main as health_check_main, run_health_checks
from .init_config import init_config, main as init_config_main
from .model_trainer import calibrate, main as model_trainer_main
from .backtest import main as backtest_main, run_backtest
from .dashboard import main as dashboard_main

__all__ = [
    "health_check_main",
    "run_health_checks",
    "init_config",
    "init_config_main",
    "calibrate",
    "model_trainer_main",
    "backtest_main",
    "run_backtest",
    "dashboard_main",
]
