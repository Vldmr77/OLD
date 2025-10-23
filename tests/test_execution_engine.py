import asyncio

from scalp_system.config.base import RiskLimits
from scalp_system.execution.executor import ExecutionEngine
from scalp_system.risk.engine import RiskEngine


def test_execution_engine_cancel_all_orders_paper_mode():
    engine = ExecutionEngine(lambda: None, RiskEngine(RiskLimits()), mode="development")
    cancelled = asyncio.run(engine.cancel_all_orders())
    assert cancelled == 0
