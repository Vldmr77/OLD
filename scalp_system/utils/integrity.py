"""Runtime integrity helpers for recovery workflows."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..data.engine import DataEngine

LOGGER = logging.getLogger(__name__)


async def check_data_integrity(engine: DataEngine, *, max_staleness: float = 5.0) -> bool:
    """Validate cached order books before resuming after reconnects."""

    missing: list[str] = []
    stale: list[str] = []
    empty: list[str] = []
    now = datetime.now(timezone.utc)
    for figi in engine.active_instruments():
        order_book = engine.get_order_book(figi)
        if order_book is None:
            missing.append(figi)
            continue
        if not order_book.bids or not order_book.asks:
            empty.append(figi)
        age = abs((now - order_book.timestamp).total_seconds())
        if age > max_staleness:
            stale.append(figi)
    if missing or stale or empty:
        LOGGER.warning(
            "Data integrity check failed missing=%s stale=%s empty=%s",
            missing,
            stale,
            empty,
        )
        return False
    return True


__all__ = ["check_data_integrity"]
