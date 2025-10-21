import asyncio
from datetime import datetime, timedelta, timezone

from scalp_system.data.engine import DataEngine
from scalp_system.data.models import OrderBook, OrderBookLevel
from scalp_system.utils.integrity import check_data_integrity


def test_check_data_integrity_success():
    engine = DataEngine()
    engine.update_active_instruments(["FIGI"])
    now = datetime.now(timezone.utc)
    book = OrderBook(
        figi="FIGI",
        timestamp=now,
        bids=(OrderBookLevel(price=100.0, quantity=10.0),),
        asks=(OrderBookLevel(price=100.5, quantity=10.0),),
        depth=1,
    )
    engine.ingest_order_book(book)
    assert asyncio.run(check_data_integrity(engine))


def test_check_data_integrity_detects_stale():
    engine = DataEngine()
    engine.update_active_instruments(["FIGI"])
    old = datetime.now(timezone.utc) - timedelta(seconds=10)
    book = OrderBook(
        figi="FIGI",
        timestamp=old,
        bids=(OrderBookLevel(price=100.0, quantity=10.0),),
        asks=(OrderBookLevel(price=100.5, quantity=10.0),),
        depth=1,
    )
    engine.ingest_order_book(book)
    assert not asyncio.run(check_data_integrity(engine, max_staleness=1.0))


def test_check_data_integrity_missing_snapshot():
    engine = DataEngine()
    engine.update_active_instruments(["FIGI"])
    assert not asyncio.run(check_data_integrity(engine))
