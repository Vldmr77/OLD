from datetime import datetime, timezone

from scalp_system.data.engine import DataEngine
from scalp_system.data.models import OrderBook, OrderBookLevel


def make_order_book(figi: str, price: float, spread: float = 0.05) -> OrderBook:
    bids = tuple(
        OrderBookLevel(price=price - i * spread, quantity=100 - i) for i in range(1, 6)
    )
    asks = tuple(
        OrderBookLevel(price=price + i * spread, quantity=100 - i) for i in range(1, 6)
    )
    return OrderBook(
        figi=figi,
        timestamp=datetime.now(timezone.utc),
        bids=bids,
        asks=asks,
        depth=10,
    )


def test_data_engine_caches_and_history():
    engine = DataEngine(ttl_seconds=1.0, max_instruments=10, history_size=5)
    engine.update_active_instruments(["AAA"])
    for i in range(5):
        engine.ingest_order_book(make_order_book("AAA", 100 + i * 0.01))
    assert len(engine.history("AAA")) == 5
    current = engine.get_order_book("AAA")
    assert current is not None
    assert current.figi == "AAA"


def test_rotation_replaces_inactive_instrument():
    engine = DataEngine(
        ttl_seconds=60.0,
        max_instruments=5,
        history_size=10,
        monitored_instruments=["BBB", "CCC"],
    )
    engine.update_active_instruments(["AAA"])
    for _ in range(6):
        book = make_order_book("AAA", 100.0, spread=0.2)
        engine.ingest_order_book(book)
    replacements = engine.rotate_instruments()
    assert replacements
    assert replacements[0][0] == "AAA"


def test_resynchronise_clears_caches():
    engine = DataEngine(ttl_seconds=60.0, max_instruments=5, history_size=10)
    engine.update_active_instruments(["AAA"])
    engine.ingest_order_book(make_order_book("AAA", 100.0))
    assert engine.get_order_book("AAA") is not None
    assert engine.history("AAA")
    engine.resynchronise()
    assert engine.get_order_book("AAA") is None
    assert engine.history("AAA") == ()
