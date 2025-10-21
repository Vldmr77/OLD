from datetime import datetime, timezone

from datetime import datetime, timezone

from scalp_system.data.engine import DataEngine
from scalp_system.data.models import Candle, OrderBook, OrderBookLevel, Trade


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
    trade = Trade(
        figi="AAA",
        timestamp=datetime.now(timezone.utc),
        price=100.0,
        quantity=1.0,
        direction="buy",
    )
    engine.ingest_trades("AAA", [trade])
    assert engine.recent_trades("AAA")[-1] == trade
    candle = Candle(
        figi="AAA",
        timestamp=datetime.now(timezone.utc),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=10.0,
    )
    engine.ingest_candles("AAA", [candle])
    assert engine.recent_candles("AAA")[-1] == candle


def test_rotation_replaces_inactive_instrument():
    engine = DataEngine(
        ttl_seconds=60.0,
        max_instruments=20,
        history_size=10,
        monitored_instruments=[f"FIGI{i}" for i in range(1, 20)],
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


def test_seed_instruments_enforces_limits():
    monitored = [f"FIGI{i:03d}" for i in range(30)]
    engine = DataEngine(
        ttl_seconds=60.0,
        max_instruments=50,
        history_size=10,
        max_active_instruments=15,
        monitor_pool_size=30,
    )
    engine.seed_instruments(monitored[:10], monitored)
    assert len(engine.active_instruments()) == 15
    assert len(engine.monitored_instruments()) == 30
    assert set(engine.active_instruments()).issubset(set(monitored))
