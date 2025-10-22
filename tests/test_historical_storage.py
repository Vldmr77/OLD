import json
import sqlite3
from datetime import datetime, timezone

from scalp_system.data.models import Candle, OrderBook, OrderBookLevel, Trade
from scalp_system.features.pipeline import FeatureVector
from scalp_system.storage.historical import HistoricalDataStorage


def _make_storage(tmp_path):
    return HistoricalDataStorage(
        tmp_path,
        orderbook_path=tmp_path / "orderbooks.db",
        trades_path=tmp_path / "trades",
        candles_path=tmp_path / "candles",
        features_path=tmp_path / "features",
        enable_zstd=False,
        parquet_compression="snappy",
    )


def test_store_order_book_and_trades(tmp_path):
    storage = _make_storage(tmp_path)
    book = OrderBook(
        figi="FIGI1",
        timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
        bids=(OrderBookLevel(price=100.0, quantity=5.0),),
        asks=(OrderBookLevel(price=101.0, quantity=3.0),),
        depth=20,
    )
    storage.store_order_book(book)
    with sqlite3.connect(tmp_path / "orderbooks.db") as conn:
        row = conn.execute("SELECT figi, depth, compressed, payload FROM l2").fetchone()
    assert row[0] == "FIGI1"
    assert row[1] == 20
    assert row[2] == 0  # compression disabled in test
    payload = json.loads(row[3].decode("utf-8"))
    assert payload["bids"][0] == [100.0, 5.0]
    assert payload["asks"][0] == [101.0, 3.0]

    trade = Trade(
        figi="FIGI1",
        timestamp=datetime(2023, 1, 1, 12, tzinfo=timezone.utc),
        price=101.2345,
        quantity=7.5,
        direction="buy",
    )
    storage.store_trades([trade])
    trade_file = tmp_path / "trades" / "FIGI1.csv"
    lines = trade_file.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "timestamp,price,quantity,direction"
    assert "buy" in lines[1]


def test_store_candles_and_features(tmp_path):
    storage = _make_storage(tmp_path)
    candle = Candle(
        figi="FIGI2",
        timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
        open=99.0,
        high=105.0,
        low=98.0,
        close=100.5,
        volume=1500.0,
    )
    storage.store_candles([candle])
    candle_files = list((tmp_path / "candles" / "FIGI2").glob("*"))
    assert candle_files, "candle file must be created"
    candle_file = candle_files[0]
    if candle_file.suffix == ".jsonl":
        content = candle_file.read_text(encoding="utf-8").strip().splitlines()
        record = json.loads(content[0])
        assert record["close"] == 100.5
    else:  # pragma: no cover - exercised when pyarrow is installed
        assert candle_file.suffix == ".parquet"
        assert candle_file.stat().st_size > 0

    vector = FeatureVector(figi="FIGI2", timestamp=1672617600.0, features=[0.1, 0.2, 0.3])
    storage.store_features(vector)
    feature_files = list((tmp_path / "features" / "FIGI2").glob("*"))
    assert feature_files, "feature file must be created"
    feature_file = feature_files[0]
    if feature_file.suffix == ".jsonl":
        feature_record = json.loads(feature_file.read_text(encoding="utf-8").strip())
        assert feature_record["feature_00"] == 0.1
        assert feature_record["timestamp"] == 1672617600.0
    else:  # pragma: no cover - exercised when pyarrow is installed
        assert feature_file.suffix == ".parquet"
        assert feature_file.stat().st_size > 0
