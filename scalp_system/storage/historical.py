"""Historical storage utilities covering order books, trades, candles, and features."""
from __future__ import annotations

import csv
import json
import logging
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from ..data.models import Candle, OrderBook, Trade
from ..features.pipeline import FeatureVector

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import zstandard as zstd  # type: ignore
except ImportError:  # pragma: no cover
    zstd = None  # type: ignore


def _format_float(value: float, precision: int) -> str:
    text = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    return text or "0"


@dataclass
class _ParquetSink:
    """Utility that writes records to parquet when possible, otherwise JSONL."""

    compression: str = "snappy"

    def __post_init__(self) -> None:
        try:  # pragma: no cover - optional dependency
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except ImportError:  # pragma: no cover
            self._pa = None
            self._pq = None
        else:  # pragma: no cover - exercised only when pyarrow available
            self._pa = pa
            self._pq = pq

    @property
    def uses_parquet(self) -> bool:
        return getattr(self, "_pa", None) is not None and getattr(self, "_pq", None) is not None

    def write(self, directory: Path, stem: str, records: Sequence[dict]) -> Path:
        if not records:
            raise ValueError("records must not be empty")
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        if self.uses_parquet:
            target = directory / f"{stem}_{timestamp}.parquet"
            table = self._pa.Table.from_pylist(list(records))  # type: ignore[attr-defined]
            self._pq.write_table(table, target, compression=self.compression)  # type: ignore[attr-defined]
            return target
        target = directory / f"{stem}_{timestamp}.jsonl"
        with target.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return target


class HistoricalDataStorage:
    """Persists market data artefacts in the formats required by the specification."""

    def __init__(
        self,
        base_path: Path,
        *,
        orderbook_path: Path,
        trades_path: Path,
        candles_path: Path,
        features_path: Path,
        enable_zstd: bool = True,
        parquet_compression: str = "snappy",
    ) -> None:
        self._orderbook_path = orderbook_path
        self._trades_path = trades_path
        self._candles_path = candles_path
        self._features_path = features_path
        self._use_zstd = enable_zstd and zstd is not None
        self._parquet_sink = _ParquetSink(compression=parquet_compression)
        for directory in [
            self._orderbook_path.parent,
            self._trades_path,
            self._candles_path,
            self._features_path,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        self._initialise_orderbook_table()
        if not self._parquet_sink.uses_parquet:
            LOGGER.warning(
                "pyarrow not installed; candles and features will be stored as JSONL backups"
            )
        if enable_zstd and zstd is None:
            LOGGER.warning("zstandard not installed; order book payloads stored without compression")

    # ------------------------------------------------------------------
    # Order books
    # ------------------------------------------------------------------
    def store_order_book(self, order_book: OrderBook) -> None:
        payload = {
            "bids": [[level.price, level.quantity] for level in order_book.bids],
            "asks": [[level.price, level.quantity] for level in order_book.asks],
        }
        blob = json.dumps(payload).encode("utf-8")
        compressed = 0
        if self._use_zstd and blob:
            compressor = zstd.ZstdCompressor(level=3)  # type: ignore[attr-defined]
            blob = compressor.compress(blob)
            compressed = 1
        with sqlite3.connect(self._orderbook_path) as conn:
            conn.execute(
                """
                INSERT INTO l2(figi, timestamp, depth, payload, compressed)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    order_book.figi,
                    order_book.timestamp.isoformat(),
                    order_book.depth,
                    sqlite3.Binary(blob),
                    compressed,
                ),
            )
            conn.commit()

    def _initialise_orderbook_table(self) -> None:
        with sqlite3.connect(self._orderbook_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS l2 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    figi TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    depth INTEGER NOT NULL,
                    payload BLOB NOT NULL,
                    compressed INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_l2_figi_ts ON l2(figi, timestamp)"
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------
    def store_trades(self, trades: Iterable[Trade]) -> None:
        buckets: dict[str, List[Trade]] = defaultdict(list)
        for trade in trades:
            buckets[trade.figi].append(trade)
        for figi, entries in buckets.items():
            path = self._trades_path / f"{figi}.csv"
            new_file = not path.exists()
            with path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                if new_file:
                    writer.writerow(["timestamp", "price", "quantity", "direction"])
                for trade in entries:
                    writer.writerow(
                        [
                            trade.timestamp.isoformat(),
                            _format_float(trade.price, 10),
                            _format_float(trade.quantity, 8),
                            trade.direction,
                        ]
                    )

    # ------------------------------------------------------------------
    # Candles & features (Parquet / JSONL fallback)
    # ------------------------------------------------------------------
    def store_candles(self, candles: Iterable[Candle]) -> None:
        buckets: dict[str, List[dict]] = defaultdict(list)
        for candle in candles:
            buckets[candle.figi].append(
                {
                    "figi": candle.figi,
                    "timestamp": candle.timestamp.isoformat(),
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                }
            )
        for figi, records in buckets.items():
            self._parquet_sink.write(self._candles_path / figi, figi, records)

    def store_features(self, vector: FeatureVector) -> None:
        record = {
            "figi": vector.figi,
            "timestamp": vector.timestamp,
        }
        for idx, value in enumerate(vector.features):
            record[f"feature_{idx:02d}"] = float(value)
        self._parquet_sink.write(self._features_path / vector.figi, vector.figi, [record])


__all__ = ["HistoricalDataStorage"]
