"""Streaming clients for market data."""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import random
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, Deque, Iterable, Optional

from .models import OrderBook, OrderBookLevel
from ..broker.tinkoff import ensure_sdk_available, open_async_client

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from tinkoff.invest import OrderBookInstrument
    from tinkoff.invest.async_services import AsyncMarketDataStreamManager
except ImportError:  # pragma: no cover
    OrderBookInstrument = None  # type: ignore
    AsyncMarketDataStreamManager = None  # type: ignore


class MarketDataStream:
    """High level wrapper over the tinkoff investments market data stream."""

    def __init__(
        self,
        *,
        token: str,
        use_sandbox: bool,
        instruments: Iterable[str],
        depth: int,
        reconnect_backoff: float = 1.0,
    ) -> None:
        ensure_sdk_available()
        self._token = token
        self._use_sandbox = use_sandbox
        self._instruments = list(instruments)
        self._depth = depth
        self._reconnect_backoff = reconnect_backoff
        self._client_cm = None
        self._client = None
        self._stream: Optional[AsyncMarketDataStreamManager] = None

    async def __aenter__(self) -> "MarketDataStream":
        self._client_cm = open_async_client(self._token, use_sandbox=self._use_sandbox)
        self._client = await self._client_cm.__aenter__()
        self._stream = self._client.create_market_data_stream()
        await self._subscribe_order_book()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._stream:
            self._stream.stop()
        if self._client_cm is not None:
            await self._client_cm.__aexit__(exc_type, exc, tb)
        self._client = None
        self._client_cm = None

    async def _subscribe_order_book(self) -> None:
        assert self._stream is not None and OrderBookInstrument is not None
        order_book_instruments = [
            OrderBookInstrument(figi=figi, depth=self._depth) for figi in self._instruments
        ]
        self._stream.order_book.subscribe(order_book_instruments)

    async def order_books(self) -> AsyncIterator[OrderBook]:
        assert self._stream is not None
        async for response in self._stream:
            if response.order_book:
                ob = response.order_book
                yield OrderBook(
                    figi=ob.figi,
                    timestamp=ob.time.replace(tzinfo=timezone.utc),
                    bids=tuple(OrderBookLevel(price=level.price, quantity=level.quantity) for level in ob.bids),
                    asks=tuple(OrderBookLevel(price=level.price, quantity=level.quantity) for level in ob.asks),
                    depth=ob.depth,
                )


class RollingOrderBookBuffer:
    """Maintains a rolling window of order books for feature engineering."""

    def __init__(self, *, maxlen: int) -> None:
        self._buffer: Deque[OrderBook] = deque(maxlen=maxlen)

    def append(self, order_book: OrderBook) -> None:
        self._buffer.append(order_book)

    def window(self) -> tuple[OrderBook, ...]:
        return tuple(self._buffer)

    def is_ready(self) -> bool:
        return len(self._buffer) == self._buffer.maxlen


async def iterate_stream(
    stream_factory,
    *,
    delay: float = 0.0,
    integrity_check: Optional[Callable[[], Awaitable[bool]]] = None,
    on_failure: Optional[Callable[[Exception], Awaitable[None] | None]] = None,
    on_recovery: Optional[Callable[[], Awaitable[None] | None]] = None,
    stop_event: Optional[asyncio.Event] = None,
) -> AsyncIterator[OrderBook]:
    backoff = delay or 1.0
    recovering = False
    while True:
        if stop_event and stop_event.is_set():
            break
        try:
            async with stream_factory() as stream:
                async for order_book in stream.order_books():
                    if stop_event and stop_event.is_set():
                        return
                    backoff = delay or 1.0
                    if recovering and on_recovery is not None:
                        await _maybe_await(on_recovery)
                        recovering = False
                    yield order_book
            break
        except Exception as exc:  # pragma: no cover - defensive branch
            LOGGER.exception("Market data stream error: %s", exc)
            if on_failure is not None:
                await _maybe_await(on_failure, exc)
            if integrity_check is not None:
                try:
                    ok = await integrity_check()
                    if not ok:
                        LOGGER.warning("Integrity check failed during reconnect")
                except Exception:  # pragma: no cover - defensive guard
                    LOGGER.exception("Integrity check raised an exception")
            if stop_event and stop_event.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            recovering = True


async def _maybe_await(callback, *args):
    result = callback(*args)
    if inspect.isawaitable(result):
        await result


class OfflineMarketDataStream:
    """Synthetic stream used when broker tokens are unavailable."""

    def __init__(
        self,
        *,
        instruments: Iterable[str],
        depth: int,
        dataset_path: Path | None = None,
        interval: float = 0.5,
    ) -> None:
        self._instruments = list(instruments) or ["DEMO_FIGI"]
        self._depth = max(1, int(depth))
        self._dataset_path = Path(dataset_path) if dataset_path else None
        self._interval = max(0.0, float(interval))
        self._rng = random.Random(42)
        self._books_cache: Optional[list[OrderBook]] = None

    async def __aenter__(self) -> "OfflineMarketDataStream":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._books_cache = None

    async def order_books(self) -> AsyncIterator[OrderBook]:
        if self._dataset_path and self._dataset_path.exists():
            books = self._load_dataset()
            if books:
                while True:
                    for book in books:
                        yield book
                        if self._interval:
                            await asyncio.sleep(self._interval)
                    if self._interval:
                        await asyncio.sleep(self._interval)
        while True:
            for figi in self._instruments:
                yield self._synthetic_book(figi)
                if self._interval:
                    await asyncio.sleep(self._interval)

    def _load_dataset(self) -> list[OrderBook]:
        if self._books_cache is not None:
            return self._books_cache
        books: list[OrderBook] = []
        if not self._dataset_path or not self._dataset_path.exists():
            self._books_cache = []
            return self._books_cache
        with self._dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                book = self._parse_payload(payload)
                if book is not None:
                    books.append(book)
        self._books_cache = books
        return books

    def _synthetic_book(self, figi: str) -> OrderBook:
        base = 90.0 + (self._rng.random() * 20.0)
        timestamp = datetime.utcnow().replace(tzinfo=timezone.utc)
        tick_size = max(0.01, base * 0.0005)
        bids = []
        asks = []
        mid_price = base * (1 + self._rng.uniform(-0.005, 0.005))
        for level in range(self._depth):
            offset = tick_size * (level + 1)
            quantity = max(1.0, self._rng.uniform(10.0, 50.0))
            bids.append(OrderBookLevel(price=mid_price - offset, quantity=quantity))
            asks.append(OrderBookLevel(price=mid_price + offset, quantity=quantity))
        return OrderBook(
            figi=figi,
            timestamp=timestamp,
            bids=tuple(bids),
            asks=tuple(asks),
            depth=self._depth,
        )

    def _parse_payload(self, payload: dict[str, object]) -> Optional[OrderBook]:
        figi = payload.get("figi")
        bids = payload.get("bids")
        asks = payload.get("asks")
        timestamp_raw = payload.get("timestamp")
        if not figi or not isinstance(figi, str) or not bids or not asks or not timestamp_raw:
            return None
        try:
            timestamp = datetime.fromisoformat(str(timestamp_raw))
        except ValueError:
            return None
        bid_levels = []
        ask_levels = []
        for price, qty in bids:
            try:
                bid_levels.append(
                    OrderBookLevel(price=float(price), quantity=float(qty))
                )
            except (TypeError, ValueError):
                continue
        for price, qty in asks:
            try:
                ask_levels.append(
                    OrderBookLevel(price=float(price), quantity=float(qty))
                )
            except (TypeError, ValueError):
                continue
        if not bid_levels or not ask_levels:
            return None
        depth = min(len(bid_levels), len(ask_levels), self._depth)
        return OrderBook(
            figi=figi,
            timestamp=timestamp.replace(tzinfo=timezone.utc),
            bids=tuple(bid_levels[:depth]),
            asks=tuple(ask_levels[:depth]),
            depth=depth,
        )


__all__ = [
    "MarketDataStream",
    "OfflineMarketDataStream",
    "RollingOrderBookBuffer",
    "iterate_stream",
]
