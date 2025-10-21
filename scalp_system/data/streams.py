"""Streaming clients for market data."""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from typing import AsyncIterator, Awaitable, Callable, Deque, Iterable, Optional

from .models import OrderBook, OrderBookLevel

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from tinkoff.invest import AsyncClient, OrderBookInstrument, SubscribeOrderBookRequest
    from tinkoff.invest.async_services import AsyncMarketDataStreamService
except ImportError:  # pragma: no cover
    AsyncClient = None  # type: ignore
    OrderBookInstrument = None  # type: ignore
    SubscribeOrderBookRequest = None  # type: ignore
    AsyncMarketDataStreamService = None  # type: ignore


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
        if AsyncClient is None:
            raise RuntimeError("tinkoff-investments SDK is required for market data streaming")
        self._token = token
        self._use_sandbox = use_sandbox
        self._instruments = list(instruments)
        self._depth = depth
        self._reconnect_backoff = reconnect_backoff
        self._client: Optional[AsyncClient] = None
        self._stream: Optional[AsyncMarketDataStreamService] = None

    async def __aenter__(self) -> "MarketDataStream":
        assert AsyncClient is not None
        target = "sandbox" if self._use_sandbox else None
        self._client = AsyncClient(self._token, target=target)
        self._stream = self._client.create_market_data_stream()
        await self._subscribe_order_book()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._stream:
            await self._stream.close()
        if self._client:
            await self._client.close()

    async def _subscribe_order_book(self) -> None:
        assert self._stream is not None and OrderBookInstrument is not None and SubscribeOrderBookRequest is not None
        order_book_instruments = [
            OrderBookInstrument(figi=figi, depth=self._depth) for figi in self._instruments
        ]
        request = SubscribeOrderBookRequest(
            subscription_action=SubscribeOrderBookRequest.SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE,
            instruments=order_book_instruments,
        )
        await self._stream.order_book_subscribe(request)

    async def order_books(self) -> AsyncIterator[OrderBook]:
        assert self._stream is not None
        while True:
            try:
                response = await self._stream.__anext__()
            except StopAsyncIteration:
                LOGGER.info("Market data stream finished")
                return
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
) -> AsyncIterator[OrderBook]:
    backoff = delay or 1.0
    while True:
        try:
            async with stream_factory() as stream:
                async for order_book in stream.order_books():
                    backoff = delay or 1.0
                    yield order_book
            break
        except Exception as exc:  # pragma: no cover - defensive branch
            LOGGER.exception("Market data stream error: %s", exc)
            if integrity_check is not None:
                try:
                    ok = await integrity_check()
                    if not ok:
                        LOGGER.warning("Integrity check failed during reconnect")
                except Exception:  # pragma: no cover - defensive guard
                    LOGGER.exception("Integrity check raised an exception")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)


__all__ = ["MarketDataStream", "RollingOrderBookBuffer", "iterate_stream"]
