"""Helpers around the tinkoff-investments SDK."""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from ..data.models import Candle, OrderBook, OrderBookLevel, Trade

try:  # pragma: no cover - runtime import
    from tinkoff.invest import AsyncClient, CandleInterval
except ImportError:  # pragma: no cover - handled via ensure_sdk_available
    AsyncClient = None  # type: ignore
    CandleInterval = None  # type: ignore


class TinkoffSDKUnavailable(RuntimeError):
    """Raised when the tinkoff-investments SDK is not importable."""


def ensure_sdk_available() -> None:
    """Ensure the tinkoff-investments SDK is present."""

    if AsyncClient is None:
        raise TinkoffSDKUnavailable(
            "tinkoff-investments SDK is required. Install the bundled wheel via "
            "`pip install tinkoff_investments-0.2.0b117-py3-none-any.whl`."
        )


@asynccontextmanager
async def open_async_client(token: str, *, use_sandbox: bool) -> AsyncIterator["AsyncClient"]:
    """Context manager that yields an authenticated ``AsyncClient`` instance."""

    ensure_sdk_available()
    assert AsyncClient is not None  # for type-checkers
    target: Optional[str] = "sandbox" if use_sandbox else None
    client = AsyncClient(token, target=target)
    try:
        yield client
    finally:
        await client.close()


class AsyncRateLimiter:
    """Asynchronous token bucket rate limiter."""

    def __init__(self, rate_per_minute: int) -> None:
        self._interval = 60.0 / max(1, rate_per_minute)
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


@dataclass
class TinkoffAPI:
    """High-level convenience facade around the tinkoff-investments SDK."""

    token: str
    use_sandbox: bool = True
    account_id: Optional[str] = None
    rate_limit_per_minute: int = 120

    def __post_init__(self) -> None:
        self._limiter = AsyncRateLimiter(self.rate_limit_per_minute)

    @asynccontextmanager
    async def client(self) -> AsyncIterator["AsyncClient"]:
        """Yield a configured ``AsyncClient`` instance."""

        async with open_async_client(self.token, use_sandbox=self.use_sandbox) as client:
            yield client

    async def ping(self) -> None:
        """Perform a lightweight call to verify connectivity."""

        async with self.client() as client:
            await self._limiter.acquire()
            await client.users.get_accounts()

    async def fetch_order_book(self, figi: str, depth: int = 10) -> OrderBook:
        """Retrieve an order book snapshot."""

        async with self.client() as client:
            await self._limiter.acquire()
            response = await client.market_data.get_order_book(figi=figi, depth=depth)
        bids = tuple(
            OrderBookLevel(price=_to_float(level.price), quantity=_to_float(level.quantity))
            for level in getattr(response, "bids", [])
        )
        asks = tuple(
            OrderBookLevel(price=_to_float(level.price), quantity=_to_float(level.quantity))
            for level in getattr(response, "asks", [])
        )
        return OrderBook(
            figi=response.figi,
            timestamp=_ensure_timestamp(getattr(response, "time", None)),
            bids=bids,
            asks=asks,
            depth=response.depth,
        )

    async def fetch_trades(self, figi: str, limit: int = 100) -> list[Trade]:
        """Fetch the most recent trades for a FIGI."""

        async with self.client() as client:
            await self._limiter.acquire()
            response = await client.market_data.get_last_trades(figi=figi)
        trades = []
        for trade in getattr(response, "trades", [])[-limit:]:
            direction_value = str(getattr(trade.direction, "name", trade.direction)).lower()
            direction = "buy" if "buy" in direction_value else "sell"
            trades.append(
                Trade(
                    figi=figi,
                    timestamp=_ensure_timestamp(getattr(trade, "time", None)),
                    price=_to_float(getattr(trade, "price", 0.0)),
                    quantity=_to_float(getattr(trade, "quantity", 0.0)),
                    direction=direction,
                )
            )
        return trades

    async def fetch_candles(
        self,
        figi: str,
        from_time: datetime,
        to_time: datetime,
        interval: str,
    ) -> list[Candle]:
        """Load historical candles for the provided time window."""

        async with self.client() as client:
            await self._limiter.acquire()
            response = await client.market_data.get_candles(
                figi=figi,
                from_=from_time,
                to=to_time,
                interval=self._resolve_candle_interval(interval),
            )
        candles = []
        for item in getattr(response, "candles", []):
            candles.append(
                Candle(
                    figi=figi,
                    timestamp=_ensure_timestamp(getattr(item, "time", None)),
                    open=_to_float(getattr(item, "open", 0.0)),
                    high=_to_float(getattr(item, "high", 0.0)),
                    low=_to_float(getattr(item, "low", 0.0)),
                    close=_to_float(getattr(item, "close", 0.0)),
                    volume=float(getattr(item, "volume", 0.0)),
                )
            )
        return candles

    async def fetch_portfolio(self) -> dict:
        """Retrieve the current portfolio state if an account is configured."""

        if not self.account_id:
            return {}
        async with self.client() as client:
            await self._limiter.acquire()
            response = await client.operations.get_portfolio(account_id=self.account_id)
        positions = []
        for position in getattr(response, "positions", []):
            positions.append(
                {
                    "figi": getattr(position, "figi", None),
                    "quantity": _to_float(getattr(position, "quantity", 0.0)),
                    "average_price": _to_float(
                        getattr(position, "average_position_price", 0.0)
                    ),
                }
            )
        return {
            "total_amount": _to_float(
                getattr(response, "total_amount_portfolio", 0.0)
            ),
            "positions": positions,
        }

    def _resolve_candle_interval(self, interval: str):
        if CandleInterval is None:
            return interval
        mapping = {
            "1min": CandleInterval.CANDLE_INTERVAL_1_MIN,
            "5min": CandleInterval.CANDLE_INTERVAL_5_MIN,
            "15min": CandleInterval.CANDLE_INTERVAL_15_MIN,
            "hour": CandleInterval.CANDLE_INTERVAL_HOUR,
            "day": CandleInterval.CANDLE_INTERVAL_DAY,
        }
        return mapping.get(interval.lower(), CandleInterval.CANDLE_INTERVAL_1_MIN)


def _to_float(value) -> float:
    if value is None:
        return 0.0
    if hasattr(value, "units") and hasattr(value, "nano"):
        return float(value.units) + float(value.nano) / 1_000_000_000
    return float(value)


def _ensure_timestamp(value) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


__all__ = [
    "AsyncRateLimiter",
    "TinkoffAPI",
    "TinkoffSDKUnavailable",
    "ensure_sdk_available",
    "open_async_client",
]
