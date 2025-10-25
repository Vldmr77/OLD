"""Helpers around the tinkoff-investments SDK."""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterator, Optional

from ..data.models import Candle, OrderBook, OrderBookLevel, Trade

try:  # pragma: no cover - fallback for optional SDK constants
    from tinkoff.invest.constants import (
        INVEST_GRPC_API,
        INVEST_GRPC_API_SANDBOX,
    )
except ImportError:  # pragma: no cover
    INVEST_GRPC_API = "invest-public-api.tinkoff.ru"
    INVEST_GRPC_API_SANDBOX = "sandbox-invest-public-api.tinkoff.ru"

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

@dataclass
class BrokerConnectionOptions:
    """Connection preferences for reaching the Tinkoff gRPC endpoints."""

    mode: str = "auto"
    proxy_url: Optional[str] = None
    no_proxy: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        mode = (self.mode or "auto").strip().lower()
        if mode not in {"auto", "direct", "proxy"}:
            mode = "auto"
        self.mode = mode
        proxy = (self.proxy_url or "").strip()
        self.proxy_url = proxy or None
        if isinstance(self.no_proxy, (list, set)):
            hosts = [str(item).strip() for item in self.no_proxy if str(item).strip()]
        elif isinstance(self.no_proxy, tuple):
            hosts = [str(item).strip() for item in self.no_proxy if str(item).strip()]
        elif self.no_proxy:
            hosts = [str(self.no_proxy).strip()]
        else:
            hosts = []
        self.no_proxy = tuple(dict.fromkeys(hosts))


@contextmanager
def _grpc_environment(
    target_host: str, options: Optional[BrokerConnectionOptions]
) -> Iterator[None]:
    """Temporarily adjust environment variables for gRPC connections."""

    saved: dict[str, Optional[str]] = {}

    def _set_env(key: str, value: Optional[str]) -> None:
        saved.setdefault(key, os.environ.get(key))
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    cert_path = (
        os.environ.get("REQUESTS_CA_BUNDLE")
        or os.environ.get("SSL_CERT_FILE")
        or os.environ.get("PIP_CERT")
        or os.environ.get("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH")
    )
    if cert_path and os.path.exists(cert_path):
        _set_env("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", cert_path)

    no_proxy_entries = [target_host, f"{target_host}:443"]
    if options:
        no_proxy_entries.extend(options.no_proxy)
    existing_entries = [
        entry.strip()
        for entry in os.environ.get("grpc.no_proxy", "").split(",")
        if entry.strip()
    ]
    combined_entries = list(dict.fromkeys(existing_entries + no_proxy_entries))
    if combined_entries:
        _set_env("grpc.no_proxy", ",".join(combined_entries))

    mode = options.mode if options else "auto"

    std_proxy_keys = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
    grpc_proxy_keys = ("grpc.http_proxy", "grpc.https_proxy", "grpc.HTTP_PROXY", "grpc.HTTPS_PROXY")

    if mode == "direct":
        for key in (*std_proxy_keys, *grpc_proxy_keys):
            _set_env(key, None)
        for key in ("no_proxy", "NO_PROXY"):
            existing = [
                entry.strip()
                for entry in os.environ.get(key, "").split(",")
                if entry.strip()
            ]
            combined = list(dict.fromkeys(existing + no_proxy_entries))
            if combined:
                _set_env(key, ",".join(combined))
            else:
                _set_env(key, None)
        _set_env("GRPC_PROXY_EXP", None)
    else:
        proxy_value = options.proxy_url if options and options.proxy_url else None
        if proxy_value:
            for key in grpc_proxy_keys:
                _set_env(key, proxy_value)
        else:
            for std_key, grpc_key in zip(std_proxy_keys, grpc_proxy_keys):
                if std_key in os.environ:
                    _set_env(grpc_key, os.environ[std_key])

    try:
        yield None
    finally:
        for key, original in saved.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


@asynccontextmanager
async def open_async_client(
    token: str,
    *,
    use_sandbox: bool,
    connection_options: Optional[BrokerConnectionOptions] = None,
) -> AsyncIterator[Any]:
    """Yield an authenticated asynchronous client from the SDK.

    The upstream ``tinkoff-investments`` package exposes ``AsyncClient`` purely as a
    context manager that returns ``AsyncServices`` from ``__aenter__`` and does not
    implement a ``close`` coroutine. Our original implementation attempted to manage
    the lifecycle manually and called ``await client.close()``, which raises an
    ``AttributeError`` on shutdown. To mirror the semantics of ``async with
    AsyncClient(...)`` we simply delegate to the SDK's context manager and yield the
    opened services instance.
    """

    ensure_sdk_available()
    assert AsyncClient is not None  # for type-checkers
    target_host = INVEST_GRPC_API_SANDBOX if use_sandbox else INVEST_GRPC_API
    target = f"{target_host}:443"
    options = []
    if connection_options and connection_options.mode == "direct":
        options.append(("grpc.enable_http_proxy", 0))
    elif connection_options and connection_options.proxy_url:
        options.append(("grpc.enable_http_proxy", 1))

    with _grpc_environment(target_host, connection_options):
        kwargs: dict[str, Any] = {"target": target}
        if options:
            kwargs["options"] = options
        client = AsyncClient(token, **kwargs)
        if hasattr(client, "__aenter__") and hasattr(client, "__aexit__"):
            async with client as services:
                yield services
        else:  # pragma: no cover - fallback for mocked clients without context manager
            try:
                yield client
            finally:
                close = getattr(client, "close", None)
                if close is not None:
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result


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
    connection_mode: str = "auto"
    proxy_url: Optional[str] = None
    no_proxy: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self._limiter = AsyncRateLimiter(self.rate_limit_per_minute)
        self._connection_options = BrokerConnectionOptions(
            mode=self.connection_mode,
            proxy_url=self.proxy_url,
            no_proxy=self.no_proxy,
        )

    @asynccontextmanager
    async def client(self) -> AsyncIterator[Any]:
        """Yield a configured ``AsyncClient`` instance."""

        async with open_async_client(
            self.token,
            use_sandbox=self.use_sandbox,
            connection_options=self._connection_options,
        ) as client:
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
    "BrokerConnectionOptions",
    "AsyncRateLimiter",
    "TinkoffAPI",
    "TinkoffSDKUnavailable",
    "ensure_sdk_available",
    "open_async_client",
]
