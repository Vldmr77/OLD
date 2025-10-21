"""Helpers around the tinkoff-investments SDK."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional

try:  # pragma: no cover - runtime import
    from tinkoff.invest import AsyncClient
except ImportError:  # pragma: no cover - handled via ensure_sdk_available
    AsyncClient = None  # type: ignore


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


@dataclass
class TinkoffAPI:
    """High-level convenience facade around the tinkoff-investments SDK."""

    token: str
    use_sandbox: bool = True

    @asynccontextmanager
    async def client(self) -> AsyncIterator["AsyncClient"]:
        """Yield a configured ``AsyncClient`` instance."""

        async with open_async_client(self.token, use_sandbox=self.use_sandbox) as client:
            yield client

    async def ping(self) -> None:
        """Perform a lightweight call to verify connectivity."""

        async with self.client() as client:
            await client.users.get_accounts()

    async def fetch_order_book(self, figi: str, depth: int = 10) -> dict:
        """Retrieve an order book snapshot for UI previews or diagnostics."""

        async with self.client() as client:
            response = await client.market_data.get_order_book(figi=figi, depth=depth)
        return {
            "figi": response.figi,
            "depth": response.depth,
            "last_price": float(response.last_price or 0),
            "bids": [
                {"price": float(level.price), "quantity": level.quantity} for level in response.bids
            ],
            "asks": [
                {"price": float(level.price), "quantity": level.quantity} for level in response.asks
            ],
        }


__all__ = [
    "TinkoffAPI",
    "TinkoffSDKUnavailable",
    "ensure_sdk_available",
    "open_async_client",
]
