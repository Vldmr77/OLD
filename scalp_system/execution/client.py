"""Broker client abstraction."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from tinkoff.invest import AsyncClient, OrderDirection, OrderType
except ImportError:  # pragma: no cover
    AsyncClient = None  # type: ignore
    OrderDirection = None  # type: ignore
    OrderType = None  # type: ignore


@dataclass
class OrderRequest:
    figi: str
    quantity: int
    price: float
    direction: "OrderDirection"
    order_type: "OrderType"
    account_id: Optional[str] = None


class BrokerClient:
    def __init__(self, token: str, *, sandbox: bool, account_id: Optional[str] = None) -> None:
        if AsyncClient is None:
            raise RuntimeError("tinkoff-investments SDK is required for order execution")
        self._token = token
        self._sandbox = sandbox
        self._account_id = account_id
        self._client: Optional[AsyncClient] = None

    async def __aenter__(self) -> "BrokerClient":
        assert AsyncClient is not None
        target = "sandbox" if self._sandbox else None
        self._client = AsyncClient(self._token, target=target)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client:
            await self._client.close()

    async def place_order(self, order: OrderRequest) -> None:
        if not self._client:
            raise RuntimeError("BrokerClient is not connected")
        LOGGER.info(
            "Placing order figi=%s qty=%s direction=%s", order.figi, order.quantity, order.direction.name
        )
        await self._client.orders.post_order(
            figi=order.figi,
            quantity=order.quantity,
            price=order.price,
            direction=order.direction,
            order_type=order.order_type,
            account_id=order.account_id or self._account_id,
        )


__all__ = ["BrokerClient", "OrderRequest"]
