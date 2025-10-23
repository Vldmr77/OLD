"""Broker client abstraction."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from tinkoff.invest import OrderDirection, OrderType
except ImportError:  # pragma: no cover
    OrderDirection = None  # type: ignore
    OrderType = None  # type: ignore

from ..broker.tinkoff import (
    AsyncRateLimiter,
    BrokerConnectionOptions,
    ensure_sdk_available,
    open_async_client,
)


@dataclass
class OrderRequest:
    figi: str
    quantity: int
    price: float
    direction: "OrderDirection"
    order_type: "OrderType"
    account_id: Optional[str] = None


class BrokerClient:
    def __init__(
        self,
        token: str,
        *,
        sandbox: bool,
        account_id: Optional[str] = None,
        orders_per_minute: int = 60,
        connection_options: BrokerConnectionOptions | None = None,
    ) -> None:
        ensure_sdk_available()
        self._token = token
        self._sandbox = sandbox
        self._account_id = account_id
        self._client_cm = None
        self._client = None
        self._limiter = AsyncRateLimiter(max(1, orders_per_minute))
        self._connection_options = connection_options

    async def __aenter__(self) -> "BrokerClient":
        self._client_cm = open_async_client(
            self._token,
            use_sandbox=self._sandbox,
            connection_options=self._connection_options,
        )
        self._client = await self._client_cm.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client_cm is not None:
            await self._client_cm.__aexit__(exc_type, exc, tb)
        self._client = None
        self._client_cm = None

    async def place_order(self, order: OrderRequest) -> None:
        if not self._client:
            raise RuntimeError("BrokerClient is not connected")
        LOGGER.info(
            "Placing order figi=%s qty=%s direction=%s", order.figi, order.quantity, order.direction.name
        )
        await self._limiter.acquire()
        await self._client.orders.post_order(
            figi=order.figi,
            quantity=order.quantity,
            price=order.price,
            direction=order.direction,
            order_type=order.order_type,
            account_id=order.account_id or self._account_id,
        )

    async def cancel_all_orders(self) -> int:
        if not self._client:
            raise RuntimeError("BrokerClient is not connected")
        LOGGER.info("Cancelling all outstanding orders")
        await self._limiter.acquire()
        orders_api = getattr(self._client, "orders", None)
        if orders_api is None:
            raise RuntimeError("Orders API unavailable on client")
        cancel = getattr(orders_api, "post_cancel_all_orders", None)
        if cancel is None:
            cancel = getattr(orders_api, "cancel_all_orders", None)
        if cancel is None:
            raise RuntimeError("Cancel-all endpoint is not available in the SDK")
        response = await cancel(account_id=self._account_id)
        count = getattr(response, "orders_cancelled", None)
        if count is None:
            count = getattr(response, "cancelled_orders_count", None)
        try:
            return int(count or 0)
        except (TypeError, ValueError):
            return 0


__all__ = ["BrokerClient", "OrderRequest"]
