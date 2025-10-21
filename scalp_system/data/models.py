"""Domain models for market data."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(slots=True)
class OrderBookLevel:
    price: float
    quantity: float


@dataclass(slots=True)
class OrderBook:
    figi: str
    timestamp: datetime
    bids: tuple[OrderBookLevel, ...]
    asks: tuple[OrderBookLevel, ...]
    depth: int

    def spread(self) -> float:
        return self.asks[0].price - self.bids[0].price

    def mid_price(self) -> float:
        return (self.asks[0].price + self.bids[0].price) / 2


@dataclass(slots=True)
class Trade:
    figi: str
    timestamp: datetime
    price: float
    quantity: float
    direction: Literal["buy", "sell"]


@dataclass(slots=True)
class Candle:
    figi: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


__all__ = ["OrderBookLevel", "OrderBook", "Trade", "Candle"]
