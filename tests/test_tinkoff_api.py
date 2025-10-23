import asyncio
from datetime import datetime, timezone, timedelta

import pytest

from scalp_system.broker import tinkoff as tinkoff_module
from scalp_system.data.models import OrderBook


class DummyClient:
    def __init__(self, token: str, target=None) -> None:
        self.token = token
        self.target = target
        self.closed = False
        self.users = self.Users()
        self.market_data = self.MarketData()
        self.operations = self.Operations()

    async def close(self) -> None:
        self.closed = True

    class Users:  # type: ignore[no-redef]
        async def get_accounts(self) -> None:
            return None

    class MarketData:  # type: ignore[no-redef]
        async def get_order_book(self, figi: str, depth: int):
            class Level:
                def __init__(self, price: float, quantity: float) -> None:
                    self.price = price
                    self.quantity = quantity

            class Response:
                def __init__(self) -> None:
                    self.figi = figi
                    self.depth = depth
                    self.time = datetime.now(timezone.utc)
                    self.bids = [Level(100.0, 5.0)]
                    self.asks = [Level(100.1, 5.0)]

            return Response()

        async def get_last_trades(self, figi: str):
            class Direction:
                def __init__(self, name: str) -> None:
                    self.name = name

            class Trade:
                def __init__(self) -> None:
                    self.direction = Direction("BUY")
                    self.time = datetime.now(timezone.utc)
                    self.price = 101.0
                    self.quantity = 1

            class Response:
                def __init__(self) -> None:
                    self.trades = [Trade()]

            return Response()

        async def get_candles(self, figi: str, from_, to, interval):
            class Candle:
                def __init__(self) -> None:
                    self.time = datetime.now(timezone.utc)
                    self.open = 99.0
                    self.high = 102.0
                    self.low = 98.0
                    self.close = 100.5
                    self.volume = 10

            class Response:
                def __init__(self) -> None:
                    self.candles = [Candle()]

            return Response()

    class Operations:  # type: ignore[no-redef]
        async def get_portfolio(self, account_id: str):
            class Position:
                def __init__(self) -> None:
                    self.figi = "FIGI1"
                    self.quantity = 2
                    self.average_position_price = 100.0

            class Response:
                def __init__(self) -> None:
                    self.positions = [Position()]
                    self.total_amount_portfolio = 200.0

            return Response()


def test_ensure_sdk_available_raises(monkeypatch):
    monkeypatch.setattr(tinkoff_module, "AsyncClient", None)
    with pytest.raises(tinkoff_module.TinkoffSDKUnavailable):
        tinkoff_module.ensure_sdk_available()


def test_open_async_client(monkeypatch):
    monkeypatch.setattr(tinkoff_module, "AsyncClient", DummyClient)

    async def run():
        async with tinkoff_module.open_async_client("TOKEN", use_sandbox=True) as client:
            assert isinstance(client, DummyClient)
            assert client.token == "TOKEN"
            assert client.target.endswith("sandbox-invest-public-api.tinkoff.ru:443")

    asyncio.run(run())


def test_tinkoff_api_fetch(monkeypatch):
    monkeypatch.setattr(tinkoff_module, "AsyncClient", DummyClient)
    api = tinkoff_module.TinkoffAPI(
        "TOKEN", account_id="ACC", rate_limit_per_minute=100_000
    )

    async def run():
        order_book = await api.fetch_order_book("BBG000000001", depth=20)
        assert isinstance(order_book, OrderBook)
        assert order_book.figi == "BBG000000001"
        trades = await api.fetch_trades("BBG000000001", limit=10)
        assert trades and trades[0].figi == "BBG000000001"
        candles = await api.fetch_candles(
            "BBG000000001",
            datetime.now(timezone.utc) - timedelta(minutes=5),
            datetime.now(timezone.utc),
            "1min",
        )
        assert candles and candles[0].figi == "BBG000000001"
        portfolio = await api.fetch_portfolio()
        assert portfolio["positions"][0]["figi"] == "FIGI1"

    asyncio.run(run())
