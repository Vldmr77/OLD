import asyncio

import pytest

from scalp_system.broker import tinkoff as tinkoff_module


class DummyClient:
    def __init__(self, token: str, target=None) -> None:
        self.token = token
        self.target = target
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    class users:  # type: ignore[no-redef]
        @staticmethod
        async def get_accounts() -> None:
            return None

    class market_data:  # type: ignore[no-redef]
        @staticmethod
        async def get_order_book(figi: str, depth: int):
            class Response:
                def __init__(self) -> None:
                    self.figi = figi
                    self.depth = depth
                    self.last_price = 123.45
                    self.bids = []
                    self.asks = []

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
            assert client.target == "sandbox"

    asyncio.run(run())


def test_tinkoff_api_fetch(monkeypatch):
    monkeypatch.setattr(tinkoff_module, "AsyncClient", DummyClient)
    api = tinkoff_module.TinkoffAPI("TOKEN")

    async def run():
        snapshot = await api.fetch_order_book("BBG000000001", depth=20)
        assert snapshot["figi"] == "BBG000000001"
        assert snapshot["depth"] == 20

    asyncio.run(run())
