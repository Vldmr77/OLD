import asyncio

from scalp_system.data.streams import OfflineMarketDataStream


def test_offline_stream_generates_synthetic_books():
    async def runner():
        stream = OfflineMarketDataStream(instruments=["FIGI1"], depth=3, interval=0)
        async with stream as ctx:
            books = []
            async for book in ctx.order_books():
                books.append(book)
                if len(books) >= 2:
                    break
        return books

    books = asyncio.run(runner())
    assert all(book.figi == "FIGI1" for book in books)
    assert all(len(book.bids) == 3 and len(book.asks) == 3 for book in books)


def test_offline_stream_replays_dataset(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        """
        {"figi": "FIGI2", "timestamp": "2023-01-01T00:00:00", "bids": [[100.0, 10.0]], "asks": [[101.0, 9.0]]}
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    async def runner():
        stream = OfflineMarketDataStream(
            instruments=[], depth=1, dataset_path=dataset, interval=0
        )
        async with stream as ctx:
            generator = ctx.order_books()
            return await generator.__anext__()

    book = asyncio.run(runner())
    assert book.figi == "FIGI2"
    assert book.depth == 1
    assert book.timestamp.tzinfo is not None
