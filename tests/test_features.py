from datetime import datetime, timezone

from scalp_system.config.base import FeatureConfig
from scalp_system.data.engine import MarketIndicators
from scalp_system.data.models import OrderBook, OrderBookLevel
from scalp_system.features.pipeline import FeaturePipeline


def make_book(price: float) -> OrderBook:
    bids = tuple(OrderBookLevel(price=price - i * 0.1, quantity=10 + i) for i in range(10))
    asks = tuple(OrderBookLevel(price=price + i * 0.1, quantity=9 + i) for i in range(10))
    return OrderBook(figi="FIGI", timestamp=datetime.now(timezone.utc), bids=bids, asks=asks, depth=10)


def test_feature_pipeline_generates_vector():
    pipeline = FeaturePipeline(FeatureConfig(lob_levels=5))
    books = [make_book(100 + i * 0.05) for i in range(5)]
    market = MarketIndicators(market_volatility=0.12, market_liquidity=1500.0, market_index=0.9)
    features = pipeline.transform(books, market=market)
    assert features.figi == "FIGI"
    assert len(features.features) >= 28
    assert all(value == value for value in features.features)
