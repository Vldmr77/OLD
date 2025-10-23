"""Feature engineering pipeline."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from ..config.base import FeatureConfig
from ..data.engine import MarketIndicators
from ..data.models import OrderBook


@dataclass
class FeatureVector:
    figi: str
    timestamp: float
    features: List[float]


class FeaturePipeline:
    """Transforms raw order books into dense feature vectors."""

    def __init__(self, config: FeatureConfig) -> None:
        self._config = config
        self._last_vector: Optional[FeatureVector] = None

    def transform(
        self,
        books: Iterable[OrderBook],
        market: Optional[MarketIndicators] = None,
    ) -> FeatureVector:
        books = list(books)
        if not books:
            raise ValueError("No order books provided")
        latest = books[-1]
        feature_list: List[float] = []
        feature_list.extend(self._base_features(books))
        feature_list.extend(self._derived_features(books))
        feature_list.extend(self._adaptive_features(books))
        feature_list.extend(self._volume_profile(latest))
        feature_list.extend(self._depth_deltas(latest))
        if market:
            feature_list.extend(
                [
                    market.market_volatility,
                    market.market_liquidity,
                    market.market_index,
                ]
            )
        vector = FeatureVector(figi=latest.figi, timestamp=latest.timestamp.timestamp(), features=feature_list)
        self._last_vector = vector
        return vector

    def reset_cache(self) -> None:
        """Clear any cached feature state before reloading models."""

        self._last_vector = None

    def _micro_price(self, book: OrderBook) -> float:
        bids = book.bids[: self._config.lob_levels]
        asks = book.asks[: self._config.lob_levels]
        weighted_bid = sum(level.price * level.quantity for level in bids)
        weighted_ask = sum(level.price * level.quantity for level in asks)
        total_volume = sum(level.quantity for level in bids + asks) or 1.0
        return (weighted_bid + weighted_ask) / total_volume

    def _spread(self, book: OrderBook) -> float:
        return book.spread()

    def _volatility(self, books: Sequence[OrderBook]) -> float:
        if len(books) < 2:
            return 0.0
        mids = [book.mid_price() for book in books]
        deltas = [abs(curr - prev) for prev, curr in zip(mids[:-1], mids[1:])]
        if not deltas:
            return 0.0
        return statistics.fmean(deltas) / (mids[-1] or 1.0)

    def _imbalance(self, book: OrderBook) -> float:
        bid_volume = sum(level.quantity for level in book.bids[: self._config.lob_levels])
        ask_volume = sum(level.quantity for level in book.asks[: self._config.lob_levels])
        total = bid_volume + ask_volume
        if not total:
            return 0.0
        return (bid_volume - ask_volume) / total

    def _base_features(self, books: Sequence[OrderBook]) -> List[float]:
        latest = books[-1]
        features = [
            self._imbalance(latest),
            self._micro_price(latest) if self._config.include_micro_price else 0.0,
            self._spread(latest),
            self._volatility(books),
        ]
        return features

    def _derived_features(self, books: Sequence[OrderBook]) -> List[float]:
        if len(books) < 2:
            return [0.0, 0.0]
        latest_imbalance = self._imbalance(books[-1])
        prev_imbalance = self._imbalance(books[-2])
        ofi = (latest_imbalance - prev_imbalance) / (prev_imbalance + 1e-8)
        gradient = (latest_imbalance - prev_imbalance) / (abs(latest_imbalance) + 1e-8)
        return [ofi, gradient]

    def _adaptive_features(self, books: Sequence[OrderBook]) -> List[float]:
        latest = books[-1]
        vol = self._volatility(books)
        imbalance = self._imbalance(latest)
        if vol > 0.15:
            return [imbalance * 1.5, 0.0]
        return [0.0, imbalance * 0.7]

    def _depth_deltas(self, book: OrderBook) -> List[float]:
        bids = [level.quantity for level in book.bids[: self._config.lob_levels]]
        asks = [level.quantity for level in book.asks[: self._config.lob_levels]]
        return [bid - ask for bid, ask in zip(bids, asks)]

    def _volume_profile(self, book: OrderBook) -> List[float]:
        bids = book.bids[: self._config.lob_levels]
        asks = book.asks[: self._config.lob_levels]
        bid_total = sum(level.quantity for level in bids) or 1.0
        ask_total = sum(level.quantity for level in asks) or 1.0
        bid_density = [level.quantity / bid_total for level in bids]
        ask_density = [level.quantity / ask_total for level in asks]
        # pad to ensure deterministic length
        if len(bid_density) < self._config.lob_levels:
            bid_density.extend([0.0] * (self._config.lob_levels - len(bid_density)))
        if len(ask_density) < self._config.lob_levels:
            ask_density.extend([0.0] * (self._config.lob_levels - len(ask_density)))
        return [bid_total, ask_total, *bid_density, *ask_density]


__all__ = ["FeaturePipeline", "FeatureVector"]
