"""Feature engineering pipeline."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from ..config.base import FeatureConfig
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

    def transform(self, books: Iterable[OrderBook]) -> FeatureVector:
        books = list(books)
        if not books:
            raise ValueError("No order books provided")
        latest = books[-1]
        feature_list: List[float] = []
        if self._config.include_micro_price:
            feature_list.append(self._micro_price(latest))
        if self._config.include_order_flow_imbalance:
            feature_list.extend(self._order_flow_imbalance(books))
        if self._config.include_volatility_cluster:
            feature_list.append(self._volatility_cluster(books))
        feature_list.extend(self._depth_deltas(latest))
        return FeatureVector(figi=latest.figi, timestamp=latest.timestamp.timestamp(), features=feature_list)

    def _micro_price(self, book: OrderBook) -> float:
        bids = book.bids[: self._config.lob_levels]
        asks = book.asks[: self._config.lob_levels]
        weighted_bid = sum(level.price * level.quantity for level in bids)
        weighted_ask = sum(level.price * level.quantity for level in asks)
        total_volume = sum(level.quantity for level in bids + asks) or 1.0
        return (weighted_bid + weighted_ask) / total_volume

    def _depth_deltas(self, book: OrderBook) -> List[float]:
        bids = [level.quantity for level in book.bids[: self._config.lob_levels]]
        asks = [level.quantity for level in book.asks[: self._config.lob_levels]]
        return [bid - ask for bid, ask in zip(bids, asks)]

    def _order_flow_imbalance(self, books: Sequence[OrderBook]) -> List[float]:
        if len(books) < 2:
            return [0.0, 0.0]
        imbalances = []
        for prev, curr in zip(books[:-1], books[1:]):
            prev_bid = prev.bids[0].quantity
            curr_bid = curr.bids[0].quantity
            prev_ask = prev.asks[0].quantity
            curr_ask = curr.asks[0].quantity
            imbalance = (curr_bid - prev_bid) - (curr_ask - prev_ask)
            imbalances.append(imbalance)
        mean_value = statistics.fmean(imbalances)
        std_dev = statistics.pstdev(imbalances) if len(imbalances) > 1 else 0.0
        return [mean_value, std_dev]

    def _volatility_cluster(self, books: Sequence[OrderBook]) -> float:
        if len(books) < 2:
            return 0.0
        returns = []
        for prev, curr in zip(books[:-1], books[1:]):
            prev_mid = prev.mid_price()
            curr_mid = curr.mid_price()
            if prev_mid <= 0 or curr_mid <= 0:
                continue
            returns.append(math.log(curr_mid / prev_mid))
        if not returns:
            return 0.0
        return statistics.pstdev(returns)


__all__ = ["FeaturePipeline", "FeatureVector"]
