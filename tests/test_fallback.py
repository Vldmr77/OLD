from datetime import datetime

import pytest

from scalp_system.config.base import FallbackConfig
from scalp_system.data.models import OrderBook, OrderBookLevel
from scalp_system.ml.fallback import FallbackSignalGenerator


def _make_order_book(figi: str, mid_price: float, bid_qty: float, ask_qty: float) -> OrderBook:
    bid_price = mid_price - 0.05
    ask_price = mid_price + 0.05
    bids = tuple(OrderBookLevel(price=bid_price, quantity=bid_qty) for _ in range(3))
    asks = tuple(OrderBookLevel(price=ask_price, quantity=ask_qty) for _ in range(3))
    return OrderBook(
        figi=figi,
        timestamp=datetime.utcnow(),
        bids=bids,
        asks=asks,
        depth=3,
    )


def test_fallback_generates_signal_when_thresholds_met():
    config = FallbackConfig(
        enabled=True,
        max_signals_per_hour=5,
        imbalance_threshold=0.5,
        volatility_threshold=0.01,
        confidence=0.75,
    )
    generator = FallbackSignalGenerator(config)
    book_old = _make_order_book("FIGI", mid_price=100.0, bid_qty=100.0, ask_qty=50.0)
    book_new = _make_order_book("FIGI", mid_price=102.0, bid_qty=120.0, ask_qty=40.0)

    decision = generator.generate([book_old, book_new], reason="exception:RuntimeError")

    assert decision is not None
    assert decision.signal.direction == 1
    assert decision.signal.confidence == pytest.approx(0.75)
    assert decision.signal.source == "fallback"
    assert decision.reason.endswith("long")


def test_fallback_respects_quota():
    config = FallbackConfig(max_signals_per_hour=1, imbalance_threshold=0.3)
    generator = FallbackSignalGenerator(config)
    window = [
        _make_order_book("FIGI", 100.0, 120.0, 40.0),
        _make_order_book("FIGI", 101.5, 120.0, 40.0),
    ]

    assert generator.generate(window, reason="exception:RuntimeError") is not None
    second = generator.generate(window, reason="exception:RuntimeError")

    assert second is None
    assert generator.last_rejection == "quota_exceeded"
    assert generator.remaining_allowance() == 0


def test_fallback_rejects_low_volatility():
    config = FallbackConfig(volatility_threshold=0.05)
    generator = FallbackSignalGenerator(config)
    window = [
        _make_order_book("FIGI", 100.0, 120.0, 40.0),
        _make_order_book("FIGI", 100.1, 120.0, 40.0),
    ]

    assert generator.generate(window, reason="exception:RuntimeError") is None
    assert generator.last_rejection == "low_volatility"
