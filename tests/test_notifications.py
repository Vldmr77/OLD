import asyncio
from urllib.parse import parse_qs

from scalp_system.config.base import NotificationConfig
from scalp_system.monitoring.notifications import NotificationDispatcher


def test_order_filled_notification_sends_payload():
    captured = {}

    def fake_http(url: str, data: bytes) -> None:
        captured["url"] = url
        captured["data"] = data

    dispatcher = NotificationDispatcher(
        NotificationConfig(
            telegram_bot_token="token",
            telegram_chat_id="chat",
            enable_sound_alerts=False,
        ),
        http_sender=fake_http,
        sound_player=lambda *args, **kwargs: None,
    )

    sent = asyncio.run(
        dispatcher.notify_order_filled("FIGI", price=101.5, confidence=0.87)
    )

    assert sent is True
    assert "token" in captured["url"]
    payload = parse_qs(captured["data"].decode("utf-8"))
    assert payload["chat_id"][0] == "chat"
    assert "ORDER_FILLED" in payload["text"][0]


def test_circuit_breaker_triggers_sound_and_cooldown():
    sounds: list[tuple[int, float]] = []

    dispatcher = NotificationDispatcher(
        NotificationConfig(
            enable_sound_alerts=True,
            cooldown_seconds=60,
        ),
        http_sender=lambda *args, **kwargs: None,
        sound_player=lambda freq, duration: sounds.append((freq, duration)),
    )

    asyncio.run(dispatcher.notify_circuit_breaker("FIGI", severity="critical"))
    asyncio.run(dispatcher.notify_circuit_breaker("FIGI", severity="critical"))

    assert sounds[0][0] == dispatcher.config.high_risk_frequency_hz
    assert len(sounds) == 1


def test_low_liquidity_notification_uses_low_frequency():
    sounds: list[tuple[int, float]] = []

    dispatcher = NotificationDispatcher(
        NotificationConfig(enable_sound_alerts=True),
        http_sender=lambda *args, **kwargs: None,
        sound_player=lambda freq, duration: sounds.append((freq, duration)),
    )

    sent = asyncio.run(dispatcher.notify_low_liquidity("FIGI", spread_bps=6.0))

    assert sent is False  # telegram not configured
    assert sounds[0][0] == dispatcher.config.low_liquidity_frequency_hz


def test_latency_notification_bypasses_cooldown_for_critical():
    calls: list[bytes] = []

    def fake_http(url: str, data: bytes) -> None:
        calls.append(data)

    dispatcher = NotificationDispatcher(
        NotificationConfig(
            telegram_bot_token="token",
            telegram_chat_id="chat",
            cooldown_seconds=60,
        ),
        http_sender=fake_http,
        sound_player=lambda *args, **kwargs: None,
    )

    asyncio.run(
        dispatcher.notify_latency_violation(
            "ml", latency_ms=25.0, threshold_ms=20.0, severity="critical"
        )
    )
    asyncio.run(
        dispatcher.notify_latency_violation(
            "ml", latency_ms=26.0, threshold_ms=20.0, severity="critical"
        )
    )

    assert len(calls) == 2
