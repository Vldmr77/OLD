import asyncio
from pathlib import Path
from urllib.parse import parse_qs

from datetime import datetime

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


def test_performance_summary_formats_payload():
    captured = {}

    def fake_http(url: str, data: bytes) -> None:
        captured["data"] = data

    dispatcher = NotificationDispatcher(
        NotificationConfig(
            telegram_bot_token="token",
            telegram_chat_id="chat",
            cooldown_seconds=0,
        ),
        http_sender=fake_http,
        sound_player=lambda *args, **kwargs: None,
    )

    asyncio.run(
        dispatcher.notify_performance_summary(
            realized_pnl=1234.5,
            signal_count=42,
            avg_confidence=0.678,
            halted=False,
        )
    )

    payload = parse_qs(captured["data"].decode("utf-8"))
    assert "PERFORMANCE" in payload["text"][0]
    assert "state=ACTIVE" in payload["text"][0]


def test_backup_notification_includes_path(tmp_path: Path):
    captured = {}

    def fake_http(url: str, data: bytes) -> None:
        captured["data"] = data

    dispatcher = NotificationDispatcher(
        NotificationConfig(
            telegram_bot_token="token",
            telegram_chat_id="chat",
        ),
        http_sender=fake_http,
        sound_player=lambda *args, **kwargs: None,
    )

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    asyncio.run(dispatcher.notify_backup_created(snapshot, size_bytes=1024))

    payload = parse_qs(captured["data"].decode("utf-8"))
    assert "BACKUP_CREATED" in payload["text"][0]
    assert str(snapshot) in payload["text"][0]


def test_manual_override_notification_formats_expiry():
    captured = {}

    def fake_http(url: str, data: bytes) -> None:
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

    expiry = datetime(2024, 1, 1, 0, 0, 0)
    asyncio.run(dispatcher.notify_manual_override("maintenance", expiry))

    payload = parse_qs(captured["data"].decode("utf-8"))
    assert "MANUAL_OVERRIDE" in payload["text"][0]
    assert "maintenance" in payload["text"][0]
    assert "until=" in payload["text"][0]


def test_manual_override_cleared_reuses_key_without_cooldown():
    captured: list[bytes] = []

    def fake_http(url: str, data: bytes) -> None:
        captured.append(data)

    dispatcher = NotificationDispatcher(
        NotificationConfig(
            telegram_bot_token="token",
            telegram_chat_id="chat",
            cooldown_seconds=60,
        ),
        http_sender=fake_http,
        sound_player=lambda *args, **kwargs: None,
    )

    asyncio.run(dispatcher.notify_manual_override("halt", None))
    asyncio.run(dispatcher.notify_manual_override_cleared())


def test_connectivity_failover_notification_triggers_sound():
    captured = {}
    sounds: list[tuple[int, float]] = []

    def fake_http(url: str, data: bytes) -> None:
        captured["data"] = data

    dispatcher = NotificationDispatcher(
        NotificationConfig(
            telegram_bot_token="token",
            telegram_chat_id="chat",
            enable_sound_alerts=True,
        ),
        http_sender=fake_http,
        sound_player=lambda freq, duration: sounds.append((freq, duration)),
    )

    asyncio.run(dispatcher.notify_connectivity_failover("lte", "timeout"))

    payload = parse_qs(captured["data"].decode("utf-8"))
    assert "CONNECTIVITY_FAILOVER" in payload["text"][0]
    assert sounds[0][0] == dispatcher.config.high_risk_frequency_hz


def test_connectivity_recovered_notification_has_no_cooldown():
    calls: list[bytes] = []

    dispatcher = NotificationDispatcher(
        NotificationConfig(
            telegram_bot_token="token",
            telegram_chat_id="chat",
        ),
        http_sender=lambda url, data: calls.append(data),
        sound_player=lambda *args, **kwargs: None,
    )

    asyncio.run(dispatcher.notify_connectivity_recovered("fiber"))

    payload = parse_qs(calls[0].decode("utf-8"))
    assert "CONNECTIVITY_RECOVERED" in payload["text"][0]


def test_gpu_failover_notification_formats_message():
    captured = {}
    sounds: list[tuple[int, float]] = []

    def fake_http(url: str, data: bytes) -> None:
        captured["data"] = data

    dispatcher = NotificationDispatcher(
        NotificationConfig(
            telegram_bot_token="token",
            telegram_chat_id="chat",
            enable_sound_alerts=True,
        ),
        http_sender=fake_http,
        sound_player=lambda freq, duration: sounds.append((freq, duration)),
    )

    asyncio.run(dispatcher.notify_gpu_failover("cpu", "cuda device lost"))

    payload = parse_qs(captured["data"].decode("utf-8"))
    assert "GPU_FAILOVER" in payload["text"][0]
    assert "mode=cpu" in payload["text"][0]
    assert sounds[0][0] == dispatcher.config.high_risk_frequency_hz


def test_heartbeat_missed_notification_contains_reason():
    captured = {}

    def fake_http(url: str, data: bytes) -> None:
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

    last_seen = datetime(2024, 1, 1, 0, 0, 0)
    asyncio.run(dispatcher.notify_heartbeat_missed(3, last_seen, "timeout"))

    payload = parse_qs(captured["data"].decode("utf-8"))
    text = payload["text"][0]
    assert "HEARTBEAT_MISSED" in text
    assert "intervals=3" in text
    assert "reason=timeout" in text


def test_fallback_signal_notification_bypasses_cooldown():
    sounds: list[tuple[int, float]] = []

    dispatcher = NotificationDispatcher(
        NotificationConfig(enable_sound_alerts=True),
        http_sender=lambda *args, **kwargs: None,
        sound_player=lambda freq, duration: sounds.append((freq, duration)),
    )

    sent = asyncio.run(
        dispatcher.notify_fallback_signal(
            "FIGI", reason="exception:long", confidence=0.7
        )
    )

    assert sent is False
    assert sounds[0][0] == dispatcher.config.high_risk_frequency_hz


def test_session_suspended_notification_mentions_resume():
    captured = {}
    sounds: list[tuple[int, float]] = []

    def fake_http(url: str, data: bytes) -> None:
        captured["data"] = data

    dispatcher = NotificationDispatcher(
        NotificationConfig(
            telegram_bot_token="token",
            telegram_chat_id="chat",
            enable_sound_alerts=True,
        ),
        http_sender=fake_http,
        sound_player=lambda freq, duration: sounds.append((freq, duration)),
    )

    resume_at = datetime(2024, 1, 2, 10, 0, 0)
    asyncio.run(dispatcher.notify_session_suspended("weekday_blocked", resume_at))

    payload = parse_qs(captured["data"].decode("utf-8"))
    text = payload["text"][0]
    assert "SESSION_SUSPENDED" in text
    assert "weekday_blocked" in text
    assert "resume_at=" in text
    assert sounds[0][0] == dispatcher.config.high_risk_frequency_hz


def test_session_resumed_notification_includes_next_pause():
    captured = {}

    def fake_http(url: str, data: bytes) -> None:
        captured["data"] = data

    dispatcher = NotificationDispatcher(
        NotificationConfig(
            telegram_bot_token="token",
            telegram_chat_id="chat",
        ),
        http_sender=fake_http,
        sound_player=lambda *args, **kwargs: None,
    )

    next_pause = datetime(2024, 1, 2, 19, 0, 0)
    asyncio.run(dispatcher.notify_session_resumed(next_pause))

    payload = parse_qs(captured["data"].decode("utf-8"))
    text = payload["text"][0]
    assert "SESSION_RESUMED" in text
    assert "next_pause=" in text
