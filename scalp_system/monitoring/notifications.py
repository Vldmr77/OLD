"""Notification helpers for messaging and audible alerts."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..config.base import NotificationConfig

LOGGER = logging.getLogger(__name__)


HttpSender = Callable[[str, bytes], None]
SoundPlayer = Callable[[int, float], None]


def _default_http_sender(url: str, data: bytes) -> None:
    request = Request(url, data=data)
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urlopen(request, timeout=5) as response:  # pragma: no cover - network I/O
        response.read()


def _default_sound_player(frequency_hz: int, duration_s: float) -> None:
    LOGGER.info("Sound alert freq=%sHz duration=%.2fs", frequency_hz, duration_s)


@dataclass
class NotificationDispatcher:
    """Routes critical system events to external notifiers."""

    config: NotificationConfig
    http_sender: HttpSender = _default_http_sender
    sound_player: SoundPlayer = _default_sound_player

    def __post_init__(self) -> None:
        self._last_sent: Dict[str, datetime] = {}

    async def notify_order_filled(self, figi: str, price: float, confidence: float) -> bool:
        message = (
            f"ORDER_FILLED {figi}: price={price:.2f} confidence={confidence:.3f}"
        )
        return await self._dispatch(
            key=f"order:{figi}",
            message=message,
            cooldown_override=0,
        )

    async def notify_circuit_breaker(self, figi: str, severity: str) -> bool:
        message = f"CIRCUIT_BREAKER {figi}: severity={severity}"
        return await self._dispatch(
            key=f"circuit:{figi}",
            message=message,
            beep_frequency=self.config.high_risk_frequency_hz,
            beep_duration=0.5,
        )

    async def notify_low_liquidity(self, figi: str, spread_bps: float) -> bool:
        message = f"LOW_LIQUIDITY {figi}: spread_bps={spread_bps:.2f}"
        return await self._dispatch(
            key=f"liquidity:{figi}",
            message=message,
            beep_frequency=self.config.low_liquidity_frequency_hz,
            beep_duration=0.3,
        )

    async def notify_latency_violation(
        self, stage: str, latency_ms: float, threshold_ms: float, severity: str
    ) -> bool:
        prefix = "LATENCY_CRITICAL" if severity == "critical" else "LATENCY_WARN"
        message = (
            f"{prefix} stage={stage} latency={latency_ms:.2f}ms threshold={threshold_ms:.2f}ms"
        )
        return await self._dispatch(
            key=f"latency:{stage}",
            message=message,
            beep_frequency=self.config.high_risk_frequency_hz
            if severity == "critical"
            else None,
            beep_duration=0.4,
            cooldown_override=0 if severity == "critical" else None,
        )

    async def notify_performance_summary(
        self,
        *,
        realized_pnl: float,
        signal_count: int,
        avg_confidence: float,
        halted: bool,
    ) -> bool:
        status = "HALTED" if halted else "ACTIVE"
        message = (
            "PERFORMANCE "
            f"pnl={realized_pnl:.2f} signals={signal_count} avg_conf={avg_confidence:.3f} "
            f"state={status}"
        )
        return await self._dispatch(
            key="performance",
            message=message,
            cooldown_override=self.config.cooldown_seconds,
        )

    async def notify_backup_created(
        self, snapshot_dir: Path, size_bytes: int
    ) -> bool:
        message = (
            "BACKUP_CREATED "
            f"path={snapshot_dir} size={size_bytes}B"
        )
        return await self._dispatch(
            key="backup",
            message=message,
            cooldown_override=0,
        )

    async def _dispatch(
        self,
        *,
        key: str,
        message: str,
        beep_frequency: Optional[int] = None,
        beep_duration: float = 0.3,
        cooldown_override: Optional[int] = None,
    ) -> bool:
        cooldown = (
            self.config.cooldown_seconds
            if cooldown_override is None
            else max(cooldown_override, 0)
        )
        if self._in_cooldown(key, cooldown):
            return False
        if beep_frequency and self.config.enable_sound_alerts:
            self.sound_player(beep_frequency, beep_duration)
        sent = await self._send_telegram(message)
        if cooldown > 0:
            self._last_sent[key] = datetime.utcnow()
        return sent

    def _in_cooldown(self, key: str, cooldown: int) -> bool:
        if cooldown <= 0:
            return False
        last = self._last_sent.get(key)
        if not last:
            return False
        return datetime.utcnow() - last < timedelta(seconds=cooldown)

    async def _send_telegram(self, message: str) -> bool:
        token = self.config.telegram_bot_token
        chat_id = self.config.telegram_chat_id
        if not token or not chat_id:
            return False
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = urlencode({"chat_id": chat_id, "text": message}).encode("utf-8")
        try:
            await asyncio.to_thread(self.http_sender, url, payload)
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.exception("Failed to send Telegram notification")
            return False
        return True


__all__ = ["NotificationDispatcher"]
