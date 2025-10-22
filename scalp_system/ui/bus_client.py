"""Event-bus utilities for the Tk dashboard."""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Callable, Optional
from urllib.error import URLError
from urllib.request import urlopen

from ..control.bus import BusClient as CoreBusClient, Event

LOGGER = logging.getLogger(__name__)


class DashboardBusClient:
    """Wrap the core bus client with helpers for the dashboard UI."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout: float = 1.0,
        events_endpoint: str | None = None,
    ) -> None:
        self._client = CoreBusClient(host, port, timeout=timeout)
        self._events_endpoint = events_endpoint
        self._stop_event = threading.Event()
        self._listener_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Bus availability
    # ------------------------------------------------------------------
    def is_up(self) -> bool:
        return self._client.check_available()

    # ------------------------------------------------------------------
    # Command emission
    # ------------------------------------------------------------------
    def emit(self, name: str, args: Optional[dict] = None) -> None:
        LOGGER.debug("Emitting dashboard command: %s", name)
        self._client.emit_event(Event(type=name, payload=args or {}))

    # ------------------------------------------------------------------
    # Event subscription
    # ------------------------------------------------------------------
    def subscribe(self, callback: Callable[[dict], None]) -> Callable[[], None]:
        """Subscribe to the optional events endpoint if configured."""

        if not self._events_endpoint:
            LOGGER.debug("No events endpoint configured; subscription skipped")
            return lambda: None

        if self._listener_thread is not None:
            raise RuntimeError("Listener already running")

        def _loop() -> None:
            LOGGER.info("Starting dashboard event listener: %s", self._events_endpoint)
            while not self._stop_event.is_set():
                try:
                    with urlopen(self._events_endpoint, timeout=5) as response:  # nosec B310
                        for line in response:
                            if self._stop_event.is_set():
                                break
                            try:
                                payload = json.loads(line.decode("utf-8"))
                            except json.JSONDecodeError:
                                continue
                            try:
                                callback(payload)
                            except Exception:  # pragma: no cover - defensive
                                LOGGER.exception("Dashboard event callback failed")
                except URLError as exc:
                    LOGGER.debug("Event stream unavailable: %s", exc)
                    time.sleep(1)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.exception("Unexpected event listener error: %s", exc)
                    time.sleep(1)

        thread = threading.Thread(target=_loop, daemon=True)
        self._listener_thread = thread
        thread.start()

        def _stop() -> None:
            self._stop_event.set()
            if thread.is_alive():
                thread.join(timeout=1)
            self._listener_thread = None
            self._stop_event.clear()

        return _stop


__all__ = ["DashboardBusClient"]
