"""Lightweight TCP event bus for orchestrator control commands."""
from __future__ import annotations

import json
import logging
import socket
import threading
from dataclasses import dataclass
from socketserver import StreamRequestHandler, ThreadingTCPServer
from typing import Callable, Dict, Optional

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    """Represents a control event emitted over the bus."""

    type: str
    payload: Optional[dict] = None

    def to_message(self) -> bytes:
        return (
            json.dumps({"type": self.type, "payload": self.payload or {}})
            + "\n"
        ).encode("utf-8")


class EventBus:
    """Threaded TCP server dispatching JSON events to registered handlers."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8787) -> None:
        self._host = host
        self._port = port
        self._handlers: Dict[str, Callable[[dict], None]] = {}
        self._server: ThreadingTCPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        if self._server is not None:
            return int(self._server.server_address[1])
        return self._port

    def register(self, event_type: str, handler: Callable[[dict], None]) -> None:
        self._handlers[event_type] = handler

    def start(self) -> None:
        if self._server is not None:
            return

        outer = self

        class _RequestHandler(StreamRequestHandler):  # type: ignore[misc]
            def handle(self) -> None:  # noqa: D401 - socketserver API
                try:
                    raw = self.rfile.readline()
                    if not raw:
                        return
                    payload = json.loads(raw.decode("utf-8"))
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.debug("Failed to decode bus payload: %s", exc)
                    return
                event_type = payload.get("type")
                data = payload.get("payload") or {}
                handler = outer._handlers.get(event_type)
                if handler is None:
                    return
                try:
                    handler(data)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.exception("Event handler failed for %s: %s", event_type, exc)

        class _Server(ThreadingTCPServer):
            allow_reuse_address = True

        server = _Server((self._host, self._port), _RequestHandler)
        self._server = server

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._thread = thread
        LOGGER.info("Event bus listening on %s:%s", *server.server_address)

    def stop(self) -> None:
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        self._server = None
        self._thread = None
        LOGGER.info("Event bus stopped")


class BusClient:
    """Client helper for emitting events to the bus."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8787, *, timeout: float = 1.0) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout

    def emit_event(self, event: Event) -> None:
        message = event.to_message()
        with socket.create_connection((self._host, self._port), self._timeout) as conn:
            conn.sendall(message)

    def check_available(self) -> bool:
        try:
            with socket.create_connection((self._host, self._port), self._timeout):
                return True
        except OSError:
            return False


__all__ = ["BusClient", "Event", "EventBus"]
