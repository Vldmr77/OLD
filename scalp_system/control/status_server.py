"""Lightweight HTTP server that exposes orchestrator status snapshots."""

from __future__ import annotations

import errno
import json
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional, Type

LOGGER = logging.getLogger(__name__)


class DashboardStatusServer:
    """Expose orchestrator status snapshots over HTTP."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        status_provider: Callable[[], dict],
    ) -> None:
        self._host = host or "127.0.0.1"
        self._port = int(port)
        self._status_provider = status_provider
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._server is not None:
            return

        outer = self

        class _Handler(BaseHTTPRequestHandler):  # type: ignore[misc]
            def do_GET(self) -> None:  # noqa: D401 - http.server API
                path = self.path.rstrip("/") or "/"
                if path in ("", "/"):
                    path = "/status"
                if path == "/status":
                    self._handle_status()
                elif path == "/health":
                    self._handle_health()
                else:
                    self.send_error(404, "Not Found")

            def log_message(self, format: str, *args) -> None:  # pragma: no cover - quiet server
                LOGGER.debug("status-server: " + format, *args)

            def _handle_status(self) -> None:
                try:
                    payload = outer._status_provider()
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.exception("Status provider failed: %s", exc)
                    body = json.dumps(
                        {
                            "error": {
                                "source": "status_provider",
                                "message": str(exc),
                            }
                        }
                    ).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body = json.dumps(payload or {}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _handle_health(self) -> None:
                body = b"{\"status\": \"ok\"}"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        class _Server(ThreadingHTTPServer):
            allow_reuse_address = True

        server = self._create_server_with_retry(_Server, _Handler)
        self._server = server
        self._port = int(server.server_address[1])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._thread = thread
        LOGGER.info("Dashboard status server listening on %s:%s", *server.server_address)

    def _create_server_with_retry(
        self,
        server_cls: Type[ThreadingHTTPServer],
        handler_cls: Type[BaseHTTPRequestHandler],
    ) -> ThreadingHTTPServer:
        if not self._port:
            return server_cls((self._host, self._port), handler_cls)

        attempts = 0
        delay = 0.05

        while attempts < 8:
            try:
                return server_cls((self._host, self._port), handler_cls)
            except OSError as exc:
                if exc.errno not in {errno.EADDRINUSE, errno.EACCES}:
                    raise
                if self._port_in_use():
                    LOGGER.error(
                        "Dashboard status port %s is already in use by another process.",
                        self._port,
                    )
                    raise
                attempts += 1
                LOGGER.warning(
                    "Dashboard status port %s unavailable (%s); retrying in %.2fs",
                    self._port,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 1.5, 0.5)

        LOGGER.warning(
            "Dashboard status port %s still unavailable after retries; falling back to an ephemeral port.",
            self._port,
        )
        return server_cls((self._host, 0), handler_cls)

    def _port_in_use(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.1)
            try:
                probe.connect((self._host, self._port))
            except OSError:
                return False
        return True

    # ------------------------------------------------------------------
    def stop(self) -> None:
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        self._server = None
        self._thread = None
        LOGGER.info("Dashboard status server stopped")

    # ------------------------------------------------------------------
    @property
    def status_endpoint(self) -> str:
        host, port = self._resolved_address
        return f"http://{host}:{port}/status"

    # ------------------------------------------------------------------
    @property
    def health_endpoint(self) -> str:
        host, port = self._resolved_address
        return f"http://{host}:{port}/health"

    # ------------------------------------------------------------------
    @property
    def _resolved_address(self) -> tuple[str, int]:
        if self._server is not None:
            host, port = self._server.server_address
            host = host or self._host
            return str(host), int(port)
        return self._host, self._port


__all__ = ["DashboardStatusServer"]
