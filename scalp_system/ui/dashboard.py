"""Lightweight HTTP dashboard without external dependencies."""
from __future__ import annotations

import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Dict

from ..storage.repository import SQLiteRepository


@dataclass
class Response:
    status_code: int
    body: bytes
    headers: Dict[str, str]

    def get_data(self, as_text: bool = False):
        return self.body.decode("utf-8") if as_text else self.body

    def json(self):
        return json.loads(self.body.decode("utf-8"))


class DashboardApp:
    def __init__(self, repository: SQLiteRepository) -> None:
        self._repository = repository

    def handle_request(self, path: str) -> Response:
        if "?" in path:
            path = path.split("?", 1)[0]
        if path in {"", "/", "/index.html"}:
            summary = self._repository.summary()
            signals = self._repository.fetch_signals(limit=50)
            html = self._render_html(summary, signals).encode("utf-8")
            return Response(HTTPStatus.OK, html, {"Content-Type": "text/html; charset=utf-8"})
        if path == "/api/signals":
            raw = self._repository.fetch_signals(limit=100)
            payload = json.dumps(
                [
                    {
                        **signal,
                        "timestamp": signal["timestamp"].isoformat()
                        if hasattr(signal["timestamp"], "isoformat")
                        else signal["timestamp"],
                    }
                    for signal in raw
                ],
                ensure_ascii=False,
            ).encode("utf-8")
            return Response(HTTPStatus.OK, payload, {"Content-Type": "application/json"})
        if path == "/api/summary":
            payload = json.dumps(self._repository.summary(), ensure_ascii=False).encode("utf-8")
            return Response(HTTPStatus.OK, payload, {"Content-Type": "application/json"})
        return Response(HTTPStatus.NOT_FOUND, b"Not Found", {"Content-Type": "text/plain"})

    def test_client(self):
        app = self

        class Client:
            def get(self, path: str):
                return app.handle_request(path)

        return Client()

    @staticmethod
    def _render_html(summary: dict, signals: list[dict]) -> str:
        rows = "".join(
            f"<tr><td>{signal['figi']}</td><td>{signal['direction']}</td><td>{signal['confidence']:.2f}</td><td>{_format_timestamp(signal['timestamp'])}</td></tr>"
            for signal in signals
        )
        latest = summary.get("latest")
        latest_block = (
            f"Latest signal: {latest['figi']} • direction {latest['direction']} • confidence "
            f"{latest['confidence']:.2f} at {latest['timestamp']}"
            if latest
            else "No signals captured yet."
        )
        return f"""
<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\">
    <title>Scalp System Dashboard</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 2rem; background: #0f172a; color: #e2e8f0; }}
      h1 {{ color: #38bdf8; }}
      table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
      th, td {{ border: 1px solid #1e293b; padding: 0.5rem; text-align: left; }}
      th {{ background: #1d4ed8; color: #f8fafc; }}
      tr:nth-child(even) {{ background: #1e293b; }}
      .summary {{ margin-bottom: 1.5rem; padding: 1rem; background: #1e293b; border-radius: 0.5rem; }}
      .badge {{ display: inline-block; padding: 0.25rem 0.5rem; border-radius: 0.5rem; background: #22c55e; color: #0f172a; font-weight: bold; }}
    </style>
  </head>
  <body>
    <h1>Scalp System Dashboard</h1>
    <div class=\"summary\">
      <p>Total persisted signals: <span class=\"badge\">{summary.get('total_signals', 0)}</span></p>
      <p>{latest_block}</p>
    </div>
    <table>
      <thead>
        <tr><th>FIGI</th><th>Direction</th><th>Confidence</th><th>Timestamp (UTC)</th></tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </body>
</html>
"""


def create_dashboard_app(repository: SQLiteRepository) -> DashboardApp:
    return DashboardApp(repository)


def run_dashboard(
    repository_path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 5000,
    background: bool = False,
) -> Thread | None:
    repository = SQLiteRepository(repository_path)
    app = create_dashboard_app(repository)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # type: ignore[override]
            response = app.handle_request(self.path)
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(response.body)

        def log_message(self, format, *args):  # pragma: no cover - suppress noisy logs
            return

    server = ThreadingHTTPServer((host, port), Handler)

    if background:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return thread

    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual shutdown
        pass
    finally:
        server.server_close()
    return None


def _format_timestamp(value) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


__all__ = ["create_dashboard_app", "run_dashboard", "DashboardApp", "Response"]
