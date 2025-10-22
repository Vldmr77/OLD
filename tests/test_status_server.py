import json
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from scalp_system.control.status_server import DashboardStatusServer


def test_status_server_serves_payload():
    payload = {"build": {"version": "x"}}
    server = DashboardStatusServer(host="127.0.0.1", port=0, status_provider=lambda: payload)
    server.start()
    try:
        with urlopen(server.status_endpoint, timeout=2) as response:
            data = json.load(response)
        assert data == payload
    finally:
        server.stop()


def test_status_server_reports_provider_errors():
    def _provider():
        raise RuntimeError("boom")

    server = DashboardStatusServer(host="127.0.0.1", port=0, status_provider=_provider)
    server.start()
    try:
        with pytest.raises(HTTPError) as excinfo:
            with urlopen(server.status_endpoint, timeout=2):
                pass
        assert excinfo.value.code == 500
        body = excinfo.value.read().decode("utf-8")
        assert "status_provider" in body
    finally:
        server.stop()
