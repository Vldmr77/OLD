from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import scalp_system.orchestrator as orchestrator


@pytest.fixture
def dummy_config() -> SimpleNamespace:
    return SimpleNamespace(
        logging=SimpleNamespace(level="INFO"),
        datafeed=SimpleNamespace(
            use_sandbox=True,
            allow_tokenless=True,
            sandbox_token=None,
            production_token=None,
        ),
        system=SimpleNamespace(mode="forward-test"),
    )


def _install_asyncio_stub(monkeypatch):
    called = {}

    async def _noop():
        return None

    monkeypatch.setattr(orchestrator, "Orchestrator", lambda config: SimpleNamespace(run=_noop))

    def _fake_run(coro):
        called["run"] = True
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(orchestrator.asyncio, "run", _fake_run)
    return called


def test_run_from_yaml_skips_sdk_when_tokenless(monkeypatch, tmp_path, dummy_config):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(orchestrator, "load_config", lambda path: dummy_config)

    called = {"ensure": 0}

    def _ensure():
        called["ensure"] += 1
        raise orchestrator.TinkoffSDKUnavailable("missing")

    monkeypatch.setattr(orchestrator, "ensure_sdk_available", _ensure)

    asyncio_calls = _install_asyncio_stub(monkeypatch)

    orchestrator.run_from_yaml(config_path)

    assert called["ensure"] == 1
    assert asyncio_calls.get("run") is True


def test_run_from_yaml_requires_sdk_when_tokens_present(monkeypatch, tmp_path, dummy_config):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")

    dummy_config.datafeed.sandbox_token = "REAL"
    monkeypatch.setattr(orchestrator, "load_config", lambda path: dummy_config)

    called = {"ensure": 0}

    def _ensure():
        called["ensure"] += 1

    monkeypatch.setattr(orchestrator, "ensure_sdk_available", _ensure)

    asyncio_calls = _install_asyncio_stub(monkeypatch)

    orchestrator.run_from_yaml(config_path)

    assert called["ensure"] == 1
    assert asyncio_calls.get("run") is True
