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


def _install_asyncio_stub(monkeypatch, run_factory=None):
    called = {}

    def _default_factory():
        async def _noop():
            return None

        return _noop

    factory = run_factory or _default_factory

    monkeypatch.setattr(
        orchestrator,
        "Orchestrator",
        lambda config, config_path=None: SimpleNamespace(run=factory()),
    )

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


def test_run_from_yaml_restarts_when_requested(monkeypatch, tmp_path, dummy_config):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(orchestrator, "load_config", lambda path: dummy_config)

    def _ensure():
        return None

    monkeypatch.setattr(orchestrator, "ensure_sdk_available", _ensure)

    runs = {"count": 0}

    def _run_factory():
        async def _run():
            runs["count"] += 1
            if runs["count"] == 1:
                raise orchestrator.RestartRequested()

        return _run

    asyncio_calls = _install_asyncio_stub(monkeypatch, run_factory=_run_factory)

    orchestrator.run_from_yaml(config_path)

    assert runs["count"] == 2
    assert asyncio_calls.get("run") is True
