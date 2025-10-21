from __future__ import annotations

from pathlib import Path

import pytest

from scalp_system.config.token_prompt import ensure_tokens_present
from scalp_system.security.key_manager import KeyManager

try:
    import yaml
except ImportError:  # pragma: no cover - pytest will fail earlier if PyYAML missing
    yaml = None


@pytest.mark.skipif(yaml is None, reason="PyYAML is required for token prompt tests")
def test_ensure_tokens_uses_env_for_sandbox(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
        datafeed:
          sandbox_token: ""
          use_sandbox: true
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("TINKOFF_SANDBOX_TOKEN", "SANDBOX123")

    ensure_tokens_present(config_path)

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["datafeed"]["sandbox_token"] == "SANDBOX123"


@pytest.mark.skipif(yaml is None, reason="PyYAML is required for token prompt tests")
def test_ensure_tokens_encrypts_when_key_available(tmp_path, monkeypatch):
    key_path = tmp_path / "key.txt"
    key = KeyManager.generate()
    key_path.write_text(key.serialise(), encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
        security:
          encryption_key_path: "{key_path}"
        datafeed:
          sandbox_token: ""
          use_sandbox: true
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("TINKOFF_SANDBOX_TOKEN", "ENCRYPTME")

    ensure_tokens_present(config_path)

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stored = data["datafeed"]["sandbox_token"]
    assert stored.startswith("enc:")
    decrypted = key.maybe_decrypt(stored)
    assert decrypted == "ENCRYPTME"


@pytest.mark.skipif(yaml is None, reason="PyYAML is required for token prompt tests")
def test_existing_token_is_preserved(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
        datafeed:
          sandbox_token: "EXISTING"
          use_sandbox: true
        """,
        encoding="utf-8",
    )
    monkeypatch.delenv("TINKOFF_SANDBOX_TOKEN", raising=False)

    ensure_tokens_present(config_path)

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["datafeed"]["sandbox_token"] == "EXISTING"
