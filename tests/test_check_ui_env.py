from __future__ import annotations

from pathlib import Path

from check_ui_env import collect_environment


def test_collect_environment_structure() -> None:
    payload = collect_environment()
    assert "python" in payload
    assert payload["python"]["compatible_3_9_plus"] in {True, False}
    assert "modules" in payload and "tkinter" in payload["modules"]
    assert "pyyaml" in payload["modules"]
    assert "config" in payload
    assert "path" in payload["config"]
    assert "bus" in payload and "reachable" in payload["bus"]
    assert "storage" in payload
    assert "signals_db" in payload["storage"]


def test_collect_environment_with_custom_config(tmp_path) -> None:
    sample = tmp_path / "config.yaml"
    sample.write_text("{}", encoding="utf-8")
    payload = collect_environment(sample)
    assert Path(payload["config"]["path"]).resolve() == sample
