from pathlib import Path

import yaml

from scalp_system.config.loader import load_config


def test_load_config_prefers_cwd_file(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "system": {"mode": "forward-test"},
                "datafeed": {"instruments": ["AAA"], "monitored_instruments": ["AAA"]},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config.system.mode == "forward-test"
    assert config.datafeed.instruments == ["AAA"]


def test_load_config_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert Path(config.dashboard.repository_path).name == "signals.db"
