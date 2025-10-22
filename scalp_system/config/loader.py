"""Config loader utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import DEFAULT_CONFIG_PATH
from .base import OrchestratorConfig

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


def load_config(path: Path | str | None = None, **overrides: Any) -> OrchestratorConfig:
    """Load the orchestrator config from YAML or JSON file with optional overrides."""
    data: dict[str, Any] = {}
    config_path: Path | None
    if path is not None:
        config_path = Path(path).expanduser()
    else:
        config_path = _discover_default_path()

    if config_path.exists():
        if config_path.suffix.lower() in {".json"}:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            if yaml is None:
                raise RuntimeError("PyYAML is required to load YAML configurations")
            with config_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
    if overrides:
        data.update(json.loads(json.dumps(overrides)))
    return OrchestratorConfig.from_dict(data)


def _discover_default_path() -> Path:
    """Locate the most appropriate configuration file when none is provided."""
    cwd = Path.cwd()
    candidate_names = (
        "config.yaml",
        "config.yml",
        "config.json",
        "config.example.yaml",
        "config.example.yml",
    )
    for name in candidate_names:
        candidate = cwd / name
        if candidate.exists():
            return candidate
    return DEFAULT_CONFIG_PATH


__all__ = ["load_config"]
