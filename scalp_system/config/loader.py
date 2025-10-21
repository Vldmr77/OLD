"""Config loader utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import OrchestratorConfig

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


def load_config(path: Path | str | None = None, **overrides: Any) -> OrchestratorConfig:
    """Load the orchestrator config from YAML or JSON file with optional overrides."""
    data: dict[str, Any] = {}
    if path:
        path_obj = Path(path).expanduser()
        if path_obj.suffix.lower() in {".json"}:
            data = json.loads(path_obj.read_text(encoding="utf-8"))
        else:
            if yaml is None:
                raise RuntimeError("PyYAML is required to load YAML configurations")
            with path_obj.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
    if overrides:
        data.update(json.loads(json.dumps(overrides)))
    return OrchestratorConfig.from_dict(data)


__all__ = ["load_config"]
