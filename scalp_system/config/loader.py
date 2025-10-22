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
    base_dir = config_path.parent if config_path is not None else Path.cwd()
    _apply_broker_tokens(data, base_dir)

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


def _apply_broker_tokens(data: dict[str, Any], base_dir: Path) -> None:
    brokers = data.get("brokers")
    if not isinstance(brokers, dict):
        return
    tinkoff = brokers.get("tinkoff")
    if not isinstance(tinkoff, dict):
        return
    tokens_file = tinkoff.get("tokens_file")
    if not tokens_file:
        return
    storage = data.get("storage") if isinstance(data.get("storage"), dict) else {}
    path = _resolve_tokens_path(
        tokens_file, storage=storage, config_dir=base_dir, prefer_existing=True
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    datafeed = data.setdefault("datafeed", {})
    if not isinstance(datafeed, dict):
        return
    for key in ("sandbox_token", "production_token"):
        if not datafeed.get(key) and payload.get(key):
            datafeed[key] = payload[key]


def _resolve_tokens_path(
    raw_path: object,
    *,
    storage: dict | None,
    config_dir: Path,
    prefer_existing: bool,
) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    base = Path.cwd()
    if isinstance(storage, dict):
        base_path_raw = storage.get("base_path")
        if base_path_raw:
            base_candidate = Path(str(base_path_raw)).expanduser()
            if not base_candidate.is_absolute():
                base_candidate = (Path.cwd() / base_candidate).resolve()
            base = base_candidate
    parts = list(path.parts)
    if parts and parts[0] == base.name:
        path = Path(*parts[1:]) if len(parts) > 1 else Path()
    resolved = (base / path).resolve()
    if not prefer_existing or resolved.exists():
        return resolved
    fallback_parts = list(path.parts)
    base_conf = config_dir
    if fallback_parts and fallback_parts[0] == base_conf.name:
        path = Path(*fallback_parts[1:]) if len(fallback_parts) > 1 else Path()
    return (base_conf / path).resolve()


__all__ = ["load_config"]
