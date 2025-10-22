"""Utility helpers for mutating orchestrator configuration files."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from .token_prompt import _dump_yaml, _ensure_config_path, _load_yaml


def _clean_instruments(values: Optional[Iterable[str]]) -> list[str]:
    if values is None:
        return []
    cleaned: list[str] = []
    for value in values:
        if value is None:
            continue
        figi = str(value).strip()
        if not figi:
            continue
        if figi not in cleaned:
            cleaned.append(figi)
    return cleaned


def update_instrument_lists(
    config_path: Path | None,
    *,
    active: Optional[Iterable[str]] = None,
    monitored: Optional[Iterable[str]] = None,
) -> bool:
    """Persist instrument lists back to the configuration file."""

    path = _ensure_config_path(config_path)
    data = _load_yaml(path)
    datafeed = data.setdefault("datafeed", {})
    if not isinstance(datafeed, dict):
        raise ValueError("datafeed section must be a mapping")

    changed = False
    if active is not None:
        cleaned_active = _clean_instruments(active)
        if datafeed.get("instruments") != cleaned_active:
            datafeed["instruments"] = cleaned_active
            changed = True
    if monitored is not None:
        cleaned_monitored = _clean_instruments(monitored)
        if datafeed.get("monitored_instruments") != cleaned_monitored:
            datafeed["monitored_instruments"] = cleaned_monitored
            changed = True

    if changed:
        _dump_yaml(path, data)
    return changed


def set_operational_mode(
    config_path: Path | None,
    *,
    mode: Optional[str] = None,
    use_sandbox: Optional[bool] = None,
) -> bool:
    """Update the system mode and sandbox flag within the configuration file."""

    path = _ensure_config_path(config_path)
    data = _load_yaml(path)
    changed = False

    if mode is not None:
        system = data.setdefault("system", {})
        if not isinstance(system, dict):
            raise ValueError("system section must be a mapping")
        if system.get("mode") != mode:
            system["mode"] = mode
            changed = True

    if use_sandbox is not None:
        datafeed = data.setdefault("datafeed", {})
        if not isinstance(datafeed, dict):
            raise ValueError("datafeed section must be a mapping")
        if bool(datafeed.get("use_sandbox")) != bool(use_sandbox):
            datafeed["use_sandbox"] = bool(use_sandbox)
            changed = True

    if changed:
        _dump_yaml(path, data)
    return changed


__all__ = ["set_operational_mode", "update_instrument_lists"]
