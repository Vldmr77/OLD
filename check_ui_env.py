"""Environment diagnostics for the scalping system UI."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict


def _discover_config_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser()
    candidates = (
        "config.yaml",
        "config.yml",
        "config.json",
        "config.example.yaml",
        "config.example.yml",
    )
    cwd = Path.cwd()
    for name in candidates:
        candidate = cwd / name
        if candidate.exists():
            return candidate
    try:
        from scalp_system.config import DEFAULT_CONFIG_PATH

        return DEFAULT_CONFIG_PATH
    except Exception:  # pragma: no cover - import guard during packaging
        return cwd / "config.yaml"


def _module_available(name: str) -> tuple[bool, str | None]:
    try:
        importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostic only
        return False, str(exc)
    return True, None


def collect_environment(config_path: Path | None = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "python": {
            "version": sys.version,
            "compatible_3_9_plus": sys.version_info >= (3, 9),
        },
        "modules": {},
        "config": {
            "path": None,
            "load_error": None,
        },
        "bus": {
            "enabled": None,
            "host": None,
            "port": None,
            "reachable": None,
        },
        "storage": {
            "signals_db": {"path": None, "exists": None},
            "health_db": {"path": None, "exists": None},
        },
    }

    for module_name in ("tkinter", "yaml"):
        available, error = _module_available(module_name)
        key = "pyyaml" if module_name == "yaml" else module_name
        result["modules"][key] = {"available": available}
        if error:
            result["modules"][key]["error"] = error

    config_location = _discover_config_path(config_path)
    result["config"]["path"] = str(config_location)

    cfg = None
    try:
        from scalp_system.config.loader import load_config

        cfg = load_config(config_location)
    except Exception as exc:  # pragma: no cover - diagnostic only
        result["config"]["load_error"] = str(exc)
    else:
        result["config"]["load_error"] = None
        try:
            repository = cfg.dashboard.repository_path
            health_db = cfg.storage.health_db
        except AttributeError:  # pragma: no cover - defensive
            repository = None
            health_db = None
        if repository is not None:
            repo_path = Path(repository)
            result["storage"]["signals_db"] = {
                "path": str(repo_path),
                "exists": repo_path.exists(),
            }
        if health_db is not None:
            health_path = Path(health_db)
            result["storage"]["health_db"] = {
                "path": str(health_path),
                "exists": health_path.exists(),
            }
        try:
            bus_cfg = cfg.bus
            result["bus"].update(
                {
                    "enabled": bool(bus_cfg.enabled),
                    "host": bus_cfg.host,
                    "port": int(bus_cfg.port),
                }
            )
            if bus_cfg.enabled:
                from scalp_system.control.bus import BusClient

                client = BusClient(host=bus_cfg.host, port=bus_cfg.port, timeout=0.5)
                result["bus"]["reachable"] = client.check_available()
            else:
                result["bus"]["reachable"] = False
        except Exception as exc:  # pragma: no cover - diagnostic only
            result["bus"]["reachable"] = False
            result["bus"]["error"] = str(exc)

    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check UI runtime prerequisites")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file (defaults to discovery logic)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    args = parser.parse_args(argv)

    payload = collect_environment(args.config)
    if args.pretty:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

