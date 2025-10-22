#!/usr/bin/env python3
"""Inject Tinkoff API tokens into a configuration file."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VENDOR_PATH = PROJECT_ROOT / "scalp_system" / "vendor"
if VENDOR_PATH.exists() and str(VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(VENDOR_PATH))

from scalp_system.config import DEFAULT_CONFIG_PATH
from scalp_system.security.key_manager import KeyManager

try:
    import yaml
except ImportError as exc:  # pragma: no cover - PyYAML is an optional dependency
    raise SystemExit("PyYAML is required to run this script") from exc


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file {path} does not exist")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")
    return data


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, indent=2, sort_keys=False, allow_unicode=True)


def _encrypt(token: str, key_path: Path | None) -> str:
    if key_path is None:
        return token
    manager = KeyManager.from_file(key_path)
    return f"enc:{manager.encrypt(token)}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write sandbox/production tokens into the configuration file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--sandbox-token",
        dest="sandbox_token",
        default=os.getenv("TINKOFF_SANDBOX_TOKEN"),
        help="Sandbox API token (default: TINKOFF_SANDBOX_TOKEN env)",
    )
    parser.add_argument(
        "--production-token",
        dest="production_token",
        default=os.getenv("TINKOFF_PRODUCTION_TOKEN"),
        help="Production API token (default: TINKOFF_PRODUCTION_TOKEN env)",
    )
    parser.add_argument(
        "--key",
        dest="key_path",
        type=Path,
        help="Optional path to Fernet key used to encrypt the tokens",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override existing tokens instead of keeping current values",
    )
    args = parser.parse_args()

    if not args.sandbox_token and not args.production_token:
        raise SystemExit("Provide at least one token via arguments or environment variables")

    config_path = args.config.expanduser().resolve()
    data = _load_yaml(config_path)
    section = data.setdefault("datafeed", {})
    if not isinstance(section, dict):
        raise ValueError("datafeed section must be a mapping")

    if args.sandbox_token:
        if args.force or not section.get("sandbox_token"):
            section["sandbox_token"] = _encrypt(args.sandbox_token, args.key_path)
    if args.production_token:
        if args.force or not section.get("production_token"):
            section["production_token"] = _encrypt(args.production_token, args.key_path)

    _dump_yaml(config_path, data)
    print(f"Updated tokens in {config_path}")


if __name__ == "__main__":
    main()
