"""Interactive helpers for ensuring API tokens exist before runtime."""
from __future__ import annotations

import os
import shutil
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, Optional

from . import DEFAULT_CONFIG_PATH
from ..security.key_manager import KeyManager

try:  # pragma: no cover - optional dependency during packaging
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime guard
    raise RuntimeError("PyYAML is required to manage configuration files") from exc

_PLACEHOLDER_VALUES = {
    "",
    "TODO",
    "PLACEHOLDER",
    "SANDBOX_TOKEN",
    "PRODUCTION_TOKEN",
    "YOUR_TOKEN",
    "YOUR_SANDBOX_TOKEN",
    "YOUR_PRODUCTION_TOKEN",
    "YOUR_ENCRYPTED_TOKEN",
    "enc:YOUR_ENCRYPTED_TOKEN",
}
_PLACEHOLDER_PREFIXES = ("enc:YOUR", "YOUR_", "REPLACE_ME", "CHANGE_ME")
_ENV_VARS = {
    "sandbox_token": "TINKOFF_SANDBOX_TOKEN",
    "production_token": "TINKOFF_PRODUCTION_TOKEN",
}


def ensure_tokens_present(config_path: Path | None) -> Path:
    """Ensure the configuration file contains broker tokens, prompting if needed."""

    path = _ensure_config_path(config_path)
    data = _load_yaml(path)
    datafeed = data.setdefault("datafeed", {})
    if not isinstance(datafeed, dict):
        raise ValueError("datafeed section must be a mapping")

    security = data.get("security", {})
    key_manager = _build_key_manager(security, config_path=path)

    use_sandbox = bool(datafeed.get("use_sandbox", True))
    fields_to_check = []
    if _token_missing(datafeed.get("sandbox_token")):
        fields_to_check.append(("sandbox_token", "sandbox"))
    if _token_missing(datafeed.get("production_token")):
        fields_to_check.append(("production_token", "production"))

    required_field = "sandbox_token" if use_sandbox else "production_token"
    if all(field != required_field for field, _ in fields_to_check):
        # Required token exists; nothing to do
        return path

    updated = False
    for field, label in fields_to_check:
        env_var = _ENV_VARS.get(field)
        token = None
        if env_var:
            token = os.getenv(env_var)
        if not token and field == required_field:
            token = _prompt_for_token(label)
        if not token:
            # Optional token missing but not provided; skip updating
            continue
        stored = _encode_token(token, key_manager)
        datafeed[field] = stored
        updated = True

    if updated:
        _dump_yaml(path, data)

    # If after updating the required token is still missing, abort with instructions
    required_value = datafeed.get(required_field)
    if _token_missing(required_value):
        raise SystemExit(
            "Broker token is required. Rerun the command and provide a valid token when prompted."
        )

    return path


def _ensure_config_path(config_path: Path | None) -> Path:
    if config_path is None:
        return DEFAULT_CONFIG_PATH
    path = Path(config_path).expanduser()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(DEFAULT_CONFIG_PATH, path)
    return path


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")
    return data


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, indent=2, sort_keys=False, allow_unicode=True)


def _build_key_manager(security_section: Any, *, config_path: Path) -> Optional[KeyManager]:
    manager: Optional[KeyManager] = None
    env_key = os.getenv("SCALP_ENCRYPTION_KEY")
    if env_key:
        try:
            manager = KeyManager.from_key_string(env_key)
        except ValueError:
            manager = None
    if manager is not None:
        return manager
    if not isinstance(security_section, dict):
        return None
    key_path_raw = security_section.get("encryption_key_path")
    if not key_path_raw:
        return None
    key_path = Path(str(key_path_raw)).expanduser()
    if not key_path.is_absolute():
        key_path = (config_path.parent / key_path).resolve()
    if not key_path.exists():
        return None
    try:
        return KeyManager.from_file(key_path)
    except ValueError:
        return None


def _token_missing(raw_value: Any) -> bool:
    if raw_value is None:
        return True
    if not isinstance(raw_value, str):
        return False
    value = raw_value.strip()
    if value in _PLACEHOLDER_VALUES:
        return True
    upper_value = value.upper()
    if upper_value in {placeholder.upper() for placeholder in _PLACEHOLDER_VALUES}:
        return True
    return any(value.startswith(prefix) for prefix in _PLACEHOLDER_PREFIXES)


def _prompt_for_token(label: str) -> str:
    prompt = f"Enter Tinkoff {label} token: "
    while True:
        try:
            token = getpass(prompt)
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("Token entry aborted") from None
        token = token.strip()
        if not token:
            print("Token cannot be empty. Please try again.")
            continue
        try:
            confirm = getpass(f"Confirm Tinkoff {label} token: ")
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("Token entry aborted") from None
        if token != confirm.strip():
            print("Tokens do not match. Please try again.")
            continue
        return token


def _encode_token(token: str, key_manager: Optional[KeyManager]) -> str:
    if key_manager is None:
        return token
    return f"enc:{key_manager.encrypt(token)}"


__all__ = ["ensure_tokens_present"]
