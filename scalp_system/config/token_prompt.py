"""Interactive helpers for ensuring API tokens exist before runtime."""
from __future__ import annotations

import json
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

    tokens_path = _resolve_tokens_file(data, config_path=path)
    if tokens_path is not None:
        tokens_from_file = _load_tokens_file(tokens_path)
        for key in ("sandbox_token", "production_token"):
            if _token_missing(datafeed.get(key)) and tokens_from_file.get(key):
                datafeed[key] = tokens_from_file[key]

    use_sandbox = bool(datafeed.get("use_sandbox", True))
    allow_tokenless = bool(datafeed.get("allow_tokenless", False))
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
        if not token and field == required_field and not allow_tokenless:
            token = _prompt_for_token(label)
        if not token:
            # Optional token missing but not provided; skip updating
            continue
        stored = _encode_token(token, key_manager)
        datafeed[field] = stored
        updated = True

    required_missing = _token_missing(datafeed.get(required_field))
    if required_missing:
        if allow_tokenless:
            if datafeed.get(required_field) is not None:
                datafeed[required_field] = None
                updated = True
            print("No broker token supplied; continuing in offline mode.")
        else:
            raise SystemExit(
                "Broker token is required. Rerun the command and provide a valid token when prompted."
            )

    if updated:
        _dump_yaml(path, data)
        if tokens_path is not None:
            _write_tokens_file(
                tokens_path,
                sandbox=datafeed.get("sandbox_token"),
                production=datafeed.get("production_token"),
            )

    return path


def token_status(config_path: Path | None) -> Dict[str, bool]:
    """Return booleans indicating whether sandbox/production tokens exist."""

    path = _ensure_config_path(config_path)
    data = _load_yaml(path)
    datafeed = data.get("datafeed", {}) if isinstance(data, dict) else {}
    sandbox_present = not _token_missing(datafeed.get("sandbox_token"))
    production_present = not _token_missing(datafeed.get("production_token"))
    tokens_path = _resolve_tokens_file(data, config_path=path)
    if tokens_path is not None:
        tokens_from_file = _load_tokens_file(tokens_path)
        if tokens_from_file.get("sandbox_token"):
            sandbox_present = True
        if tokens_from_file.get("production_token"):
            production_present = True
    return {"sandbox": sandbox_present, "production": production_present}


def store_tokens(
    config_path: Path | None,
    *,
    sandbox: Optional[str] = None,
    production: Optional[str] = None,
) -> bool:
    """Persist provided tokens back to the configuration file.

    Empty strings clear the respective value. ``None`` leaves the existing token
    untouched. Returns ``True`` when the file contents were modified.
    """

    path = _ensure_config_path(config_path)
    data = _load_yaml(path)
    datafeed = data.setdefault("datafeed", {})
    if not isinstance(datafeed, dict):
        raise ValueError("datafeed section must be a mapping")

    security = data.get("security", {})
    key_manager = _build_key_manager(security, config_path=path)

    tokens_path = _resolve_tokens_file(data, config_path=path)

    updated = False

    def _apply(field: str, value: Optional[str]) -> None:
        nonlocal updated
        if value is None:
            return
        if value == "":
            if datafeed.get(field) is not None:
                datafeed[field] = None
                updated = True
            return
        encoded = _encode_token(value, key_manager)
        if datafeed.get(field) != encoded:
            datafeed[field] = encoded
            updated = True

    _apply("sandbox_token", sandbox)
    _apply("production_token", production)

    if updated:
        _dump_yaml(path, data)
    if tokens_path is not None:
        _write_tokens_file(
            tokens_path,
            sandbox=datafeed.get("sandbox_token"),
            production=datafeed.get("production_token"),
        )

    return updated


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


def _resolve_tokens_file(data: Dict[str, Any], *, config_path: Path) -> Optional[Path]:
    brokers = data.get("brokers")
    if not isinstance(brokers, dict):
        return None
    tinkoff = brokers.get("tinkoff")
    if not isinstance(tinkoff, dict):
        return None
    raw_path = tinkoff.get("tokens_file")
    if not raw_path:
        return None
    storage = data.get("storage") if isinstance(data.get("storage"), dict) else {}
    return _resolve_tokens_path(
        raw_path, storage=storage, config_dir=config_path.parent, prefer_existing=False
    )


def _resolve_tokens_path(
    raw_path: object, *, storage: Any, config_dir: Path, prefer_existing: bool
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
    if fallback_parts and fallback_parts[0] == config_dir.name:
        path = Path(*fallback_parts[1:]) if len(fallback_parts) > 1 else Path()
    return (config_dir / path).resolve()


def _load_tokens_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle) or {}
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str) and v}


def _write_tokens_file(
    path: Path, *, sandbox: Optional[str], production: Optional[str]
) -> None:
    payload: Dict[str, Optional[str]] = {}
    if sandbox:
        payload["sandbox_token"] = sandbox
    if production:
        payload["production_token"] = production
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


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


def is_placeholder_token(raw_value: Any) -> bool:
    """Return True when the provided token is effectively missing."""

    return _token_missing(raw_value)


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


__all__ = [
    "ensure_tokens_present",
    "is_placeholder_token",
    "store_tokens",
    "token_status",
]
