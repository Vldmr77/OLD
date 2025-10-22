from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

try:  # pragma: no cover - guard for script execution
    from scalp_system.config import DEFAULT_CONFIG_PATH
    from scalp_system.config.token_prompt import ensure_tokens_present
    from scalp_system.orchestrator import run_from_yaml
except ModuleNotFoundError:  # running as `python scalp_system/__main__.py`
    import sys

    PACKAGE_ROOT = Path(__file__).resolve().parent.parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))

    from scalp_system.config import DEFAULT_CONFIG_PATH
    from scalp_system.config.token_prompt import ensure_tokens_present
    from scalp_system.orchestrator import run_from_yaml
except RuntimeError as exc:  # pragma: no cover - PyYAML missing or similar
    from scalp_system.config import DEFAULT_CONFIG_PATH
    from scalp_system.orchestrator import run_from_yaml

    def ensure_tokens_present(config_path: Path | None):  # type: ignore[override]
        resolved = (config_path or DEFAULT_CONFIG_PATH).expanduser()
        print(f"[warn] Token setup skipped: {exc}")
        return resolved


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scalping system orchestrator")
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML configuration (defaults to packaged example)",
    )
    args = parser.parse_args(argv)
    config_path = ensure_tokens_present(args.config)
    run_from_yaml(config_path)


if __name__ == "__main__":
    main()
