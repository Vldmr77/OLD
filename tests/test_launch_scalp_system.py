from __future__ import annotations

from launch_scalp_system import build_invocation


def test_default_invocation_targets_system() -> None:
    module, forwarded = build_invocation([])
    assert module == "scalp_system"
    assert forwarded == []


def test_config_path_forwarded_to_system() -> None:
    module, forwarded = build_invocation(["config.yaml"])
    assert module == "scalp_system"
    assert forwarded == ["config.yaml"]


def test_dashboard_invocation_without_separator() -> None:
    module, forwarded = build_invocation(["--dashboard", "--refresh-interval", "750"])
    assert module == "scalp_system.cli.dashboard"
    assert forwarded == ["--refresh-interval", "750"]


def test_dashboard_invocation_with_separator() -> None:
    module, forwarded = build_invocation(["--dashboard", "--", "--signal-limit", "40"])
    assert module == "scalp_system.cli.dashboard"
    assert forwarded == ["--signal-limit", "40"]
