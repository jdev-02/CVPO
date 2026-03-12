from __future__ import annotations

import json

import pytest

from cvpo.hardware import (
    PRIVACY_NOTICE,
    capability_assessment,
    detect_hardware,
    pre_run_validation,
)
from cvpo.frontends import cli


def test_privacy_notice_present_in_hardware_profile() -> None:
    hw = detect_hardware()
    assert "privacy_notice" in hw
    assert "never transmitted" in hw["privacy_notice"].lower()


def test_detect_hardware_manual_override() -> None:
    hw = detect_hardware(override_ram_gb=64.0, override_gpu_vram_gb=24.0)
    assert hw["ram_gb"] == 64.0
    assert hw["vram_gb"] == 24.0
    assert hw["gpu_detected"] is True
    assert hw["overrides_applied"] is True


def test_how_to_find_specs_present() -> None:
    hw = detect_hardware()
    assert "how_to_find_specs" in hw
    assert len(hw["how_to_find_specs"]) > 10


def test_pre_run_validation_clear() -> None:
    rows = [{"model": "yolov8n", "status": "good", "suggestion": None}]
    result = pre_run_validation(rows)
    assert result["status"] == "clear"


def test_pre_run_validation_warnings() -> None:
    rows = [{"model": "sam2_tiny", "status": "degraded", "suggestion": "No GPU detected."}]
    result = pre_run_validation(rows)
    assert result["status"] == "warnings"
    assert len(result["warnings"]) == 1


def test_pre_run_validation_blocked() -> None:
    rows = [{"model": "sam2_tiny", "status": "not_recommended", "suggestion": "RAM too low."}]
    result = pre_run_validation(rows)
    assert result["status"] == "blocked"
    assert len(result["blockers"]) == 1


def test_capability_assessment_includes_suggestions() -> None:
    hw = detect_hardware(override_ram_gb=2.0)
    rows = capability_assessment(hw)
    below_min = [r for r in rows if r["status"] == "not_recommended"]
    assert len(below_min) >= 1
    assert below_min[0]["suggestion"] is not None


def test_cli_guided_with_override_shows_inline_guidance(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "cvpo",
            "--workflow-demo",
            "--goal",
            "geese_tracking",
            "--override-ram-gb",
            "2.0",
            "--format",
            "pretty",
        ],
    )
    cli.main()
    out = capsys.readouterr().out
    assert "--override-ram-gb" in out
    assert "find your actual specs" in out.lower()


def test_cli_guided_with_override(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "cvpo",
            "--workflow-demo",
            "--goal",
            "geese_tracking",
            "--override-ram-gb",
            "2.0",
        ],
    )
    cli.main()
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["hardware_profile"]["ram_gb"] == 2.0
    assert payload["pre_run_validation"]["status"] in ("blocked", "warnings")
    assert "privacy_notice" in payload
