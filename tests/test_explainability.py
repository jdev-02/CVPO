from __future__ import annotations

import json

import pytest

from cvpo.education.explainability import get_all_explainability, get_explainability
from cvpo.frontends import cli


def test_get_explainability_returns_entry() -> None:
    entry = get_explainability("output_format")
    assert "why" in entry
    assert "when_to_change" in entry


def test_get_all_explainability_has_multiple_keys() -> None:
    all_entries = get_all_explainability()
    assert len(all_entries) >= 5
    for key, entry in all_entries.items():
        assert "why" in entry, f"Missing 'why' for {key}"


def test_cli_explain_flag(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["cvpo", "--explain", "output_format"])
    cli.main()
    out = capsys.readouterr().out
    assert "Why:" in out
    assert "When to change:" in out


def test_cli_explain_all_flag(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["cvpo", "--explain-all"])
    cli.main()
    out = capsys.readouterr().out
    assert "CVPO Settings Explainability Guide" in out
    assert "output_format" in out


def test_guided_workflow_includes_settings_explainability(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["cvpo", "--workflow-demo", "--goal", "geese_tracking"],
    )
    cli.main()
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "settings_explainability" in payload
    assert "experience_level" in payload["settings_explainability"]
