from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvpo.frontends.cli import build_parser
from cvpo.frontends import cli


def test_cli_parser_builds() -> None:
    parser = build_parser()
    args = parser.parse_args(["--version"])
    assert args.version is True


def test_cli_level0_demo_outputs_json(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["cvpo", "--level0-demo", "--labels", "goose,duck,pigeon"],
    )
    cli.main()
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["pipeline"] == "level0_classification"
    assert "top_label" in payload


def test_cli_level1_demo_outputs_json(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["cvpo", "--level1-demo"])
    cli.main()
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["pipeline"] == "level1_detect_segment"
    assert payload["mask_count"] >= 1


def test_cli_level2_demo_outputs_json(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["cvpo", "--level2-demo", "--labels", "goose,duck,pigeon"])
    cli.main()
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["pipeline"] == "level2_detect_segment_classify"
    assert payload["class_count"] >= 1


def test_cli_level3_demo_outputs_json(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["cvpo", "--level3-demo"])
    cli.main()
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["workflow"] == "level3_detect_segment_classify_track"
    assert payload["frame_count"] == 8


def test_cli_guided_workflow_outputs_assessment(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "cvpo",
            "--workflow-demo",
            "--goal",
            "geese_tracking",
            "--frontend-choice",
            "gradio",
            "--skip-socratic",
        ],
    )
    cli.main()
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["mode"] == "guided_workflow"
    assert payload["frontend_choice"] == "gradio"
    assert "decision_path" in payload
    assert "tradeoff_cards" in payload
    assert "hardware_profile" in payload
    assert "capability_assessment" in payload
    assert "honest_assessment" in payload
    assert payload["socratic"]["skipped"] is True


def test_cli_guided_workflow_pretty_format(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "cvpo",
            "--workflow-demo",
            "--goal",
            "geese_tracking",
            "--frontend-choice",
            "cli",
            "--format",
            "pretty",
        ],
    )
    cli.main()
    out = capsys.readouterr().out
    assert "CVPO Guided Workflow" in out
    assert "Decision Path" in out
    assert "Tradeoff Cards" in out
    assert "Hardware Capability" in out
    assert "Honest Assessment" in out
    assert "Socratic Check" in out
    assert "References:" in out


def test_cli_summary_format(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "cvpo",
            "--workflow-demo",
            "--goal",
            "geese_tracking",
            "--format",
            "summary",
        ],
    )
    cli.main()
    out = capsys.readouterr().out
    assert "CVPO Summary" in out
    assert "--format pretty" in out or "--format guided" in out


def test_cli_save_report_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "cvpo",
            "--level0-demo",
            "--labels",
            "goose,duck,pigeon",
            "--save-report",
            str(report_path),
        ],
    )
    cli.main()
    _ = capsys.readouterr()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["pipeline"] == "level0_classification"


def test_cli_parser_accepts_launch_gradio_flag() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["--launch-gradio"])
    assert args.launch_gradio is True


def test_cli_benchmark_outputs_payload(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "cvpo",
            "--benchmark",
            "--benchmark-workflow",
            "level0",
            "--benchmark-repeats",
            "2",
            "--benchmark-warmup",
            "0",
        ],
    )
    cli.main()
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "benchmark" in payload
    assert payload["benchmark"]["workflow"] == "level0"
