"""Notebook helpers for CVPO demos."""

from __future__ import annotations

from typing import Any

from cvpo.frontends.cli import run_guided_workflow, _to_pretty

def notebook_bootstrap_text() -> str:
    return (
        "CVPO notebook flow: INTRO -> EXPECTATIONS -> CONFIG -> WARMUP -> BUILD "
        "-> MEASURE -> ANALYSIS -> CONFIRMATION"
    )


def run_guided_notebook_demo(
    goal: str = "geese_tracking",
    frontend_choice: str = "notebook",
    experience_level: str = "beginner",
    skip_socratic: bool = False,
    labels: list[str] | None = None,
    max_frames: int = 8,
) -> dict[str, Any]:
    if labels is None:
        labels = ["goose", "duck", "pigeon", "crow"]
    return run_guided_workflow(
        goal=goal,
        frontend_choice=frontend_choice,
        experience_level=experience_level,
        skip_socratic=skip_socratic,
        input_image=None,
        input_video=None,
        labels=labels,
        max_frames=max_frames,
    )


def pretty_notebook_summary(payload: dict[str, Any]) -> str:
    return _to_pretty(payload)
