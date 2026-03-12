from __future__ import annotations

from cvpo.frontends.notebook import pretty_notebook_summary, run_guided_notebook_demo


def test_notebook_guided_demo_returns_payload() -> None:
    payload = run_guided_notebook_demo(goal="image_labeling", frontend_choice="notebook")
    assert payload["mode"] == "guided_workflow"
    assert payload["frontend_choice"] == "notebook"


def test_notebook_pretty_summary_contains_header() -> None:
    payload = run_guided_notebook_demo(goal="image_labeling", frontend_choice="notebook")
    summary = pretty_notebook_summary(payload)
    assert "CVPO Guided Workflow" in summary
