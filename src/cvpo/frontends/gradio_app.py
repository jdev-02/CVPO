"""Gradio app for guided CVPO frontend."""

from __future__ import annotations

import json
from typing import Any

from cvpo.frontends.cli import run_guided_workflow, _to_pretty


def _run_guided_from_ui(
    goal: str,
    frontend_choice: str,
    experience_level: str,
    skip_socratic: bool,
    labels: str,
    max_frames: int,
) -> tuple[str, str]:
    parsed_labels = [label.strip() for label in labels.split(",") if label.strip()]
    payload = run_guided_workflow(
        goal=goal,
        frontend_choice=frontend_choice,
        experience_level=experience_level,
        skip_socratic=skip_socratic,
        input_image=None,
        input_video=None,
        labels=parsed_labels,
        max_frames=max_frames,
    )
    return _to_pretty(payload), json.dumps(payload, indent=2)


def create_app() -> Any:
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError(
            "Gradio is not installed. Install with: pip install gradio"
        ) from exc

    with gr.Blocks(title="CVPO Guided Workflow") as app:
        gr.Markdown("# CVPO Guided Workflow")
        gr.Markdown(
            "Deterministic guided run with frontend choice, honest assessment, "
            "hardware capability, Socratic output, and execution results."
        )
        with gr.Row():
            goal = gr.Dropdown(
                choices=["geese_tracking", "image_labeling"],
                value="geese_tracking",
                label="Goal",
            )
            frontend_choice = gr.Dropdown(
                choices=["cli", "gradio", "notebook"],
                value="gradio",
                label="Frontend Choice",
            )
            experience_level = gr.Dropdown(
                choices=["beginner", "intermediate", "advanced"],
                value="beginner",
                label="Experience Level",
            )
        with gr.Row():
            labels = gr.Textbox(
                value="goose,duck,pigeon,crow",
                label="Candidate Labels (comma-separated)",
            )
            max_frames = gr.Slider(minimum=1, maximum=300, value=8, step=1, label="Max Frames")
            skip_socratic = gr.Checkbox(value=False, label="Skip Socratic")

        run_btn = gr.Button("Run Guided Workflow")
        pretty_out = gr.Textbox(label="Pretty Output", lines=24)
        json_out = gr.Code(label="Raw JSON", language="json")
        run_btn.click(
            fn=_run_guided_from_ui,
            inputs=[goal, frontend_choice, experience_level, skip_socratic, labels, max_frames],
            outputs=[pretty_out, json_out],
        )
    return app


def launch(share: bool = False) -> None:
    app = create_app()
    app.launch(share=share)
