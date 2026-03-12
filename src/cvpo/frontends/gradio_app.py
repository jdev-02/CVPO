"""Gradio walkthrough UI for CVPO guided workflow."""

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
) -> tuple[str, str, str, str, str, str, str]:
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

    overview = (
        f"Goal: {payload.get('goal')}\n"
        f"Experience: {payload.get('experience_level')}\n"
        f"Frontend: {payload.get('frontend_choice')} — {payload.get('frontend_description')}\n\n"
        f"Privacy: {payload.get('privacy_notice', '')}"
    )

    assessment = payload.get("honest_assessment", {})
    assessment_text = f"Summary: {assessment.get('summary', 'n/a')}\n\n"
    for key, title in [("works_well", "Works Well"), ("limitations", "Limitations"),
                       ("fit_for_purpose", "Fit For Purpose"), ("not_fit_for", "Not Fit For")]:
        values = assessment.get(key, [])
        if values:
            assessment_text += f"{title}:\n"
            for v in values:
                assessment_text += f"  - {v}\n"
    cites = assessment.get("citations", [])
    if cites:
        assessment_text += "\nReferences:\n"
        for c in cites:
            if isinstance(c, dict):
                assessment_text += f"  {c.get('short', '?')} — {c.get('url', '')}\n"

    decision = payload.get("decision_path", {})
    decision_text = (
        f"Pipeline: {decision.get('pipeline_level', 'n/a')}\n"
        f"CV Tasks: {', '.join(decision.get('cv_tasks', []))}\n"
        f"Reasoning: {decision.get('reasoning', 'n/a')}\n"
        f"Back navigation: {decision.get('back_navigation_hint', 'n/a')}"
    )

    hw = payload.get("hardware_profile", {})
    cap = payload.get("capability_assessment", [])
    val = payload.get("pre_run_validation", {})
    hw_text = (
        f"System: {hw.get('system')}\nCPU: {hw.get('cpu')}\n"
        f"Cores: {hw.get('logical_cores')}\nRAM: {hw.get('ram_gb')} GB\n"
        f"GPU: {'Yes' if hw.get('gpu_detected') else 'No'}\n\n"
    )
    for row in cap:
        hw_text += f"- {row.get('model')} ({row.get('stage')}): {row.get('status')}"
        if row.get("suggestion"):
            hw_text += f"\n  >> {row['suggestion']}"
        hw_text += "\n"
    hw_text += f"\nValidation: {val.get('status', 'n/a')} — {val.get('proceed_message', '')}"

    cards = payload.get("tradeoff_cards", [])
    cards_text = ""
    for card in cards:
        cards_text += f"[{card.get('stage')}] {card.get('model')}:\n"
        cards_text += f"  Tradeoff: {card.get('tradeoff')}\n"
        cards_text += f"  Guidance: {card.get('guidance')}\n\n"

    socratic = payload.get("socratic", {})
    socratic_text = (
        f"Question: {socratic.get('question', 'n/a')}\n\n"
        f"{'[SKIPPED]' if socratic.get('skipped') else ''}\n\n"
        f"Answer: {socratic.get('answer', 'n/a')}\n\n"
    )
    scites = socratic.get("citations", [])
    if scites:
        socratic_text += "References:\n"
        for c in scites:
            if isinstance(c, dict):
                socratic_text += f"  {c.get('short', '?')} — {c.get('url', '')}\n"

    result_text = json.dumps(payload.get("run_result", {}), indent=2)

    return overview, assessment_text, decision_text, hw_text, cards_text, socratic_text, result_text


def create_app() -> Any:
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError(
            "Gradio is not installed. Install with: pip install gradio"
        ) from exc

    with gr.Blocks(title="CVPO Guided Workflow") as app:
        gr.Markdown("# CVPO — Computer Vision Pipeline Orchestrator")
        gr.Image(value="assets/cvpo_mascot.png", show_label=False, height=200)

        with gr.Row():
            goal = gr.Dropdown(
                choices=[
                    "image_labeling", "object_finding", "object_boundaries",
                    "video_tracking", "counting_over_time", "species_identification",
                    "geese_tracking",
                ],
                value="species_identification",
                label="What are you trying to do?",
            )
            experience_level = gr.Dropdown(
                choices=["beginner", "intermediate", "advanced"],
                value="beginner",
                label="Your experience level",
            )
            frontend_choice = gr.Dropdown(
                choices=["cli", "gradio", "notebook"],
                value="gradio",
                label="Preferred interface",
            )

        with gr.Row():
            labels = gr.Textbox(value="capybara,beaver,otter,bear,dog", label="Candidate Labels")
            max_frames = gr.Slider(minimum=1, maximum=300, value=8, step=1, label="Max Frames (video)")
            skip_socratic = gr.Checkbox(value=False, label="Skip Socratic Questions")

        run_btn = gr.Button("Run Guided Workflow", variant="primary")

        with gr.Tabs():
            with gr.Tab("Overview"):
                overview_out = gr.Textbox(label="Overview", lines=6)
            with gr.Tab("Honest Assessment"):
                assessment_out = gr.Textbox(label="Honest Assessment", lines=12)
            with gr.Tab("Decision Path"):
                decision_out = gr.Textbox(label="Decision Path", lines=6)
            with gr.Tab("Hardware"):
                hw_out = gr.Textbox(label="Hardware Capability", lines=12)
            with gr.Tab("Tradeoff Cards"):
                cards_out = gr.Textbox(label="Tradeoff Cards", lines=10)
            with gr.Tab("Socratic Check"):
                socratic_out = gr.Textbox(label="Socratic Check", lines=10)
            with gr.Tab("Run Result"):
                result_out = gr.Code(label="Run Result JSON", language="json")

        run_btn.click(
            fn=_run_guided_from_ui,
            inputs=[goal, frontend_choice, experience_level, skip_socratic, labels, max_frames],
            outputs=[overview_out, assessment_out, decision_out, hw_out, cards_out, socratic_out, result_out],
        )

    return app


def launch(share: bool = False) -> None:
    app = create_app()
    app.launch(share=share)
