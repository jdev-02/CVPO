"""CLI entrypoint for CVPO."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
from pathlib import Path
from typing import Any

import numpy as np

from cvpo.core.data_types import PipelineContext
from cvpo.education import get_socratic_block, get_tradeoff_cards, onboarding_questions, resolve_goal
from cvpo.education.explainability import get_all_explainability, get_explainability
from cvpo.hardware import capability_assessment, detect_hardware, pre_run_validation
from cvpo.core.stage import StageConfig
from cvpo.pipelines import build_level0_pipeline, build_level1_pipeline, build_level2_pipeline
from cvpo.stages import ByteTrackStage, YOLOv8Stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cvpo", description="CVPO CLI")
    parser.add_argument("--version", action="store_true", help="Show CVPO version and exit.")
    parser.add_argument(
        "--level0-demo",
        action="store_true",
        help="Run deterministic Level 0 single-image classification demo.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="goose,duck,pigeon,crow",
        help="Comma-separated candidate labels for Level 0 demo.",
    )
    parser.add_argument(
        "--level1-demo",
        action="store_true",
        help="Run deterministic Level 1 detection->segmentation demo.",
    )
    parser.add_argument(
        "--level2-demo",
        action="store_true",
        help="Run deterministic Level 2 detection->segmentation->classification demo.",
    )
    parser.add_argument(
        "--level3-demo",
        action="store_true",
        help="Run deterministic full workflow demo across synthetic video frames.",
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default=None,
        help="Optional path to an input image for Level 0/1/2 demos.",
    )
    parser.add_argument(
        "--input-video",
        type=str,
        default=None,
        help="Optional path to an input video for Level 3 demo.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=8,
        help="Max frames to process for video demo (Level 3).",
    )
    parser.add_argument(
        "--workflow-demo",
        action="store_true",
        help="Run guided deterministic workflow with frontend choice and honest assessment.",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="geese_tracking",
        choices=["geese_tracking", "image_labeling"],
        help="Problem goal for guided workflow.",
    )
    parser.add_argument(
        "--frontend-choice",
        type=str,
        default="cli",
        choices=["cli", "gradio", "notebook"],
        help="Frontend selection for guided workflow.",
    )
    parser.add_argument(
        "--experience-level",
        type=str,
        default="beginner",
        choices=["beginner", "intermediate", "advanced"],
        help="User experience level for guided workflow output.",
    )
    parser.add_argument(
        "--skip-socratic",
        action="store_true",
        help="Skip Socratic question prompts in guided workflow output.",
    )
    parser.add_argument(
        "--real-models",
        action="store_true",
        help="Use real model backends (requires model dependencies installed).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "pretty", "summary", "guided"],
        help="Output format: json, pretty (full), summary (compact), guided (step-by-step).",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Optional path to save run report (.json or .md).",
    )
    parser.add_argument(
        "--launch-gradio",
        action="store_true",
        help="Launch Gradio guided UI.",
    )
    parser.add_argument(
        "--gradio-share",
        action="store_true",
        help="Enable Gradio public sharing.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark suite for a selected workflow.",
    )
    parser.add_argument(
        "--benchmark-workflow",
        type=str,
        default="guided_geese",
        choices=["level0", "level1", "level2", "level3", "guided_geese"],
        help="Workflow target for benchmark run.",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=5,
        help="Number of timed benchmark repetitions.",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=1,
        help="Number of warmup runs before timed repetitions.",
    )
    parser.add_argument(
        "--benchmark-env-tag",
        type=str,
        default="local",
        help="Environment label for cross-machine comparison.",
    )
    parser.add_argument(
        "--benchmark-baseline",
        type=str,
        default=None,
        help="Optional baseline JSON file for regression check.",
    )
    parser.add_argument(
        "--benchmark-regression-threshold",
        type=float,
        default=15.0,
        help="Allowed mean-latency regression percent against baseline.",
    )
    parser.add_argument(
        "--explain",
        type=str,
        default=None,
        help="Show WHY a setting exists and when to change it. Example: --explain output_format",
    )
    parser.add_argument(
        "--explain-all",
        action="store_true",
        help="Show explainability cards for every configurable setting.",
    )
    parser.add_argument(
        "--override-ram-gb",
        type=float,
        default=None,
        help="Manually specify RAM in GB (overrides auto-detection).",
    )
    parser.add_argument(
        "--override-gpu-vram-gb",
        type=float,
        default=None,
        help="Manually specify GPU VRAM in GB (overrides auto-detection).",
    )
    parser.add_argument(
        "--override-gpu",
        action="store_true",
        default=None,
        help="Manually declare GPU is present (overrides auto-detection).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.version:
        print("cvpo 0.1.0")
        return
    if args.explain_all:
        _print_all_explainability()
        return
    if args.explain:
        _print_explainability(args.explain)
        return
    if args.launch_gradio:
        from cvpo.frontends.gradio_app import launch as launch_gradio

        launch_gradio(share=args.gradio_share)
        return
    if args.benchmark:
        from cvpo.benchmark import BenchmarkConfig, regression_check, run_benchmark

        labels = [label.strip() for label in args.labels.split(",") if label.strip()]
        config = BenchmarkConfig(
            workflow=args.benchmark_workflow,
            repeats=args.benchmark_repeats,
            warmup=args.benchmark_warmup,
            labels=labels,
            max_frames=args.max_frames,
            env_tag=args.benchmark_env_tag,
        )
        payload: dict[str, Any] = {"benchmark": run_benchmark(config)}
        if args.benchmark_baseline:
            baseline_payload = _read_json(args.benchmark_baseline)
            baseline = baseline_payload.get("benchmark", baseline_payload)
            payload["regression"] = regression_check(
                current_result=payload["benchmark"],
                baseline_result=baseline,
                allowed_regression_pct=args.benchmark_regression_threshold,
            )
        _emit(payload, output_format=args.format)
        _save_report(payload, output_format=args.format, report_path=args.save_report)
        _save_benchmark_timeseries_csv(payload, report_path=args.save_report)
        return
    cls_backend = "siglip" if args.real_models else "deterministic"
    det_backend = "yolo" if args.real_models else "deterministic"
    seg_backend = "deterministic"

    if args.level0_demo:
        labels = [label.strip() for label in args.labels.split(",") if label.strip()]
        pipeline = build_level0_pipeline(candidate_labels=labels, backend=cls_backend)
        image = _load_image_or_synthetic(args.input_image)
        context = PipelineContext(run_id="cli-level0", input_source=args.input_image or "synthetic", frontend="cli")
        result = pipeline.run(image, context)
        payload = {
            "pipeline": pipeline.name,
            "real_models": args.real_models,
            "top_label": result.classes[0].label,
            "confidence": result.classes[0].confidence,
            "scores": dict(result.classes[0].scores),
        }
        _emit(payload, output_format=args.format)
        _save_report(payload, output_format=args.format, report_path=args.save_report)
        return
    if args.level1_demo:
        pipeline = build_level1_pipeline(
            detection_backend=det_backend,
            segmentation_backend=seg_backend,
        )
        image = _load_image_or_synthetic(args.input_image, make_bright_square=not args.real_models)
        context = PipelineContext(run_id="cli-level1", input_source=args.input_image or "synthetic", frontend="cli")
        result = pipeline.run(image, context)
        payload = {
            "pipeline": pipeline.name,
            "real_models": args.real_models,
            "mask_count": len(result.masks),
            "first_mask_pixels": int(result.masks[0].mask.sum()) if result.masks else 0,
        }
        _emit(payload, output_format=args.format)
        _save_report(payload, output_format=args.format, report_path=args.save_report)
        return
    if args.level2_demo:
        labels = [label.strip() for label in args.labels.split(",") if label.strip()]
        pipeline = build_level2_pipeline(
            candidate_labels=labels,
            detection_backend=det_backend,
            segmentation_backend=seg_backend,
            classification_backend=cls_backend,
        )
        image = _load_image_or_synthetic(args.input_image, make_bright_square=True)
        context = PipelineContext(run_id="cli-level2", input_source="synthetic", frontend="cli")
        result = pipeline.run(image, context)
        payload = {
            "pipeline": pipeline.name,
            "class_count": len(result.classes),
            "top_labels": [entry.label for entry in result.classes],
        }
        _emit(payload, output_format=args.format)
        _save_report(payload, output_format=args.format, report_path=args.save_report)
        return
    if args.level3_demo:
        payload = run_level3_demo(input_video=args.input_video, max_frames=args.max_frames)
        _emit(payload, output_format=args.format)
        _save_report(payload, output_format=args.format, report_path=args.save_report)
        return
    if args.workflow_demo:
        payload = run_guided_workflow(
            goal=args.goal,
            frontend_choice=args.frontend_choice,
            experience_level=args.experience_level,
            skip_socratic=args.skip_socratic,
            input_image=args.input_image,
            input_video=args.input_video,
            labels=[label.strip() for label in args.labels.split(",") if label.strip()],
            max_frames=args.max_frames,
            override_ram_gb=args.override_ram_gb,
            override_gpu_vram_gb=args.override_gpu_vram_gb,
            override_gpu_detected=args.override_gpu if args.override_gpu else None,
        )
        _emit(payload, output_format=args.format)
        _save_report(payload, output_format=args.format, report_path=args.save_report)
        return
    parser.print_help()


def run_level3_demo(input_video: str | None = None, max_frames: int = 8) -> dict:
    labels = ["goose", "duck", "pigeon", "crow"]
    level2 = build_level2_pipeline(
        candidate_labels=labels,
        detection_backend="deterministic",
        segmentation_backend="deterministic",
        classification_backend="deterministic",
    )
    tracker = ByteTrackStage(
        StageConfig(name="tracking", model_name="bytetrack", params={"max_distance": 45.0})
    )
    detector = YOLOv8Stage(
        StageConfig(name="detection", model_name="yolov8", params={"backend": "deterministic"})
    )

    from cvpo.core.data_types import TrackState

    per_frame = []
    track_state = TrackState()
    if input_video:
        frames = _load_video_frames(input_video, max_frames=max_frames)
    else:
        frames = _synthetic_video_frames(frame_count=max_frames)

    for frame_idx, frame in enumerate(frames):
        ctx = PipelineContext(
            run_id=f"cli-level3-{frame_idx}",
            input_source="synthetic-video",
            frontend="cli",
            metadata={"frame_index": frame_idx, "track_state": track_state},
        )
        cls_result = level2.run(frame, ctx)
        det_result = detector.run(frame, ctx)
        trk_result = tracker.run(det_result, ctx)
        track_state = ctx.metadata["track_state"]
        per_frame.append(
            {
                "frame_index": frame_idx,
                "detections": len(det_result.detections),
                "classes": [entry.label for entry in cls_result.classes],
                "track_ids": [track.track_id for track in trk_result.tracks],
            }
        )

    unique_track_ids = sorted({track_id for row in per_frame for track_id in row["track_ids"]})
    return {
        "workflow": "level3_detect_segment_classify_track",
        "frame_count": len(per_frame),
        "unique_track_ids": unique_track_ids,
        "input_video": input_video,
        "per_frame": per_frame,
    }


def run_guided_workflow(
    goal: str,
    frontend_choice: str,
    experience_level: str,
    skip_socratic: bool,
    input_image: str | None,
    input_video: str | None,
    labels: list[str],
    max_frames: int,
    override_ram_gb: float | None = None,
    override_gpu_vram_gb: float | None = None,
    override_gpu_detected: bool | None = None,
) -> dict[str, Any]:
    frontend_info = _frontend_info(frontend_choice)
    decision = resolve_goal(goal)
    tradeoff_cards = get_tradeoff_cards(goal)
    hardware = detect_hardware(
        override_ram_gb=override_ram_gb,
        override_gpu_vram_gb=override_gpu_vram_gb,
        override_gpu_detected=override_gpu_detected,
    )
    capability = capability_assessment(hardware)
    validation = pre_run_validation(capability)
    assessment = _honest_assessment(goal)
    socratic = get_socratic_block(
        goal=goal,
        experience_level=experience_level,
        skip_socratic=skip_socratic,
    )

    if goal == "geese_tracking":
        run_payload = run_level3_demo(input_video=input_video, max_frames=max_frames)
    elif goal == "image_labeling":
        pipeline = build_level0_pipeline(candidate_labels=labels, backend="deterministic")
        image = _load_image_or_synthetic(input_image)
        ctx = PipelineContext(
            run_id="guided-level0",
            input_source=input_image or "synthetic",
            frontend=frontend_choice,
        )
        result = pipeline.run(image, ctx)
        run_payload = {
            "workflow": "level0_classification",
            "top_label": result.classes[0].label,
            "confidence": result.classes[0].confidence,
            "scores": dict(result.classes[0].scores),
        }
    else:  # pragma: no cover
        raise ValueError(f"Unsupported goal: {goal}")

    active_settings_context = {
        "experience_level": get_explainability("experience_level"),
        "frontend_choice": get_explainability("frontend_choice"),
        "skip_socratic": get_explainability("skip_socratic"),
        "max_frames": get_explainability("max_frames"),
        "candidate_labels": get_explainability("candidate_labels"),
    }

    return {
        "mode": "guided_workflow",
        "goal": goal,
        "experience_level": experience_level,
        "frontend_choice": frontend_choice,
        "frontend_description": frontend_info,
        "privacy_notice": hardware.get("privacy_notice", ""),
        "decision_path": decision,
        "tradeoff_cards": tradeoff_cards,
        "hardware_profile": hardware,
        "capability_assessment": capability,
        "pre_run_validation": validation,
        "honest_assessment": assessment,
        "socratic": socratic,
        "settings_explainability": active_settings_context,
        "run_result": run_payload,
    }


def _frontend_info(frontend_choice: str) -> str:
    for question in onboarding_questions():
        if question.id != "frontend_preference":
            continue
        for option in question.options:
            if option.id == frontend_choice:
                return option.description
    return "Unknown frontend."


def _honest_assessment(goal: str) -> dict[str, Any]:
    from cvpo.education.citations import cite

    if goal == "geese_tracking":
        return {
            "summary": "Moderate confidence for park geese concentration analysis.",
            "works_well": [
                "Detects bird-like objects in outdoor scenes reliably.",
                (
                    "Keeps track of each object across video frames so you don't "
                    "double-count — technically called multi-object tracking."
                ),
                "Produces movement trends and concentration indicators over time.",
            ],
            "limitations": [
                (
                    "May not distinguish goose species precisely without a specialized "
                    "bird model. The general detector groups all birds together."
                ),
                (
                    "When two birds cross paths closely, the tracker may swap their "
                    "identities — called an ID switch."
                ),
                (
                    "Dense flocks (many birds very close together) may be detected as "
                    "fewer objects than actually present — called merged detections."
                ),
            ],
            "fit_for_purpose": [
                "Park pattern analysis",
                "Operational trend monitoring",
            ],
            "not_fit_for": [
                "Scientific census-grade measurement",
                "Regulatory reporting requiring species-certified precision",
            ],
            "citations": cite("coco", "bytetrack"),
        }
    return {
        "summary": "High confidence for single-image labeling tasks.",
        "works_well": [
            "Maps an image to your provided labels without any model retraining.",
            (
                "Returns a confidence score for each label showing the model's "
                "relative preference — technically a probability distribution."
            ),
        ],
        "limitations": [
            (
                "Does not tell you WHERE objects are in the image — only WHAT the "
                "image contains overall. This is classification, not detection."
            ),
            (
                "Accuracy depends on how well your candidate labels describe what's "
                "in the image. Vague or overlapping labels reduce clarity."
            ),
        ],
        "fit_for_purpose": ["Quick image triage", "Concept demos"],
        "not_fit_for": ["Object counting", "Multi-object tracking"],
        "citations": cite("siglip"),
    }


def _emit(payload: dict[str, Any], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(payload, indent=2))
        return
    if output_format == "summary":
        print(_to_summary(payload))
        return
    if output_format == "guided":
        _print_guided(payload)
        return
    print(_to_pretty(payload))


def _save_report(payload: dict[str, Any], output_format: str, report_path: str | None) -> None:
    if not report_path:
        return
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    if suffix == ".json":
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return
    if suffix == ".md":
        timestamp = datetime.now(UTC).isoformat()
        pretty = _to_pretty(payload)
        md = (
            "# CVPO Run Report\n\n"
            f"- Generated: `{timestamp}`\n"
            f"- Output Format: `{output_format}`\n\n"
            "## Summary\n\n"
            "```text\n"
            f"{pretty}\n"
            "```\n\n"
            "## Raw JSON\n\n"
            "```json\n"
            f"{json.dumps(payload, indent=2)}\n"
            "```\n"
        )
        target.write_text(md, encoding="utf-8")
        return
    raise ValueError("Unsupported report extension. Use .json or .md.")


def _read_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Baseline file not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def _save_benchmark_timeseries_csv(payload: dict[str, Any], report_path: str | None) -> None:
    if not report_path or "benchmark" not in payload:
        return
    benchmark = payload["benchmark"]
    times = benchmark.get("timeseries_ms", [])
    if not isinstance(times, list):
        return

    target = Path(report_path)
    stem = target.stem
    parent = target.parent
    csv_path = parent / f"{stem}_timeseries.csv"
    lines = ["iteration,elapsed_ms"]
    for idx, elapsed in enumerate(times):
        lines.append(f"{idx},{float(elapsed):.6f}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_cites(citations: list) -> list[str]:
    lines: list[str] = []
    for entry in citations:
        if isinstance(entry, dict):
            lines.append(f"  {entry.get('short', '?')} — {entry.get('url', '')}")
        else:
            lines.append(f"  {entry}")
    return lines


def _render_section(title: str, body_lines: list[str]) -> list[str]:
    lines = [title, "-" * len(title)]
    lines.extend(body_lines)
    lines.append("")
    return lines


def _guided_sections(payload: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Return named sections for progressive or full rendering."""
    sections: list[tuple[str, list[str]]] = []

    overview_lines = [
        f"Goal: {payload.get('goal')}",
        f"Experience: {payload.get('experience_level')}",
        f"Frontend: {payload.get('frontend_choice')} - {payload.get('frontend_description')}",
    ]
    privacy = payload.get("privacy_notice", "")
    if privacy:
        overview_lines.append("")
        overview_lines.append(f"Privacy: {privacy}")
    sections.append(("Overview", overview_lines))

    assessment = payload.get("honest_assessment", {})
    body: list[str] = [f"Summary: {assessment.get('summary', 'n/a')}"]
    for key, title in [
        ("works_well", "Works Well"),
        ("limitations", "Limitations"),
        ("fit_for_purpose", "Fit For Purpose"),
        ("not_fit_for", "Not Fit For"),
    ]:
        values = assessment.get(key, [])
        if values:
            body.append(f"{title}:")
            for value in values:
                body.append(f"  - {value}")
    body.append("References:")
    body.extend(_format_cites(assessment.get("citations", [])))
    sections.append(("Honest Assessment", body))

    decision = payload.get("decision_path", {})
    body = [f"Pipeline Level: {decision.get('pipeline_level', 'n/a')}"]
    tasks = decision.get("cv_tasks", [])
    if tasks:
        body.append(f"CV Tasks: {', '.join(tasks)}")
    body.append(f"Reasoning: {decision.get('reasoning', 'n/a')}")
    body.append(f"Back Navigation Hint: {decision.get('back_navigation_hint', 'n/a')}")
    sections.append(("Decision Path", body))

    hardware = payload.get("hardware_profile", {})
    body = [
        f"System: {hardware.get('system', 'n/a')}",
        f"CPU: {hardware.get('cpu', 'n/a')}",
        f"Logical Cores: {hardware.get('logical_cores', 'n/a')}",
        f"RAM (GB): {hardware.get('ram_gb', 'unknown')}",
        f"GPU Detected: {hardware.get('gpu_detected', False)}",
    ]
    if hardware.get("overrides_applied"):
        body.append("(Manual overrides applied — auto-detected values replaced.)")
    body.append(f"How to verify: {hardware.get('how_to_find_specs', 'n/a')}")
    body.append("")
    for row in payload.get("capability_assessment", []):
        line = (
            f"- {row.get('model')} ({row.get('stage')}): "
            f"status={row.get('status')}, "
            f"ram_ok={row.get('ram_ok')}, "
            f"gpu={row.get('gpu_note')} / present={row.get('gpu_present')}"
        )
        body.append(line)
        if row.get("suggestion"):
            body.append(f"  >> {row['suggestion']}")
    sections.append(("Hardware Capability", body))

    validation = payload.get("pre_run_validation", {})
    if validation:
        body = [f"Status: {validation.get('status', 'n/a')}"]
        body.append(f"Message: {validation.get('proceed_message', '')}")
        for blocker in validation.get("blockers", []):
            body.append(f"  BLOCKER: {blocker}")
        for warning in validation.get("warnings", []):
            body.append(f"  WARNING: {warning}")
        if validation.get("status") in ("blocked", "warnings"):
            hw = payload.get("hardware_profile", {})
            body.append("")
            body.append(
                "If these readings seem wrong, you can correct them with override flags:"
            )
            body.append(f"  --override-ram-gb XX     (detected: {hw.get('ram_gb', 'unknown')} GB)")
            body.append(f"  --override-gpu-vram-gb XX")
            body.append(f"  --override-gpu           (to declare a GPU is present)")
            body.append("")
            body.append(
                f"To find your actual specs: {hw.get('how_to_find_specs', 'Check your system settings.')}"
            )
        sections.append(("Pre-Run Validation", body))

    body = []
    for card in payload.get("tradeoff_cards", []):
        body.append(f"[{card.get('stage')}] {card.get('model')}:")
        body.append(f"  Tradeoff: {card.get('tradeoff')}")
        body.append(f"  Guidance: {card.get('guidance')}")
        cites = card.get("citations", [])
        if cites:
            body.append("  References:")
            body.extend(["  " + line for line in _format_cites(cites)])
    sections.append(("Tradeoff Cards", body))

    socratic = payload.get("socratic", {})
    body = [
        f"Question: {socratic.get('question', 'n/a')}",
        f"Skipped: {socratic.get('skipped', False)}",
        f"Answer: {socratic.get('answer', 'n/a')}",
    ]
    body.append("References:")
    body.extend(_format_cites(socratic.get("citations", [])))
    sections.append(("Socratic Check", body))

    settings_ctx = payload.get("settings_explainability", {})
    if settings_ctx:
        body = []
        for skey, sval in settings_ctx.items():
            if not sval:
                continue
            body.append(f"[{skey}] {sval.get('setting', skey)}:")
            body.append(f"  Why: {sval.get('why', 'n/a')}")
            body.append(f"  When to change: {sval.get('when_to_change', 'n/a')}")
        sections.append(("Settings Explainability", body))

    run_result = payload.get("run_result", {})
    sections.append(("Run Result", [json.dumps(run_result, indent=2)]))

    return sections


def _to_pretty(payload: dict[str, Any]) -> str:
    if payload.get("mode") == "guided_workflow":
        lines = ["CVPO Guided Workflow", "====================", ""]
        for title, body in _guided_sections(payload):
            lines.extend(_render_section(title, body))
        return "\n".join(lines)

    lines = ["CVPO Result", "==========="]
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            lines.append(f"{key}:")
            lines.append(json.dumps(value, indent=2))
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _to_summary(payload: dict[str, Any]) -> str:
    if payload.get("mode") == "guided_workflow":
        assessment = payload.get("honest_assessment", {})
        decision = payload.get("decision_path", {})
        hardware = payload.get("hardware_profile", {})
        run_result = payload.get("run_result", {})
        lines = [
            "CVPO Summary",
            "============",
            f"Goal: {payload.get('goal')}",
            f"Experience: {payload.get('experience_level')}",
            f"Frontend: {payload.get('frontend_choice')}",
            f"Assessment: {assessment.get('summary', 'n/a')}",
            f"Pipeline: {decision.get('pipeline_level', 'n/a')}",
            f"Hardware: {hardware.get('cpu', '?')} / {hardware.get('ram_gb', '?')} GB RAM / GPU={'yes' if hardware.get('gpu_detected') else 'no'}",
            f"Result: {run_result.get('workflow', 'n/a')} ({run_result.get('frame_count', '?')} frames)",
            "",
            "Use --format pretty for full details or --format guided for step-by-step walkthrough.",
        ]
        return "\n".join(lines)

    lines = ["CVPO Summary", "============"]
    for key, value in payload.items():
        if not isinstance(value, (dict, list)):
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _print_guided(payload: dict[str, Any]) -> None:
    if payload.get("mode") != "guided_workflow":
        print(_to_pretty(payload))
        return

    sections = _guided_sections(payload)
    total = len(sections)
    for idx, (title, body) in enumerate(sections, 1):
        print(f"\n[Step {idx}/{total}] {title}")
        print("=" * (len(title) + 12))
        for line in body:
            print(line)
        if idx < total:
            try:
                input("\nPress Enter to continue...")
            except EOFError:
                print()
    print("\nWorkflow complete.")


def _load_image_or_synthetic(
    input_image: str | None,
    make_bright_square: bool = False,
) -> np.ndarray:
    if input_image:
        return _load_image(input_image)
    image = np.full((224, 224, 3), fill_value=180, dtype=np.uint8)
    if make_bright_square:
        image[:] = 0
        image[80:160, 90:170] = 255
    return image


def _load_image(path: str) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "Image loading requires pillow. Install with: pip install -e .[models]"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(p).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _load_video_frames(path: str, max_frames: int) -> list[np.ndarray]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(
            "Video loading requires opencv-python. Install with: pip install opencv-python"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frames: list[np.ndarray] = []
    try:
        while len(frames) < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb.astype(np.uint8))
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    return frames


def _synthetic_video_frames(frame_count: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for frame_idx in range(frame_count):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        x = 30 + frame_idx * 12
        frame[90:140, x : x + 40] = 255
        frames.append(frame)
    return frames


def _print_explainability(key: str) -> None:
    entry = get_explainability(key)
    if not entry:
        print(f"Unknown setting key: '{key}'")
        print("Use --explain-all to see all available settings.")
        return
    print(f"Setting: {entry.get('setting', key)}")
    print(f"Options: {entry.get('options', 'n/a')}")
    print(f"Why: {entry.get('why', 'n/a')}")
    print(f"When to change: {entry.get('when_to_change', 'n/a')}")


def _print_all_explainability() -> None:
    print("CVPO Settings Explainability Guide")
    print("===================================")
    for key, entry in get_all_explainability().items():
        print(f"\n[{key}] {entry.get('setting', key)}")
        print(f"  Options: {entry.get('options', 'n/a')}")
        print(f"  Why: {entry.get('why', 'n/a')}")
        print(f"  When to change: {entry.get('when_to_change', 'n/a')}")


if __name__ == "__main__":
    main()
