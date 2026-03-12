"""Explainability registry: WHY you would toggle every setting, feature, or measure."""

from __future__ import annotations

from typing import Any


EXPLAINABILITY: dict[str, dict[str, Any]] = {
    "experience_level": {
        "setting": "Experience Level",
        "options": ["beginner", "intermediate", "advanced"],
        "why": (
            "Controls how much detail and technical vocabulary the system uses. "
            "Pick beginner if you want plain-language explanations with technical "
            "terms introduced gently. Pick advanced if you already know CV and "
            "want concise technical output without extra context."
        ),
        "when_to_change": (
            "Change this if the output feels too simple (move up) or too dense (move down). "
            "You can re-run the same workflow at a different level without losing results."
        ),
    },
    "frontend_choice": {
        "setting": "Frontend Choice",
        "options": ["cli", "gradio", "notebook"],
        "why": (
            "Determines how you interact with CVPO. CLI is fastest for scripting "
            "and automation. Gradio gives you a visual browser-based interface that "
            "is easiest for demos. Notebook provides a step-by-step educational "
            "flow where you can inspect and modify code at each stage."
        ),
        "when_to_change": (
            "Use CLI when you want to batch-process or pipe output into other tools. "
            "Use Gradio when showing someone else or exploring visually. "
            "Use Notebook when you want to learn by reading and modifying code."
        ),
    },
    "output_format": {
        "setting": "Output Format",
        "options": ["json", "pretty", "summary", "guided"],
        "why": (
            "Controls how results are displayed. JSON is structured data for "
            "programmatic use. Pretty prints all sections at once for quick review. "
            "Summary gives a compact one-screen overview. Guided walks you through "
            "one section at a time so you can absorb each part before moving on."
        ),
        "when_to_change": (
            "Use JSON when feeding output into scripts or saving reports. "
            "Use summary for a quick sanity check. "
            "Use guided when running CVPO for the first time or presenting to someone. "
            "Use pretty when you know the workflow and want everything visible."
        ),
    },
    "skip_socratic": {
        "setting": "Skip Socratic Questions",
        "options": [True, False],
        "why": (
            "Socratic questions test your understanding of the CV concepts behind "
            "each pipeline stage. The pre-authored answer is always shown regardless. "
            "Skipping just suppresses the 'try to answer' prompt."
        ),
        "when_to_change": (
            "Skip when you already understand the concepts or are running in batch mode. "
            "Keep enabled when learning or when you want the system to prompt reflection."
        ),
    },
    "max_frames": {
        "setting": "Max Frames",
        "options": "integer (default 8)",
        "why": (
            "Limits how many video frames the pipeline processes. Processing fewer "
            "frames is faster and uses less memory. Processing more frames gives "
            "better temporal coverage and more reliable tracking statistics."
        ),
        "when_to_change": (
            "Increase for longer videos or when you need full temporal coverage. "
            "Decrease when testing, when hardware is limited, or when you only need "
            "a quick sample of results."
        ),
    },
    "candidate_labels": {
        "setting": "Candidate Labels",
        "options": "comma-separated strings",
        "why": (
            "These are the categories the classification model scores each image "
            "or crop against. The model does not invent labels — it only picks from "
            "what you provide. More specific, non-overlapping labels produce clearer "
            "results. Vague or redundant labels dilute confidence scores."
        ),
        "when_to_change": (
            "Change when you know what specific things you are looking for. "
            "For example, use 'golden retriever, labrador, poodle' instead of "
            "'dog, pet, animal'. Add labels to distinguish between things the "
            "detector groups together (like 'bird' -> 'goose, duck, pigeon')."
        ),
    },
    "detection_backend": {
        "setting": "Detection Backend",
        "options": ["deterministic", "yolo"],
        "why": (
            "The deterministic backend runs without downloading any model weights "
            "and produces predictable synthetic output — useful for testing, "
            "development, and environments without internet access. The yolo "
            "backend uses a real YOLOv8 model for actual object detection."
        ),
        "when_to_change": (
            "Use deterministic for development, CI testing, and offline demos. "
            "Switch to yolo when you want real detection results on real images."
        ),
    },
    "segmentation_backend": {
        "setting": "Segmentation Backend",
        "options": ["deterministic", "sam2"],
        "why": (
            "The deterministic backend creates rectangular masks from detection "
            "boxes — fast and predictable but not pixel-accurate. The sam2 backend "
            "uses Meta's Segment Anything Model 2 for precise object outlines."
        ),
        "when_to_change": (
            "Use deterministic when you only need bounding-box-level analysis or "
            "when testing pipeline logic. Switch to sam2 when precise object "
            "boundaries matter (area measurement, occlusion analysis)."
        ),
    },
    "classification_backend": {
        "setting": "Classification Backend",
        "options": ["deterministic", "siglip"],
        "why": (
            "The deterministic backend produces consistent synthetic scores for "
            "testing. The siglip backend uses Google's SigLIP model for real "
            "zero-shot image classification against your candidate labels."
        ),
        "when_to_change": (
            "Use deterministic for pipeline development and reproducible tests. "
            "Switch to siglip when you need real classification results."
        ),
    },
    "save_report": {
        "setting": "Save Report",
        "options": [".json", ".md"],
        "why": (
            "Saves a persistent record of the run — useful for comparing results "
            "across different configurations, tracking performance over time, or "
            "building presentation evidence. JSON is machine-readable. Markdown "
            "is human-readable with both narrative and raw data."
        ),
        "when_to_change": (
            "Enable when you need an artifact trail for presentations, audits, or "
            "cross-environment comparisons. Disable for quick exploratory runs."
        ),
    },
    "benchmark_regression_threshold": {
        "setting": "Benchmark Regression Threshold",
        "options": "percentage (default 15.0)",
        "why": (
            "Sets how much slower a run can be compared to baseline before it is "
            "flagged as a regression. A threshold of 15% means if mean latency "
            "increases by more than 15% versus baseline, the check fails."
        ),
        "when_to_change": (
            "Tighten (lower the number) for performance-sensitive deployments. "
            "Loosen (raise the number) when running on variable hardware or when "
            "small fluctuations are expected."
        ),
    },
}


def get_explainability(setting_key: str) -> dict[str, Any]:
    return EXPLAINABILITY.get(setting_key, {})


def get_all_explainability() -> dict[str, dict[str, Any]]:
    return EXPLAINABILITY
