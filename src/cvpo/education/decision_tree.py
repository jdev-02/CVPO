"""Deterministic onboarding and discovery questions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass(slots=True)
class QuestionOption:
    id: str
    label: str
    description: str


@dataclass(slots=True)
class Question:
    id: str
    prompt: str
    options: List[QuestionOption] = field(default_factory=list)
    allow_multiple: bool = False


def onboarding_questions() -> List[Question]:
    """Return deterministic onboarding questions shown to all users."""

    return [
        Question(
            id="experience_level",
            prompt="What's your experience with computer vision?",
            options=[
                QuestionOption("beginner", "Beginner", "I am new to computer vision."),
                QuestionOption(
                    "intermediate",
                    "Intermediate",
                    "I know Python/ML basics but not end-to-end CV pipelines.",
                ),
                QuestionOption(
                    "advanced",
                    "Advanced",
                    "I already know CV and want direct control.",
                ),
            ],
        ),
        Question(
            id="frontend_preference",
            prompt="Which interface do you want to use for CVPO?",
            options=[
                QuestionOption(
                    "cli",
                    "CLI",
                    "Fast, scriptable, and best for automation or power users.",
                ),
                QuestionOption(
                    "gradio",
                    "Web UI (Gradio)",
                    "Visual, guided, and easiest for beginners and demos.",
                ),
                QuestionOption(
                    "notebook",
                    "Jupyter Notebook",
                    "Step-by-step educational flow with code and explanations.",
                ),
            ],
        ),
    ]


def resolve_goal(goal: str) -> dict[str, Any]:
    """
    Deterministically map a user goal to CV task decomposition and pipeline level.

    This powers the "Step 2/3 correction" behavior in a non-LLM path.
    """
    if goal == "geese_tracking":
        return {
            "goal": goal,
            "cv_tasks": ["detection", "segmentation", "classification", "tracking"],
            "pipeline_level": "level3_detect_segment_classify_track",
            "reasoning": (
                "Tracking geese concentration requires object localization, identity continuity, "
                "and temporal aggregation."
            ),
            "back_navigation_hint": (
                "If your true need is just species label on a single image, switch to image_labeling."
            ),
        }
    return {
        "goal": goal,
        "cv_tasks": ["classification"],
        "pipeline_level": "level0_classification",
        "reasoning": "Single-image labeling maps directly to classification.",
        "back_navigation_hint": (
            "If you need where objects are located, return to discovery and choose detection/tracking."
        ),
    }
