"""Deterministic onboarding, discovery questions, and goal resolution."""

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


def discovery_questions() -> List[Question]:
    """Return problem-discovery questions that map to goals."""

    return [
        Question(
            id="problem_type",
            prompt="What are you trying to do?",
            options=[
                QuestionOption(
                    "image_labeling",
                    "Label images",
                    "I want to know what's in a photo — assign a category or label.",
                ),
                QuestionOption(
                    "object_finding",
                    "Find objects in images",
                    "I want to find WHERE specific things are in an image and draw boxes around them.",
                ),
                QuestionOption(
                    "object_boundaries",
                    "Get precise object boundaries",
                    "I want pixel-perfect outlines around objects, not just rectangles.",
                ),
                QuestionOption(
                    "video_tracking",
                    "Track objects in video",
                    "I want to follow objects across video frames and keep consistent IDs.",
                ),
                QuestionOption(
                    "counting_over_time",
                    "Count things over time",
                    "I want to count how many of something appear across a video and see patterns.",
                ),
                QuestionOption(
                    "species_identification",
                    "Identify specific species or types",
                    "I want to detect objects AND identify exactly what species or subtype each one is.",
                ),
            ],
        ),
    ]


GOAL_MAP: dict[str, dict[str, Any]] = {
    "image_labeling": {
        "goal": "image_labeling",
        "cv_tasks": ["classification"],
        "pipeline_level": "level0_classification",
        "reasoning": (
            "Single-image labeling — telling you what's in a photo overall — maps "
            "directly to classification. One model looks at the whole image and picks "
            "the best-matching label from your candidates."
        ),
        "back_navigation_hint": (
            "If you need to know WHERE objects are (not just what the image contains), "
            "go back and choose 'Find objects in images' instead."
        ),
    },
    "object_finding": {
        "goal": "object_finding",
        "cv_tasks": ["detection"],
        "pipeline_level": "level1_detect_segment",
        "reasoning": (
            "Finding objects means locating them with bounding boxes — technically "
            "called object detection. The detector scans the image and outputs a box, "
            "label, and confidence for each object it finds."
        ),
        "back_navigation_hint": (
            "If you only need a single label for the whole image, go back and choose "
            "'Label images'. If you need precise outlines, choose 'Get precise object boundaries'."
        ),
    },
    "object_boundaries": {
        "goal": "object_boundaries",
        "cv_tasks": ["detection", "segmentation"],
        "pipeline_level": "level1_detect_segment",
        "reasoning": (
            "Precise boundaries require two steps: first detect where objects are "
            "(bounding boxes), then segment each one to get pixel-level outlines — "
            "technically called instance segmentation."
        ),
        "back_navigation_hint": (
            "If rectangles are good enough, go back and choose 'Find objects in images'."
        ),
    },
    "video_tracking": {
        "goal": "video_tracking",
        "cv_tasks": ["detection", "tracking"],
        "pipeline_level": "level3_detect_segment_classify_track",
        "reasoning": (
            "Tracking objects in video requires detection on each frame plus a tracking "
            "algorithm that maintains consistent identities — technically called "
            "multi-object tracking (MOT). Each object gets a unique ID that persists "
            "across frames."
        ),
        "back_navigation_hint": (
            "If you only need to analyze single images (not video), go back and choose "
            "a simpler task."
        ),
    },
    "counting_over_time": {
        "goal": "counting_over_time",
        "cv_tasks": ["detection", "segmentation", "classification", "tracking"],
        "pipeline_level": "level3_detect_segment_classify_track",
        "reasoning": (
            "Counting over time requires the full pipeline: detect objects, track them "
            "across frames so you don't double-count, and optionally classify each one. "
            "This is the most complete pipeline CVPO offers."
        ),
        "back_navigation_hint": (
            "If you're counting in a single image (not over time), detection alone may "
            "be sufficient — go back and choose 'Find objects in images'."
        ),
    },
    "species_identification": {
        "goal": "species_identification",
        "cv_tasks": ["detection", "segmentation", "classification"],
        "pipeline_level": "level2_detect_segment_classify",
        "reasoning": (
            "Species identification combines detection (find the animal) with "
            "classification (determine the species). The detector finds WHERE animals "
            "are, then the classifier determines WHAT each one is using your provided "
            "species labels. This is the pattern that corrects YOLO's generic labels "
            "with CLIP's open-vocabulary specificity."
        ),
        "back_navigation_hint": (
            "If you don't need to locate objects and just want to label a whole image, "
            "go back and choose 'Label images'."
        ),
    },
    "geese_tracking": {
        "goal": "geese_tracking",
        "cv_tasks": ["detection", "segmentation", "classification", "tracking"],
        "pipeline_level": "level3_detect_segment_classify_track",
        "reasoning": (
            "Tracking geese concentration requires object localization, species-level "
            "classification (to distinguish geese from other birds), identity continuity "
            "across frames, and temporal aggregation."
        ),
        "back_navigation_hint": (
            "If your true need is just a species label on a single image, switch to "
            "'Label images' or 'Identify specific species or types'."
        ),
    },
}


def resolve_goal(goal: str) -> dict[str, Any]:
    """Deterministically map a user goal to CV task decomposition and pipeline level."""
    if goal in GOAL_MAP:
        return GOAL_MAP[goal]
    return GOAL_MAP["image_labeling"]
