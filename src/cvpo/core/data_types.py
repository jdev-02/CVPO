"""Typed data contracts for deterministic stage-to-stage flow.

Functional programming audit notes:
- All artifact dataclasses use slots=True for memory efficiency.
- Scalar-only dataclasses (BBox, Detection, Classification, Track, SegmentationPrompt)
  use frozen=True for true immutability.
- Array-carrying dataclasses cannot be frozen (numpy arrays are mutable by nature)
  but are treated as logically immutable — stages and connectors MUST NOT mutate
  incoming artifacts. They always produce new objects.
- TrackState is an explicit, pass-through data structure — tracking state is not
  hidden inside a stage object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(slots=True)
class PipelineContext:
    """Run-level metadata and output selection."""

    run_id: str
    input_source: str
    frontend: str = "gradio"
    output_modes: List[str] = field(default_factory=lambda: ["visual", "data"])
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box in absolute pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True, slots=True)
class Detection:
    """Single detection produced by a detector model. Immutable."""

    label: str
    confidence: float
    box: BBox


@dataclass(slots=True)
class DetectionResult:
    """Detection stage output. Image payload prevents freezing."""

    image: np.ndarray
    detections: List[Detection]
    model_name: str


@dataclass(slots=True)
class SegmentationMask:
    """Mask tied to a detection or region."""

    mask: np.ndarray
    confidence: float
    detection_index: Optional[int] = None


@dataclass(slots=True)
class SegmentationResult:
    """Segmentation stage output."""

    image: np.ndarray
    masks: List[SegmentationMask]
    model_name: str


@dataclass(frozen=True, slots=True)
class SegmentationPrompt:
    """Prompt region used to drive segmentation. Immutable."""

    box: BBox
    label: str = "object"
    confidence: float = 1.0
    detection_index: Optional[int] = None


@dataclass(slots=True)
class SegmentationPromptSet:
    """Connector output that feeds segmentation stages."""

    image: np.ndarray
    prompts: List[SegmentationPrompt]
    source_model_name: str


@dataclass(slots=True)
class ClassificationRegion:
    """Region crop sent to the classification stage."""

    crop: np.ndarray
    detection_index: Optional[int] = None


@dataclass(slots=True)
class ClassificationInputSet:
    """Connector output for region-level classification."""

    image: np.ndarray
    regions: List[ClassificationRegion]
    source_model_name: str


@dataclass(frozen=True, slots=True)
class Classification:
    """Classification result for an image or region. Immutable."""

    label: str
    confidence: float
    candidate_labels: tuple = ()
    scores: tuple = ()
    detection_index: Optional[int] = None


@dataclass(slots=True)
class ClassificationResult:
    """Classification stage output."""

    image: np.ndarray
    classes: List[Classification]
    model_name: str


@dataclass(frozen=True, slots=True)
class Track:
    """Tracked object entry. Immutable."""

    track_id: int
    label: str
    confidence: float
    box: BBox


@dataclass(slots=True)
class TrackingResult:
    """Tracking stage output."""

    frame: np.ndarray
    frame_index: int
    tracks: List[Track]
    model_name: str


@dataclass(frozen=True, slots=True)
class TrackState:
    """Explicit tracker memory passed in and out of tracking stages.

    Making this a frozen dataclass enforces that tracking state is
    never mutated in-place — a new TrackState is always produced.
    """

    next_id: int = 1
    active: tuple = ()


StageArtifact = (
    DetectionResult
    | SegmentationPromptSet
    | ClassificationInputSet
    | SegmentationResult
    | ClassificationResult
    | TrackingResult
    | np.ndarray
)
