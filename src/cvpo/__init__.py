"""CVPO: Computer Vision Pipeline Orchestrator."""

from cvpo.core.pipeline import Pipeline
from cvpo.core.stage import Stage
from cvpo.core.connector import Connector
from cvpo.core.data_types import (
    ClassificationResult,
    DetectionResult,
    PipelineContext,
    SegmentationResult,
    StageArtifact,
    TrackingResult,
    TrackState,
)

__all__ = [
    "Pipeline",
    "Stage",
    "Connector",
    "PipelineContext",
    "StageArtifact",
    "DetectionResult",
    "SegmentationResult",
    "ClassificationResult",
    "TrackingResult",
    "TrackState",
]
