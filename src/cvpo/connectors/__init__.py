"""Deterministic connectors between known model-stage pairs."""

from cvpo.connectors.detection_to_segmentation import (
    DetectionToSegmentationConnector,
    default_detection_to_segmentation_connector,
)
from cvpo.connectors.segmentation_to_classification import (
    SegmentationToClassificationConnector,
    default_segmentation_to_classification_connector,
)

__all__ = [
    "DetectionToSegmentationConnector",
    "default_detection_to_segmentation_connector",
    "SegmentationToClassificationConnector",
    "default_segmentation_to_classification_connector",
]
