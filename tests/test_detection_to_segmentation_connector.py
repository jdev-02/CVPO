from __future__ import annotations

import numpy as np

from cvpo.connectors import default_detection_to_segmentation_connector
from cvpo.core.data_types import BBox, Detection, DetectionResult, SegmentationPromptSet


def test_connector_converts_detections_to_prompts() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    det = Detection(label="bird", confidence=0.9, box=BBox(1, 2, 10, 12))
    result = DetectionResult(image=image, detections=[det], model_name="yolov8")
    connector = default_detection_to_segmentation_connector()

    adapted = connector.adapt(result)
    assert isinstance(adapted, SegmentationPromptSet)
    assert len(adapted.prompts) == 1
    assert adapted.prompts[0].label == "bird"
