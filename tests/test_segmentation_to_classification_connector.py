from __future__ import annotations

import numpy as np

from cvpo.connectors import default_segmentation_to_classification_connector
from cvpo.core.data_types import SegmentationMask, SegmentationResult


def test_connector_creates_classification_regions() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 7:15] = 1
    seg_result = SegmentationResult(
        image=image,
        masks=[SegmentationMask(mask=mask, confidence=0.9, detection_index=0)],
        model_name="sam2",
    )

    connector = default_segmentation_to_classification_connector()
    adapted = connector.adapt(seg_result)
    assert len(adapted.regions) == 1
    assert adapted.regions[0].crop.size > 0
