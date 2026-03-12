from __future__ import annotations

import numpy as np

from cvpo.core.data_types import PipelineContext, SegmentationResult
from cvpo.pipelines import build_level1_pipeline


def test_level1_pipeline_runs_detection_to_segmentation() -> None:
    pipeline = build_level1_pipeline(
        detection_backend="deterministic",
        segmentation_backend="deterministic",
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[20:40, 22:45] = 255
    context = PipelineContext(run_id="t-level1", input_source="unit-test", frontend="cli")

    result = pipeline.run(image, context)
    assert isinstance(result, SegmentationResult)
    assert len(result.masks) >= 1
    assert int(result.masks[0].mask.sum()) > 0
