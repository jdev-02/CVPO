from __future__ import annotations

import numpy as np

from cvpo.core.data_types import ClassificationResult, PipelineContext
from cvpo.pipelines import build_level2_pipeline


def test_level2_pipeline_runs_three_stage_flow() -> None:
    pipeline = build_level2_pipeline(
        candidate_labels=["goose", "duck", "pigeon"],
        detection_backend="deterministic",
        segmentation_backend="deterministic",
        classification_backend="deterministic",
    )
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[20:50, 30:60] = 255
    ctx = PipelineContext(run_id="t-level2", input_source="unit-test", frontend="cli")

    result = pipeline.run(image, ctx)
    assert isinstance(result, ClassificationResult)
    assert len(result.classes) >= 1
