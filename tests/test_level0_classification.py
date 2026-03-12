from __future__ import annotations

import numpy as np

from cvpo.core.data_types import ClassificationResult, PipelineContext
from cvpo.pipelines import build_level0_pipeline


def test_level0_pipeline_returns_classification_result() -> None:
    pipeline = build_level0_pipeline(
        candidate_labels=["goose", "duck", "pigeon"], backend="deterministic"
    )
    image = np.full((64, 64, 3), fill_value=200, dtype=np.uint8)
    ctx = PipelineContext(run_id="t-level0", input_source="unit-test", frontend="cli")

    result = pipeline.run(image, ctx)
    assert isinstance(result, ClassificationResult)
    assert len(result.classes) == 1
    assert result.classes[0].label in {"goose", "duck", "pigeon"}
    scores_dict = dict(result.classes[0].scores)
    assert abs(sum(scores_dict.values()) - 1.0) < 1e-6


def test_level0_is_deterministic_for_same_input() -> None:
    labels = ["goose", "duck", "pigeon"]
    pipeline = build_level0_pipeline(candidate_labels=labels, backend="deterministic")
    image = np.full((32, 32, 3), fill_value=123, dtype=np.uint8)
    ctx = PipelineContext(run_id="t-deterministic", input_source="unit-test", frontend="cli")

    result_a = pipeline.run(image, ctx)
    result_b = pipeline.run(image, ctx)
    assert result_a.classes[0].scores == result_b.classes[0].scores
