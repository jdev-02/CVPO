"""Level 0 pipeline: true hello-world single-image classification."""

from __future__ import annotations

from cvpo.core.pipeline import Pipeline, PipelineStep
from cvpo.core.stage import StageConfig
from cvpo.stages.classification import SigLIPStage


def build_level0_pipeline(
    candidate_labels: list[str],
    backend: str = "deterministic",
) -> Pipeline:
    stage = SigLIPStage(
        StageConfig(
            name="classification",
            model_name="siglip",
            params={"candidate_labels": candidate_labels, "backend": backend},
        )
    )
    pipeline = Pipeline(name="level0_classification")
    pipeline.add_step(PipelineStep(stage=stage))
    return pipeline
