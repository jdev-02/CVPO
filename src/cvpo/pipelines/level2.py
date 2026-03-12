"""Level 2 pipeline: detection -> segmentation -> classification."""

from __future__ import annotations

from cvpo.connectors import (
    default_detection_to_segmentation_connector,
    default_segmentation_to_classification_connector,
)
from cvpo.core.pipeline import Pipeline, PipelineStep
from cvpo.core.stage import StageConfig
from cvpo.stages import SAM2Stage, SigLIPStage, YOLOv8Stage


def build_level2_pipeline(
    candidate_labels: list[str],
    detection_backend: str = "deterministic",
    segmentation_backend: str = "deterministic",
    classification_backend: str = "deterministic",
) -> Pipeline:
    detect_stage = YOLOv8Stage(
        StageConfig(
            name="detection",
            model_name="yolov8",
            params={"backend": detection_backend, "label": "bird"},
        )
    )
    seg_stage = SAM2Stage(
        StageConfig(
            name="segmentation",
            model_name="sam2",
            params={"backend": segmentation_backend},
        )
    )
    cls_stage = SigLIPStage(
        StageConfig(
            name="classification",
            model_name="siglip",
            params={
                "backend": classification_backend,
                "candidate_labels": candidate_labels,
            },
        )
    )

    pipeline = Pipeline(name="level2_detect_segment_classify")
    pipeline.add_step(
        PipelineStep(
            stage=detect_stage,
            connector_to_next=default_detection_to_segmentation_connector(),
        )
    )
    pipeline.add_step(
        PipelineStep(
            stage=seg_stage,
            connector_to_next=default_segmentation_to_classification_connector(),
        )
    )
    pipeline.add_step(PipelineStep(stage=cls_stage))
    return pipeline
