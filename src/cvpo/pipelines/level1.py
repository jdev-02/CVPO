"""Level 1 pipeline: detection -> segmentation."""

from __future__ import annotations

from cvpo.connectors import default_detection_to_segmentation_connector
from cvpo.core.pipeline import Pipeline, PipelineStep
from cvpo.core.stage import StageConfig
from cvpo.stages import SAM2Stage, YOLOv8Stage


def build_level1_pipeline(
    detection_backend: str = "deterministic",
    segmentation_backend: str = "deterministic",
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
    connector = default_detection_to_segmentation_connector()

    pipeline = Pipeline(name="level1_detect_segment")
    pipeline.add_step(PipelineStep(stage=detect_stage, connector_to_next=connector))
    pipeline.add_step(PipelineStep(stage=seg_stage))
    return pipeline
