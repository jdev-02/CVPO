"""Connector: detection output -> segmentation prompts.

Pure function: creates new SegmentationPromptSet from DetectionResult.
Does not mutate the input artifact.

Research backing:
- Box-prompted segmentation is SAM2's native API. Detection bounding boxes
  are passed directly as box prompts to the segmentation model.
  Ravi et al. 2024 (SAM 2): https://arxiv.org/abs/2408.00714
- Two-stage detect-then-segment pattern originates from Mask R-CNN.
  He et al. 2017: https://arxiv.org/abs/1703.06870
"""

from __future__ import annotations

from cvpo.core.connector import Connector, ConnectorSpec
from cvpo.core.data_types import DetectionResult, SegmentationPrompt, SegmentationPromptSet, StageArtifact


class DetectionToSegmentationConnector(Connector):
    def adapt(self, artifact: StageArtifact) -> StageArtifact:
        if not isinstance(artifact, DetectionResult):
            raise ValueError("DetectionToSegmentationConnector expects DetectionResult.")

        prompts = [
            SegmentationPrompt(
                box=detection.box,
                label=detection.label,
                confidence=detection.confidence,
                detection_index=index,
            )
            for index, detection in enumerate(artifact.detections)
        ]
        return SegmentationPromptSet(
            image=artifact.image,
            prompts=prompts,
            source_model_name=artifact.model_name,
        )


def default_detection_to_segmentation_connector() -> DetectionToSegmentationConnector:
    return DetectionToSegmentationConnector(
        ConnectorSpec(
            name="detection_to_segmentation",
            source_stage="detection",
            target_stage="segmentation",
        )
    )
