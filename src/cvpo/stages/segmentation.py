"""Segmentation stage implementations."""

from __future__ import annotations

from typing import List

import numpy as np

from cvpo.core.data_types import (
    PipelineContext,
    SegmentationMask,
    SegmentationPromptSet,
    SegmentationResult,
    StageArtifact,
)
from cvpo.core.stage import Stage, StageConfig


class SAM2Stage(Stage):
    """
    Segmentation stage with deterministic and placeholder SAM2 backend.

    Backends:
    - deterministic: rectangular masks from prompt boxes.
    - sam2: reserved for real SAM2 integration.
    """

    def __init__(self, config: StageConfig) -> None:
        super().__init__(config)
        self.backend = str(config.params.get("backend", "deterministic"))

    def validate_input(self, artifact: StageArtifact) -> None:
        if not isinstance(artifact, SegmentationPromptSet):
            raise ValueError("SAM2Stage expects SegmentationPromptSet input.")

    def run(self, artifact: StageArtifact, context: PipelineContext) -> StageArtifact:
        prompt_set = artifact
        assert isinstance(prompt_set, SegmentationPromptSet)
        if self.backend == "deterministic":
            masks = self._segment_deterministic(prompt_set)
        elif self.backend == "sam2":
            masks = self._segment_sam2(prompt_set)
        else:
            raise ValueError(f"Unknown backend '{self.backend}'.")
        return SegmentationResult(image=prompt_set.image, masks=masks, model_name=self.config.model_name)

    @staticmethod
    def _segment_deterministic(prompt_set: SegmentationPromptSet) -> List[SegmentationMask]:
        h, w = prompt_set.image.shape[:2]
        masks: List[SegmentationMask] = []
        for prompt in prompt_set.prompts:
            mask = np.zeros((h, w), dtype=np.uint8)
            x1 = max(0, int(prompt.box.x1))
            y1 = max(0, int(prompt.box.y1))
            x2 = min(w - 1, int(prompt.box.x2))
            y2 = min(h - 1, int(prompt.box.y2))
            mask[y1 : y2 + 1, x1 : x2 + 1] = 1
            masks.append(
                SegmentationMask(
                    mask=mask,
                    confidence=prompt.confidence,
                    detection_index=prompt.detection_index,
                )
            )
        return masks

    @staticmethod
    def _segment_sam2(prompt_set: SegmentationPromptSet) -> List[SegmentationMask]:
        # Kept explicit so behavior stays deterministic unless user opts in.
        raise RuntimeError(
            "SAM2 backend is not wired yet. Use backend='deterministic' for v1 flow."
        )
