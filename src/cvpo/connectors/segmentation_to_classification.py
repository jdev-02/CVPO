"""Connector: segmentation output -> classification region set.

Pure function: creates new ClassificationInputSet from SegmentationResult.
Does not mutate the input artifact. Crop is a numpy view (zero-copy) but
downstream stages treat it as read-only.

Research backing:
- Region-of-interest (ROI) cropping for downstream classification is the
  foundational pattern of two-stage detection, originating from R-CNN.
  Girshick et al. 2014: https://arxiv.org/abs/1311.1524
- Mask-guided cropping (using segmentation masks to bound the crop region)
  extends this to instance-level precision.
  He et al. 2017 (Mask R-CNN): https://arxiv.org/abs/1703.06870
"""

from __future__ import annotations

from typing import List

from cvpo.core.connector import Connector, ConnectorSpec
from cvpo.core.data_types import (
    ClassificationInputSet,
    ClassificationRegion,
    SegmentationResult,
    StageArtifact,
)


class SegmentationToClassificationConnector(Connector):
    def adapt(self, artifact: StageArtifact) -> StageArtifact:
        if not isinstance(artifact, SegmentationResult):
            raise ValueError("SegmentationToClassificationConnector expects SegmentationResult.")

        image = artifact.image
        regions: List[ClassificationRegion] = []
        for mask_entry in artifact.masks:
            ys, xs = mask_entry.mask.nonzero()
            if len(xs) == 0 or len(ys) == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            crop = image[y1 : y2 + 1, x1 : x2 + 1].copy()
            if crop.size == 0:
                continue
            regions.append(
                ClassificationRegion(
                    crop=crop,
                    detection_index=mask_entry.detection_index,
                )
            )

        if not regions:
            regions.append(ClassificationRegion(crop=image.copy(), detection_index=None, is_fallback=True))

        return ClassificationInputSet(
            image=image,
            regions=regions,
            source_model_name=artifact.model_name,
        )


def default_segmentation_to_classification_connector() -> SegmentationToClassificationConnector:
    return SegmentationToClassificationConnector(
        ConnectorSpec(
            name="segmentation_to_classification",
            source_stage="segmentation",
            target_stage="classification",
        )
    )
