"""Detection stage implementations."""

from __future__ import annotations

from typing import List

import numpy as np

from cvpo.core.data_types import BBox, Detection, DetectionResult, PipelineContext, StageArtifact
from cvpo.core.stage import Stage, StageConfig


class YOLOv8Stage(Stage):
    """
    Detection stage with deterministic and optional real backends.

    Backends:
    - deterministic: threshold-based detector for local deterministic flows.
    - yolo: optional ultralytics-backed inference.
    """

    def __init__(self, config: StageConfig) -> None:
        super().__init__(config)
        self.backend = str(config.params.get("backend", "deterministic"))
        self.threshold = int(config.params.get("threshold", 200))
        self.label = str(config.params.get("label", "object"))

    def validate_input(self, artifact: StageArtifact) -> None:
        if not isinstance(artifact, np.ndarray):
            raise ValueError("YOLOv8Stage expects an image as numpy.ndarray.")
        if artifact.ndim not in (2, 3):
            raise ValueError("Input image must be HxW or HxWxC.")

    def run(self, artifact: StageArtifact, context: PipelineContext) -> StageArtifact:
        image = artifact
        assert isinstance(image, np.ndarray)
        if self.backend == "deterministic":
            detections = self._detect_deterministic(image)
        elif self.backend == "yolo":
            detections = self._detect_yolo(image)
        else:
            raise ValueError(f"Unknown backend '{self.backend}'.")
        return DetectionResult(image=image, detections=detections, model_name=self.config.model_name)

    def _detect_deterministic(self, image: np.ndarray) -> List[Detection]:
        if image.ndim == 2:
            gray = image
        else:
            gray = image[..., :3].mean(axis=2)
        y_idx, x_idx = np.where(gray >= self.threshold)
        if len(x_idx) == 0:
            # Fallback deterministic box in center so downstream stages can still run.
            h, w = gray.shape
            cx, cy = w // 2, h // 2
            half_w, half_h = max(4, w // 8), max(4, h // 8)
            box = BBox(
                x1=max(0, cx - half_w),
                y1=max(0, cy - half_h),
                x2=min(w - 1, cx + half_w),
                y2=min(h - 1, cy + half_h),
            )
            return [Detection(label=self.label, confidence=0.51, box=box)]

        box = BBox(
            x1=float(np.min(x_idx)),
            y1=float(np.min(y_idx)),
            x2=float(np.max(x_idx)),
            y2=float(np.max(y_idx)),
        )
        return [Detection(label=self.label, confidence=0.9, box=box)]

    @staticmethod
    def _detect_yolo(image: np.ndarray) -> List[Detection]:
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("YOLO backend requires optional dependency 'ultralytics'.") from exc

        model = YOLO("yolov8n.pt")
        results = model(image)
        detections: List[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                detections.append(
                    Detection(
                        label=str(names.get(cls_id, f"class_{cls_id}")),
                        confidence=conf,
                        box=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )
        return detections
