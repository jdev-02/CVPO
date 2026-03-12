"""Classification stage implementations."""

from __future__ import annotations

import hashlib
from typing import Dict, List

import numpy as np

from cvpo.core.data_types import (
    Classification,
    ClassificationInputSet,
    ClassificationResult,
    PipelineContext,
    StageArtifact,
)
from cvpo.core.stage import Stage, StageConfig


class SigLIPStage(Stage):
    """
    Level-0 classification stage.

    Backends:
    - "deterministic": pure-python deterministic scorer for zero-cost local runs.
    - "siglip": real SigLIP inference via transformers (optional dependency).
    """

    def __init__(self, config: StageConfig) -> None:
        super().__init__(config)
        self.backend = str(config.params.get("backend", "deterministic"))
        labels = config.params.get("candidate_labels", [])
        if not isinstance(labels, list) or not labels:
            raise ValueError("SigLIPStage requires non-empty 'candidate_labels' in config.params.")
        self.candidate_labels: List[str] = [str(label) for label in labels]

    def validate_input(self, artifact: StageArtifact) -> None:
        if isinstance(artifact, np.ndarray):
            if artifact.ndim not in (2, 3):
                raise ValueError("Input image must be HxW or HxWxC.")
            return
        if isinstance(artifact, ClassificationInputSet):
            if not artifact.regions:
                raise ValueError("ClassificationInputSet.regions cannot be empty.")
            return
        raise ValueError("SigLIPStage expects ndarray or ClassificationInputSet.")

    def run(self, artifact: StageArtifact, context: PipelineContext) -> StageArtifact:
        if isinstance(artifact, ClassificationInputSet):
            image = artifact.image
            classes: List[Classification] = []
            for region in artifact.regions:
                scores = self._predict(region.crop)
                top_label = max(scores, key=scores.get)
                top_score = float(scores[top_label])
                classes.append(self._make_classification(
                    top_label, top_score, scores, detection_index=region.detection_index,
                ))
            return ClassificationResult(image=image, classes=classes, model_name=self.config.model_name)

        image = artifact
        assert isinstance(image, np.ndarray)
        scores = self._predict(image)
        top_label = max(scores, key=scores.get)
        top_score = float(scores[top_label])
        return ClassificationResult(
            image=image,
            classes=[self._make_classification(top_label, top_score, scores)],
            model_name=self.config.model_name,
        )

    def _make_classification(
        self,
        label: str,
        confidence: float,
        scores: Dict[str, float],
        detection_index: int | None = None,
    ) -> Classification:
        return Classification(
            label=label,
            confidence=confidence,
            candidate_labels=tuple(self.candidate_labels),
            scores=tuple(sorted(scores.items())),
            detection_index=detection_index,
        )

    def _predict(self, image: np.ndarray) -> Dict[str, float]:
        if self.backend == "deterministic":
            return self._predict_scores_deterministic(image, self.candidate_labels)
        if self.backend == "siglip":
            return self._predict_scores_siglip(image, self.candidate_labels)
        raise ValueError(f"Unknown backend '{self.backend}'.")

    @staticmethod
    def _predict_scores_deterministic(image: np.ndarray, labels: List[str]) -> Dict[str, float]:
        """Deterministic heuristic scorer used for Day-2 runnable baseline."""
        if image.ndim == 2:
            image_rgb = np.stack([image, image, image], axis=-1).astype(np.float32)
        else:
            image_rgb = image[..., :3].astype(np.float32)

        mean_rgb = image_rgb.mean(axis=(0, 1)) / 255.0
        std_rgb = image_rgb.std(axis=(0, 1)) / 255.0
        image_feature = np.concatenate([mean_rgb, std_rgb])
        norm = float(np.linalg.norm(image_feature) + 1e-8)
        image_feature = image_feature / norm

        raw_scores: Dict[str, float] = {}
        for label in labels:
            digest = hashlib.sha256(label.encode("utf-8")).digest()
            vec = np.array([digest[i] for i in range(6)], dtype=np.float32) / 255.0
            vec_norm = float(np.linalg.norm(vec) + 1e-8)
            vec = vec / vec_norm
            # Map cosine similarity from [-1,1] to [0,1].
            score = float((np.dot(image_feature, vec) + 1.0) / 2.0)
            raw_scores[label] = score

        # Re-normalize into a probability-like distribution for readability.
        total = sum(raw_scores.values()) + 1e-12
        return {label: float(score / total) for label, score in raw_scores.items()}

    @staticmethod
    def _predict_scores_siglip(image: np.ndarray, labels: List[str]) -> Dict[str, float]:
        """Optional real SigLIP inference path."""
        try:
            from PIL import Image
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "SigLIP backend requires optional dependencies (transformers, pillow, torch)."
            ) from exc

        model_id = "google/siglip-base-patch16-224"
        classifier = pipeline(
            task="zero-shot-image-classification",
            model=model_id,
        )
        pil_image = Image.fromarray(image.astype(np.uint8))
        outputs = classifier(images=pil_image, candidate_labels=labels)
        if not isinstance(outputs, list):
            outputs = [outputs]

        # transformers returns highest-first scores.
        scores = {entry["label"]: float(entry["score"]) for entry in outputs}
        # Ensure every label appears in output map.
        for label in labels:
            scores.setdefault(label, 0.0)
        total = sum(scores.values()) + 1e-12
        return {label: float(score / total) for label, score in scores.items()}
