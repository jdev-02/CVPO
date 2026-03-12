"""Integration tests using real model backends on real images.

These tests require model dependencies (torch, transformers, ultralytics).
They are skipped automatically when dependencies are not installed.
They validate that the pipeline produces CORRECT results, not just that it runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

LIBS = Path(__file__).resolve().parents[1] / "libs"
CAPYBARA = LIBS / "capybara.jpg"
COW = LIBS / "cow.jpg"
GG_BRIDGE = LIBS / "gg_bridge.jpg"

try:
    import torch
    from transformers import CLIPModel

    HAS_CLIP = True
except Exception:
    HAS_CLIP = False

try:
    from ultralytics import YOLO

    HAS_YOLO = True
except Exception:
    HAS_YOLO = False


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


needs_clip = pytest.mark.skipif(not HAS_CLIP, reason="CLIP dependencies not installed")
needs_yolo = pytest.mark.skipif(not HAS_YOLO, reason="ultralytics not installed")
needs_both = pytest.mark.skipif(
    not (HAS_CLIP and HAS_YOLO), reason="Requires both CLIP and YOLO"
)


# ---------------------------------------------------------------------------
# Level 0: CLIP classification correctness
# ---------------------------------------------------------------------------


@needs_clip
@pytest.mark.skipif(not CAPYBARA.exists(), reason="capybara.jpg not in libs/")
def test_clip_classifies_capybara_correctly() -> None:
    from cvpo.core.data_types import ClassificationResult, PipelineContext
    from cvpo.pipelines import build_level0_pipeline

    pipeline = build_level0_pipeline(
        candidate_labels=["capybara", "beaver", "otter", "bear", "dog"],
        backend="siglip",
    )
    image = _load_image(CAPYBARA)
    result = pipeline.run(
        image, PipelineContext(run_id="test-real-l0", input_source="capybara.jpg", frontend="cli")
    )
    assert isinstance(result, ClassificationResult)
    assert result.classes[0].label == "capybara"
    assert result.classes[0].confidence > 0.9


@needs_clip
@pytest.mark.skipif(not COW.exists(), reason="cow.jpg not in libs/")
def test_clip_classifies_cow_correctly() -> None:
    from cvpo.core.data_types import ClassificationResult, PipelineContext
    from cvpo.pipelines import build_level0_pipeline

    pipeline = build_level0_pipeline(
        candidate_labels=["cow", "horse", "dog", "sheep", "goat"],
        backend="siglip",
    )
    image = _load_image(COW)
    result = pipeline.run(
        image, PipelineContext(run_id="test-real-cow", input_source="cow.jpg", frontend="cli")
    )
    assert isinstance(result, ClassificationResult)
    assert result.classes[0].label == "cow"
    assert result.classes[0].confidence > 0.9


@needs_clip
@pytest.mark.skipif(not GG_BRIDGE.exists(), reason="gg_bridge.jpg not in libs/")
def test_clip_classifies_bridge_correctly() -> None:
    from cvpo.core.data_types import ClassificationResult, PipelineContext
    from cvpo.pipelines import build_level0_pipeline

    pipeline = build_level0_pipeline(
        candidate_labels=["bridge", "mountain", "beach", "city", "forest"],
        backend="siglip",
    )
    image = _load_image(GG_BRIDGE)
    result = pipeline.run(
        image, PipelineContext(run_id="test-real-bridge", input_source="gg_bridge.jpg", frontend="cli")
    )
    assert isinstance(result, ClassificationResult)
    assert result.classes[0].label == "bridge"
    assert result.classes[0].confidence > 0.9


# ---------------------------------------------------------------------------
# Level 0: Label sensitivity (contrastive scoring behavior)
# ---------------------------------------------------------------------------


@needs_clip
@pytest.mark.skipif(not CAPYBARA.exists(), reason="capybara.jpg not in libs/")
def test_clip_scores_shift_when_correct_label_removed() -> None:
    """When 'capybara' is removed from labels, the model picks the next-closest animal."""
    from cvpo.core.data_types import PipelineContext
    from cvpo.pipelines import build_level0_pipeline

    pipeline = build_level0_pipeline(
        candidate_labels=["beaver", "otter", "bear", "dog"],
        backend="siglip",
    )
    image = _load_image(CAPYBARA)
    result = pipeline.run(
        image, PipelineContext(run_id="test-shift", input_source="capybara.jpg", frontend="cli")
    )
    assert result.classes[0].label != "capybara"
    assert result.classes[0].label in {"beaver", "otter", "bear", "dog"}


# ---------------------------------------------------------------------------
# YOLOv8: Detection presence
# ---------------------------------------------------------------------------


@needs_yolo
@pytest.mark.skipif(not CAPYBARA.exists(), reason="capybara.jpg not in libs/")
def test_yolo_detects_object_in_capybara() -> None:
    from cvpo.core.data_types import DetectionResult, PipelineContext
    from cvpo.core.stage import StageConfig
    from cvpo.stages import YOLOv8Stage

    stage = YOLOv8Stage(
        StageConfig(name="det", model_name="yolov8", params={"backend": "yolo"})
    )
    image = _load_image(CAPYBARA)
    result = stage.run(
        image, PipelineContext(run_id="test-yolo", input_source="capybara.jpg", frontend="cli")
    )
    assert isinstance(result, DetectionResult)
    assert len(result.detections) >= 1
    assert result.detections[0].confidence > 0.5


@needs_yolo
@pytest.mark.skipif(not GG_BRIDGE.exists(), reason="gg_bridge.jpg not in libs/")
def test_yolo_detects_something_in_bridge_scene() -> None:
    from cvpo.core.data_types import DetectionResult, PipelineContext
    from cvpo.core.stage import StageConfig
    from cvpo.stages import YOLOv8Stage

    stage = YOLOv8Stage(
        StageConfig(name="det", model_name="yolov8", params={"backend": "yolo"})
    )
    image = _load_image(GG_BRIDGE)
    result = stage.run(
        image, PipelineContext(run_id="test-yolo-bridge", input_source="gg_bridge.jpg", frontend="cli")
    )
    assert isinstance(result, DetectionResult)
    assert len(result.detections) >= 1


# ---------------------------------------------------------------------------
# Level 2: Pipeline composition (detect -> segment -> classify)
# ---------------------------------------------------------------------------


@needs_both
@pytest.mark.skipif(not CAPYBARA.exists(), reason="capybara.jpg not in libs/")
def test_level2_pipeline_corrects_yolo_label_on_capybara() -> None:
    """The core value prop: YOLO says 'bear', pipeline says 'capybara'."""
    from cvpo.core.data_types import ClassificationResult, PipelineContext
    from cvpo.pipelines import build_level2_pipeline

    pipeline = build_level2_pipeline(
        candidate_labels=["capybara", "beaver", "otter", "bear", "dog"],
        detection_backend="yolo",
        segmentation_backend="deterministic",
        classification_backend="siglip",
    )
    image = _load_image(CAPYBARA)
    result = pipeline.run(
        image, PipelineContext(run_id="test-l2-cap", input_source="capybara.jpg", frontend="cli")
    )
    assert isinstance(result, ClassificationResult)
    assert len(result.classes) >= 1
    assert result.classes[0].label == "capybara"


@needs_both
@pytest.mark.skipif(not COW.exists(), reason="cow.jpg not in libs/")
def test_level2_pipeline_on_cow_does_not_break() -> None:
    """Even when YOLO detects nothing, pipeline falls back gracefully."""
    from cvpo.core.data_types import ClassificationResult, PipelineContext
    from cvpo.pipelines import build_level2_pipeline

    pipeline = build_level2_pipeline(
        candidate_labels=["cow", "horse", "dog", "sheep", "goat"],
        detection_backend="yolo",
        segmentation_backend="deterministic",
        classification_backend="siglip",
    )
    image = _load_image(COW)
    result = pipeline.run(
        image, PipelineContext(run_id="test-l2-cow", input_source="cow.jpg", frontend="cli")
    )
    assert isinstance(result, ClassificationResult)
    assert len(result.classes) >= 1
    assert result.classes[0].label == "cow"
