"""Edge case tests: ambiguous labels, empty detections, low-res, multiple objects.

Tests both deterministic and (where available) real model behavior under
non-ideal conditions. Documents expected degradation patterns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cvpo.core.data_types import (
    BBox,
    ClassificationResult,
    Detection,
    DetectionResult,
    PipelineContext,
    SegmentationResult,
)
from cvpo.core.stage import StageConfig
from cvpo.pipelines import build_level0_pipeline, build_level1_pipeline, build_level2_pipeline
from cvpo.stages import SigLIPStage, YOLOv8Stage

LIBS = Path(__file__).resolve().parents[1] / "libs"

try:
    from transformers import CLIPModel

    HAS_CLIP = True
except Exception:
    HAS_CLIP = False

try:
    from ultralytics import YOLO

    HAS_YOLO = True
except Exception:
    HAS_YOLO = False

needs_clip = pytest.mark.skipif(not HAS_CLIP, reason="CLIP deps not installed")
needs_yolo = pytest.mark.skipif(not HAS_YOLO, reason="ultralytics not installed")


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Ambiguous / overlapping labels
# ---------------------------------------------------------------------------


@needs_clip
@pytest.mark.skipif(not (LIBS / "capybara.jpg").exists(), reason="capybara.jpg missing")
def test_ambiguous_labels_produce_distributed_scores() -> None:
    """Overlapping labels dilute confidence — documented behavior, not a bug."""
    pipeline = build_level0_pipeline(
        candidate_labels=["large rodent", "small bear", "brown animal", "mammal", "pet"],
        backend="siglip",
    )
    image = _load_image(LIBS / "capybara.jpg")
    ctx = PipelineContext(run_id="edge-ambig", input_source="capybara.jpg", frontend="cli")
    result = pipeline.run(image, ctx)
    scores = dict(result.classes[0].scores)
    top_conf = result.classes[0].confidence
    assert top_conf < 0.9, (
        f"Expected lower confidence with ambiguous labels, got {top_conf:.3f}"
    )


@needs_clip
@pytest.mark.skipif(not (LIBS / "capybara.jpg").exists(), reason="capybara.jpg missing")
def test_single_label_produces_high_confidence() -> None:
    """With only one candidate, all probability mass goes to it."""
    pipeline = build_level0_pipeline(
        candidate_labels=["animal"],
        backend="siglip",
    )
    image = _load_image(LIBS / "capybara.jpg")
    ctx = PipelineContext(run_id="edge-single", input_source="capybara.jpg", frontend="cli")
    result = pipeline.run(image, ctx)
    assert result.classes[0].confidence > 0.99


# ---------------------------------------------------------------------------
# Empty / blank images
# ---------------------------------------------------------------------------


def test_deterministic_pipeline_handles_blank_image() -> None:
    """A fully black image should still produce a result, not crash."""
    pipeline = build_level2_pipeline(
        candidate_labels=["nothing", "object", "pattern"],
        detection_backend="deterministic",
        segmentation_backend="deterministic",
        classification_backend="deterministic",
    )
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    ctx = PipelineContext(run_id="edge-blank", input_source="synthetic", frontend="cli")
    result = pipeline.run(blank, ctx)
    assert isinstance(result, ClassificationResult)
    assert len(result.classes) >= 1


@needs_clip
def test_real_clip_handles_blank_image() -> None:
    """CLIP should still return scores on a blank image — no crash."""
    pipeline = build_level0_pipeline(
        candidate_labels=["blank", "object", "animal"],
        backend="siglip",
    )
    blank = np.zeros((224, 224, 3), dtype=np.uint8)
    ctx = PipelineContext(run_id="edge-blank-clip", input_source="synthetic", frontend="cli")
    result = pipeline.run(blank, ctx)
    assert isinstance(result, ClassificationResult)


# ---------------------------------------------------------------------------
# Low resolution
# ---------------------------------------------------------------------------


@needs_clip
@pytest.mark.skipif(not (LIBS / "capybara.jpg").exists(), reason="capybara.jpg missing")
def test_low_resolution_still_classifies() -> None:
    """Downscaling to 32x32 should still produce a result, possibly less confident."""
    from PIL import Image

    img = Image.open(LIBS / "capybara.jpg").convert("RGB").resize((32, 32))
    arr = np.array(img, dtype=np.uint8)
    pipeline = build_level0_pipeline(
        candidate_labels=["capybara", "beaver", "dog"],
        backend="siglip",
    )
    ctx = PipelineContext(run_id="edge-lowres", input_source="capybara_32x32", frontend="cli")
    result = pipeline.run(arr, ctx)
    assert isinstance(result, ClassificationResult)
    assert result.classes[0].label in {"capybara", "beaver", "dog"}


# ---------------------------------------------------------------------------
# Multiple detections
# ---------------------------------------------------------------------------


def test_connector_handles_multiple_detections() -> None:
    """Multiple detections should produce multiple classification results."""
    from cvpo.connectors import (
        default_detection_to_segmentation_connector,
        default_segmentation_to_classification_connector,
    )

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[10:40, 10:50] = 200
    image[60:90, 120:180] = 150

    det_result = DetectionResult(
        image=image,
        detections=[
            Detection(label="obj_a", confidence=0.9, box=BBox(10, 10, 50, 40)),
            Detection(label="obj_b", confidence=0.8, box=BBox(120, 60, 180, 90)),
        ],
        model_name="yolov8",
    )

    conn1 = default_detection_to_segmentation_connector()
    prompt_set = conn1.adapt(det_result)
    assert len(prompt_set.prompts) == 2

    from cvpo.stages import SAM2Stage

    seg_stage = SAM2Stage(
        StageConfig(name="seg", model_name="sam2", params={"backend": "deterministic"})
    )
    ctx = PipelineContext(run_id="edge-multi", input_source="synthetic", frontend="cli")
    seg_result = seg_stage.run(prompt_set, ctx)
    assert isinstance(seg_result, SegmentationResult)
    assert len(seg_result.masks) == 2

    conn2 = default_segmentation_to_classification_connector()
    cls_input = conn2.adapt(seg_result)
    assert len(cls_input.regions) == 2

    cls_stage = SigLIPStage(
        StageConfig(
            name="cls",
            model_name="siglip",
            params={"backend": "deterministic", "candidate_labels": ["a", "b"]},
        )
    )
    cls_result = cls_stage.run(cls_input, ctx)
    assert isinstance(cls_result, ClassificationResult)
    assert len(cls_result.classes) == 2


# ---------------------------------------------------------------------------
# YOLO on no-object scenes
# ---------------------------------------------------------------------------


@needs_yolo
def test_yolo_on_uniform_image_produces_no_detections() -> None:
    """A uniform color image should yield zero or very few detections."""
    stage = YOLOv8Stage(
        StageConfig(name="det", model_name="yolov8", params={"backend": "yolo"})
    )
    uniform = np.full((224, 224, 3), fill_value=128, dtype=np.uint8)
    ctx = PipelineContext(run_id="edge-uniform", input_source="synthetic", frontend="cli")
    result = stage.run(uniform, ctx)
    assert isinstance(result, DetectionResult)
    assert len(result.detections) <= 2
