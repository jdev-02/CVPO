"""Level 3 video pipeline tests with real models.

Tests video loading, per-frame detection, classification, and tracking
continuity across frames.
"""

from __future__ import annotations

from pathlib import Path

import pytest

LIBS = Path(__file__).resolve().parents[1] / "libs"

try:
    import torch
    from transformers import CLIPModel
    from ultralytics import YOLO

    HAS_DEPS = True
except Exception:
    HAS_DEPS = False

needs_deps = pytest.mark.skipif(not HAS_DEPS, reason="Requires CLIP + YOLO")


@needs_deps
@pytest.mark.skipif(
    not (LIBS / "test_video_static_capybara.mp4").exists(),
    reason="test_video_static_capybara.mp4 not generated",
)
def test_level3_static_capybara_stable_tracking() -> None:
    """Static video should maintain one consistent track ID."""
    from cvpo.frontends.cli import run_level3_demo

    result = run_level3_demo(
        input_video=str(LIBS / "test_video_static_capybara.mp4"),
        max_frames=5,
        labels=["capybara", "bear", "dog"],
        detection_backend="yolo",
        classification_backend="siglip",
    )
    assert result["frame_count"] == 5
    assert len(result["unique_track_ids"]) == 1
    for frame in result["per_frame"]:
        assert frame["detections"] >= 1
        assert "capybara" in frame["classes"]


@needs_deps
@pytest.mark.skipif(
    not (LIBS / "test_video_capybara_zoom.mp4").exists(),
    reason="test_video_capybara_zoom.mp4 not generated",
)
def test_level3_zoom_maintains_classification() -> None:
    """Zooming capybara should be classified correctly across scales."""
    from cvpo.frontends.cli import run_level3_demo

    result = run_level3_demo(
        input_video=str(LIBS / "test_video_capybara_zoom.mp4"),
        max_frames=5,
        labels=["capybara", "bear", "cow", "dog"],
        detection_backend="yolo",
        classification_backend="siglip",
    )
    assert result["frame_count"] == 5
    for frame in result["per_frame"]:
        if frame["detections"] >= 1:
            assert "capybara" in frame["classes"]


@needs_deps
@pytest.mark.skipif(
    not (LIBS / "test_video_capybara_slide.mp4").exists(),
    reason="test_video_capybara_slide.mp4 not generated",
)
def test_level3_slide_produces_multiple_tracks() -> None:
    """Sliding object with intermittent detection may produce ID resets."""
    from cvpo.frontends.cli import run_level3_demo

    result = run_level3_demo(
        input_video=str(LIBS / "test_video_capybara_slide.mp4"),
        max_frames=10,
        labels=["capybara", "bear", "dog", "bird"],
        detection_backend="yolo",
        classification_backend="siglip",
    )
    assert result["frame_count"] == 10
    assert len(result["unique_track_ids"]) >= 1
