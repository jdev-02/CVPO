# Level 3 Video Pipeline Validation

## Test Videos Generated

| Video | Description | Frames | Purpose |
|-------|-------------|--------|---------|
| test_video_capybara_slide.mp4 | Capybara image sliding across dark background | 60 | Motion tracking, entry/exit |
| test_video_cow_pan.mp4 | Camera pan across cow image | 50 | Simulated camera movement |
| test_video_multi_objects.mp4 | Capybara + cow entering from opposite sides over bridge | 80 | Multi-object tracking |
| test_video_static_capybara.mp4 | Static capybara frame held steady | 40 | Tracking stability baseline |
| test_video_capybara_zoom.mp4 | Capybara zooming from small to full frame | 50 | Scale-varying detection |

## Results: Real YOLOv8 + CLIP + Deterministic Tracking

### Capybara Slide (motion tracking)
- YOLO detected "sheep" and "bear" (closed vocabulary — no "capybara" in COCO)
- CLIP correctly classified every detection as "capybara"
- **5 unique track IDs** across 10 frames — the sliding motion + intermittent
  detection caused ID resets. This is expected behavior: when YOLO misses a frame
  (detection count drops to 0), the tracker loses the object and assigns a new ID
  when it reappears.
- Frame 1 had zero detections — CLIP fallback classified the whole frame as "bird"
  (a degenerate result from the dark background). This surfaces a UX improvement
  opportunity: flag fallback classifications distinctly from detection-driven ones.

### Static Capybara (stability baseline)
- YOLO detected 1 object consistently on every frame
- CLIP classified as "capybara" on every frame
- **Track ID 1 maintained across all 5 frames** — perfect stability when the
  object doesn't move. This validates that the tracker's explicit state passing
  works correctly in steady-state conditions.

### Capybara Zoom (scale variation)
- YOLO detected the capybara across scale changes (small to large)
- CLIP classified as "capybara" on every frame
- **Track ID 1 maintained across all 8 frames** — the tracker handles gradual
  scale changes well because the centroid moves smoothly.

## Key Findings

### 1. Track ID Stability Depends on Detection Consistency
When YOLO detects an object on every frame, tracking is stable (one consistent ID).
When detection is intermittent (object partially visible, low contrast), the tracker
assigns new IDs on reappearance. This is inherent to tracking-by-detection and is
well-documented in the SORT/ByteTrack literature.

### 2. CLIP Correctly Classifies Across Scales
The capybara zoom test shows CLIP maintaining correct "capybara" classification
from small (30% of frame) to full frame. CLIP's internal resizing handles scale
variation well.

### 3. Fallback Classification Needs Distinct Flagging
When YOLO detects nothing, the pipeline falls back to classifying the whole frame.
This can produce misleading results (dark background -> "bird"). The guided workflow
should distinguish "detection-driven classification" from "fallback whole-image
classification" in its output.

## Research Context
- Track ID fragmentation under intermittent detection: Bewley et al. 2016 (SORT)
- Scale-invariant classification: Radford et al. 2021 (CLIP)
- Detection consistency as tracker input quality: Zhang et al. 2022 (ByteTrack)
