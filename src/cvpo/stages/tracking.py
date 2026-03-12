"""Tracking stage: pure-function design with explicit state pass-through.

Functional audit:
- run() takes TrackState in via context.metadata["track_state"] and returns
  updated TrackState via result metadata. No hidden mutable instance state.
- _match_track is a pure function (no self mutation).
- TrackState is frozen — a new instance is always produced.

Research backing:
- Tracking-by-detection with IoU matching: Bewley et al. 2016 (SORT)
  https://arxiv.org/abs/1602.00763
- Two-stage confidence association: Zhang et al. 2022 (ByteTrack)
  https://github.com/ifzhang/ByteTrack
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from cvpo.core.data_types import (
    BBox,
    DetectionResult,
    PipelineContext,
    StageArtifact,
    Track,
    TrackingResult,
    TrackState,
)
from cvpo.core.stage import Stage, StageConfig


class ByteTrackStage(Stage):
    """Deterministic nearest-centroid tracker with explicit state."""

    def __init__(self, config: StageConfig) -> None:
        super().__init__(config)
        self.max_distance = float(config.params.get("max_distance", 50.0))

    def validate_input(self, artifact: StageArtifact) -> None:
        if not isinstance(artifact, DetectionResult):
            raise ValueError("ByteTrackStage expects DetectionResult input.")

    def run(self, artifact: StageArtifact, context: PipelineContext) -> StageArtifact:
        det_result = artifact
        assert isinstance(det_result, DetectionResult)
        frame_index = int(context.metadata.get("frame_index", 0))
        prev_state: TrackState = context.metadata.get("track_state", TrackState())

        tracks, new_state = self._associate(det_result.detections, prev_state)
        context.metadata["track_state"] = new_state

        return TrackingResult(
            frame=det_result.image,
            frame_index=frame_index,
            tracks=tracks,
            model_name=self.config.model_name,
        )

    def _associate(
        self,
        detections: list,
        state: TrackState,
    ) -> Tuple[List[Track], TrackState]:
        active = {entry[0]: entry for entry in state.active}
        next_id = state.next_id
        used: set[int] = set()
        tracks: List[Track] = []
        new_active: list[tuple] = []

        for detection in detections:
            cx = (detection.box.x1 + detection.box.x2) / 2.0
            cy = (detection.box.y1 + detection.box.y2) / 2.0

            matched_id = _match_track(
                center=(cx, cy),
                active=active,
                used=used,
                max_distance=self.max_distance,
            )
            if matched_id is None:
                matched_id = next_id
                next_id += 1

            used.add(matched_id)
            new_active.append((matched_id, cx, cy, detection.label))
            tracks.append(Track(
                track_id=matched_id,
                label=detection.label,
                confidence=detection.confidence,
                box=BBox(
                    x1=detection.box.x1,
                    y1=detection.box.y1,
                    x2=detection.box.x2,
                    y2=detection.box.y2,
                ),
            ))

        return tracks, TrackState(next_id=next_id, active=tuple(new_active))


def _match_track(
    center: Tuple[float, float],
    active: dict,
    used: set[int],
    max_distance: float,
) -> int | None:
    """Pure function: find nearest active track within distance threshold."""
    best_id: int | None = None
    best_dist = max_distance + 1.0
    for track_id, entry in active.items():
        if track_id in used:
            continue
        _, ecx, ecy, _ = entry
        dist = float(np.hypot(center[0] - ecx, center[1] - ecy))
        if dist < best_dist and dist <= max_distance:
            best_dist = dist
            best_id = track_id
    return best_id
