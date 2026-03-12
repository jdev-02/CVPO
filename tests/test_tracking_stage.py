from __future__ import annotations

import numpy as np

from cvpo.core.data_types import BBox, Detection, DetectionResult, PipelineContext, TrackState
from cvpo.core.stage import StageConfig
from cvpo.stages import ByteTrackStage


def test_tracker_keeps_same_id_for_small_motion() -> None:
    tracker = ByteTrackStage(
        StageConfig(name="tracking", model_name="bytetrack", params={"max_distance": 30.0})
    )

    frame0 = np.zeros((50, 50, 3), dtype=np.uint8)
    det0 = DetectionResult(
        image=frame0,
        detections=[Detection(label="bird", confidence=0.9, box=BBox(10, 10, 20, 20))],
        model_name="yolov8",
    )
    ctx0 = PipelineContext(run_id="r0", input_source="test", metadata={"frame_index": 0})
    res0 = tracker.run(det0, ctx0)
    track_id0 = res0.tracks[0].track_id

    frame1 = np.zeros((50, 50, 3), dtype=np.uint8)
    det1 = DetectionResult(
        image=frame1,
        detections=[Detection(label="bird", confidence=0.88, box=BBox(12, 10, 22, 20))],
        model_name="yolov8",
    )
    ctx1 = PipelineContext(
        run_id="r1",
        input_source="test",
        metadata={"frame_index": 1, "track_state": ctx0.metadata["track_state"]},
    )
    res1 = tracker.run(det1, ctx1)
    track_id1 = res1.tracks[0].track_id

    assert track_id0 == track_id1


def test_tracker_state_is_explicit_and_frozen() -> None:
    state = TrackState(next_id=5, active=((1, 10.0, 10.0, "bird"),))
    assert state.next_id == 5
    try:
        state.next_id = 6  # type: ignore[misc]
        assert False, "TrackState should be frozen"
    except AttributeError:
        pass
