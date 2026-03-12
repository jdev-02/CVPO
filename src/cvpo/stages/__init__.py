"""Stage implementations."""

from cvpo.stages.classification import SigLIPStage
from cvpo.stages.detection import YOLOv8Stage
from cvpo.stages.segmentation import SAM2Stage
from cvpo.stages.tracking import ByteTrackStage

__all__ = ["SigLIPStage", "YOLOv8Stage", "SAM2Stage", "ByteTrackStage"]
