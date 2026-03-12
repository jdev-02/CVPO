"""Hardware detection, model capability assessment, and pre-run validation."""

from cvpo.hardware.detect import PRIVACY_NOTICE, HOW_TO_FIND_SPECS, detect_hardware
from cvpo.hardware.requirements import MODEL_REQUIREMENTS, capability_assessment, pre_run_validation

__all__ = [
    "PRIVACY_NOTICE",
    "HOW_TO_FIND_SPECS",
    "detect_hardware",
    "MODEL_REQUIREMENTS",
    "capability_assessment",
    "pre_run_validation",
]
