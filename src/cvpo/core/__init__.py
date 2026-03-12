"""Core CVPO abstractions."""

from cvpo.core.connector import Connector, ConnectorSpec
from cvpo.core.pipeline import Pipeline, PipelineStep
from cvpo.core.stage import Stage, StageConfig

__all__ = [
    "Connector",
    "ConnectorSpec",
    "Pipeline",
    "PipelineStep",
    "Stage",
    "StageConfig",
]
