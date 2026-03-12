"""Stage interface for deterministic pipeline execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from cvpo.core.data_types import PipelineContext, StageArtifact


@dataclass(slots=True)
class StageConfig:
    """Configuration block attached to a stage instance."""

    name: str
    model_name: str
    params: Dict[str, Any] = field(default_factory=dict)


class Stage(ABC):
    """Base class for every executable stage in CVPO."""

    def __init__(self, config: StageConfig) -> None:
        self.config = config

    @abstractmethod
    def validate_input(self, artifact: StageArtifact) -> None:
        """Raise ValueError if input artifact is invalid for this stage."""

    @abstractmethod
    def run(self, artifact: StageArtifact, context: PipelineContext) -> StageArtifact:
        """Execute deterministic stage logic and return the next artifact."""
