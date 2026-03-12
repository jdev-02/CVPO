"""Connector abstraction for deterministic stage adaptation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from cvpo.core.data_types import StageArtifact


@dataclass(slots=True)
class ConnectorSpec:
    """Explicit source-target contract for a connector."""

    name: str
    source_stage: str
    target_stage: str


class Connector(ABC):
    """Transforms stage output into the next stage input."""

    def __init__(self, spec: ConnectorSpec) -> None:
        self.spec = spec

    @abstractmethod
    def adapt(self, artifact: StageArtifact) -> StageArtifact:
        """Return a deterministic adaptation of the input artifact."""
