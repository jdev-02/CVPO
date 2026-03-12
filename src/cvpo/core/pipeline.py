"""Linear deterministic pipeline executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from cvpo.core.connector import Connector
from cvpo.core.data_types import PipelineContext, StageArtifact
from cvpo.core.stage import Stage


@dataclass(slots=True)
class PipelineStep:
    """One stage and an optional connector to the next stage."""

    stage: Stage
    connector_to_next: Connector | None = None


@dataclass(slots=True)
class Pipeline:
    """Executes a linear chain of stages."""

    name: str
    steps: List[PipelineStep] = field(default_factory=list)

    def add_step(self, step: PipelineStep) -> None:
        self.steps.append(step)

    def run(self, initial_artifact: StageArtifact, context: PipelineContext) -> StageArtifact:
        artifact = initial_artifact
        for step in self.steps:
            step.stage.validate_input(artifact)
            artifact = step.stage.run(artifact, context)
            if step.connector_to_next is not None:
                artifact = step.connector_to_next.adapt(artifact)
        return artifact
