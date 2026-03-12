from __future__ import annotations

import numpy as np

from cvpo.core.connector import Connector, ConnectorSpec
from cvpo.core.data_types import PipelineContext
from cvpo.core.pipeline import Pipeline, PipelineStep
from cvpo.core.stage import Stage, StageConfig


class IncrementStage(Stage):
    def validate_input(self, artifact):
        if not isinstance(artifact, np.ndarray):
            raise ValueError("Expected ndarray artifact.")

    def run(self, artifact, context):
        return artifact + 1


class DoubleConnector(Connector):
    def adapt(self, artifact):
        return artifact * 2


def test_pipeline_executes_stage_and_connector() -> None:
    stage = IncrementStage(StageConfig(name="inc", model_name="noop"))
    connector = DoubleConnector(
        ConnectorSpec(name="double", source_stage="inc", target_stage="end")
    )
    pipeline = Pipeline(name="test")
    pipeline.add_step(PipelineStep(stage=stage, connector_to_next=connector))

    ctx = PipelineContext(run_id="r1", input_source="test")
    output = pipeline.run(np.array([1, 2, 3]), ctx)
    assert output.tolist() == [4, 6, 8]
