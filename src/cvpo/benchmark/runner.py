"""Benchmark runner and regression checks for CVPO workflows."""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from cvpo.core.data_types import PipelineContext
from cvpo.frontends.cli import run_guided_workflow, run_level3_demo
from cvpo.hardware import detect_hardware
from cvpo.pipelines import build_level0_pipeline, build_level1_pipeline, build_level2_pipeline


@dataclass(slots=True)
class BenchmarkConfig:
    workflow: str
    repeats: int = 5
    warmup: int = 1
    labels: list[str] | None = None
    max_frames: int = 8
    env_tag: str = "local"


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    labels = config.labels or ["goose", "duck", "pigeon", "crow"]
    fn = _resolve_workflow_callable(config.workflow, labels=labels, max_frames=config.max_frames)

    for _ in range(config.warmup):
        fn()

    times_ms: list[float] = []
    for _ in range(config.repeats):
        start = time.perf_counter()
        fn()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)

    times_np = np.array(times_ms, dtype=np.float64)
    return {
        "benchmark_version": "v1",
        "workflow": config.workflow,
        "env_tag": config.env_tag,
        "repeats": config.repeats,
        "warmup": config.warmup,
        "metrics": {
            "mean_ms": float(times_np.mean()),
            "std_ms": float(times_np.std()),
            "p50_ms": float(np.percentile(times_np, 50)),
            "p90_ms": float(np.percentile(times_np, 90)),
            "p99_ms": float(np.percentile(times_np, 99)),
            "min_ms": float(times_np.min()),
            "max_ms": float(times_np.max()),
        },
        "timeseries_ms": [float(v) for v in times_ms],
        "environment": _environment_snapshot(config.env_tag),
    }


def regression_check(
    current_result: dict[str, Any],
    baseline_result: dict[str, Any],
    allowed_regression_pct: float = 15.0,
) -> dict[str, Any]:
    current_mean = float(current_result["metrics"]["mean_ms"])
    baseline_mean = float(baseline_result["metrics"]["mean_ms"])
    if baseline_mean <= 0:
        return {
            "status": "invalid_baseline",
            "message": "Baseline mean must be > 0.",
            "current_mean_ms": current_mean,
            "baseline_mean_ms": baseline_mean,
        }
    delta_pct = ((current_mean - baseline_mean) / baseline_mean) * 100.0
    passed = delta_pct <= allowed_regression_pct
    return {
        "status": "pass" if passed else "fail",
        "allowed_regression_pct": allowed_regression_pct,
        "delta_pct": float(delta_pct),
        "current_mean_ms": current_mean,
        "baseline_mean_ms": baseline_mean,
    }


def _resolve_workflow_callable(
    workflow: str, labels: list[str], max_frames: int
) -> Callable[[], dict[str, Any]]:
    if workflow == "level0":
        pipeline = build_level0_pipeline(candidate_labels=labels, backend="deterministic")
        image = np.full((224, 224, 3), fill_value=180, dtype=np.uint8)

        def _run() -> dict[str, Any]:
            result = pipeline.run(
                image,
                PipelineContext(run_id="bench-l0", input_source="synthetic", frontend="cli"),
            )
            return {"classes": len(result.classes)}

        return _run

    if workflow == "level1":
        pipeline = build_level1_pipeline(
            detection_backend="deterministic", segmentation_backend="deterministic"
        )
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        image[80:160, 90:170] = 255

        def _run() -> dict[str, Any]:
            result = pipeline.run(
                image,
                PipelineContext(run_id="bench-l1", input_source="synthetic", frontend="cli"),
            )
            return {"masks": len(result.masks)}

        return _run

    if workflow == "level2":
        pipeline = build_level2_pipeline(
            candidate_labels=labels,
            detection_backend="deterministic",
            segmentation_backend="deterministic",
            classification_backend="deterministic",
        )
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        image[70:160, 80:170] = 255

        def _run() -> dict[str, Any]:
            result = pipeline.run(
                image,
                PipelineContext(run_id="bench-l2", input_source="synthetic", frontend="cli"),
            )
            return {"classes": len(result.classes)}

        return _run

    if workflow == "level3":
        return lambda: run_level3_demo(input_video=None, max_frames=max_frames)

    if workflow == "guided_geese":
        return lambda: run_guided_workflow(
            goal="geese_tracking",
            frontend_choice="cli",
            experience_level="beginner",
            skip_socratic=True,
            input_image=None,
            input_video=None,
            labels=labels,
            max_frames=max_frames,
        )

    raise ValueError(f"Unsupported workflow '{workflow}'.")


def _environment_snapshot(env_tag: str) -> dict[str, Any]:
    return {
        "env_tag": env_tag,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hardware": detect_hardware(),
    }
