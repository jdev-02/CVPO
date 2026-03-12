from __future__ import annotations

from cvpo.benchmark import BenchmarkConfig, regression_check, run_benchmark


def test_run_benchmark_returns_metrics() -> None:
    result = run_benchmark(BenchmarkConfig(workflow="level0", repeats=2, warmup=0))
    assert "metrics" in result
    assert result["metrics"]["mean_ms"] >= 0
    assert len(result["timeseries_ms"]) == 2


def test_regression_check_pass_and_fail() -> None:
    baseline = {"metrics": {"mean_ms": 10.0}}
    current_fast = {"metrics": {"mean_ms": 10.5}}
    current_slow = {"metrics": {"mean_ms": 20.0}}

    pass_result = regression_check(current_fast, baseline, allowed_regression_pct=15.0)
    fail_result = regression_check(current_slow, baseline, allowed_regression_pct=15.0)

    assert pass_result["status"] == "pass"
    assert fail_result["status"] == "fail"
