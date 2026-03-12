# Day 9 Benchmark + Regression Log (2026-03-10)

## Objective
Add deterministic benchmarking and regression checks for workflow performance tracking across environments.

## What Was Added
- `src/cvpo/benchmark/runner.py`
  - benchmark config
  - timed run loop (warmup + repeats)
  - metrics: mean/std/p50/p90/p99/min/max
  - environment snapshot with hardware metadata
  - regression check vs baseline (`pass`/`fail`)
- CLI benchmark mode:
  - `--benchmark`
  - `--benchmark-workflow`
  - `--benchmark-repeats`
  - `--benchmark-warmup`
  - `--benchmark-env-tag`
  - `--benchmark-baseline`
  - `--benchmark-regression-threshold`
- Report support:
  - JSON/Markdown report via `--save-report`
  - auto-generated benchmark timeseries CSV adjacent to saved report

## Why This Matters
Provides repeatable, machine-comparable measurements and makes performance regressions visible during iterative development.
