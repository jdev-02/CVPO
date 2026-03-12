# Day 6 Hardware Capability Log (2026-03-10)

## Objective
Add hardware-awareness output to guided workflow runs.

## What Was Added
- `src/cvpo/hardware/detect.py`
  - runtime hardware profile detection (platform, CPU, logical cores, RAM, GPU flag placeholder)
- `src/cvpo/hardware/requirements.py`
  - model requirement table
  - capability assessment function producing per-model status
- Guided CLI output now includes:
  - Hardware Capability Card
  - Model Requirement Table

## Status Mapping
- `good`: hardware meets practical guidance
- `degraded`: runnable but likely slower or less ideal (for example, no GPU where recommended)
- `not_recommended`: below minimum memory guidance

## Note
GPU/VRAM probing is currently conservative and defaults to CPU-first unless explicit GPU detection is added.
