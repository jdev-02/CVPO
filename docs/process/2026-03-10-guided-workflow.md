# Guided Workflow Command Log (2026-03-10)

## Objective
Provide a single user-facing command that feels like the product experience:
- frontend choice captured
- deterministic problem mapping
- always-on honest assessment
- optional Socratic skip
- workflow execution in same command

## What Was Added
- CLI flag: `--workflow-demo`
- Goal routing:
  - `geese_tracking` -> full Level 3 execution path
  - `image_labeling` -> Level 0 execution path
- Frontend explanation integrated from onboarding question bank
- Honest assessment card always emitted
- Socratic block always emitted with skip support
- Optional report export:
  - `--save-report path.json` for structured artifact capture
  - `--save-report path.md` for presentation-friendly narrative + raw JSON

## Manual Example
`python run_cvpo.py --workflow-demo --goal geese_tracking --frontend-choice gradio --skip-socratic`

## Cleanup Plan
Report export is intentionally lightweight and can be removed after class deck completion if no longer needed.
