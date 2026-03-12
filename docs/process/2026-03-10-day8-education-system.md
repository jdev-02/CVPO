# Day 8 Education System Log (2026-03-10)

## Objective
Expand deterministic educational scaffolding:
- decision-path mapping
- tiered Socratic banks
- tradeoff cards tied to goals/model stages

## What Was Added
- `src/cvpo/education/questions.py`
  - question bank by goal and experience level
- `src/cvpo/education/tradeoffs.py`
  - stage/model tradeoff cards by goal
- `resolve_goal(...)` in `decision_tree.py`
  - deterministic goal -> task decomposition -> pipeline level mapping
- Guided workflow output now includes:
  - `decision_path`
  - `tradeoff_cards`
  - tier-aware Socratic block

## Why It Matters
This moves CVPO from static prompts to a structured teaching system that adapts by user level without needing LLM calls.
