# Day 1 Foundation Log (2026-03-05)

## Objective
Scaffold a deterministic, procedural CVPO repository foundation:
- Python package layout
- Core stage/connector/pipeline abstractions
- Typed data contracts
- Baseline tests
- Process logging for presentation traceability

## Decisions Implemented
1. **Python packaging** uses `pyproject.toml` with Python `>=3.10`.
2. **Deterministic core interfaces**:
   - `Stage` with explicit input validation and deterministic `run()`.
   - `Connector` with explicit deterministic `adapt()`.
   - `Pipeline` as a **linear chain** executor for v1.
3. **Hybrid data contract**:
   - Typed dataclasses with numpy-backed payloads for model artifacts.
4. **Frontend strategy**:
   - CLI, Gradio, and notebook modules scaffolded as thin wrappers.
5. **Repo hygiene start**:
   - `.gitignore` created with `libs/` excluded from distribution workflows.

## Artifacts Added
- `pyproject.toml`
- `.gitignore`
- `src/cvpo/` package with `core/` and `frontends/`
- `src/cvpo/education/decision_tree.py` onboarding question skeleton
- `tests/` initial unit tests
- `docs/process/` log directory

## UX Update (User Feedback Integrated)
- Added deterministic onboarding question for **frontend selection**.
- The system now asks users whether they prefer CLI, Gradio, or Notebook and explains the tradeoffs of each.

## Why This Matters
This establishes a stable deterministic execution substrate before model-specific integration.
Every later feature (YOLO, SAM2, SigLIP, ByteTrack) plugs into the same typed stage and connector contracts.

## Next (Day 2)
Implement Level 0 hello world:
- SigLIP stage wrapper
- Single-image classification flow
- Unit tests for stage behavior
- Beginner educational content seed
