# Manual Workflow Readiness Log (2026-03-10)

## Objective
Enable true manual end-to-end testing by a user without internal PYTHONPATH hacks.

## What Was Added
1. Module entrypoint:
   - `src/cvpo/__main__.py` so users can run `python -m cvpo ...`
2. CLI file/media input support:
   - `--input-image` for Level 0/1/2
   - `--input-video` for Level 3
   - `--max-frames` for bounded video processing
3. Graceful dependency errors:
   - Image loading requires pillow
   - Video loading requires OpenCV
   - Clear install guidance shown in runtime errors
4. Optional dependency updates:
   - `opencv-python` included in `models` extra

## Manual Run Paths
- Synthetic baseline:
  - `cvpo --level0-demo ...`
  - `cvpo --level1-demo`
  - `cvpo --level2-demo ...`
  - `cvpo --level3-demo`
- Real data:
  - `cvpo --level0-demo --input-image "..."`
  - `cvpo --level3-demo --input-video "..."`

## Notes
The deterministic defaults still run without external model downloads.
Real model backends are opt-in and added incrementally.

Environment note:
- Editable install may fail in restricted/private package-index environments.
- Added `run_cvpo.py` so manual testing is still one command without installation.
