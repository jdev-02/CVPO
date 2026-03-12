# Hardware Graceful Failure + Privacy + Override + Mascot Log

## Objective
Implement four parallel improvements:
1. Hardware graceful failure (warn before running, suggest alternatives, never crash)
2. Privacy notice for hardware reading
3. Manual hardware override + OS-specific guidance
4. CVPO mascot image

## What Was Added

### 1. Hardware Graceful Failure
- `pre_run_validation()` in `src/cvpo/hardware/requirements.py`
  - Returns `clear`, `warnings`, or `blocked` status
  - Blockers and warnings list with actionable suggestions per model
  - `proceed_message` explains what the status means
- Capability assessment rows now include `suggestion` field with specific guidance
- Pre-Run Validation section added to guided workflow output (pretty and guided formats)

### 2. Privacy Notice
- `PRIVACY_NOTICE` constant in `src/cvpo/hardware/detect.py`
- Embedded in hardware profile dict and shown in Overview section of guided output
- Text: "CVPO reads your CPU, RAM, and GPU specs to recommend appropriate model sizes. This data stays on your machine and is never transmitted or stored beyond this session."

### 3. Manual Hardware Override
- CLI flags: `--override-ram-gb`, `--override-gpu-vram-gb`, `--override-gpu`
- `--show-hw-help` prints OS-specific instructions for finding hardware specs
- `HOW_TO_FIND_SPECS` dict covers macOS, Linux, Windows
- `detect_hardware()` accepts optional override parameters
- Override status shown in guided output when applied

### 4. CVPO Mascot
- Generated proprietary image: `assets/cvpo_mascot.png`
- Golden robot character with camera-lens eyes, holding magnifying glass
- Holographic pipeline screens showing detection, segmentation, classification
- Suitable for README header, Gradio welcome screen, and presentations

## Testing
- `tests/test_hardware_enhanced.py` covers all new functionality
- Privacy notice presence, override behavior, pre-run validation states,
  suggestion generation, CLI flag handling
