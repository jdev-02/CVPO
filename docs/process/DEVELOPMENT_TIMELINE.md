# CVPO Development Timeline

## Overview

CVPO was built iteratively from first principles over multiple sessions, starting
from a blank repo with only a README and LICENSE.

## Phase 1: Planning (Session 1)

**Approach**: Plan mode — no code written. Pure design iteration.

1. Defined the product vision: educational CV pipeline orchestrator
2. Established three-tier user experience (beginner/intermediate/advanced)
3. Validated "orchestrator" as the correct terminology
4. Defined the core constraint: procedural, deterministic, zero-cost
5. Chose NVIDIA-style state machine as the structural pattern
6. Vetted candidate models: YOLOv8, SAM2, SigLIP, ByteTrack, DINOv2, CLIP
7. Triaged 43 class reference files to identify covered topics and gaps
8. Identified four critical knowledge gaps: SAM2, CLIP/SigLIP, ByteTrack internals, YOLOv8 architecture
9. Found and cataloged vetted sources for all four gaps
10. Designed the beginner user experience walkthrough (geese tracking scenario)
11. Decided on dual-register language (plain + technical terms together)
12. Produced 10-day implementation plan

## Phase 2: Core Architecture (Day 1-5)

**Day 1**: Scaffolded repo structure, defined Stage/Connector/Pipeline abstractions,
typed data contracts, dev environment, process journal.

**Day 2**: Level 0 hello world — SigLIPStage with deterministic backend,
first runnable CLI demo.

**Day 3**: Level 1 — YOLOv8Stage, SAM2Stage, first connector
(detection -> segmentation prompts).

**Day 4**: Level 2 — segmentation -> classification connector,
3-model pipeline end-to-end.

**Day 5**: Level 3 — ByteTrackStage, video frame processing,
full 4-model deterministic workflow.

## Phase 3: Product Features (Day 6-8)

**Day 6**: Hardware detection, model requirement tables, capability assessment.

**Day 7**: All three frontends (CLI, Gradio, Notebook) as thin wrappers over
shared core workflow.

**Day 8**: Education system — decision tree, tiered Socratic banks, tradeoff cards,
dual-register language rewrite, citation URLs.

## Phase 4: Quality + Polish (Day 9-10)

**Day 9**: Benchmark suite with regression checks, report export.

**Day 10**: Repo hygiene checker, release checklist, settings explainability.

## Phase 5: Real Model Validation

**Functional programming audit**: Made all scalar types frozen/immutable, tracking
state explicit, connectors pure, research citations embedded in code.

**Real model wiring**:
- CLIP (openai/clip-vit-base-patch32) for zero-shot classification
- YOLOv8-nano for object detection
- Validated on capybara.jpg, cow.jpg, gg_bridge.jpg, trees_forest.jpg

**Key finding**: Pipeline produces semantically correct results that neither model
achieves alone. YOLO (closed vocabulary, 80 COCO classes) says "bear" for a capybara.
CLIP (open vocabulary) correctly says "capybara". Through the pipeline: correct
localization AND correct classification.

**Edge case validation**: 7 edge cases tested (ambiguous labels, blank images,
low resolution, multiple detections, uniform images, no-detection fallback,
non-COCO objects). Pipeline never crashes, degrades gracefully.

## Metrics

- Total tests: 58 (all passing)
- Test categories: unit (core abstractions), integration (pipeline flows),
  real-model (CLIP + YOLO correctness), edge cases
- Linter diagnostics: clean throughout
- Repo hygiene check: passing
- Process log entries: 13 dated documents

## Key Design Decisions (and Why)

1. **Deterministic core, real models optional**: Anyone can run the full demo workflow
   without downloading model weights. Real backends switch on with `--real-models`.

2. **Functional programming audit**: Immutable data contracts, pure connectors, explicit
   tracking state. Makes the pipeline predictable and testable.

3. **Dual-register language**: Every explanation uses plain language first, then
   introduces the technical term. Serves beginners and experts simultaneously.

4. **Always-on honest assessment**: Every pipeline run shows what works, what doesn't,
   and what the pipeline is NOT suitable for. Users are never surprised.

5. **Contextual guidance over standalone commands**: Hardware override instructions and
   spec-finding help only appear when a problem is detected — not as flags users need
   to discover.

6. **CLIP instead of SigLIP**: SigLIP has a tokenizer compatibility bug in transformers
   5.x. CLIP provides the same contrastive vision-language capability and works reliably.
   Both are documented as valid backends.
