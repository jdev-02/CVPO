# CVPO Presentation Summary

## One-Line Pitch
CVPO is a zero-cost, deterministic computer vision pipeline orchestrator that
composes open-source models and educates users from first principles.

## Architecture (for slide diagram)

```
User Goal → Decision Tree → Pipeline Builder
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              [Detection]    [Segmentation]   [Classification]    [Tracking]
              YOLOv8-nano      SAM2-tiny       CLIP/SigLIP       ByteTrack
                    │               │               │               │
                    └── Connector ──┴── Connector ──┘               │
                                                    └── Connector ──┘
                                                            │
                                                    ┌───────┴───────┐
                                                    ▼               ▼
                                              [Results]     [Honest Assessment]
```

Each Connector is a pure function that reshapes one model's output into
the next model's expected input — no LLM needed, fully deterministic.

## Key Finding: The Capybara Value Proposition

| Method | What it says | Correct? |
|--------|-------------|----------|
| YOLO alone | "bear" (90%) | Wrong species, right location |
| CLIP alone | "capybara" (99.8%) | Right species, no location |
| CVPO pipeline | "capybara" at [0,118,678,675] | Right species AND right location |

**Why**: YOLO is a closed-vocabulary detector (80 COCO classes — no "capybara").
CLIP is an open-vocabulary classifier (trained on 400M image-text pairs).
The connector crops YOLO's detection region and feeds it to CLIP.
Neither model alone gives the full answer.

## Cross-Image Validation

| Image | YOLO | CLIP L0 | Pipeline L2 | Insight |
|-------|------|---------|-------------|---------|
| capybara.jpg | "bear" | "capybara" (99.8%) | "capybara" | Pipeline corrects closed-vocab |
| cow.jpg | No detection | "cow" (97.9%) | "cow" (fallback) | Graceful degradation |
| gg_bridge.jpg | "boat" | "bridge" (98.4%) | "bridge" | Detection focus ≠ scene subject |
| trees_forest.jpg | n/a | "forest" (97.6%) | n/a | Clean baseline |

## Video Tracking Validation

| Video | Track Stability | Classification | Finding |
|-------|----------------|----------------|---------|
| Static capybara | 1 ID, perfect | "capybara" every frame | Baseline stability works |
| Capybara zoom | 1 ID, stable | "capybara" at all scales | Scale-invariant classification |
| Capybara slide | 5 IDs (fragmented) | "capybara" when detected | Intermittent detection → ID resets |

## Edge Case Summary

| Edge Case | Result |
|-----------|--------|
| Ambiguous labels | Confidence drops, distributed scores |
| Blank image | No crash, low-confidence output |
| Low resolution (32x32) | Still classifies, reduced accuracy |
| Multiple detections | Each gets independent classification |
| Uniform image (no objects) | Zero YOLO detections, no crash |

## Test Metrics

- **61 tests passing** (unit + integration + real-model + edge case + video)
- **8 real-model integration tests** validating actual CLIP + YOLO correctness
- **5 test videos** for tracking validation
- **43 fast deterministic tests** for CI/development
- All tests, hygiene check, and lints clean

## Technology Stack

| Component | Model | License | Cost |
|-----------|-------|---------|------|
| Detection | YOLOv8-nano (Ultralytics) | AGPL-3.0 | Free |
| Classification | CLIP (OpenAI) | MIT | Free |
| Segmentation | SAM2 interface (Meta) | Apache-2.0 | Free |
| Tracking | ByteTrack-inspired | MIT | Free |
| Core framework | CVPO | Apache-2.0 | Free |

## Design Principles

1. **Deterministic**: Same input → same output. No LLM in the core.
2. **Functional**: Pure connectors, frozen data types, explicit state.
3. **Educational**: Dual-register language, Socratic questions, honest assessments.
4. **Hardware-aware**: Auto-detection, pre-run validation, contextual guidance.
5. **Zero-cost**: All models free, runs on CPU, no API keys required.

## Iteration Process

- Started from blank repo (README + LICENSE)
- Planning phase: 6 design sessions with UX walkthroughs
- 10-day build plan executed with process logs at each phase
- Functional programming audit tightened immutability and purity
- Real model validation proved core value proposition
- 15+ process documents capturing every decision and finding
