# Pipeline Value Proposition Validation

## Finding

A multi-model pipeline produces semantically correct results that neither model
achieves alone. This was validated using a capybara image — a non-standard animal
not present in standard detection vocabularies.

## Test Case: capybara.jpg

### Single Model Results (Each Alone)

**YOLOv8-nano (detection only):**
- Output: "bear" (90% confidence)
- Why wrong: YOLOv8 is trained on COCO's 80 fixed categories. "Capybara" is not
  one of them. "Bear" is the closest large brown animal in COCO's vocabulary.
- What it got right: bounding box [0, 118, 678, 675] — correct spatial localization.

**CLIP (classification only, Level 0):**
- Output: "capybara" (99.8% confidence)
- Why correct: CLIP was trained on 400M image-text pairs from the internet, learning
  open-vocabulary visual concepts. It has seen enough "capybara" images to recognize one.
- What it cannot do alone: no spatial localization — it only says WHAT, not WHERE.

### Multi-Model Pipeline Result (CVPO Level 2)

**YOLOv8 -> Connector -> SAM2 -> Connector -> CLIP:**
- YOLOv8 detected an object at [0, 118, 678, 675] (labeled "bear")
- Connector cropped and reformatted the detection region
- CLIP classified the crop as "capybara" (correct)
- Final output: correctly localized AND correctly classified

## Why This Matters

| Capability | YOLO Alone | CLIP Alone | CVPO Pipeline |
|------------|-----------|------------|---------------|
| Spatial localization (WHERE) | Yes | No | Yes |
| Semantic correctness (WHAT) | Limited to 80 COCO classes | Open vocabulary | Open vocabulary |
| Capybara identification | No ("bear") | Yes | Yes |
| Capybara localization | Yes (box correct) | No | Yes |

The pipeline's increase in semantic correctness over individual models is the core
value proposition of CVPO. The connector pattern (crop detection region -> feed to
classifier) is the standard two-stage detect-then-classify approach from R-CNN
(Girshick et al. 2014), now accessible through a composable typed interface.

## Why Capybara Is a Good Test Case

Capybara is not in COCO's 80-class vocabulary, making it an ideal edge case:
- It exposes the closed-vocabulary limitation of standard detectors.
- It validates that open-vocabulary classification (CLIP/SigLIP) compensates.
- It is visually distinctive enough that CLIP identifies it with very high confidence.
- It demonstrates the practical scenario users will encounter: "my object isn't in
  the detector's built-in categories."

## Research Backing

- Closed-vocabulary detection: Lin et al. 2014 (COCO), Jocher et al. 2023 (YOLOv8)
- Open-vocabulary classification: Radford et al. 2021 (CLIP), Zhai et al. 2023 (SigLIP)
- Two-stage detect-then-classify pattern: Girshick et al. 2014 (R-CNN)
- Connector design: SAM2 box prompts (Ravi et al. 2024), ROI cropping (He et al. 2017)
