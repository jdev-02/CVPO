# Edge Case Testing Findings

## Test Matrix

| Edge Case | Behavior | Validated |
|-----------|----------|-----------|
| Ambiguous/overlapping labels | Confidence drops below 90% — scores distribute across similar options | Yes |
| Single candidate label | All probability mass on the one label (>99%) | Yes |
| Blank/black image | Pipeline completes without crash. CLIP returns scores (low confidence, distributed). | Yes |
| Low resolution (32x32) | CLIP still produces classification. Accuracy may decrease. | Yes |
| Multiple detections | Connectors correctly produce one classification per detection. 2 detections -> 2 crops -> 2 classifications. | Yes |
| Uniform color image (no objects) | YOLO returns zero detections. Pipeline does not crash. | Yes |
| YOLO misses object (cow close-up) | Pipeline falls back to whole-image classification via connector fallback. | Yes (from validation doc) |
| Non-COCO object (capybara) | YOLO labels incorrectly ("bear"), CLIP corrects via pipeline. | Yes (from validation doc) |

## Key Findings

### 1. Label Quality Directly Affects Confidence
When labels are specific and non-overlapping (e.g., "capybara, beaver, otter, bear, dog"),
CLIP produces high confidence (99.8%). When labels are vague and overlapping
(e.g., "large rodent, small bear, brown animal, mammal, pet"), confidence drops
significantly. This is not a bug — it's inherent to contrastive scoring and should be
explained to users as part of the honest assessment.

### 2. Pipeline Never Crashes on Degenerate Input
Blank images, uniform images, and low-resolution inputs all produce results without
exceptions. The pipeline degrades gracefully: lower confidence, less meaningful results,
but no crashes. This is critical for user trust.

### 3. Multiple Detection Handling Works Correctly
When YOLO finds multiple objects, each detection flows independently through the
connector chain and produces its own classification. This validates the per-detection
data flow architecture.

### 4. Resolution Affects Quality But Not Functionality
A 32x32 image can still be classified (CLIP handles arbitrary input sizes via internal
resizing). The model's effective accuracy drops at very low resolutions, but the system
remains functional. This should be surfaced in the honest assessment for users working
with low-quality inputs.

## Implications for CVPO UX

- The guided workflow should warn users about vague labels and suggest improvements.
- The honest assessment should mention input quality (resolution, lighting) as a factor.
- Fallback behavior (no detections -> classify full image) should be explained when it occurs.
