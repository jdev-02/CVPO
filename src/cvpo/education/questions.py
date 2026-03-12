"""Tiered Socratic question banks with dual-register answers and citation URLs."""

from __future__ import annotations

from typing import Any

from cvpo.education.citations import cite


QUESTION_BANK: dict[str, dict[str, dict[str, Any]]] = {
    "geese_tracking": {
        "beginner": {
            "question": (
                "Why might it be okay to skip some video frames when tracking "
                "slow-moving geese, and what could go wrong if you skip too many?"
            ),
            "answer": (
                "If objects move slowly, you can process every 2nd or 3rd frame "
                "instead of every single one — this is called frame subsampling. "
                "It saves processing time because the computer does less work. "
                "The risk is that if two birds cross paths quickly during a skipped "
                "moment, the system might mix up which bird is which — in tracking "
                "terminology this is called an ID switch."
            ),
            "citations": cite("sort", "bytetrack"),
        },
        "intermediate": {
            "question": (
                "Why does a tracker's ability to keep consistent identities depend "
                "on how much objects overlap between the frames it actually processes?"
            ),
            "answer": (
                "Tracking algorithms match detections across frames by checking how "
                "much bounding boxes overlap — measured by a metric called IoU "
                "(Intersection over Union). They also predict where an object should "
                "appear next using motion models (like a Kalman filter). When you "
                "skip frames, objects move farther between processed frames, overlap "
                "drops, and the matching becomes ambiguous — especially in crowded "
                "scenes where multiple objects are close together."
            ),
            "citations": cite("sort", "bytetrack"),
        },
        "advanced": {
            "question": (
                "How does ByteTrack's two-stage association of high- and "
                "low-confidence detections improve identity metrics (IDF1) "
                "while controlling false positives?"
            ),
            "answer": (
                "ByteTrack first matches high-confidence detections to existing "
                "tracks using IoU. Unmatched tracks are then associated with "
                "lower-confidence detections — boxes the detector is less sure "
                "about but that are often real objects seen through occlusion or "
                "blur. This second pass recovers trajectories that would otherwise "
                "be lost, improving IDF1 (a metric measuring identity preservation) "
                "without adding a heavy re-identification network. The low-confidence "
                "candidates are filtered by motion consistency, which limits false "
                "positive risk."
            ),
            "citations": cite("bytetrack"),
        },
    },
    "image_labeling": {
        "beginner": {
            "question": (
                "When the system gives a confidence score for each label, "
                "what does that number actually mean?"
            ),
            "answer": (
                "The confidence score shows how strongly the model prefers one "
                "label over the others you provided — think of it as a relative "
                "ranking, not an absolute guarantee. A score of 0.85 for 'goose' "
                "means the model thinks 'goose' fits better than the other options, "
                "but it does not mean there is an 85% chance it is correct in the "
                "real world. In technical terms, this is called a softmax "
                "probability distribution over the candidate label set."
            ),
            "citations": cite("clip", "siglip"),
        },
        "intermediate": {
            "question": (
                "Why does changing the list of candidate labels change the "
                "confidence scores, even for labels you did not add or remove?"
            ),
            "answer": (
                "The model scores each label relative to all other candidates — "
                "this is called comparative or contrastive scoring. Adding a new "
                "label changes the pool that scores are normalized against, which "
                "can shift every score up or down. For example, adding 'hawk' to "
                "['goose', 'duck'] might lower 'goose' even though the image has "
                "not changed. Technically, the normalization denominator changes, "
                "redistributing probability mass."
            ),
            "citations": cite("clip", "siglip"),
        },
        "advanced": {
            "question": (
                "How do the geometry of the embedding space and the exact wording "
                "of text prompts affect zero-shot classification calibration?"
            ),
            "answer": (
                "Vision-language models like CLIP and SigLIP project images and "
                "text into a shared embedding space. The classification score is "
                "essentially a distance (cosine similarity) between the image "
                "embedding and each text embedding. Slight changes in prompt "
                "wording — for example 'a photo of a goose' versus just 'goose' "
                "— shift the text vector direction, which can change which image "
                "regions are effectively matched. This is why prompt engineering "
                "matters for zero-shot models: the embedding alignment is sensitive "
                "to phrasing, and scores are not inherently calibrated to real-world "
                "probabilities."
            ),
            "citations": cite("clip", "siglip"),
        },
    },
    "object_finding": {
        "beginner": {
            "question": (
                "What is the difference between knowing WHAT is in an image and "
                "knowing WHERE it is?"
            ),
            "answer": (
                "Classification tells you what the image contains overall — for example "
                "'this is a photo of a dog.' Detection goes further: it tells you WHERE "
                "each object is by drawing a bounding box (rectangle) around it. "
                "Technically, detection outputs coordinates (x1, y1, x2, y2) for each "
                "object, plus a label and a confidence score."
            ),
            "citations": cite("yolov8", "coco"),
        },
        "intermediate": {
            "question": (
                "Why do detection models output confidence scores, and how does the "
                "confidence threshold affect what you see?"
            ),
            "answer": (
                "The model assigns a confidence score to each detection, representing "
                "how sure it is that an object exists at that location. A higher "
                "threshold (e.g., 0.7) shows fewer but more reliable detections. A "
                "lower threshold (e.g., 0.3) shows more detections but includes less "
                "certain ones — technically called trading precision for recall."
            ),
            "citations": cite("yolov8", "coco"),
        },
        "advanced": {
            "question": (
                "How does anchor-free detection in YOLOv8 differ from anchor-based "
                "approaches, and what are the practical implications?"
            ),
            "answer": (
                "Anchor-based detectors (like earlier YOLO versions) predefine a set "
                "of box templates at each grid cell and predict offsets from them. "
                "YOLOv8's anchor-free approach predicts object centers directly, "
                "eliminating the need to tune anchor sizes for each dataset. This "
                "simplifies training and generalizes better to unusual aspect ratios."
            ),
            "citations": cite("yolov8"),
        },
    },
    "object_boundaries": {
        "beginner": {
            "question": (
                "Why would you need precise outlines instead of just rectangles "
                "around objects?"
            ),
            "answer": (
                "Rectangles (bounding boxes) include background pixels around the "
                "object. Precise outlines — called segmentation masks — follow the "
                "exact shape of the object pixel by pixel. This matters when you need "
                "to measure an object's area, remove the background, or distinguish "
                "overlapping objects."
            ),
            "citations": cite("sam2"),
        },
        "intermediate": {
            "question": (
                "How does SAM2 use a detection bounding box as a 'prompt' to "
                "produce a segmentation mask?"
            ),
            "answer": (
                "SAM2 is a promptable segmentation model — you tell it roughly where "
                "to look by providing a bounding box (or a point click), and it "
                "produces a precise pixel-level mask within that region. The box acts "
                "as spatial guidance so the model knows which object to segment."
            ),
            "citations": cite("sam2"),
        },
        "advanced": {
            "question": (
                "What are the memory and compute tradeoffs of running SAM2 on every "
                "detection versus selective segmentation?"
            ),
            "answer": (
                "SAM2's ViT encoder is the expensive part — it processes the full "
                "image once. The lightweight mask decoder then runs per-prompt. For "
                "many detections, the encoder cost is amortized but decoder calls "
                "add up. Selective segmentation (only segmenting high-confidence or "
                "high-priority detections) reduces total compute."
            ),
            "citations": cite("sam2"),
        },
    },
    "video_tracking": {
        "beginner": {
            "question": (
                "Why can't you just detect objects on each frame separately — why "
                "do you need tracking?"
            ),
            "answer": (
                "Detection on each frame tells you 'there is a dog here' but doesn't "
                "tell you if it's the SAME dog as the previous frame. Tracking assigns "
                "a persistent ID to each object — technically called multi-object "
                "tracking (MOT) — so you can follow individual objects over time and "
                "avoid counting the same object twice."
            ),
            "citations": cite("sort", "bytetrack"),
        },
        "intermediate": {
            "question": (
                "How does a tracking-by-detection approach like ByteTrack maintain "
                "object identities across frames?"
            ),
            "answer": (
                "ByteTrack runs detection on each frame, then matches new detections "
                "to existing tracks using spatial overlap — measured by IoU "
                "(Intersection over Union) — and motion prediction from a Kalman "
                "filter. If a detection overlaps enough with a predicted track "
                "position, it's assigned the same ID."
            ),
            "citations": cite("sort", "bytetrack"),
        },
        "advanced": {
            "question": (
                "What causes track ID switches and fragmentation, and how does "
                "ByteTrack's two-pass association mitigate this?"
            ),
            "answer": (
                "ID switches occur when two objects cross paths and the IoU-based "
                "matcher swaps their assignments. Fragmentation happens when an "
                "object is temporarily undetected (occlusion, blur) and gets a new "
                "ID when it reappears. ByteTrack's two-pass approach — matching "
                "high-confidence detections first, then recovering low-confidence "
                "ones — reduces fragmentation by keeping tracks alive through noisy "
                "detection frames."
            ),
            "citations": cite("bytetrack"),
        },
    },
    "counting_over_time": {
        "beginner": {
            "question": (
                "If you want to count how many geese visit a park throughout the "
                "day, why isn't a simple frame-by-frame count enough?"
            ),
            "answer": (
                "A frame-by-frame count would count the same goose multiple times — "
                "once per frame it appears in. Tracking solves this by assigning each "
                "goose a unique ID. You count unique IDs instead of total detections, "
                "giving you an accurate count of individual geese rather than an "
                "inflated number."
            ),
            "citations": cite("sort", "bytetrack"),
        },
        "intermediate": {
            "question": (
                "What is the difference between 'total detections across frames' and "
                "'unique tracked objects,' and why does the distinction matter?"
            ),
            "answer": (
                "Total detections counts every bounding box on every frame — if a "
                "goose appears in 100 frames, it contributes 100 detections. Unique "
                "tracked objects counts distinct IDs — the same goose across 100 "
                "frames is 1 unique object. For population estimates, you need the "
                "latter. For activity metrics (like total bird-seconds of presence), "
                "you might want both."
            ),
            "citations": cite("bytetrack"),
        },
        "advanced": {
            "question": (
                "How would you estimate counting accuracy and what metrics quantify "
                "tracker-induced error?"
            ),
            "answer": (
                "MOTA (Multi-Object Tracking Accuracy) captures false positives, "
                "misses, and ID switches in a single metric. IDF1 measures identity "
                "preservation — how often a track correctly maintains one ID for one "
                "real object. HOTA balances detection accuracy and association "
                "accuracy. For counting specifically, the error is: unique_ids - "
                "true_count, driven primarily by fragmentation (overcount) and "
                "merged detections (undercount)."
            ),
            "citations": cite("bytetrack"),
        },
    },
    "species_identification": {
        "beginner": {
            "question": (
                "Why can't the object detector tell you the species directly — why "
                "do you need a second model?"
            ),
            "answer": (
                "Object detectors like YOLO are trained on a fixed list of categories "
                "(COCO has 80, like 'bird', 'car', 'dog'). They can't distinguish "
                "between species within a category — YOLO sees 'bird' but can't tell "
                "you if it's a goose, duck, or pigeon. A second model (CLIP) can "
                "classify with any labels you provide — this is called open-vocabulary "
                "or zero-shot classification."
            ),
            "citations": cite("yolov8", "clip", "coco"),
        },
        "intermediate": {
            "question": (
                "How does the detect-then-classify pipeline maintain spatial "
                "information while adding semantic specificity?"
            ),
            "answer": (
                "The detector provides spatial localization (bounding box). The "
                "connector crops the image region inside each box. The classifier "
                "scores each crop against your labels. The final output preserves "
                "both the spatial coordinates from detection and the specific label "
                "from classification — technically called ROI-level classification, "
                "originating from the R-CNN architecture."
            ),
            "citations": cite("clip", "yolov8"),
        },
        "advanced": {
            "question": (
                "What are the failure modes of detect-then-classify versus "
                "end-to-end fine-tuned detection?"
            ),
            "answer": (
                "Detect-then-classify fails when: (1) the detector misses the object "
                "entirely (recall failure), (2) the bounding box crop is too loose "
                "and includes confusing context, or (3) the classifier's open-vocab "
                "capability doesn't cover the visual distinction needed. End-to-end "
                "fine-tuned detection avoids the two-model overhead but requires "
                "labeled training data for each target class and loses the flexibility "
                "of open-vocabulary labeling."
            ),
            "citations": cite("clip", "yolov8", "coco"),
        },
    },
}


def get_socratic_block(goal: str, experience_level: str, skip_socratic: bool) -> dict[str, Any]:
    goal_bank = QUESTION_BANK.get(goal, QUESTION_BANK["image_labeling"])
    level_bank = goal_bank.get(experience_level, goal_bank["beginner"])
    return {
        "question": level_bank["question"],
        "skipped": skip_socratic,
        "answer": level_bank["answer"],
        "citations": level_bank["citations"],
    }
