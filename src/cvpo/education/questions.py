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
