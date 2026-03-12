"""Deterministic tradeoff cards by goal and model stage with dual-register language."""

from __future__ import annotations

from typing import Any

from cvpo.education.citations import cite


TRADEOFF_CARDS: dict[str, list[dict[str, Any]]] = {
    "geese_tracking": [
        {
            "stage": "detection",
            "model": "yolov8n",
            "tradeoff": (
                "Faster processing but may miss small or partially hidden objects. "
                "In technical terms: higher throughput, lower recall on small targets."
            ),
            "guidance": (
                "Use the nano (smallest) model on machines without a GPU. "
                "If you have a GPU and need more accuracy, scale up to a larger model size."
            ),
            "citations": cite("yolov8"),
        },
        {
            "stage": "segmentation",
            "model": "sam2_tiny",
            "tradeoff": (
                "Draws precise outlines around objects instead of just rectangles, "
                "but takes more processing power. Technically: pixel-level masks at "
                "higher compute cost."
            ),
            "guidance": (
                "Turn on when exact object boundaries matter (for example, measuring "
                "area coverage). Turn off for lightweight counting where rectangles are enough."
            ),
            "citations": cite("sam2"),
        },
        {
            "stage": "tracking",
            "model": "bytetrack",
            "tradeoff": (
                "Keeps consistent labels on each object across frames when motion is "
                "smooth, but may mix up identities when objects cross paths closely. "
                "Technically: stable ID assignment via IoU matching with motion priors."
            ),
            "guidance": (
                "Works well for slow-moving subjects like geese. Only skip frames when "
                "objects move slowly relative to the camera's field of view."
            ),
            "citations": cite("bytetrack", "sort"),
        },
    ],
    "image_labeling": [
        {
            "stage": "classification",
            "model": "siglip_base",
            "tradeoff": (
                "Can classify images using any labels you provide — no retraining needed. "
                "This is called zero-shot classification. Accuracy depends on how clearly "
                "your labels describe distinct categories."
            ),
            "guidance": (
                "Use specific, non-overlapping labels for clearest results. For example, "
                "'golden retriever, poodle, beagle' works better than 'dog, pet, animal'."
            ),
            "citations": cite("siglip", "clip"),
        },
    ],
    "object_finding": [
        {
            "stage": "detection",
            "model": "yolov8n",
            "tradeoff": (
                "Faster processing but may miss small or partially hidden objects. "
                "In technical terms: higher throughput, lower recall on small targets."
            ),
            "guidance": (
                "Use nano for quick exploration. Scale to small/medium/large model "
                "sizes for better accuracy when you have more compute available."
            ),
            "citations": cite("yolov8"),
        },
    ],
    "object_boundaries": [
        {
            "stage": "detection",
            "model": "yolov8n",
            "tradeoff": (
                "Provides the initial location boxes that guide segmentation. "
                "Missed detections mean missed segmentations."
            ),
            "guidance": "Use a higher-recall detector setting when boundary completeness matters.",
            "citations": cite("yolov8"),
        },
        {
            "stage": "segmentation",
            "model": "sam2_tiny",
            "tradeoff": (
                "Produces pixel-perfect outlines but at higher compute cost than "
                "rectangles. Technically: instance-level masks vs bounding boxes."
            ),
            "guidance": (
                "Enable when exact shape matters (area measurement, background removal). "
                "Skip when rectangles are sufficient."
            ),
            "citations": cite("sam2"),
        },
    ],
    "video_tracking": [
        {
            "stage": "detection",
            "model": "yolov8n",
            "tradeoff": "Detection consistency directly affects tracking quality.",
            "guidance": "Prioritize consistent detection across frames over single-frame accuracy.",
            "citations": cite("yolov8"),
        },
        {
            "stage": "tracking",
            "model": "bytetrack",
            "tradeoff": (
                "Maintains object identities through smooth motion but may lose track "
                "during rapid direction changes or heavy occlusion — called ID switches."
            ),
            "guidance": "Works best for steady motion. Increase max_distance for fast-moving objects.",
            "citations": cite("bytetrack", "sort"),
        },
    ],
    "counting_over_time": [
        {
            "stage": "detection",
            "model": "yolov8n",
            "tradeoff": "Missed detections lead to undercounting. False positives lead to overcounting.",
            "guidance": "Tune confidence threshold based on whether undercounting or overcounting matters more.",
            "citations": cite("yolov8"),
        },
        {
            "stage": "tracking",
            "model": "bytetrack",
            "tradeoff": (
                "Track fragmentation (one object getting multiple IDs) inflates counts. "
                "Merged tracks (two objects sharing one ID) reduce counts."
            ),
            "guidance": "Use lower max_distance for crowded scenes to reduce ID sharing.",
            "citations": cite("bytetrack", "sort"),
        },
        {
            "stage": "classification",
            "model": "siglip_base",
            "tradeoff": "Adds species-level precision but increases per-frame processing time.",
            "guidance": "Only add classification if you need to count specific types, not just total objects.",
            "citations": cite("siglip", "clip"),
        },
    ],
    "species_identification": [
        {
            "stage": "detection",
            "model": "yolov8n",
            "tradeoff": (
                "Finds where animals are, but labels them with generic COCO categories "
                "(e.g., 'bird' instead of 'goose'). The classifier corrects this."
            ),
            "guidance": "The detector only needs to find objects — species accuracy comes from the classifier.",
            "citations": cite("yolov8", "coco"),
        },
        {
            "stage": "classification",
            "model": "siglip_base",
            "tradeoff": (
                "Open-vocabulary species classification without retraining. Accuracy "
                "depends on how visually distinct the species are and how specific "
                "your labels are."
            ),
            "guidance": (
                "Provide specific species names as labels. If the model confuses two "
                "species, try adding distinguishing context like 'Canada goose' instead "
                "of just 'goose'."
            ),
            "citations": cite("clip", "siglip"),
        },
    ],
}


def get_tradeoff_cards(goal: str) -> list[dict[str, Any]]:
    return TRADEOFF_CARDS.get(goal, TRADEOFF_CARDS["image_labeling"])
