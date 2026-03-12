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
}


def get_tradeoff_cards(goal: str) -> list[dict[str, Any]]:
    return TRADEOFF_CARDS.get(goal, TRADEOFF_CARDS["image_labeling"])
