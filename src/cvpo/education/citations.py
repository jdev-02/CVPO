"""Centralized citation registry with URLs."""

from __future__ import annotations


CITATIONS: dict[str, dict[str, str]] = {
    "coco": {
        "short": "Lin et al. 2014 (COCO dataset)",
        "url": "https://arxiv.org/abs/1405.0312",
    },
    "bytetrack": {
        "short": "Zhang et al. 2022 (ByteTrack)",
        "url": "https://github.com/ifzhang/ByteTrack",
    },
    "sort": {
        "short": "Bewley et al. 2016 (SORT)",
        "url": "https://arxiv.org/abs/1602.00763",
    },
    "clip": {
        "short": "Radford et al. 2021 (CLIP)",
        "url": "https://openai.com/research/clip",
    },
    "siglip": {
        "short": "Zhai et al. 2023 (SigLIP)",
        "url": "https://arxiv.org/abs/2303.15343",
    },
    "sam2": {
        "short": "Ravi et al. 2024 (SAM 2)",
        "url": "https://arxiv.org/abs/2408.00714",
    },
    "yolov8": {
        "short": "Jocher et al. 2023 (YOLOv8)",
        "url": "https://docs.ultralytics.com/models/yolov8/",
    },
}


def cite(*keys: str) -> list[dict[str, str]]:
    return [CITATIONS[k] for k in keys if k in CITATIONS]


def format_citations(entries: list[dict[str, str]]) -> str:
    parts = [f"{e['short']} — {e['url']}" for e in entries]
    return "\n".join(parts)
