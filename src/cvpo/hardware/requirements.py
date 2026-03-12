"""Model requirement tables, capability assessment, and pre-run validation."""

from __future__ import annotations

from typing import Any


MODEL_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "yolov8n": {
        "stage": "detection",
        "min_ram_gb": 4.0,
        "recommended_gpu": False,
        "recommended_vram_gb": None,
        "notes": "CPU-friendly baseline detector.",
    },
    "sam2_tiny": {
        "stage": "segmentation",
        "min_ram_gb": 8.0,
        "recommended_gpu": True,
        "recommended_vram_gb": 6.0,
        "notes": "Segmentation benefits strongly from GPU acceleration.",
    },
    "siglip_base": {
        "stage": "classification",
        "min_ram_gb": 6.0,
        "recommended_gpu": False,
        "recommended_vram_gb": None,
        "notes": "Runs on CPU; GPU improves throughput.",
    },
    "bytetrack": {
        "stage": "tracking",
        "min_ram_gb": 4.0,
        "recommended_gpu": False,
        "recommended_vram_gb": None,
        "notes": "Tracking-by-detection, typically CPU-manageable.",
    },
}


def capability_assessment(hardware: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate hardware against model requirements."""
    ram_gb = hardware.get("ram_gb")
    gpu = bool(hardware.get("gpu_detected"))
    vram_gb = hardware.get("vram_gb")

    rows: list[dict[str, Any]] = []
    for model_name, req in MODEL_REQUIREMENTS.items():
        ram_ok = (ram_gb is None) or (ram_gb >= req["min_ram_gb"])
        gpu_note = "recommended" if req["recommended_gpu"] else "optional"
        gpu_ok = True
        if req["recommended_gpu"] and not gpu:
            gpu_ok = False

        vram_ok = True
        rec_vram = req["recommended_vram_gb"]
        if rec_vram is not None and vram_gb is not None:
            vram_ok = vram_gb >= rec_vram

        status = "good"
        if not ram_ok:
            status = "not_recommended"
        elif not gpu_ok:
            status = "degraded"
        elif not vram_ok:
            status = "degraded"

        suggestion = None
        if status == "not_recommended":
            suggestion = (
                f"Your RAM ({ram_gb} GB) is below the minimum ({req['min_ram_gb']} GB) "
                f"for {model_name}. Consider using a smaller model variant or a machine "
                f"with more memory."
            )
        elif status == "degraded" and not gpu_ok:
            suggestion = (
                f"{model_name} runs best with a GPU but none was detected. It will still "
                f"work on CPU but expect slower performance. Consider a machine with a "
                f"dedicated GPU for production use."
            )
        elif status == "degraded" and not vram_ok:
            suggestion = (
                f"{model_name} recommends {rec_vram} GB VRAM but you have {vram_gb} GB. "
                f"Consider a smaller model variant or expect potential out-of-memory errors."
            )

        rows.append({
            "model": model_name,
            "stage": req["stage"],
            "status": status,
            "ram_ok": ram_ok,
            "gpu_note": gpu_note,
            "gpu_present": gpu,
            "vram_ok": vram_ok,
            "notes": req["notes"],
            "suggestion": suggestion,
        })
    return rows


def pre_run_validation(
    capability: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check capability assessment for blockers before pipeline execution.

    Returns a validation result with status, warnings, and blockers.
    Never silently crashes — always gives the user actionable information.
    """
    blockers: list[str] = []
    warnings: list[str] = []

    for row in capability:
        if row["status"] == "not_recommended":
            blockers.append(
                f"[{row['model']}] {row.get('suggestion', 'Below minimum requirements.')}"
            )
        elif row["status"] == "degraded":
            warnings.append(
                f"[{row['model']}] {row.get('suggestion', 'Performance may be reduced.')}"
            )

    if blockers:
        status = "blocked"
    elif warnings:
        status = "warnings"
    else:
        status = "clear"

    return {
        "status": status,
        "blockers": blockers,
        "warnings": warnings,
        "proceed_message": (
            "All models are compatible with your hardware."
            if status == "clear"
            else (
                "Some models have reduced performance on your hardware. "
                "You can proceed, but results may be slower than expected."
                if status == "warnings"
                else (
                    "One or more models may not run on your hardware. "
                    "You can still proceed (at your own risk) or choose "
                    "alternative model sizes."
                )
            )
        ),
    }
