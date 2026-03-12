"""Hardware detection with privacy notice, manual override, and OS-specific guidance."""

from __future__ import annotations

import os
import platform
from typing import Any

PRIVACY_NOTICE = (
    "CVPO reads your CPU, RAM, and GPU specs to recommend appropriate model sizes. "
    "This data stays on your machine and is never transmitted or stored beyond this session."
)

HOW_TO_FIND_SPECS: dict[str, str] = {
    "Darwin": (
        "macOS: Open Apple menu > About This Mac, or run 'system_profiler SPHardwareDataType' "
        "in Terminal to see CPU, memory, and GPU details."
    ),
    "Linux": (
        "Linux: Run 'lscpu' for CPU info, 'free -h' for RAM, and 'nvidia-smi' for GPU. "
        "Or check /proc/cpuinfo and /proc/meminfo."
    ),
    "Windows": (
        "Windows: Open Settings > System > About for CPU/RAM, or run 'systeminfo' in Command Prompt. "
        "For GPU: open Device Manager > Display adapters, or run 'nvidia-smi' if available."
    ),
}


def detect_hardware(
    override_ram_gb: float | None = None,
    override_gpu_vram_gb: float | None = None,
    override_gpu_detected: bool | None = None,
) -> dict[str, Any]:
    """Return hardware profile with optional manual overrides.

    Overrides take precedence over auto-detection when provided.
    """
    cpu_name = platform.processor() or platform.machine() or "unknown"
    logical_cores = os.cpu_count() or 1
    system = platform.system()

    ram_gb: float | None = None
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        ram_gb = None

    if override_ram_gb is not None:
        ram_gb = override_ram_gb

    gpu_detected = False
    vram_gb: float | None = None
    if override_gpu_detected is not None:
        gpu_detected = override_gpu_detected
    if override_gpu_vram_gb is not None:
        vram_gb = override_gpu_vram_gb
        if vram_gb and vram_gb > 0:
            gpu_detected = True

    return {
        "platform": platform.platform(),
        "system": system,
        "cpu": cpu_name,
        "logical_cores": logical_cores,
        "ram_gb": ram_gb,
        "gpu_detected": gpu_detected,
        "vram_gb": vram_gb,
        "privacy_notice": PRIVACY_NOTICE,
        "how_to_find_specs": HOW_TO_FIND_SPECS.get(system, HOW_TO_FIND_SPECS["Linux"]),
        "overrides_applied": any(v is not None for v in [override_ram_gb, override_gpu_vram_gb, override_gpu_detected]),
    }
