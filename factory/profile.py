"""Hardware profiling and appliance-bandwidth adjustment.

Responsibilities:
    - Measure model size on disk and peak device memory.
    - Measure throughput on the dev machine, then scale to the target appliance
      profile: adjusted = measured * (target_bandwidth / dev_bandwidth).
      T4 (320 GB/s) -> appliance (273 GB/s) = 0.85.
    - Always emit both `measured_*` and `adjusted_*`, clearly labelled.
    - Assert the device actually in use. A silent CPU fallback invalidates every
      latency number, so it must be logged loudly.

Note: this module shadows the stdlib `profile` only inside the package
namespace; absolute imports elsewhere still resolve the stdlib module.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def bandwidth_ratio(config: Mapping[str, Any]) -> float:
    """target_bandwidth / dev_bandwidth, from `hardware` in the config."""
    hw = config["hardware"]
    profiles = hw["profiles"]
    return profiles[hw["target_profile"]]["bandwidth_gbs"] / profiles[hw["dev_profile"]]["bandwidth_gbs"]


def require_device(config: Mapping[str, Any]) -> str:
    """Return "cuda" or raise. Never silently fall back to CPU when CUDA is required."""
    import torch

    if torch.cuda.is_available():
        log.info("device: cuda (%s)", torch.cuda.get_device_name(0))
        return "cuda"
    if config.get("runtime", {}).get("require_cuda", True):
        raise RuntimeError(
            "CUDA is required (runtime.require_cuda: true) but unavailable. Refusing to fall "
            "back to CPU: every latency number would be invalid. Use a GPU runtime."
        )
    log.warning("!!! RUNNING ON CPU: latency and memory numbers are NOT comparable !!!")
    return "cpu"


def size_on_disk(path: str | Path) -> int:
    """Total bytes of all files under `path`."""
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def peak_memory_bytes(device: str) -> int:
    if device != "cuda":
        return 0
    import torch

    return int(torch.cuda.max_memory_allocated())


def adjust_latency(latency: Mapping[str, float], ratio: float, device: str) -> dict[str, Any]:
    """Label measured latencies and add bandwidth-adjusted estimates.

    Latency scales inversely with bandwidth, so adjusted = measured / ratio.
    """
    out: dict[str, Any] = {"measured_on": device, "bandwidth_ratio": ratio}
    for key in ("latency_p50_ms", "latency_p95_ms"):
        out[f"measured_{key}"] = latency[key]
        out[f"adjusted_{key}"] = latency[key] / ratio
    return out
