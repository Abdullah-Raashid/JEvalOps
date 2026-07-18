from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class AcceleratorInfo:
    requested_device: str
    resolved_device: str
    accelerator: str
    cuda_available: bool
    mps_available: bool
    apple_metal_enabled: bool
    mps_fallback_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def configure_apple_metal_fallback(enabled: bool = True) -> None:
    if enabled:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def resolve_device(requested: str, torch, *, enable_mps_fallback: bool = True):
    configure_apple_metal_fallback(enable_mps_fallback)
    normalized = requested.lower()
    if normalized == "metal":
        normalized = "mps"
    if normalized == "auto":
        if torch.cuda.is_available():
            normalized = "cuda"
        elif _mps_available(torch):
            normalized = "mps"
        else:
            normalized = "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch does not see any CUDA device.")
    if normalized == "mps" and not _mps_available(torch):
        raise RuntimeError("Apple Metal/MPS was requested, but PyTorch does not see an available MPS device.")
    return torch.device(normalized)


def describe_accelerator(requested: str, resolved_device, torch) -> AcceleratorInfo:
    resolved = str(resolved_device)
    accelerator = "apple_metal_mps" if resolved_device.type == "mps" else resolved_device.type
    return AcceleratorInfo(
        requested_device=requested,
        resolved_device=resolved,
        accelerator=accelerator,
        cuda_available=torch.cuda.is_available(),
        mps_available=_mps_available(torch),
        apple_metal_enabled=resolved_device.type == "mps",
        mps_fallback_enabled=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
    )


def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def accelerator_memory_mb(device, torch) -> float | None:
    if device is None:
        return None
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    if device.type == "mps" and hasattr(torch, "mps"):
        if hasattr(torch.mps, "current_allocated_memory"):
            return torch.mps.current_allocated_memory() / (1024 * 1024)
    return None


def reset_peak_memory(device, torch) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _mps_available(torch) -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
