from __future__ import annotations

import types

import pytest

from jevalops.hardware.accelerator import describe_accelerator, resolve_device


class FakeDevice:
    def __init__(self, name: str) -> None:
        self.type = name
        self.name = name

    def __str__(self) -> str:
        return self.name


class FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


def fake_torch(*, cuda: bool = False, mps: bool = False):
    return types.SimpleNamespace(
        cuda=FakeCuda(cuda),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
        device=lambda name: FakeDevice(name),
    )


def test_resolve_metal_alias_to_mps() -> None:
    device = resolve_device("metal", fake_torch(mps=True))
    assert str(device) == "mps"


def test_auto_prefers_cuda_then_mps_then_cpu() -> None:
    assert str(resolve_device("auto", fake_torch(cuda=True, mps=True))) == "cuda"
    assert str(resolve_device("auto", fake_torch(cuda=False, mps=True))) == "mps"
    assert str(resolve_device("auto", fake_torch(cuda=False, mps=False))) == "cpu"


def test_requesting_unavailable_mps_errors() -> None:
    with pytest.raises(RuntimeError, match="Apple Metal/MPS"):
        resolve_device("mps", fake_torch(mps=False))


def test_describe_apple_metal_accelerator() -> None:
    torch = fake_torch(mps=True)
    info = describe_accelerator("mps", FakeDevice("mps"), torch)
    assert info.accelerator == "apple_metal_mps"
    assert info.apple_metal_enabled
