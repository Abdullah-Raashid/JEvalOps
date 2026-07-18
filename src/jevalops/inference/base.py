from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass
class GenerationResult:
    text: str
    latency_seconds: float
    time_to_first_token: float
    output_tokens: int
    tokens_per_second: float
    peak_memory_mb: float | None = None
    model_name: str = "unknown"
    backend_name: str = "unknown"
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelBackend(Protocol):
    model_name: str
    backend_name: str

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> GenerationResult:
        ...


def count_tokens_roughly(text: str) -> int:
    # Japanese tokenizers vary by backend; this deterministic approximation keeps reports comparable offline.
    return max(1, len(text.replace(" ", "")) // 2)
