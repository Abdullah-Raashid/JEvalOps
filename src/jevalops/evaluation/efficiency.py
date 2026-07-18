from __future__ import annotations

import math


def efficiency_adjusted_score(quality_score: float, latency_seconds: float) -> float:
    latency_ms = max(latency_seconds * 1000, 1.0)
    return quality_score / math.log1p(latency_ms)


def aggregate_efficiency(results: list[dict]) -> dict[str, float]:
    if not results:
        return {"mean_latency_seconds": 0.0, "p95_latency_seconds": 0.0, "mean_tokens_per_second": 0.0}
    latencies = sorted(item["latency_seconds"] for item in results)
    p95_index = min(len(latencies) - 1, int(len(latencies) * 0.95))
    return {
        "mean_latency_seconds": sum(latencies) / len(latencies),
        "p95_latency_seconds": latencies[p95_index],
        "mean_tokens_per_second": sum(item["tokens_per_second"] for item in results) / len(results),
    }
