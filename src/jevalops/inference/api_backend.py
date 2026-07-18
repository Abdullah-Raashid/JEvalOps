from __future__ import annotations

import json
import time
import urllib.request

from jevalops.inference.base import GenerationResult, count_tokens_roughly


class APIBackend:
    backend_name = "http_api"

    def __init__(self, model_name: str, endpoint: str, api_key: str | None = None) -> None:
        self.model_name = model_name
        self.endpoint = endpoint
        self.api_key = api_key

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> GenerationResult:
        body = json.dumps({"model": self.model_name, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}).encode()
        request = urllib.request.Request(self.endpoint, data=body, headers={"Content-Type": "application/json"})
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        latency = max(time.perf_counter() - start, 0.001)
        text = payload.get("text") or payload.get("choices", [{}])[0].get("text", "")
        output_tokens = count_tokens_roughly(text)
        return GenerationResult(
            text=text,
            latency_seconds=latency,
            time_to_first_token=payload.get("time_to_first_token", latency),
            output_tokens=payload.get("output_tokens", output_tokens),
            tokens_per_second=payload.get("tokens_per_second", output_tokens / latency),
            peak_memory_mb=payload.get("peak_memory_mb"),
            model_name=self.model_name,
            backend_name=self.backend_name,
            metadata={"endpoint": self.endpoint},
        )
