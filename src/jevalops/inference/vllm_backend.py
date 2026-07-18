from __future__ import annotations

import time

from jevalops.inference.base import GenerationResult, count_tokens_roughly


class VLLMBackend:
    backend_name = "vllm"

    def __init__(self, model_name: str, tensor_parallel_size: int = 1) -> None:
        self.model_name = model_name
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError("Install the optional 'vllm' dependencies to use VLLMBackend.") from exc
        self._sampling_params_cls = SamplingParams
        self.llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> GenerationResult:
        params = self._sampling_params_cls(max_tokens=max_tokens, temperature=temperature)
        start = time.perf_counter()
        outputs = self.llm.generate([prompt], params)
        latency = max(time.perf_counter() - start, 0.001)
        text = outputs[0].outputs[0].text.strip()
        output_tokens = count_tokens_roughly(text)
        return GenerationResult(
            text=text,
            latency_seconds=latency,
            time_to_first_token=latency,
            output_tokens=output_tokens,
            tokens_per_second=output_tokens / latency,
            peak_memory_mb=None,
            model_name=self.model_name,
            backend_name=self.backend_name,
        )
