from __future__ import annotations

import time

from jevalops.inference.base import GenerationResult, count_tokens_roughly


class HuggingFaceBackend:
    backend_name = "huggingface_transformers"

    def __init__(self, model_name: str, device_map: str = "auto") -> None:
        self.model_name = model_name
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise ImportError("Install the optional 'hf' dependencies to use HuggingFaceBackend.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
        self.pipeline = pipeline("text-generation", model=model, tokenizer=self.tokenizer)

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> GenerationResult:
        start = time.perf_counter()
        outputs = self.pipeline(prompt, max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-5))
        latency = max(time.perf_counter() - start, 0.001)
        text = outputs[0]["generated_text"][len(prompt) :].strip()
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
