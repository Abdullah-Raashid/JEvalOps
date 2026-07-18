from __future__ import annotations

import resource
import time
from pathlib import Path

from jevalops.inference.base import GenerationResult, count_tokens_roughly


class HuggingFaceBackend:
    backend_name = "huggingface_transformers"

    def __init__(
        self,
        model_name: str,
        device_map: str | None = "auto",
        adapter_path: str | Path | None = None,
        default_max_new_tokens: int = 128,
        use_fast_tokenizer: bool = True,
    ) -> None:
        self.model_name = model_name
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.default_max_new_tokens = default_max_new_tokens
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise ImportError("Install the optional 'hf' dependencies to use HuggingFaceBackend.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast_tokenizer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {"device_map": device_map} if device_map else {}
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if self.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError("Install the optional 'train' dependencies to load LoRA adapters.") from exc
            model = PeftModel.from_pretrained(model, self.adapter_path)
            self.model_name = f"{model_name}+lora:{self.adapter_path.name}"
        self.pipeline = pipeline("text-generation", model=model, tokenizer=self.tokenizer)

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> GenerationResult:
        start = time.perf_counter()
        outputs = self.pipeline(
            prompt,
            max_new_tokens=min(max_tokens, self.default_max_new_tokens),
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=self.tokenizer.pad_token_id,
            return_full_text=False,
        )
        latency = max(time.perf_counter() - start, 0.001)
        text = outputs[0]["generated_text"].strip()
        output_tokens = count_tokens_roughly(text)
        return GenerationResult(
            text=text,
            latency_seconds=latency,
            time_to_first_token=latency,
            output_tokens=output_tokens,
            tokens_per_second=output_tokens / latency,
            peak_memory_mb=_peak_memory_mb(),
            model_name=self.model_name,
            backend_name=self.backend_name,
        )


def _peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024 / 1024 if usage > 10_000_000 else usage / 1024
