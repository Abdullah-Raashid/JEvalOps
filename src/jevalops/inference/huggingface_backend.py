import resource
import time
from pathlib import Path

from jevalops.hardware.accelerator import accelerator_memory_mb, model_device, resolve_device
from jevalops.inference.base import GenerationResult, count_tokens_roughly


class HuggingFaceBackend:
    backend_name = "huggingface_transformers"

    def __init__(
        self,
        model_name: str,
        device_map: str | None = None,
        device: str = "auto",
        adapter_path: str | Path | None = None,
        default_max_new_tokens: int = 128,
        use_fast_tokenizer: bool = True,
        enable_mps_fallback: bool = True,
    ) -> None:
        self.model_name = model_name
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.default_max_new_tokens = default_max_new_tokens
        self.requested_device = device
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install the optional 'hf' dependencies to use HuggingFaceBackend.") from exc
        self._torch = torch
        self.device = resolve_device(device, torch, enable_mps_fallback=enable_mps_fallback)
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
        if not device_map:
            model.to(self.device)
        model.eval()
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> GenerationResult:
        start = time.perf_counter()
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_device = model_device(self.model) or self.device
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        generation_kwargs = {
            "max_new_tokens": min(max_tokens, self.default_max_new_tokens),
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        with self._torch.inference_mode():
            outputs = self.model.generate(**encoded, **generation_kwargs)
        prompt_tokens = encoded["input_ids"].shape[-1]
        text = self.tokenizer.decode(
            outputs[0][prompt_tokens:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        latency = max(time.perf_counter() - start, 0.001)
        text = text.strip()
        output_tokens = max(1, outputs.shape[-1] - prompt_tokens) if hasattr(outputs, "shape") else count_tokens_roughly(text)
        return GenerationResult(
            text=text,
            latency_seconds=latency,
            time_to_first_token=latency,
            output_tokens=output_tokens,
            tokens_per_second=output_tokens / latency,
            peak_memory_mb=_peak_memory_mb(self.model, self._torch),
            model_name=self.model_name,
            backend_name=self.backend_name,
            metadata={"device": str(input_device), "requested_device": self.requested_device},
        )


def _peak_memory_mb(model, torch) -> float:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = None
    accelerator_memory = accelerator_memory_mb(device, torch)
    if accelerator_memory is not None:
        return accelerator_memory
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024 / 1024 if usage > 10_000_000 else usage / 1024
