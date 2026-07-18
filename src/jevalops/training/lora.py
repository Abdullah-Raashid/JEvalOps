from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jevalops.hardware.accelerator import accelerator_memory_mb, describe_accelerator, reset_peak_memory, resolve_device


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] | None = None


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-4
    max_steps: int = 100
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    seed: int = 42
    device: str = "auto"
    mixed_precision: str = "no"
    enable_mps_fallback: bool = True
    logging_steps: int = 10
    use_fast_tokenizer: bool = True


def train_lora(
    model_name: str,
    train_jsonl: Path,
    output_dir: Path,
    *,
    validation_jsonl: Path | None = None,
    lora_config: LoRAConfig | None = None,
    training_config: TrainingConfig | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig as PeftLoraConfig
        from peft import TaskType, get_peft_model
        from torch.utils.data import DataLoader
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise ImportError("LoRA training requires `pip install -e '.[train]'`.") from exc

    lora_config = lora_config or LoRAConfig()
    training_config = training_config or TrainingConfig()
    random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(training_config.device, torch, enable_mps_fallback=training_config.enable_mps_fallback)
    model_kwargs: dict[str, Any] = {}
    if device.type == "cuda" and training_config.mixed_precision in {"fp16", "bf16"}:
        model_kwargs["torch_dtype"] = torch.float16 if training_config.mixed_precision == "fp16" else torch.bfloat16
    if device.type == "mps" and training_config.mixed_precision == "fp16":
        model_kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=training_config.use_fast_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False
    target_modules = lora_config.target_modules or infer_lora_target_modules(model_name)
    peft_config = PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.to(device)
    model.train()

    train_records = _load_sft_jsonl(train_jsonl)
    if not train_records:
        raise ValueError(f"No SFT records found at {train_jsonl}")
    train_records = _repeat_to_steps(train_records, training_config.max_steps * training_config.per_device_train_batch_size)
    dataloader = DataLoader(
        train_records,
        batch_size=training_config.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, tokenizer, training_config.max_length),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=training_config.max_steps)

    losses: list[float] = []
    reset_peak_memory(device, torch)

    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    while step < training_config.max_steps:
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / training_config.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % training_config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()) * training_config.gradient_accumulation_steps)
            step += 1
            if step >= training_config.max_steps:
                break

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics = {
        "base_model": model_name,
        "adapter_path": str(output_dir),
        "train_records": len(_load_sft_jsonl(train_jsonl)),
        "validation_records": len(_load_sft_jsonl(validation_jsonl)) if validation_jsonl else 0,
        "training_seconds": time.perf_counter() - start,
        "final_train_loss": losses[-1],
        "mean_train_loss": sum(losses) / len(losses),
        "peak_cuda_memory_mb": _peak_cuda_memory_mb(device, torch),
        "peak_accelerator_memory_mb": accelerator_memory_mb(device, torch),
        "accelerator": describe_accelerator(training_config.device, device, torch).to_dict(),
        "lora": asdict(lora_config) | {"resolved_target_modules": target_modules},
        "training": asdict(training_config) | {"resolved_device": str(device)},
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def _load_sft_jsonl(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _repeat_to_steps(records: list[dict[str, str]], target_size: int) -> list[dict[str, str]]:
    repeated: list[dict[str, str]] = []
    while len(repeated) < target_size:
        repeated.extend(records)
    return repeated[:target_size]


def _collate(batch: list[dict[str, str]], tokenizer, max_length: int) -> dict[str, Any]:
    prompts = [item["prompt"] for item in batch]
    texts = [f"{item['prompt']} {item['response']}{tokenizer.eos_token}" for item in batch]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    encoded["labels"] = encoded["input_ids"].clone()
    encoded["labels"][encoded["attention_mask"] == 0] = -100
    prompt_lengths = [
        len(tokenizer(prompt, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]) for prompt in prompts
    ]
    for row, prompt_length in enumerate(prompt_lengths):
        encoded["labels"][row, : min(prompt_length, encoded["labels"].shape[1])] = -100
    return encoded


def _peak_cuda_memory_mb(device, torch) -> float | None:
    if getattr(device, "type", None) != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def infer_lora_target_modules(model_name: str) -> list[str]:
    lowered = model_name.lower()
    if "qwen" in lowered or "llama" in lowered or "mistral" in lowered:
        return ["q_proj", "v_proj"]
    if "gpt2" in lowered:
        return ["c_attn"]
    return ["q_proj", "v_proj"]
