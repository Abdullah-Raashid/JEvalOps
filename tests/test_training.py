from __future__ import annotations

from jevalops.training.lora import infer_lora_target_modules


def test_infer_lora_targets_for_qwen() -> None:
    assert infer_lora_target_modules("Qwen/Qwen2.5-0.5B-Instruct") == ["q_proj", "v_proj"]


def test_infer_lora_targets_for_gpt2() -> None:
    assert infer_lora_target_modules("sshleifer/tiny-gpt2") == ["c_attn"]
