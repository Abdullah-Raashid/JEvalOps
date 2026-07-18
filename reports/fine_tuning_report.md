# Fine-Tuning Report

The completed Apple Metal/MPS LoRA run uses `rinna/japanese-gpt2-small` with a PEFT adapter stored at `checkpoints/rinna_japanese_gpt2_small_lora_mps`.

See `reports/rinna_lora_mps/fine_tuning_report.md` for the full before/after evaluation.

Headline result on 50 held-out test examples:

- Baseline quality: `0.3533`
- LoRA quality: `0.5507`
- Relative quality lift: `55.85%`
- P95 latency delta: `+0.0145s`
- Error count: `50 -> 23`
- Training seconds: `18.10`
- Peak accelerator memory: `515.08 MB`
- Accelerator: `apple_metal_mps`

The model is not promoted by the strict production gate because JSON schema compliance remains below the configured floor.

Apple Metal/MPS support is available for M-series Macs via `--device mps` and `--eval-device mps`. Rerun `make finetune-mps` to reproduce these metrics on a PyTorch build with MPS available.
