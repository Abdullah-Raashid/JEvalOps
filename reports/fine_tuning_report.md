# Fine-Tuning Report

The completed Apple Metal/MPS LoRA run uses `rinna/japanese-gpt2-small` with a PEFT adapter stored at `checkpoints/rinna_japanese_gpt2_small_lora_mps`.

See `reports/rinna_lora_mps_full/fine_tuning_report.md` for the full before/after evaluation.

Headline result on all 250 held-out test examples:

- Baseline quality: `0.3473`
- LoRA quality: `0.5428`
- Relative quality lift: `56.28%`
- P95 latency delta: `-0.0639s`
- Error count: `250 -> 123`
- Training seconds: `18.10`
- Peak accelerator memory: `515.08 MB`
- Accelerator: `apple_metal_mps`

The model is not promoted by the strict production gate because JSON schema compliance remains below the configured floor.

Apple Metal/MPS support is available for M-series Macs via `--device mps` and `--eval-device mps`. Rerun `make finetune-mps` to reproduce these metrics on a PyTorch build with MPS available.
