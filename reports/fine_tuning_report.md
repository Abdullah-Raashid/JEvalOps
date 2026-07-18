# Fine-Tuning Report

The completed CPU LoRA run uses `rinna/japanese-gpt2-small` with a PEFT adapter stored at `checkpoints/rinna_japanese_gpt2_small_lora`.

See `reports/rinna_lora/fine_tuning_report.md` for the full before/after evaluation.

Headline result on 50 held-out test examples:

- Baseline quality: `0.3533`
- LoRA quality: `0.4873`
- Relative quality lift: `37.92%`
- P95 latency delta: `-0.0097s`
- Error count: `50 -> 30`

The model is not promoted by the strict production gate because JSON schema compliance remains below the configured floor.
