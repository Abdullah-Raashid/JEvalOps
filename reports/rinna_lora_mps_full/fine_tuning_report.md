# Fine-Tuning Evaluation Report

Base model: `rinna/japanese-gpt2-small`
Adapter: `checkpoints/rinna_japanese_gpt2_small_lora_mps`
Evaluation examples: 250

## Before / After

- Baseline quality: 0.3473
- LoRA quality: 0.5428
- Absolute quality delta: 0.1955
- Relative quality delta: 56.28%
- Baseline P95 latency: 0.6771s
- LoRA P95 latency: 0.6132s
- P95 latency delta: -0.0639s
- Baseline error count: 250
- LoRA error count: 123

## Task Scores

| Task | Baseline | LoRA | Delta |
| --- | ---: | ---: | ---: |
| `business_rewriting` | 0.6160 | 0.9680 | 0.3520 |
| `grounded_qa` | 0.6007 | 0.8640 | 0.2633 |
| `information_extraction` | 0.0000 | 0.0000 | 0.0000 |
| `robustness` | 0.5200 | 0.8820 | 0.3620 |
| `summarization` | 0.0000 | 0.0000 | 0.0000 |

## Promotion Gates

Promoted: **False**
- `quality_improves`: True
- `no_critical_task_drop`: True
- `p95_latency_within_limit`: True
- `schema_compliance_above_floor`: False

## Training

- Training examples: 300
- Validation examples: 100
- Training seconds: 18.10
- Final train loss: 0.0211
- Mean train loss: 0.7619
- Resolved device: `mps`
- Accelerator: `apple_metal_mps`
- Peak accelerator memory: 515.08 MB
- MPS fallback enabled: True
