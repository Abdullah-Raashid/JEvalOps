# Fine-Tuning Report

Base model: `rinna/japanese-gpt2-small`
Adapter: `checkpoints/rinna_japanese_gpt2_small_lora_mps`
Evaluation examples: 50

## Before / After

- Baseline quality: 0.3533
- LoRA quality: 0.5507
- Absolute quality delta: 0.1973
- Relative quality delta: 55.85%
- Baseline P95 latency: 0.5894s
- LoRA P95 latency: 0.6038s
- P95 latency delta: 0.0145s

## Task Scores

| Task | Baseline | LoRA | Delta |
| --- | ---: | ---: | ---: |
| `business_rewriting` | 0.6133 | 0.9733 | 0.3600 |
| `grounded_qa` | 0.6033 | 0.8533 | 0.2500 |
| `information_extraction` | 0.0000 | 0.0000 | 0.0000 |
| `robustness` | 0.5500 | 0.9267 | 0.3767 |
| `summarization` | 0.0000 | 0.0000 | 0.0000 |

## Promotion Gates

Promoted: **False**
- `quality_improves`: True
- `no_critical_task_drop`: True
- `p95_latency_within_limit`: True
- `schema_compliance_above_floor`: False

## Training

- Final train loss: 0.0211
- Mean train loss: 0.7619
- Training seconds: 18.10
- Resolved device: `mps`
