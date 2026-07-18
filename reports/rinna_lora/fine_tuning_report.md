# Fine-Tuning Evaluation Report

Base model: `rinna/japanese-gpt2-small`
Adapter: `checkpoints/rinna_japanese_gpt2_small_lora`
Evaluation examples: 50

## Before / After

- Baseline quality: 0.3533
- LoRA quality: 0.4873
- Absolute quality delta: 0.1340
- Relative quality delta: 37.92%
- Baseline P95 latency: 0.6362s
- LoRA P95 latency: 0.6265s
- P95 latency delta: -0.0097s
- Baseline error count: 50
- LoRA error count: 30

## Task Scores

| Task | Baseline | LoRA | Delta |
| --- | ---: | ---: | ---: |
| `business_rewriting` | 0.6133 | 0.9867 | 0.3733 |
| `grounded_qa` | 0.6033 | 0.7233 | 0.1200 |
| `information_extraction` | 0.0000 | 0.0000 | 0.0000 |
| `robustness` | 0.5500 | 0.7267 | 0.1767 |
| `summarization` | 0.0000 | 0.0000 | 0.0000 |

## Promotion Gates

Promoted: **False**
- `quality_improves`: True
- `no_critical_task_drop`: True
- `p95_latency_within_limit`: True
- `schema_compliance_above_floor`: False
