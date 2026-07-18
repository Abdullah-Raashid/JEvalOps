# JEvalOps

Japanese Enterprise LLM Evaluation & Adaptation.

JEvalOps is a reproducible framework for curating governed Japanese enterprise datasets, benchmarking LLMs across quality and efficiency dimensions, performing Japanese-specific error analysis, and adapting compact models with targeted LoRA fine-tuning.

## Why This Exists

Japanese enterprise evaluation is hard because correct behavior depends on keigo, internal/external company perspective, omitted subjects, mixed scripts, date normalization, structured-output reliability, and grounded refusal behavior. A single aggregate score hides those failures, so JEvalOps reports task quality, latency, schema compliance, hallucination/failure-to-abstain behavior, and an explicit error taxonomy.

## Supported Tasks

- Business Japanese and keigo rewriting.
- Enterprise information extraction with JSON schema checks.
- Summarization with decisions and action items.
- Grounded question answering with abstention checks.
- Robustness for full-width/half-width text, Japanese eras, numerals, typos, and mixed terminology.

## Architecture

```mermaid
flowchart TD
    A[Raw Japanese Data] --> B[Validation: provenance, PII, Japanese ratio]
    B --> C[Normalization and Deduplication]
    C --> D[Versioned JSONL Dataset]
    D --> E[HF / vLLM / API / Local Backends]
    E --> F[Evaluation Engine]
    F --> G[Error Taxonomy and Reports]
    G --> H[Targeted SFT Dataset]
    H --> I[LoRA Fine-Tuning]
    I --> J[Regression Gates]
```

## Quickstart

```bash
python3 -m pip install -e ".[dev]"
make build-dataset
make baseline
make error-report
make report
```

The default backend is deterministic and local, so the full smoke path runs without downloading models. Use the Hugging Face or vLLM adapters once you select real model IDs and hardware.

## Fine-Tuning Result

The repository includes a CPU-run LoRA adapter for `rinna/japanese-gpt2-small`, trained on the generated SFT split and evaluated against the frozen test split.

- Evaluation slice: 50 held-out test examples.
- Baseline quality: `0.3533`.
- LoRA quality: `0.4873`.
- Relative quality lift: `+37.92%`.
- Baseline P95 latency: `0.6362s`.
- LoRA P95 latency: `0.6265s`.
- P95 latency delta: `-0.0097s`.
- Labeled error count: `50 -> 30`.

The adapter improves business rewriting, grounded QA, and robustness behavior, but it is not promoted by the production gate because JSON schema compliance for extraction/summarization remains below the configured floor. See `reports/rinna_lora/fine_tuning_report.md`.

## Apple Metal / MPS Acceleration

JEvalOps supports Apple Silicon acceleration through PyTorch MPS, Apple Metal's PyTorch backend. Device selection accepts `auto`, `cpu`, `cuda`, `mps`, and `metal`; `metal` is treated as an alias for `mps`.

Check local accelerator visibility:

```bash
python3 - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available())
print("MPS:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
PY
```

Run the same LoRA experiment on an M-series Mac GPU:

```bash
python3 -m pip install -e ".[train]"
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src python3 scripts/run_finetuning_experiment.py \
  --model-name rinna/japanese-gpt2-small \
  --adapter-dir checkpoints/rinna_japanese_gpt2_small_lora_mps \
  --reports-dir reports/rinna_lora_mps \
  --eval-limit 50 \
  --eval-max-new-tokens 48 \
  --max-steps 200 \
  --max-length 192 \
  --learning-rate 0.001 \
  --lora-r 16 \
  --lora-alpha 32 \
  --device mps \
  --eval-device mps \
  --target-modules c_attn c_proj c_fc \
  --slow-tokenizer
```

`PYTORCH_ENABLE_MPS_FALLBACK=1` lets PyTorch fall back to CPU for individual operations that are not implemented on MPS. Training metrics record the requested device, resolved device, accelerator name, MPS availability, and accelerator memory when PyTorch exposes it.

## Repository Map

- `src/jevalops/data`: Pydantic schema, JSONL validation, PII checks, normalization, deduplication.
- `src/jevalops/generation`: controlled template generation and synthetic filters.
- `src/jevalops/inference`: shared backend contract plus local, Hugging Face, vLLM, and HTTP API adapters.
- `src/jevalops/evaluation`: exact metrics, JSON/schema checks, rubric scoring, efficiency metrics.
- `src/jevalops/analysis`: Japanese-specific error taxonomy and structured failure analysis.
- `src/jevalops/training`: SFT dataset export, LoRA entrypoint, model promotion gates.
- `scripts`: reproducible CLI stages.
- `demo`: Streamlit interface.

## Reproduction

```bash
PYTHONPATH=src python3 scripts/build_dataset.py --train-size 300 --validation-size 100 --test-size 250
PYTHONPATH=src python3 scripts/run_baseline.py --dataset data/test.jsonl --output reports/baseline_results.json
PYTHONPATH=src python3 scripts/run_error_analysis.py --evaluation reports/baseline_results.json --output reports/error_analysis.json
PYTHONPATH=src python3 scripts/generate_report.py --evaluation reports/baseline_results.json --errors reports/error_analysis.json --output reports/baseline_report.md
```

LoRA reproduction:

```bash
python3 -m pip install -e ".[train]"
PYTHONPATH=src python3 scripts/run_finetuning_experiment.py \
  --model-name rinna/japanese-gpt2-small \
  --adapter-dir checkpoints/rinna_japanese_gpt2_small_lora \
  --reports-dir reports/rinna_lora \
  --eval-limit 50 \
  --eval-max-new-tokens 48 \
  --max-steps 200 \
  --max-length 192 \
  --learning-rate 0.001 \
  --lora-r 16 \
  --lora-alpha 32 \
  --device cpu \
  --target-modules c_attn c_proj c_fc \
  --slow-tokenizer
```

To log the experiment to MLflow, install `.[tracking]` and add `--mlflow-experiment jevalops-lora`.

## API

```bash
python3 -m pip install -e ".[api]"
make api
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"指示:\n取引先への丁寧な依頼文に書き換えてください。\n\n入力:\n資料見たら返事ください\n\n回答:"}'
```

## Model Backends

The `ModelBackend` contract returns text, latency, approximate time-to-first-token, output tokens, throughput, memory, model name, and backend name. Real model adapters are available for Hugging Face Transformers and vLLM; the local backend exists for CI and sanity checks.

## Promotion Rule

A candidate model is promoted only when composite quality improves by at least 5%, no task drops by more than 2%, P95 latency remains within the configured limit, and schema compliance stays above 95%.

## Limitations

- The checked-in generator is a reproducible MVP, not a substitute for a frozen native-speaker benchmark.
- Synthetic data can reflect the generator or template author’s preferences.
- Japanese politeness often has multiple valid formulations.
- Automated scores do not fully capture naturalness or cultural appropriateness.
- Enterprise workflows require domain-expert review before deployment.
