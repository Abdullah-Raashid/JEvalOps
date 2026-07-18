.PHONY: install train-install test lint build-dataset baseline error-report report api demo regression finetune finetune-mps eval-lora eval-lora-mps

install:
	python3 -m pip install -e ".[dev]"

train-install:
	python3 -m pip install -e ".[train,dev]"

test:
	PYTHONPATH=src python3 -m pytest -q

lint:
	python3 -m ruff check src scripts tests

build-dataset:
	PYTHONPATH=src python3 scripts/build_dataset.py --train-size 300 --validation-size 100 --test-size 250

baseline:
	PYTHONPATH=src python3 scripts/run_baseline.py --dataset data/test.jsonl --output reports/baseline_results.json

error-report:
	PYTHONPATH=src python3 scripts/run_error_analysis.py --evaluation reports/baseline_results.json --output reports/error_analysis.json

report:
	PYTHONPATH=src python3 scripts/generate_report.py --evaluation reports/baseline_results.json --errors reports/error_analysis.json --output reports/baseline_report.md

regression:
	PYTHONPATH=src python3 scripts/run_regression.py --baseline reports/baseline_results.json --candidate reports/baseline_results.json

finetune:
	PYTHONPATH=src python3 scripts/run_finetuning_experiment.py --model-name rinna/japanese-gpt2-small --adapter-dir checkpoints/rinna_japanese_gpt2_small_lora --reports-dir reports/rinna_lora --eval-limit 50 --eval-max-new-tokens 48 --max-steps 200 --max-length 192 --learning-rate 0.001 --lora-r 16 --lora-alpha 32 --device cpu --target-modules c_attn c_proj c_fc --slow-tokenizer

finetune-mps:
	PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src python3 scripts/run_finetuning_experiment.py --model-name rinna/japanese-gpt2-small --adapter-dir checkpoints/rinna_japanese_gpt2_small_lora_mps --reports-dir reports/rinna_lora_mps --eval-limit 50 --eval-max-new-tokens 48 --max-steps 200 --max-length 192 --learning-rate 0.001 --lora-r 16 --lora-alpha 32 --device mps --eval-device mps --target-modules c_attn c_proj c_fc --slow-tokenizer

eval-lora:
	PYTHONPATH=src python3 scripts/evaluate_lora_adapter.py --model-name rinna/japanese-gpt2-small --adapter-dir checkpoints/rinna_japanese_gpt2_small_lora --slow-tokenizer

eval-lora-mps:
	PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src python3 scripts/evaluate_lora_adapter.py --model-name rinna/japanese-gpt2-small --adapter-dir checkpoints/rinna_japanese_gpt2_small_lora --eval-device mps --slow-tokenizer

api:
	PYTHONPATH=src uvicorn jevalops.api.main:app --reload --port 8000

demo:
	PYTHONPATH=src streamlit run demo/app.py
