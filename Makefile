.PHONY: install test lint build-dataset baseline error-report report api demo regression

install:
	python3 -m pip install -e ".[dev]"

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

api:
	PYTHONPATH=src uvicorn jevalops.api.main:app --reload --port 8000

demo:
	PYTHONPATH=src streamlit run demo/app.py
