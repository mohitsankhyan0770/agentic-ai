.PHONY: install test lint format run api eval eval-ablate

install:
	pip install -e ".[dev,api]"

test:
	pytest --cov=agentic_ai --cov-report=term-missing

lint:
	ruff check . && ruff format --check .

format:
	ruff format . && ruff check . --fix

run:
	agentic-ai

api:
	uvicorn agentic_ai.api:app --reload

eval:
	python evals/run_evals.py --offline

eval-ablate:
	python evals/run_evals.py --offline --ablate
