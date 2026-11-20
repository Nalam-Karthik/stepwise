.PHONY: install db test lint api worker demo

install:
	uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -e ".[dev]"

db:
	docker compose up -d --wait postgres

test: db
	.venv/bin/pytest -q --cov=stepwise --cov-report=term-missing

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

api:
	.venv/bin/uvicorn stepwise.api:app --port 8080

worker:
	.venv/bin/python -m stepwise.worker

demo: db
	.venv/bin/python scripts/demo_crash.py
