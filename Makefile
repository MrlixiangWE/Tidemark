.PHONY: install dev test lint replay trace clean

PY ?= python

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	ruff check tidemark tests scripts

trace:
	$(PY) scripts/make_demo_trace.py --sessions 24 --seed 3 > examples/traces/demo.jsonl

replay:
	tidemark replay --trace examples/traces/demo.jsonl --load 0.6
	tidemark replay --trace examples/traces/demo.jsonl --load 0.9

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache telemetry
	find . -name __pycache__ -type d -exec rm -rf {} +
