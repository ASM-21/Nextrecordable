.PHONY: help dev-install test lint fix run-all download load build sanity train serve explore clean

CLI := python -m nextrecordable.cli

help:
	@echo "make dev-install   install with dev tools"
	@echo "make test          run the test suite (synthetic data, no network)"
	@echo "make lint          ruff"
	@echo "make run-all       download -> load -> build -> sanity -> train (real OSHA data)"
	@echo "make download | load | build | sanity | train   one step at a time"
	@echo "make serve         run the API and lookup page on http://127.0.0.1:8000"
	@echo "make clean         remove caches"

dev-install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

fix:
	ruff check . --fix

run-all:
	$(CLI) run-all

download:
	$(CLI) download summary

load:
	$(CLI) load

build:
	$(CLI) build

sanity:
	$(CLI) sanity

train:
	$(CLI) train

serve:
	$(CLI) serve

explore:
	$(CLI) explore

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name '*.egg-info' \) -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist
