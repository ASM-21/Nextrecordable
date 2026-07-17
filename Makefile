.PHONY: help install dev-install test test-fast lint fix format download load explore clean

help:
	@echo "osha-risk dev commands"
	@echo ""
	@echo "  make install       Install package (runtime deps only)"
	@echo "  make dev-install   Install package + dev deps (ruff, pytest, jupyter)"
	@echo "  make test          Run full test suite verbosely"
	@echo "  make test-fast     Run tests, quiet output, stop on first failure"
	@echo "  make lint          Run ruff check"
	@echo "  make fix           Run ruff check --fix"
	@echo "  make download      Download OSHA ITA summary data (2016-2024)"
	@echo "  make load          Load downloaded data into DuckDB"
	@echo "  make explore       Print summary stats from the loaded DB"
	@echo "  make clean         Remove caches, __pycache__, build artifacts"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test:
	pytest -v

test-fast:
	pytest -q -x

lint:
	ruff check .

fix:
	ruff check . --fix

download:
	python -m osha_risk.cli download summary

load:
	python -m osha_risk.cli load

explore:
	python -m osha_risk.cli explore

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.ruff_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist
