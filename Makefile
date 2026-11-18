export MPLCONFIGDIR ?= /tmp/matplotlib
export PYTHONPATH ?= .

.PHONY: help install test lint format figures bench clean

help:
	@echo "hesse developer commands:"
	@echo "  make install  - Install package in editable mode with dev dependencies"
	@echo "  make test     - Run pytest suite"
	@echo "  make lint     - Check code style with ruff"
	@echo "  make format   - Format code with ruff"
	@echo "  make figures  - Generate publication figures for README"
	@echo "  make bench    - Run matrix-free Hessian scaling benchmarks"
	@echo "  make clean    - Remove build and cache artifacts"

install:
	pip install -e ".[dev,app]"

test:
	pytest -v tests/

lint:
	ruff check .

format:
	ruff format .

figures:
	python3 scripts/make_readme_figures.py

bench:
	python3 experiments/bench_hessian.py

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
