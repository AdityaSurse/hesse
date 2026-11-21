export MPLCONFIGDIR ?= /tmp/matplotlib
export PYTHONPATH ?= .

.PHONY: help install test lint format figures bench clean reproduce e1 e2 e3 e4 e5 e6 summary app paper

help:
	@echo "hesse developer commands:"
	@echo "  make install    - Install package in editable mode with dev dependencies"
	@echo "  make app        - Launch Streamlit real-time telemetry dashboard"
	@echo "  make test       - Run pytest suite"
	@echo "  make lint       - Check code style with ruff"
	@echo "  make format     - Format code with ruff"
	@echo "  make figures    - Generate publication figures for README"
	@echo "  make bench      - Run matrix-free Hessian scaling benchmarks"
	@echo "  make reproduce  - Run all empirical experiments E1-E6 and generate summary.md"
	@echo "  make e1         - Run Experiment 1 (GD Edge of Stability reproduction)"
	@echo "  make e2         - Run Experiment 2 (SGD batch size scaling)"
	@echo "  make e3         - Run Experiment 3 (Adam preconditioned sharpness)"
	@echo "  make e4         - Run Experiment 4 (Lion sign-update dynamics)"
	@echo "  make e5         - Run Experiment 5 (Architectural comparison: MLP, CNN, ViT)"
	@echo "  make e6         - Run Experiment 6 (Objective geometry: MSE vs Cross-Entropy)"
	@echo "  make summary    - Compile results into experiments/results/summary.md"
	@echo "  make clean      - Remove build and cache artifacts"

install:
	pip install -e ".[dev,app]"

test:
	pytest -v tests/

lint:
	ruff check .

format:
	ruff format .

app:
	streamlit run app/dashboard.py

figures:
	python3 scripts/make_readme_figures.py

bench:
	python3 experiments/bench_hessian.py

reproduce: e1 e2 e3 e4 e5 e6 summary

e1:
	python3 experiments/run_e1_gd_eos.py

e2:
	python3 experiments/run_e2_sgd_batch.py

e3:
	python3 experiments/run_e3_adam_eos.py

e4:
	python3 experiments/run_e4_lion.py

e5:
	python3 experiments/run_e5_archs.py

e6:
	python3 experiments/run_e6_objectives.py

summary:
	python3 experiments/generate_summary.py

paper:
	@if command -v pdflatex >/dev/null 2>&1; then \
		cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex; \
	elif command -v tectonic >/dev/null 2>&1; then \
		cd paper && tectonic main.tex; \
	else \
		echo "Error: Neither pdflatex nor tectonic found. Please install a TeX engine." >&2; exit 1; \
	fi

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
