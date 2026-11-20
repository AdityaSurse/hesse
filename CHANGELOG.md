# Changelog

All notable changes to `hesse` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-20

### Added
- Real-time Streamlit telemetry dashboard (`app/dashboard.py`) with Live Run KPIs, Multi-Run Curvature comparison, Spectral evolution with SLQ density slider, 3D interactive loss landscapes, and Empirical findings gallery.
- Standalone static HTML telemetry report exporter (`hesse report --run <dir> --out <file>`) embedding interactive Plotly JS.
- End-to-end integration and smoke test suite (`tests/test_e2e_pipeline.py`) verifying train -> track -> landscape -> report workflows.
- CI automated coverage gate enforcing minimum 85% code coverage (`--cov-fail-under=85`).
- Complete public API reference manual with executable examples (`docs/API.md`).
- Citation specification (`CITATION.cff`) and developer contribution guide (`CONTRIBUTING.md`).

### Changed
- Production hardening of the entire `src/hesse` codebase: strict type annotations, equation-rich docstrings, zero dead code.
- NumPy 2.0 array casting compliance across grid landscape slicing modules.
- Upgraded package version to v1.0.0 for formal open-source release.

## [0.6.0] - 2026-11-19

### Added
- Empirical research study suite (`experiments/run_e1` through `run_e6`) testing Edge-of-Stability phenomena across GD, SGD, Adam, and Lion.
- Automatic result synthesis engine (`experiments/generate_summary.py`) generating verified multi-seed statistics in `experiments/results/summary.md`.
- Publication-quality research figure generation for Experiments E1–E6 in `docs/figures/`.

## [0.5.0] - 2026-11-19

### Added
- Loss landscape slicing engine (`src/hesse/landscape/`) implementing filter normalization (Li et al., 2018), top Hessian eigenvectors, and trajectory PCA projections.
- 1D linear interpolation ("Goodfellow slices") and 2D grid evaluation routines.
- Interactive 3D surface and 2D contour visualization generators (`src/hesse/viz/landscape_plots.py`).

## [0.4.0] - 2026-11-19

### Added
- Live curvature training harness (`src/hesse/train.py`) with divergence guard and checkpoint resumption.
- Telemetry tracker hook (`src/hesse/tracker.py`) recording step metrics, gradient norms, update norms, $\lambda_{\max}$, and computation times to `metrics.jsonl`.
- Optimizer suite (`src/hesse/optim.py`) supporting GD, SGD, Adam, AdamW, and custom Lion optimizer with sign updates and decoupled weight decay.

## [0.3.0] - 2026-11-18

### Added
- Matrix-free Hessian engine (`src/hesse/hessian/`):
  - Pearlmutter double backward HVP operator (`hvp.py`).
  - Power iteration solver with sign-aware eigenvalue estimation (`power.py`).
  - Lanczos iteration with full Gram-Schmidt reorthogonalization for top-$k$ spectra (`lanczos.py`).
  - Hutchinson's trace estimator using Rademacher probe vectors (`trace.py`).
  - Stochastic Lanczos Quadrature (SLQ) continuous empirical spectral density (`density.py`).
  - Preconditioned sharpness analyzer for Adam (`precond.py`).
  - Unified `Curvature` facade (`__init__.py`).

## [0.2.0] - 2026-11-18

### Added
- Deterministic data loaders (`src/hesse/data.py`) for CIFAR-10 subsets, MNIST, and synthetic regression/classification.
- Model architectures (`src/hesse/models.py`): MLP, SmallCNN, and TinyTransformer.
- Configuration dataclasses (`src/hesse/config.py`) with YAML serialization and validation.
- Reproducibility utilities (`src/hesse/utils.py`).

## [0.1.0] - 2026-11-18

### Added
- Initial project blueprint, mathematical specifications, and repository structure.
