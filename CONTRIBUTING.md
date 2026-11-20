# Contributing to hesse

Thank you for your interest in contributing to `hesse`! `hesse` is designed as a mathematically rigorous, reproducible toolkit for second-order loss landscape diagnostics and Edge-of-Stability research in PyTorch.

---

## 1. Development Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- PyTorch >= 2.0
- Git

### Installation
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/AdityaSurse/hesse.git
cd hesse

python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with development and dashboard extras
pip install -e ".[dev,app]"
```

Verify your installation:
```bash
make test
make lint
```

---

## 2. Coding Guidelines & Mathematical Rigor

### Code Style
- We use **Ruff** for linting and formatting with a 100-character line limit.
- Format code before committing:
  ```bash
  make format
  make lint
  ```
- **Type Annotations:** All public functions, classes, and methods must include full Python type annotations.
- **Docstrings:** Document functions and classes using Google style docstrings. Where algorithms implement mathematical methods (e.g. Pearlmutter's HVP, Lanczos recurrence, Hutchinson trace, SLQ), include the corresponding mathematical formulas and citations.

### Numerical Linear Algebra Principles
- **Never materialize full Hessians** for models with more than $10^3$ parameters. All curvature estimation must operate matrix-free via Hessian-vector products ($\mathcal{O}(P)$ memory).
- Always use **deterministic random seeds** (`hesse.utils.set_seed`).
- Evaluate curvature on **fixed evaluation subsets** so metrics remain comparable across training iterations.
- When performing weight perturbations during landscape evaluations, **guarantee bit-identical parameter restoration**.

---

## 3. Testing & Verification

Every pull request must pass the automated test suite and satisfy the coverage requirement:

```bash
# Run pytest with coverage verification
pytest -v --cov=hesse --cov-fail-under=85
```

- **Unit Tests:** Verify individual estimators against exact analytical or dense PyTorch baselines (`torch.linalg.eigh`, exact quadratic forms).
- **Integration Tests:** Verify training loops, tracking telemetry, checkpoint resumption, and report generation.

---

## 4. Submitting Pull Requests

1. **Fork & Branch:** Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Commit:** Keep commits atomic with descriptive messages.
3. **Verify:** Run the full verification suite:
   ```bash
   make lint
   make test
   ```
4. **Open PR:** Submit a Pull Request against `main`. Ensure CI checks pass cleanly.
