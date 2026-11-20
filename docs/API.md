# hesse Public API Reference

The `hesse` package exposes matrix-free curvature estimators, live training telemetry hooks, and loss-landscape visualization tools.

---

## 1. Matrix-Free Hessian Engine (`hesse.hessian`)

### `Curvature` Facade

```python
from hesse.hessian import Curvature
```

The unified facade providing zero-materialization second-order curvature operators. Evaluates Hessian operations in $\mathcal{O}(P)$ memory via Pearlmutter's double backward algorithm.

#### Constructor

```python
curv = Curvature(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data: Tuple[torch.Tensor, torch.Tensor] | DatasetBundle | DataLoader,
    backend: str = "pearlmutter",  # "pearlmutter" or "functorch"
    max_chunk_size: int = 500,
)
```

#### Methods

- **`sharpness(iters: int = 40, tol: float = 1e-4, seed: int = 42) -> PowerIterationResult`**
  Computes the top Hessian eigenvalue $\lambda_{\max}$ and corresponding eigenvector via Power Iteration:
  $$\lambda_{\max} = \frac{v^T H v}{v^T v}$$
  *Returns:* `PowerIterationResult(eigenvalue=..., eigenvector=..., n_iters=..., converged=...)`

- **`top_k(k: int = 5, m: int = 25, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]`**
  Computes the top-$k$ Ritz eigenvalues and Ritz vectors using Lanczos iteration with full Gram-Schmidt reorthogonalization.
  *Returns:* `(ritz_values, ritz_vectors)` where `ritz_values` has shape `(k,)` and `ritz_vectors` has shape `(k, P)`.

- **`trace(n_samples: int = 30, seed: int = 42) -> Tuple[float, float]`**
  Estimates $\operatorname{tr}(H)$ via Hutchinson's estimator with Rademacher random probe vectors $v \in \{-1, +1\}^P$:
  $$\operatorname{tr}(H) \approx \frac{1}{M} \sum_{i=1}^M v_i^T H v_i$$
  *Returns:* `(estimated_trace, standard_error)`

- **`density(n_probes: int = 15, m: int = 30, sigma: float = 0.05, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]`**
  Estimates the empirical spectral density $\rho(\lambda) = \frac{1}{P} \sum_{i=1}^P \delta(\lambda - \lambda_i)$ using Stochastic Lanczos Quadrature (SLQ) with Gaussian kernel smoothing.
  *Returns:* `(eigenvalue_grid, spectral_density)`

- **`preconditioned_sharpness(optimizer: torch.optim.Optimizer, iters: int = 40, tol: float = 1e-4, seed: int = 42) -> Tuple[float, float]`**
  Computes Adam preconditioned sharpness $\lambda_{\max}(P_t^{-1} H)$ and analytical stability threshold $\frac{2+2\beta_1}{(1-\beta_1)\eta}$.
  *Returns:* `(preconditioned_sharpness, stability_threshold)`

#### Example

```python
import torch
import torch.nn as nn
from hesse.hessian import Curvature

model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))
loss_fn = nn.CrossEntropyLoss()
data = (torch.randn(64, 16), torch.randint(0, 2, (64,)))

curv = Curvature(model=model, loss_fn=loss_fn, data=data)

# Compute top eigenvalue
res = curv.sharpness(iters=30)
print(f"Sharpness lambda_max: {res.eigenvalue:.4f}")

# Compute top-3 eigenvalues
ritz_vals, ritz_vecs = curv.top_k(k=3, m=15)
print(f"Top 3 eigenvalues: {ritz_vals.tolist()}")

# Compute trace
tr, std_err = curv.trace(n_samples=20)
print(f"Hessian Trace: {tr:.2f} +/- {std_err:.2f}")
```

---

## 2. Training & Curvature Telemetry (`hesse.train`, `hesse.tracker`)

### `train(config: RunConfig, overrides: dict | None = None, resume_from: str | Path | None = None) -> dict`

Executes the optimization loop with integrated real-time curvature telemetry and checkpointing.

```python
from hesse.config import DataConfig, ModelConfig, OptimConfig, RunConfig, TrackerConfig
from hesse.train import train

cfg = RunConfig(
    run_name="runs/cifar_gd_demo",
    data=DataConfig(dataset="cifar10", subset_size=5000, loss_type="mse"),
    model=ModelConfig(architecture="mlp", hidden_dims=[512, 256], activation="tanh"),
    optim=OptimConfig(optimizer="gd", lr=0.05),
    tracker=TrackerConfig(
        track_interval=10,
        compute_sharpness=True,
        compute_topk=5,
        compute_trace=True,
    ),
    steps=100,
    checkpoint_interval=20,
)

summary = train(cfg)
print(f"Training status: {summary['status']}, Final Loss: {summary['final_loss']:.4f}")
```

---

## 3. Loss Landscape Slicing (`hesse.landscape`)

### Direction Generators

```python
from hesse.landscape.directions import (
    create_random_direction,
    create_hessian_directions,
    create_pca_directions,
    get_directions,
)
```

- **`create_random_direction(model: nn.Module, seed: int = 42, filter_normalized: bool = True) -> torch.Tensor`**
  Generates random Gaussian perturbations rescaled per filter/neuron to match target parameter norms (Li et al., 2018):
  $$d_{i,j} \leftarrow d_{i,j} \frac{\|\theta_{i,j}\|_2}{\|d_{i,j}\|_2}$$

- **`create_hessian_directions(model: nn.Module, loss_fn: Callable, data: Any, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]`**
  Extracts the top-2 Hessian eigenvectors using Lanczos iteration.

- **`create_pca_directions(checkpoints: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]`**
  Extracts orthonormal principal component axes from optimization trajectory differences $(\theta_t - \theta^*)$.

- **`get_directions(method: str, model: nn.Module, ...) -> Tuple[torch.Tensor, torch.Tensor]`**
  Unified dispatcher supporting `'random'`, `'hessian'`, and `'pca'`.

### Grid Evaluation

```python
from hesse.landscape.grid import evaluate_1d, evaluate_2d
```

- **`evaluate_1d(model, loss_fn, data, d, alphas) -> dict`**
  Evaluates loss along a 1D slice $\theta(\alpha) = \theta^* + \alpha \cdot d$.

- **`evaluate_2d(model, loss_fn, data, d1, d2, alphas, betas, show_progress=True) -> dict`**
  Evaluates loss surface over a 2D mesh $\theta(\alpha, \beta) = \theta^* + \alpha \cdot d_1 + \beta \cdot d_2$. Guarantees bit-identical parameter restoration upon completion.

---

## 4. Report Exporter (`hesse.report`)

### `generate_report(run_dir: str | Path, out_path: str | Path | None = None) -> Path`

Generates a self-contained, offline HTML report embedding Plotly JS:

```python
from hesse.report import generate_report

report_file = generate_report("runs/cifar_gd_demo", out_path="report.html")
print(f"Report exported to {report_file}")
```
