"""Unit tests for the unified Curvature facade."""

from __future__ import annotations

import torch
import torch.nn as nn

from hesse.hessian import Curvature
from hesse.utils import set_seed


def test_curvature_facade_end_to_end() -> None:
    """Verify that Curvature facade exposes all curvature metrics cleanly."""
    set_seed(42)
    model = nn.Sequential(
        nn.Linear(4, 6),
        nn.ReLU(),
        nn.Linear(6, 2),
    )
    x = torch.randn(15, 4)
    y = torch.randn(15, 2)
    loss_fn = nn.MSELoss()

    curv = Curvature(model=model, loss_fn=loss_fn, data=(x, y))

    # 1. Sharpness
    sharpness = curv.sharpness(iters=20, tol=1e-4, seed=42)
    assert isinstance(sharpness, float)
    assert sharpness > 0

    # 2. Top-k
    ritz_vals, ritz_vecs = curv.top_k(k=3, m=10, seed=42)
    assert ritz_vals.shape == (3,)
    assert ritz_vecs.shape == (curv.op.dim, 3)

    # 3. Trace
    trace_est, std_err = curv.trace(n_samples=20, seed=42)
    assert isinstance(trace_est, float)
    assert isinstance(std_err, float)

    # 4. Density
    grid, density = curv.density(n_probes=5, m=8, sigma=0.2, n_grid=50, seed=42)
    assert grid.shape == (50,)
    assert density.shape == (50,)
    integral = torch.trapezoid(density, grid).item()
    assert abs(integral - 1.0) < 1e-3

    # 5. Preconditioned Sharpness
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    prec_sharp, bound = curv.preconditioned_sharpness(optimizer=opt, iters=20, seed=42)
    assert isinstance(prec_sharp, float)
    assert isinstance(bound, float)
    assert bound > 0
