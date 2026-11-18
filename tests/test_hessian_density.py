"""Unit tests for Stochastic Lanczos Quadrature (SLQ) spectral density estimation."""

from __future__ import annotations

import torch

from hesse.hessian.density import stochastic_lanczos_quadrature
from hesse.hessian.hvp import HVPOperator
from hesse.utils import set_seed


def test_slq_density_integrates_to_one_and_covers_extremes() -> None:
    """Verify SLQ density integrates to 1 and has non-zero mass near extreme eigenvalues."""
    set_seed(42)
    dim = 25
    # Construct SPD matrix with known extreme eigenvalues: lambda_min = 2.0, lambda_max = 12.0
    gen = torch.Generator().manual_seed(42)
    M = torch.randn(dim, dim, generator=gen, dtype=torch.float64)
    Q, _ = torch.linalg.qr(M)
    evals = torch.linspace(2.0, 12.0, dim, dtype=torch.float64)
    A = (Q @ torch.diag(evals) @ Q.T).to(torch.float32)
    op = HVPOperator.from_matrix(A)

    grid, density = stochastic_lanczos_quadrature(
        op,
        n_probes=15,
        m=20,
        sigma=0.3,
        n_grid=250,
        seed=42,
    )

    # 1. Integrates to 1 within 1e-3
    integral = torch.trapezoid(density, grid).item()
    assert abs(integral - 1.0) < 1e-3, f"SLQ density integral was {integral}, expected 1.0 +- 1e-3"

    # 2. Non-zero mass near extreme eigenvalues: lambda_min = 2.0 and lambda_max = 12.0
    idx_min = torch.argmin(torch.abs(grid - 2.0)).item()
    idx_max = torch.argmin(torch.abs(grid - 12.0)).item()

    assert density[idx_min].item() > 1e-4, "Expected non-zero density mass near lambda_min"
    assert density[idx_max].item() > 1e-4, "Expected non-zero density mass near lambda_max"


def test_slq_density_determinism() -> None:
    """Verify that SLQ density is deterministic for a fixed seed."""
    A = torch.diag(torch.tensor([1.0, 3.0, 5.0, 7.0]))
    op = HVPOperator.from_matrix(A)

    grid1, dens1 = stochastic_lanczos_quadrature(op, n_probes=5, m=4, seed=55)
    grid2, dens2 = stochastic_lanczos_quadrature(op, n_probes=5, m=4, seed=55)

    assert torch.equal(grid1, grid2)
    assert torch.equal(dens1, dens2)
