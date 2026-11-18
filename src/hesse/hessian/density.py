"""Stochastic Lanczos Quadrature (SLQ) for empirical spectral density estimation."""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

import torch

from hesse.hessian.hvp import HVPOperator

logger = logging.getLogger(__name__)


def stochastic_lanczos_quadrature(
    op: HVPOperator,
    n_probes: int = 10,
    m: int = 30,
    sigma: float = 0.1,
    n_grid: int = 200,
    seed: int = 42,
    grid_range: Tuple[float, float] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate the continuous eigenvalue spectral density rho(lambda) using SLQ.

    Combines n_probes runs of m-step Lanczos iterations with Gaussian quadrature,
    convolving discrete Ritz delta distributions with a Gaussian kernel of width sigma.

    Args:
        op: Hessian-vector product operator.
        n_probes: Number of independent random probe vectors (M).
        m: Lanczos Krylov subspace dimension per probe.
        sigma: Standard deviation (bandwidth) of the smoothing Gaussian kernel.
        n_grid: Number of discretization points along the eigenvalue spectrum grid.
        seed: Random seed for probe generation.
        grid_range: Optional explicit (min_val, max_val) evaluation range.

    Returns:
        Tuple of (grid, density) tensors of shape (n_grid,), where the density
        integrates to 1.0 via trapezoidal quadrature.

    Raises:
        ValueError: If n_probes, m, sigma, or n_grid are invalid.
    """
    if n_probes <= 0 or m <= 0 or sigma <= 0 or n_grid <= 1:
        raise ValueError(
            f"Invalid parameters: n_probes={n_probes}, m={m}, sigma={sigma}, n_grid={n_grid}"
        )

    m = min(m, op.dim)
    gen = torch.Generator(device=op.device)
    gen.manual_seed(seed)

    all_nodes: List[torch.Tensor] = []
    all_weights: List[torch.Tensor] = []

    for _ in range(n_probes):
        # Generate random normalized Rademacher probe vector
        z = (
            torch.randint(
                0,
                2,
                (op.dim,),
                generator=gen,
                device=op.device,
                dtype=torch.int64,
            ).to(op.dtype)
            * 2.0
            - 1.0
        )

        v = z / torch.linalg.vector_norm(z)

        # Run m-step Lanczos
        basis: List[torch.Tensor] = [v]
        alphas: List[torch.Tensor] = []
        betas: List[torch.Tensor] = []
        m_actual = m

        for j in range(m):
            v_j = basis[j]
            w = op.matvec(v_j)
            if j > 0:
                w = w - betas[j - 1] * basis[j - 1]
            alpha_j = torch.dot(v_j, w)
            alphas.append(alpha_j)
            w = w - alpha_j * v_j

            # Reorthogonalize
            for _ in range(2):
                for i in range(j + 1):
                    w = w - torch.dot(basis[i], w) * basis[i]

            beta_j = torch.linalg.vector_norm(w)
            if j < m - 1:
                if beta_j < 1e-12:
                    m_actual = j + 1
                    break
                betas.append(beta_j)
                basis.append(w / beta_j)
            else:
                m_actual = m

        # Build tridiagonal matrix T
        T = torch.zeros((m_actual, m_actual), device=op.device, dtype=op.dtype)
        for i in range(m_actual):
            T[i, i] = alphas[i]
            if i < len(betas):
                T[i, i + 1] = betas[i]
                T[i + 1, i] = betas[i]

        evals, evecs = torch.linalg.eigh(T)
        # Weights are the squared first components of eigenvectors: tau_k = (evecs[0, k])^2
        weights = evecs[0, :] ** 2

        all_nodes.append(evals)
        all_weights.append(weights)

    flat_nodes = torch.cat(all_nodes)
    flat_weights = torch.cat(all_weights)

    # Determine evaluation grid range
    if grid_range is not None:
        min_lambda, max_lambda = grid_range
    else:
        min_lambda = flat_nodes.min().item() - 3.0 * sigma
        max_lambda = flat_nodes.max().item() + 3.0 * sigma

    grid = torch.linspace(min_lambda, max_lambda, n_grid, device=op.device, dtype=op.dtype)

    # Gaussian kernel density smoothing:
    # density(lambda) = 1/M * sum_i w_i * (1/(sqrt(2pi)*s)) * exp(-(lambda - node_i)^2 / (2*s^2))
    # Vectorized computation: shape (n_grid, total_nodes)
    diff = grid.unsqueeze(1) - flat_nodes.unsqueeze(0)  # (n_grid, N)
    kernel = torch.exp(-0.5 * (diff / sigma) ** 2) / (math.sqrt(2.0 * math.pi) * sigma)
    # Sum weighted contributions and divide by number of probes
    density = torch.sum(kernel * flat_weights.unsqueeze(0), dim=1) / float(n_probes)

    # Normalize so the integral over the grid equals 1.0
    integral = torch.trapezoid(density, grid)
    if integral > 0:
        density = density / integral

    return grid, density


# Expose alias
spectral_density = stochastic_lanczos_quadrature
