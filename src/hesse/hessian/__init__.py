"""Matrix-free Hessian curvature estimation engine for neural networks."""

from __future__ import annotations

import logging
from typing import Any, Callable, Tuple

import torch
import torch.nn as nn

from hesse.hessian.density import (
    spectral_density,
    stochastic_lanczos_quadrature,
)
from hesse.hessian.hvp import HVPOperator, hvp
from hesse.hessian.lanczos import LanczosResult, lanczos
from hesse.hessian.power import PowerIterationResult, power_iteration
from hesse.hessian.precond import (
    PreconditionedResult,
    extract_adam_preconditioner,
    preconditioned_sharpness,
)
from hesse.hessian.trace import TraceResult, hutchinson_trace

logger = logging.getLogger(__name__)


class Curvature:
    """Unified matrix-free curvature facade for PyTorch neural networks.

    Provides a clean, unified API for computing sharpness (top eigenvalue),
    top-k eigenspectrum, Hutchinson trace, continuous spectral density (SLQ),
    and Adam preconditioned sharpness.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        data: Any,
        use_torch_func: bool = False,
        chunk_size: int | None = None,
    ) -> None:
        """Initialize Curvature analysis harness.

        Args:
            model: Target PyTorch neural network.
            loss_fn: Loss function mapping (outputs, targets) to a scalar tensor.
            data: Evaluation data as (x, y), DataLoader, or DatasetBundle.
            use_torch_func: If True, uses forward-over-reverse jvp of grad.
            chunk_size: Optional chunk size for sub-batching large datasets.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.data = data
        self.use_torch_func = use_torch_func
        self.chunk_size = chunk_size

        self.op = HVPOperator(
            model=model,
            loss_fn=loss_fn,
            data=data,
            use_torch_func=use_torch_func,
            chunk_size=chunk_size,
        )

    def hvp(self, v: torch.Tensor) -> torch.Tensor:
        """Compute the matrix-free Hessian-vector product H * v.

        Args:
            v: 1D probe tensor of shape (P,).

        Returns:
            1D result tensor of shape (P,).
        """
        return self.op.matvec(v)

    def sharpness(
        self,
        iters: int = 100,
        tol: float = 1e-4,
        seed: int = 42,
    ) -> float:
        """Compute the top Hessian eigenvalue lambda_max (sharpness).

        Args:
            iters: Maximum number of power iteration steps.
            tol: Relative change stopping tolerance.
            seed: Random seed for probe initialization.

        Returns:
            Scalar float representing lambda_max.
        """
        res = power_iteration(self.op, iters=iters, tol=tol, seed=seed)
        return res.eigenvalue

    def top_k(
        self,
        k: int = 5,
        m: int = 30,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute top-k eigenvalues and Ritz vectors using Lanczos iteration.

        Args:
            k: Number of extremal eigenvalues to return.
            m: Krylov subspace dimension (m >= k).
            seed: Random seed for Lanczos starting vector.

        Returns:
            Tuple of (ritz_values, ritz_vectors) where ritz_values has shape (k,)
            and ritz_vectors has shape (P, k).
        """
        res = lanczos(self.op, k=k, m=m, seed=seed)
        return res.ritz_values, res.ritz_vectors

    def trace(
        self,
        n_samples: int = 100,
        seed: int = 42,
    ) -> Tuple[float, float]:
        """Estimate the Hessian trace tr(H) via Hutchinson's estimator.

        Args:
            n_samples: Number of random Rademacher probe evaluations.
            seed: Random seed for probe generation.

        Returns:
            Tuple of (trace_estimate, standard_error).
        """
        res = hutchinson_trace(self.op, n_samples=n_samples, seed=seed)
        return res.estimate, res.std_err

    def density(
        self,
        n_probes: int = 10,
        m: int = 30,
        sigma: float = 0.1,
        n_grid: int = 200,
        seed: int = 42,
        grid_range: Tuple[float, float] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute continuous eigenvalue spectral density rho(lambda) using SLQ.

        Args:
            n_probes: Number of random probe vectors.
            m: Lanczos Krylov subspace dimension per probe.
            sigma: Smoothing Gaussian kernel bandwidth.
            n_grid: Discretization resolution.
            seed: Random seed for probes.
            grid_range: Optional evaluation interval (min, max).

        Returns:
            Tuple of (grid, density) normalized so that integral equals 1.0.
        """
        return stochastic_lanczos_quadrature(
            self.op,
            n_probes=n_probes,
            m=m,
            sigma=sigma,
            n_grid=n_grid,
            seed=seed,
            grid_range=grid_range,
        )

    def preconditioned_sharpness(
        self,
        optimizer: torch.optim.Optimizer | None = None,
        iters: int = 100,
        tol: float = 1e-4,
        seed: int = 42,
        lr: float | None = None,
        beta1: float | None = None,
    ) -> Tuple[float, float]:
        """Compute Adam preconditioned sharpness and theoretical stability bound.

        Solves for lambda_max(P^{-1/2} H P^{-1/2}) and computes the stability threshold.

        Args:
            optimizer: Adam/AdamW optimizer instance holding second-moment state.
            iters: Maximum power iteration steps.
            tol: Power iteration convergence tolerance.
            seed: Random seed for power iteration probe.
            lr: Optional learning rate override.
            beta1: Optional first-moment momentum beta1 override.

        Returns:
            Tuple of (preconditioned_sharpness, stability_threshold).
        """
        res = preconditioned_sharpness(
            self.op,
            optimizer=optimizer,
            iters=iters,
            tol=tol,
            seed=seed,
            lr=lr,
            beta1=beta1,
        )
        return res.sharpness, res.threshold


__all__ = [
    "Curvature",
    "HVPOperator",
    "LanczosResult",
    "PowerIterationResult",
    "PreconditionedResult",
    "TraceResult",
    "extract_adam_preconditioner",
    "hutchinson_trace",
    "hvp",
    "lanczos",
    "power_iteration",
    "preconditioned_sharpness",
    "spectral_density",
    "stochastic_lanczos_quadrature",
]
