"""Power iteration spectral solver for top Hessian eigenvalue (sharpness)."""

from __future__ import annotations

import logging
from typing import Iterator

import torch

from hesse.hessian.hvp import HVPOperator

logger = logging.getLogger(__name__)


class PowerIterationResult:
    """Result container for power iteration with tuple unpacking compatibility.

    Attributes:
        eigenvalue: Top eigenvalue with largest absolute magnitude (signed).
        eigenvector: Corresponding normalized unit eigenvector of shape (dim,).
        n_iters: Number of iterations executed until convergence or limit.
        largest_magnitude: Synonym for eigenvalue (largest magnitude, signed).
        largest_algebraic: Largest algebraic (most positive) eigenvalue.
    """

    def __init__(
        self,
        eigenvalue: float,
        eigenvector: torch.Tensor,
        n_iters: int,
        largest_algebraic: float | None = None,
    ) -> None:
        self.eigenvalue = float(eigenvalue)
        self.eigenvector = eigenvector
        self.n_iters = int(n_iters)
        self.largest_magnitude = float(eigenvalue)
        self.largest_algebraic = (
            float(largest_algebraic) if largest_algebraic is not None else float(eigenvalue)
        )

    def __iter__(self) -> Iterator[float | torch.Tensor | int]:
        """Allow 3-tuple unpacking: (eigenvalue, eigenvector, n_iters)."""
        yield self.eigenvalue
        yield self.eigenvector
        yield self.n_iters

    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int) -> float | torch.Tensor | int:
        return (self.eigenvalue, self.eigenvector, self.n_iters)[idx]

    def __repr__(self) -> str:
        return (
            f"PowerIterationResult(eigenvalue={self.eigenvalue:.6f}, "
            f"largest_algebraic={self.largest_algebraic:.6f}, "
            f"n_iters={self.n_iters})"
        )


def power_iteration(
    op: HVPOperator,
    iters: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
) -> PowerIterationResult:
    """Compute the top Hessian eigenvalue using randomized power iteration.

    Iterates v_{k+1} = H v_k / ||H v_k||, monitoring the relative change in Rayleigh
    quotient lambda = v_k^T H v_k. Distinguishes between largest-magnitude and
    largest-algebraic eigenvalues using shifted iteration if negative curvature dominates.

    Args:
        op: Hessian-vector product operator.
        iters: Maximum number of power iteration steps.
        tol: Relative change tolerance for early convergence termination.
        seed: Random seed for initial probe vector.

    Returns:
        PowerIterationResult holding eigenvalue, eigenvector, n_iters, and largest_algebraic.
    """
    gen = torch.Generator(device=op.device)
    gen.manual_seed(seed)

    # Initialize random unit vector
    v = torch.randn(op.dim, generator=gen, device=op.device, dtype=op.dtype)
    v_norm = torch.linalg.vector_norm(v)
    if v_norm == 0:
        v = torch.ones(op.dim, device=op.device, dtype=op.dtype)
        v_norm = torch.linalg.vector_norm(v)
    v = v / v_norm

    prev_lam = 0.0
    actual_iters = 0

    for step in range(1, iters + 1):
        actual_iters = step
        w = op.matvec(v)
        lam = torch.dot(v, w).item()

        w_norm = torch.linalg.vector_norm(w)
        if w_norm == 0:
            break

        v = w / w_norm

        if step > 1:
            rel_change = abs(lam - prev_lam) / (abs(lam) + 1e-12)
            if rel_change < tol:
                break
        prev_lam = lam

    # Final Rayleigh quotient on converged vector
    w_final = op.matvec(v)
    lam_final = torch.dot(v, w_final).item()

    # Determine largest algebraic eigenvalue:
    # If lam_final >= 0, largest magnitude is already positive (or zero).
    # If lam_final < 0, negative curvature has larger magnitude than positive curvature.
    # We run shifted power iteration on (H + |lam_final| I) to locate largest algebraic eigenvalue.
    largest_alg = lam_final
    if lam_final < 0:
        shift = abs(lam_final)

        def _shifted_matvec(u: torch.Tensor) -> torch.Tensor:
            return op.matvec(u) + shift * u

        shifted_op = HVPOperator.from_callable(
            dim=op.dim,
            fn=_shifted_matvec,
            device=op.device,
            dtype=op.dtype,
        )
        res_shifted = power_iteration(shifted_op, iters=iters, tol=tol, seed=seed + 1)
        largest_alg = res_shifted.eigenvalue - shift

    return PowerIterationResult(
        eigenvalue=lam_final,
        eigenvector=v,
        n_iters=actual_iters,
        largest_algebraic=largest_alg,
    )
