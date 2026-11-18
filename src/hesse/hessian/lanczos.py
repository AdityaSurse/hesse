"""Lanczos iteration with full reorthogonalization for top-k spectrum and Ritz vectors."""

from __future__ import annotations

import logging
from typing import Iterator, List

import torch

from hesse.hessian.hvp import HVPOperator

logger = logging.getLogger(__name__)


class LanczosResult:
    """Result container for Lanczos iteration with 2-tuple unpacking compatibility.

    Attributes:
        ritz_values: 1D tensor of shape (k,) with top-k Ritz eigenvalues (descending).
        ritz_vectors: 2D tensor of shape (P, k) with corresponding Ritz vectors.
        tridiagonal: 2D tensor of shape (m, m) representing tridiagonal matrix T.
        basis: 2D tensor of shape (P, m) containing orthonormal Krylov basis vectors V.
        m: Actual number of Lanczos steps completed.
    """

    def __init__(
        self,
        ritz_values: torch.Tensor,
        ritz_vectors: torch.Tensor,
        tridiagonal: torch.Tensor,
        basis: torch.Tensor,
        m: int,
    ) -> None:
        self.ritz_values = ritz_values
        self.ritz_vectors = ritz_vectors
        self.tridiagonal = tridiagonal
        self.basis = basis
        self.m = m

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Allow 2-tuple unpacking: (ritz_values, ritz_vectors)."""
        yield self.ritz_values
        yield self.ritz_vectors

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> torch.Tensor:
        return (self.ritz_values, self.ritz_vectors)[idx]

    def __repr__(self) -> str:
        return (
            f"LanczosResult(k={len(self.ritz_values)}, m={self.m}, "
            f"top_value={self.ritz_values[0].item() if len(self.ritz_values) > 0 else 0.0:.4f})"
        )


def lanczos(
    op: HVPOperator,
    k: int = 5,
    m: int = 30,
    seed: int = 42,
) -> LanczosResult:
    """Compute top-k eigenvalues and Ritz vectors using m-step Lanczos iteration.

    Implements full Gram-Schmidt reorthogonalization at each iteration to maintain
    numerical orthogonality of the Krylov subspace basis.

    Args:
        op: Hessian-vector product operator.
        k: Number of extremal eigenvalues and Ritz vectors to return.
        m: Maximum Krylov subspace dimension (m >= k).
        seed: Random seed for initial probe vector.

    Returns:
        LanczosResult holding ritz_values, ritz_vectors, tridiagonal matrix T, and basis V.

    Raises:
        ValueError: If k > m or k <= 0 or m <= 0.
    """
    if k <= 0 or m <= 0:
        raise ValueError(f"k and m must be positive integers, got k={k}, m={m}")

    m = min(m, op.dim)
    k = min(k, m)

    gen = torch.Generator(device=op.device)
    gen.manual_seed(seed)

    # Initial random normalized unit vector
    v1 = torch.randn(op.dim, generator=gen, device=op.device, dtype=op.dtype)
    v1_norm = torch.linalg.vector_norm(v1)
    if v1_norm == 0:
        v1 = torch.ones(op.dim, device=op.device, dtype=op.dtype)
        v1_norm = torch.linalg.vector_norm(v1)
    v1 = v1 / v1_norm

    basis_vectors: List[torch.Tensor] = [v1]
    alphas: List[torch.Tensor] = []
    betas: List[torch.Tensor] = []

    m_actual = m

    for j in range(m):
        v_j = basis_vectors[j]
        w = op.matvec(v_j)

        if j > 0:
            beta_prev = betas[j - 1]
            w = w - beta_prev * basis_vectors[j - 1]

        alpha_j = torch.dot(v_j, w)
        alphas.append(alpha_j)
        w = w - alpha_j * v_j

        # Full reorthogonalization (Twice Gram-Schmidt against all previous basis vectors)
        for _ in range(2):
            for i in range(j + 1):
                proj = torch.dot(basis_vectors[i], w)
                w = w - proj * basis_vectors[i]

        beta_j = torch.linalg.vector_norm(w)

        if j < m - 1:
            if beta_j < 1e-12:
                # Invariant Krylov subspace discovered; terminate early
                m_actual = j + 1
                break
            betas.append(beta_j)
            basis_vectors.append(w / beta_j)
        else:
            m_actual = m

    # Construct symmetric tridiagonal matrix T of shape (m_actual, m_actual)
    T = torch.zeros((m_actual, m_actual), device=op.device, dtype=op.dtype)
    for i in range(m_actual):
        T[i, i] = alphas[i]
        if i < len(betas):
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]

    # Diagonalize T using symmetric eigendecomposition
    evals, evecs = torch.linalg.eigh(T)

    # Sort Ritz values in descending algebraic order
    sort_idx = torch.argsort(evals, descending=True)
    evals_sorted = evals[sort_idx]
    evecs_sorted = evecs[:, sort_idx]

    # Select top-k Ritz values
    top_ritz_values = evals_sorted[:k]

    # Lift Ritz vectors back to full parameter space: Y = V @ U
    V_matrix = torch.stack(basis_vectors[:m_actual], dim=1)  # shape (P, m_actual)
    top_evecs_subspace = evecs_sorted[:, :k]  # shape (m_actual, k)
    top_ritz_vectors = V_matrix @ top_evecs_subspace  # shape (P, k)

    # Normalize columns of Ritz vectors
    top_ritz_vectors = top_ritz_vectors / (
        torch.linalg.vector_norm(top_ritz_vectors, dim=0, keepdim=True) + 1e-12
    )

    return LanczosResult(
        ritz_values=top_ritz_values,
        ritz_vectors=top_ritz_vectors,
        tridiagonal=T,
        basis=V_matrix,
        m=m_actual,
    )
