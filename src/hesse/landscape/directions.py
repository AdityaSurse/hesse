"""Direction generators for loss-landscape slicing: filter-normalized, Hessian, and PCA."""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence, Tuple

import torch
import torch.nn as nn

from hesse.hessian import Curvature

logger = logging.getLogger(__name__)


def create_random_direction(
    model: nn.Module,
    seed: int | None = None,
    filter_normalized: bool = True,
) -> torch.Tensor:
    """Generate a random parameter perturbation direction vector.

    Under filter-wise normalization (Li et al., 2018), each 2D/4D filter or neuron
    slice is scaled to match the Frobenius norm of its corresponding trained weight slice,
    eliminating scale invariance artifacts. Biases and 1D parameters are zeroed.

    Args:
        model: Target PyTorch model.
        seed: Random seed for reproducibility.
        filter_normalized: If True, applies filter-wise normalization; else unit Gaussian.

    Returns:
        1D tensor of shape (P,) representing the perturbation direction vector.
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    slices: list[torch.Tensor] = []

    for p in model.parameters():
        if not p.requires_grad:
            continue

        if filter_normalized:
            if p.ndim >= 2:
                d_p = torch.zeros_like(p)
                for j in range(p.shape[0]):
                    w_slice = p[j]
                    w_norm = torch.linalg.norm(w_slice)
                    g_slice = torch.randn(
                        w_slice.shape,
                        generator=gen,
                        dtype=p.dtype,
                        device=p.device,
                    )
                    g_norm = torch.linalg.norm(g_slice)
                    d_p[j] = (g_slice / (g_norm + 1e-12)) * w_norm
                slices.append(d_p.reshape(-1))
            else:
                # Biases and 1D parameters are zeroed out as in Li et al.
                slices.append(torch.zeros_like(p).reshape(-1))
        else:
            g = torch.randn(p.shape, generator=gen, dtype=p.dtype, device=p.device)
            slices.append(g.reshape(-1))

    flat_dir = torch.cat(slices)
    if not filter_normalized:
        norm = torch.linalg.vector_norm(flat_dir)
        if norm > 0:
            flat_dir = flat_dir / norm

    return flat_dir


def create_hessian_directions(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data: Any,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the top-2 Hessian eigenvectors q_1 and q_2 using Lanczos iteration.

    Args:
        model: Target PyTorch model.
        loss_fn: Loss function.
        data: Evaluation dataset or batch for curvature calculation.
        seed: Random seed for Lanczos initialization.

    Returns:
        Tuple of (q_1, q_2) 1D tensors of shape (P,), representing orthonormal directions.
    """
    curv = Curvature(model=model, loss_fn=loss_fn, data=data)
    _, ritz_vectors = curv.top_k(k=2, m=25, seed=seed)

    q1 = ritz_vectors[:, 0].contiguous()
    q2 = ritz_vectors[:, 1].contiguous()

    # Ensure unit norm
    q1 = q1 / (torch.linalg.vector_norm(q1) + 1e-12)
    q2 = q2 / (torch.linalg.vector_norm(q2) + 1e-12)

    return q1, q2


def create_pca_directions(
    checkpoints: Sequence[torch.Tensor] | torch.Tensor,
    reference_theta: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract the top-2 principal components from an optimization trajectory.

    Given checkpoints theta_0, ..., theta_T and reference theta* (defaulting to theta_T),
    computes the SVD of the centered trajectory matrix M = [theta_t - theta*].

    Args:
        checkpoints: Sequence of 1D parameter vectors along the training path.
        reference_theta: Reference parameter vector (defaults to checkpoints[-1]).

    Returns:
        Tuple of (u_1, u_2) 1D tensors of shape (P,), representing orthonormal PCA directions.

    Raises:
        ValueError: If fewer than 2 checkpoints are provided.
    """
    if len(checkpoints) < 2:
        raise ValueError(f"Trajectory PCA requires at least 2 checkpoints, got {len(checkpoints)}")

    flat_ckpts = [ckpt.detach().reshape(-1) for ckpt in checkpoints]
    ref = reference_theta.detach().reshape(-1) if reference_theta is not None else flat_ckpts[-1]

    # Difference vectors: delta_t = theta_t - ref
    diffs = [ckpt - ref for ckpt in flat_ckpts]
    # Matrix of shape (P, T)
    M = torch.stack(diffs, dim=1).to(dtype=torch.float32)

    # Compute SVD: M = U @ S @ V^T
    # Columns of U are principal direction vectors
    U, _, _ = torch.linalg.svd(M, full_matrices=False)

    p_dim = U.shape[0]
    rank = U.shape[1]

    if rank >= 2:
        u1 = U[:, 0]
        u2 = U[:, 1]
    elif rank == 1:
        u1 = U[:, 0]
        # Construct an orthogonal vector to u1
        rand_v = torch.randn(p_dim, device=U.device, dtype=U.dtype)
        rand_v = rand_v - torch.dot(u1, rand_v) * u1
        u2 = rand_v / (torch.linalg.vector_norm(rand_v) + 1e-12)
    else:
        u1 = torch.zeros(p_dim, device=U.device, dtype=U.dtype)
        u1[0] = 1.0
        u2 = torch.zeros(p_dim, device=U.device, dtype=U.dtype)
        u2[1] = 1.0

    u1 = u1 / (torch.linalg.vector_norm(u1) + 1e-12)
    u2 = u2 / (torch.linalg.vector_norm(u2) + 1e-12)

    return u1, u2


def get_directions(
    method: str,
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    data: Any | None = None,
    checkpoints: Sequence[torch.Tensor] | None = None,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified direction dispatcher for loss landscape visualization.

    Args:
        method: Direction generation technique ('random', 'hessian', 'pca').
        model: Target PyTorch model.
        loss_fn: Loss function (required for 'hessian').
        data: Evaluation dataset (required for 'hessian').
        checkpoints: Trajectory checkpoints (required for 'pca').
        seed: Random seed.

    Returns:
        Tuple of (d_1, d_2) direction vectors.
    """
    m = method.lower()
    if m == "random":
        d1 = create_random_direction(model, seed=seed, filter_normalized=True)
        d2 = create_random_direction(model, seed=seed + 1, filter_normalized=True)
        return d1, d2
    if m == "hessian":
        if loss_fn is None or data is None:
            raise ValueError("Hessian directions require both loss_fn and data.")
        return create_hessian_directions(model, loss_fn, data, seed=seed)
    if m == "pca":
        if checkpoints is None or len(checkpoints) < 2:
            raise ValueError("PCA directions require at least 2 checkpoints.")
        return create_pca_directions(checkpoints)

    raise ValueError(
        f"Unknown direction method '{method}'. Expected 'random', 'hessian', or 'pca'."
    )
