"""Unit tests for Hessian-Vector Product (HVP) operators and Pearlmutter / torch.func agreement."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hesse.hessian.hvp import HVPOperator, hvp
from hesse.utils import flatten_params, set_seed


class QuadraticModule(nn.Module):
    """Linear quadratic module L(theta) = 1/2 theta^T A theta."""

    def __init__(self, A: torch.Tensor, theta_init: torch.Tensor | None = None) -> None:
        super().__init__()
        self.A = nn.Parameter(A, requires_grad=False)
        dim = A.shape[0]
        init_val = theta_init if theta_init is not None else torch.randn(dim)
        self.theta = nn.Parameter(init_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns theta directly for dummy loss evaluation
        return self.theta


def make_random_spd(dim: int, seed: int = 42) -> torch.Tensor:
    """Generate a symmetric positive definite (SPD) matrix with known spectrum."""
    gen = torch.Generator().manual_seed(seed)
    M = torch.randn(dim, dim, generator=gen, dtype=torch.float64)
    Q, _ = torch.linalg.qr(M)
    # Known positive eigenvalues
    evals = torch.linspace(1.0, 10.0, dim, dtype=torch.float64)
    A = Q @ torch.diag(evals) @ Q.T
    return (A + A.T) / 2.0


def test_hvp_quadratic_agrees_with_matrix_product() -> None:
    """Prove that HVP on L = 1/2 theta^T A theta matches A @ v exactly."""
    set_seed(101)
    dim = 8
    A = make_random_spd(dim, seed=101).to(torch.float32)
    model = QuadraticModule(A)

    def quad_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.dot(pred, A @ pred)

    dummy_x = torch.zeros(1, 1)
    dummy_y = torch.zeros(1, 1)

    v = torch.randn(dim)
    expected_hv = A @ v

    # Pearlmutter HVP
    actual_hv = hvp(model, quad_loss, (dummy_x, dummy_y), v, use_torch_func=False)
    rel_error = torch.linalg.vector_norm(actual_hv - expected_hv) / torch.linalg.vector_norm(
        expected_hv
    )
    assert rel_error.item() < 1e-5


def test_hvp_pearlmutter_vs_torch_func_agreement() -> None:
    """Verify that double backward and torch.func (jvp of grad) agree to 1e-5."""
    set_seed(202)
    model = nn.Sequential(
        nn.Linear(6, 12),
        nn.Tanh(),
        nn.Linear(12, 4),
    )
    x = torch.randn(20, 6)
    y = torch.randn(20, 4)
    loss_fn = nn.MSELoss()

    flat_p = flatten_params(model)
    v = torch.randn_like(flat_p)

    hvp_pearlmutter = hvp(model, loss_fn, (x, y), v, use_torch_func=False)
    hvp_func = hvp(model, loss_fn, (x, y), v, use_torch_func=True)

    max_diff = torch.max(torch.abs(hvp_pearlmutter - hvp_func)).item()
    rel_diff = (
        torch.linalg.vector_norm(hvp_pearlmutter - hvp_func)
        / torch.linalg.vector_norm(hvp_pearlmutter)
    ).item()

    assert max_diff < 1e-5
    assert rel_diff < 1e-5


def test_hvp_chunking_matches_full_batch() -> None:
    """Verify that chunked evaluation matches unchunked full-batch HVP."""
    set_seed(303)
    model = nn.Sequential(
        nn.Linear(5, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    x = torch.randn(50, 5)
    y = torch.randint(0, 2, (50,))
    loss_fn = nn.CrossEntropyLoss()

    v = torch.randn(sum(p.numel() for p in model.parameters()))

    hvp_full = hvp(model, loss_fn, (x, y), v, chunk_size=None)
    hvp_chunked = hvp(model, loss_fn, (x, y), v, chunk_size=10)

    rel_error = (
        torch.linalg.vector_norm(hvp_full - hvp_chunked) / torch.linalg.vector_norm(hvp_full)
    ).item()
    assert rel_error < 1e-5


def test_hvp_operator_counter_and_from_matrix() -> None:
    """Test HVPOperator counter incrementation and from_matrix factory."""
    A = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
    op = HVPOperator.from_matrix(A)
    assert op.dim == 2
    assert op.n_calls == 0

    v = torch.tensor([1.0, 2.0])
    res = op.matvec(v)
    assert torch.allclose(res, torch.tensor([5.0, 5.0]))
    assert op.n_calls == 1

    op(v)
    assert op.n_calls == 2
    op.reset_counter()
    assert op.n_calls == 0


def test_hvp_dimension_mismatch_raises() -> None:
    """Test that invalid probe vector dimensions raise ValueError."""
    model = nn.Linear(4, 2)
    loss_fn = nn.MSELoss()
    x = torch.randn(2, 4)
    y = torch.randn(2, 2)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        hvp(model, loss_fn, (x, y), torch.randn(5))
