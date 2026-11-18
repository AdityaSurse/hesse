"""Unit tests for Adam preconditioned sharpness and stability boundary estimation."""

from __future__ import annotations

import torch
import torch.nn as nn

from hesse.hessian.hvp import HVPOperator
from hesse.hessian.power import power_iteration
from hesse.hessian.precond import extract_adam_preconditioner, preconditioned_sharpness
from hesse.utils import set_seed


def test_preconditioned_sharpness_identity_matches_plain_sharpness() -> None:
    """Prove that preconditioned sharpness equals plain sharpness when P is the identity."""
    set_seed(42)
    # Symmetric positive-definite matrix
    A = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    op = HVPOperator.from_matrix(A)

    # Standard sharpness
    plain_sharpness, _, _ = power_iteration(op, iters=100, tol=1e-6, seed=42)

    # Preconditioned sharpness with P = I
    identity_p = torch.ones(2)
    res_precond = preconditioned_sharpness(
        op,
        preconditioner=identity_p,
        iters=100,
        tol=1e-6,
        seed=42,
        lr=0.01,
        beta1=0.9,
    )

    rel_err = abs(res_precond.sharpness - plain_sharpness) / plain_sharpness
    assert rel_err < 1e-5, f"Expected equivalence with P=I, got diff {rel_err}"


def test_adam_threshold_formula() -> None:
    """Verify that theoretical threshold matches (2 + 2*beta1) / ((1 - beta1) * lr)."""
    lr = 0.005
    beta1 = 0.9
    expected_threshold = (2.0 + 2.0 * 0.9) / ((1.0 - 0.9) * 0.005)  # 3.8 / (0.1 * 0.005) = 7600.0

    model = nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))
    x = torch.randn(10, 4)
    y = torch.randn(10, 2)
    loss_fn = nn.MSELoss()

    op = HVPOperator(model=model, loss_fn=loss_fn, data=(x, y))
    res = preconditioned_sharpness(op, optimizer=opt)

    assert abs(res.threshold - expected_threshold) < 1e-4


def test_adam_preconditioner_extraction_after_step() -> None:
    """Test extracting non-trivial Adam preconditioner after an optimizer step."""
    set_seed(77)
    model = nn.Linear(3, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    x = torch.randn(8, 3)
    y = torch.randn(8, 2)

    # Perform one step to populate optimizer state
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.step()

    p_diag = extract_adam_preconditioner(model, opt)
    assert p_diag.numel() == 8  # 3*2 + 2
    # Preconditioner entries should be strictly positive
    assert torch.all(p_diag > 0)
    # Should not be all 1.0 (indicating non-trivial v_hat was read)
    assert not torch.allclose(p_diag, torch.ones_like(p_diag))


def test_preconditioned_sharpness_determinism() -> None:
    """Verify preconditioned sharpness is deterministic for fixed seed."""
    A = torch.tensor([[5.0, 2.0], [2.0, 4.0]])
    op = HVPOperator.from_matrix(A)
    p = torch.tensor([1.5, 0.8])

    res1 = preconditioned_sharpness(op, preconditioner=p, iters=30, seed=12)
    res2 = preconditioned_sharpness(op, preconditioner=p, iters=30, seed=12)

    assert res1.sharpness == res2.sharpness
    assert res1.threshold == res2.threshold
    assert torch.equal(res1.eigenvector, res2.eigenvector)
