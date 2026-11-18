"""Unit tests for Power Iteration and Lanczos Krylov solvers."""

from __future__ import annotations

import torch
import torch.nn as nn

from hesse.hessian.hvp import HVPOperator
from hesse.hessian.lanczos import lanczos
from hesse.hessian.power import power_iteration
from hesse.utils import set_seed


def make_spd_matrix(
    dim: int, evals_range: tuple[float, float] = (1.0, 10.0), seed: int = 42
) -> torch.Tensor:
    """Generate a symmetric positive-definite matrix with exact known eigenvalues."""
    gen = torch.Generator().manual_seed(seed)
    M = torch.randn(dim, dim, generator=gen, dtype=torch.float64)
    Q, _ = torch.linalg.qr(M)
    evals = torch.linspace(evals_range[0], evals_range[1], dim, dtype=torch.float64)
    A = Q @ torch.diag(evals) @ Q.T
    return ((A + A.T) / 2.0).to(torch.float32)


def compute_exact_hessian_mlp(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Materialize the exact Hessian matrix of a small model using torch.autograd.functional."""
    from torch.func import functional_call

    param_names = [k for k, _ in model.named_parameters() if _.requires_grad]
    param_tensors = [p for p in model.parameters() if p.requires_grad]
    flat_params = torch.cat([p.detach().reshape(-1) for p in param_tensors])

    def loss_from_flat(th: torch.Tensor) -> torch.Tensor:
        offset = 0
        p_dict = {}
        for name, p in zip(param_names, param_tensors):
            n = p.numel()
            p_dict[name] = th[offset : offset + n].view_as(p)
            offset += n
        preds = functional_call(model, p_dict, (x,))
        if isinstance(loss_fn, nn.MSELoss) and loss_fn.reduction == "mean":
            return nn.functional.mse_loss(preds, y, reduction="none").mean()
        return loss_fn(preds, y)

    H = torch.autograd.functional.hessian(loss_from_flat, flat_params)
    return (H + H.T) / 2.0


def test_power_iteration_and_lanczos_on_quadratic() -> None:
    """Verify Power Iteration and Lanczos recover true top eigenvalues within 1e-4 error."""
    set_seed(42)
    dim = 20
    true_max_eval = 25.0
    A = make_spd_matrix(dim, evals_range=(2.0, true_max_eval), seed=42)
    op = HVPOperator.from_matrix(A)

    # 1. Power Iteration
    lam_pi, _, _ = power_iteration(op, iters=150, tol=1e-6, seed=42)
    rel_error_pi = abs(lam_pi - true_max_eval) / true_max_eval
    assert rel_error_pi < 1e-4, f"Power iteration rel error {rel_error_pi} exceeds 1e-4"

    # 2. Lanczos top-3
    ritz_vals, _ = lanczos(op, k=3, m=15, seed=42)
    rel_error_lanczos_top1 = abs(ritz_vals[0].item() - true_max_eval) / true_max_eval
    assert rel_error_lanczos_top1 < 1e-4, (
        f"Lanczos top-1 rel error {rel_error_lanczos_top1} exceeds 1e-4"
    )

    # Verify descending ordering of Ritz values
    assert ritz_vals[0] >= ritz_vals[1] >= ritz_vals[2]


def test_lanczos_against_exact_hessian_on_tiny_mlp() -> None:
    """Verify Lanczos matches top-5 eigenvalues of exact Hessian on tiny MLP (< 200 params)."""
    set_seed(123)
    # Architecture: 4 -> 8 -> 2 => (4*8 + 8) + (8*2 + 2) = 40 + 18 = 58 parameters (< 200)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.Tanh(),
        nn.Linear(8, 2),
    )
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 200, f"Expected < 200 parameters, got {total_params}"

    x = torch.randn(30, 4)
    y = torch.randn(30, 2)
    loss_fn = nn.MSELoss()

    # Compute exact Hessian
    H_exact = compute_exact_hessian_mlp(model, loss_fn, x, y)
    exact_evals = torch.linalg.eigvalsh(H_exact)
    # Sort exact eigenvalues in descending order
    exact_evals_desc = torch.flip(exact_evals, dims=[0])
    exact_top5 = exact_evals_desc[:5]

    # Compute top-5 Ritz values using matrix-free Lanczos
    op = HVPOperator(model=model, loss_fn=loss_fn, data=(x, y))
    ritz_vals, _ = lanczos(op, k=5, m=30, seed=42)

    for i in range(5):
        target = exact_top5[i].item()
        approx = ritz_vals[i].item()
        # Relative error with small epsilon in denominator
        rel_err = abs(approx - target) / (abs(target) + 1e-8)
        assert rel_err < 1e-3, (
            f"Ritz value {i} mismatch: approx={approx}, exact={target}, rel_err={rel_err}"
        )


def test_power_iteration_sign_handling() -> None:
    """Test that power iteration handles sign and reports largest magnitude and algebraic."""
    # Diagonal matrix with dominant negative eigenvalue -10 and positive eigenvalue 3
    A = torch.diag(torch.tensor([-10.0, 3.0, 1.0, -2.0]))
    op = HVPOperator.from_matrix(A)

    res = power_iteration(op, iters=100, tol=1e-5, seed=42)
    # Largest magnitude is -10.0
    assert abs(res.eigenvalue - (-10.0)) < 1e-3
    assert abs(res.largest_magnitude - (-10.0)) < 1e-3
    # Largest algebraic is +3.0
    assert abs(res.largest_algebraic - 3.0) < 1e-3


def test_spectral_estimators_determinism() -> None:
    """Verify that Power Iteration and Lanczos produce identical outputs for fixed seed."""
    A = make_spd_matrix(15, seed=99)
    op = HVPOperator.from_matrix(A)

    # Power Iteration determinism
    res1_pi = power_iteration(op, iters=50, seed=77)
    res2_pi = power_iteration(op, iters=50, seed=77)
    assert res1_pi.eigenvalue == res2_pi.eigenvalue
    assert torch.equal(res1_pi.eigenvector, res2_pi.eigenvector)

    # Lanczos determinism
    res1_lz = lanczos(op, k=4, m=10, seed=77)
    res2_lz = lanczos(op, k=4, m=10, seed=77)
    assert torch.equal(res1_lz.ritz_values, res2_lz.ritz_values)
    assert torch.equal(res1_lz.ritz_vectors, res2_lz.ritz_vectors)
