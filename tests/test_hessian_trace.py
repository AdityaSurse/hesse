"""Unit tests for Hutchinson's randomized trace estimator."""

from __future__ import annotations

import torch
import torch.nn as nn

from hesse.hessian.hvp import HVPOperator
from hesse.hessian.trace import hutchinson_trace
from hesse.utils import set_seed


def test_hutchinson_trace_within_three_standard_errors() -> None:
    """Verify Hutchinson trace matches exact Hessian trace on tiny MLP within 3 standard errors."""
    from torch.func import functional_call

    set_seed(42)
    model = nn.Sequential(
        nn.Linear(3, 6),
        nn.Tanh(),
        nn.Linear(6, 2),
    )
    # Total params: (3*6+6) + (6*2+2) = 24 + 14 = 38 (< 200)
    x = torch.randn(25, 3)
    y = torch.randn(25, 2)
    loss_fn = nn.MSELoss()

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
        return nn.functional.mse_loss(preds, y, reduction="none").mean()

    H_exact = torch.autograd.functional.hessian(loss_from_flat, flat_params)
    exact_trace = torch.trace((H_exact + H_exact.T) / 2.0).item()

    op = HVPOperator(model=model, loss_fn=loss_fn, data=(x, y))
    # Run Hutchinson with 120 probes
    res = hutchinson_trace(op, n_samples=120, seed=42)

    diff = abs(res.estimate - exact_trace)
    threshold = 3.0 * res.std_err

    assert diff <= threshold, (
        f"Hutchinson trace {res.estimate:.4f} deviated from exact {exact_trace:.4f} "
        f"by {diff:.4f}, which exceeds 3 std_err ({threshold:.4f})"
    )


def test_hutchinson_trace_determinism() -> None:
    """Verify that Hutchinson trace estimator is completely deterministic for fixed seed."""
    A = torch.randn(10, 10)
    A = A @ A.T
    op = HVPOperator.from_matrix(A)

    res1 = hutchinson_trace(op, n_samples=30, seed=123)
    res2 = hutchinson_trace(op, n_samples=30, seed=123)

    assert res1.estimate == res2.estimate
    assert res1.std_err == res2.std_err
    assert res1.samples == res2.samples
