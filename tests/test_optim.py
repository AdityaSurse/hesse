"""Unit tests for optimizer factory, stability thresholds, and Lion implementation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hesse.config import OptimConfig
from hesse.optim import Lion, get_optimizer, get_stability_threshold


def test_lion_two_step_hand_computed() -> None:
    """Verify Lion matches hand-computed two-step numerical derivation exactly.

    Derivation:
        theta_0 = 1.0, lr = 0.1, beta1 = 0.9, beta2 = 0.8, weight_decay = 0.1, m_0 = 0.0

        Step 1 (grad = 2.0):
            theta -> theta * (1 - lr * wd) = 1.0 * 0.99 = 0.99
            c_1 = beta1 * m_0 + (1 - beta1) * g_1 = 0.9 * 0 + 0.1 * 2.0 = 0.2
            update = sign(0.2) = 1.0
            theta_1 = 0.99 - lr * update = 0.99 - 0.1 * 1.0 = 0.89
            m_1 = beta2 * m_0 + (1 - beta2) * g_1 = 0.8 * 0 + 0.2 * 2.0 = 0.4

        Step 2 (grad = -1.0):
            theta -> theta_1 * 0.99 = 0.89 * 0.99 = 0.8811
            c_2 = beta1 * m_1 + (1 - beta1) * g_2 = 0.9 * 0.4 + 0.1 * (-1.0) = 0.26
            update = sign(0.26) = 1.0
            theta_2 = 0.8811 - lr * update = 0.8811 - 0.1 * 1.0 = 0.7811
            m_2 = beta2 * m_1 + (1 - beta2) * g_2 = 0.8 * 0.4 + 0.2 * (-1.0) = 0.12
    """
    theta = torch.tensor([1.0], requires_grad=True)
    opt = Lion([theta], lr=0.1, betas=(0.9, 0.8), weight_decay=0.1)

    # Step 1
    theta.grad = torch.tensor([2.0])
    opt.step()
    assert abs(theta.item() - 0.89) < 1e-6
    assert abs(opt.state[theta]["exp_avg"].item() - 0.4) < 1e-6

    # Step 2
    theta.grad = torch.tensor([-1.0])
    opt.step()
    assert abs(theta.item() - 0.7811) < 1e-6
    assert abs(opt.state[theta]["exp_avg"].item() - 0.12) < 1e-6


def test_optimizer_factory_dispatch() -> None:
    """Verify get_optimizer constructs GD, SGD, Adam, AdamW, and Lion correctly."""
    model = nn.Linear(4, 2)

    # GD
    opt_gd = get_optimizer(model, OptimConfig(optimizer="gd", lr=0.05))
    assert isinstance(opt_gd, torch.optim.SGD)
    assert opt_gd.param_groups[0]["momentum"] == 0.0

    # SGD
    opt_sgd = get_optimizer(model, OptimConfig(optimizer="sgd", lr=0.01, momentum=0.9))
    assert isinstance(opt_sgd, torch.optim.SGD)
    assert opt_sgd.param_groups[0]["momentum"] == 0.9

    # Adam
    opt_adam = get_optimizer(model, OptimConfig(optimizer="adam", lr=1e-3))
    assert isinstance(opt_adam, torch.optim.Adam)

    # AdamW
    opt_adamw = get_optimizer(model, OptimConfig(optimizer="adamw", lr=1e-3, weight_decay=0.01))
    assert isinstance(opt_adamw, torch.optim.AdamW)

    # Lion
    opt_lion = get_optimizer(model, OptimConfig(optimizer="lion", lr=1e-4))
    assert isinstance(opt_lion, Lion)


def test_stability_threshold_calculations() -> None:
    """Verify stability thresholds for GD/SGD (2/lr) and Adam (38/lr at beta1=0.9)."""
    assert get_stability_threshold("gd", lr=0.05) == pytest.approx(40.0)
    assert get_stability_threshold("sgd", lr=0.02) == pytest.approx(100.0)

    # Adam: (2 + 2*0.9) / ((1 - 0.9) * 0.01) = 3.8 / 0.001 = 3800.0
    assert get_stability_threshold("adam", lr=0.01, beta1=0.9) == pytest.approx(3800.0)
    assert get_stability_threshold("adamw", lr=0.01, beta1=0.9) == pytest.approx(3800.0)

    # Lion: None
    assert get_stability_threshold("lion", lr=0.01) is None
