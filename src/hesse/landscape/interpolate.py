"""Linear 1D interpolation between initial and final checkpoints (Goodfellow et al., 2014)."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Sequence

import numpy as np
import torch
import torch.nn as nn

from hesse.landscape.grid import _evaluate_on_data
from hesse.utils import flatten_params, unflatten_params

logger = logging.getLogger(__name__)


def interpolate_1d(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data: Any,
    theta_init: torch.Tensor,
    theta_final: torch.Tensor,
    alphas: Sequence[float] | np.ndarray = np.linspace(-0.5, 1.5, 50),
) -> Dict[str, Any]:
    """Evaluate loss and accuracy along the linear interpolation path between two checkpoints.

    theta(alpha) = (1 - alpha) * theta_init + alpha * theta_final
    At alpha = 0.0: theta = theta_init
    At alpha = 1.0: theta = theta_final

    Args:
        model: Target PyTorch model.
        loss_fn: Loss function.
        data: Evaluation data.
        theta_init: 1D parameter vector at initialization.
        theta_final: 1D parameter vector at convergence / checkpoint.
        alphas: Interpolation grid coordinates (default: -0.5 to 1.5).

    Returns:
        Dictionary holding 'alphas', 'loss', and optional 'acc' arrays.
    """
    alphas_arr = np.array(alphas, dtype=np.float32)
    original_weights = [p.clone() for p in model.parameters()]
    flat_orig = flatten_params(model).clone()

    was_training = model.training
    model.eval()

    th_init = theta_init.to(device=flat_orig.device, dtype=flat_orig.dtype).reshape(-1)
    th_final = theta_final.to(device=flat_orig.device, dtype=flat_orig.dtype).reshape(-1)

    losses: list[float] = []
    accuracies: list[float | None] = []

    try:
        with torch.no_grad():
            for alpha in alphas_arr:
                theta_alpha = (1.0 - float(alpha)) * th_init + float(alpha) * th_final
                unflatten_params(model, theta_alpha)
                l_val, a_val = _evaluate_on_data(model, loss_fn, data)
                losses.append(l_val)
                accuracies.append(a_val)
    finally:
        with torch.no_grad():
            for p, orig in zip(model.parameters(), original_weights):
                p.copy_(orig)
            for p, orig in zip(model.parameters(), original_weights):
                assert torch.equal(p, orig), "Model weights were not restored bit-identically!"
        model.train(was_training)

    return {
        "alphas": alphas_arr,
        "loss": np.array(losses, dtype=np.float32),
        "acc": (
            np.array(accuracies, dtype=np.float32)
            if all(a is not None for a in accuracies)
            else None
        ),
    }
