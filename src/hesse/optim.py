"""Optimizer factory for GD, SGD, Adam, AdamW, and Lion, plus stability thresholds."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch.optim import Optimizer

from hesse.config import OptimConfig

logger = logging.getLogger(__name__)


class Lion(Optimizer):
    """Lion (EvoLved Sign Momentum) optimizer from Chen et al. (2023).

    Update rule:
        c_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        theta_t = theta_{t-1} * (1 - lr * weight_decay) - lr * sign(c_t)
        m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        """Initialize Lion optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            lr: Learning rate (default: 1e-4).
            betas: Coefficients (beta1, beta2) for candidate and momentum tracking.
            weight_decay: Decoupled weight decay coefficient.
        """
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0) or not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                # Decoupled weight decay
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)

                # Candidate update step using sign
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1).sign()
                p.add_(update, alpha=-lr)

                # Momentum accumulation
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


def get_optimizer(
    model_or_params: nn.Module | Iterable[torch.Tensor],
    config: OptimConfig | None = None,
    **kwargs: Any,
) -> Optimizer:
    """Build an optimizer instance from an OptimConfig or keyword overrides.

    Args:
        model_or_params: Target PyTorch model or parameter iterable.
        config: OptimConfig specification. If None, defaults are used.
        **kwargs: Parameter overrides (e.g. lr=0.01, optimizer="sgd").

    Returns:
        Configured PyTorch Optimizer instance.
    """
    if config is None:
        config = OptimConfig(**kwargs)
    elif kwargs:
        cfg_dict = {
            "optimizer": config.optimizer,
            "lr": config.lr,
            "momentum": config.momentum,
            "beta1": config.beta1,
            "beta2": config.beta2,
            "weight_decay": config.weight_decay,
            "eps": config.eps,
        }
        cfg_dict.update(kwargs)
        config = OptimConfig(**cfg_dict)

    if isinstance(model_or_params, nn.Module):
        params = [p for p in model_or_params.parameters() if p.requires_grad]
    else:
        params = list(model_or_params)

    opt_name = config.optimizer.lower()

    if opt_name == "gd":
        # Full-batch Gradient Descent: SGD with zero momentum
        return torch.optim.SGD(
            params,
            lr=config.lr,
            momentum=0.0,
            weight_decay=config.weight_decay,
        )
    if opt_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if opt_name == "adam":
        return torch.optim.Adam(
            params,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    if opt_name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    if opt_name == "lion":
        return Lion(
            params,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {config.optimizer}")


def get_stability_threshold(
    optimizer_name: str,
    lr: float,
    beta1: float = 0.9,
) -> float | None:
    """Compute the linear stability threshold for a given optimizer.

    For GD / SGD:
        2 / eta
    For Adam / AdamW:
        (2 + 2 * beta1) / ((1 - beta1) * eta), equals 38 / eta for beta1=0.9
    For Lion:
        None (sign update breaks linear quadratic stability)

    Args:
        optimizer_name: Identifier of optimizer ('gd', 'sgd', 'adam', 'adamw', 'lion').
        lr: Learning rate eta.
        beta1: First moment momentum parameter (default: 0.9).

    Returns:
        Scalar threshold float, or None if undefined.
    """
    name = optimizer_name.lower()
    if lr <= 0.0:
        return None

    if name in {"gd", "sgd"}:
        return 2.0 / lr

    if name in {"adam", "adamw"}:
        if beta1 >= 1.0 or beta1 < 0.0:
            return None
        return (2.0 + 2.0 * beta1) / ((1.0 - beta1) * lr)

    if name == "lion":
        return None

    return None
