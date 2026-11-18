"""Utility helpers for reproducibility, device management, and parameter manipulation."""

from __future__ import annotations

import os
import random
from typing import Generator

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch with deterministic flags.

    Args:
        seed: Integer random seed for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Enable deterministic algorithms where supported, warning if an op lacks one
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def get_device(preferred: str = "auto") -> torch.device:
    """Resolve compute device based on preference and available hardware.

    Args:
        preferred: Device request string ('auto', 'cpu', 'cuda', 'mps').

    Returns:
        Selected torch.device instance.
    """
    pref = preferred.lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if pref in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device("cpu")


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the total number of parameters in a neural network module.

    Args:
        model: Target PyTorch neural network.
        trainable_only: If True, count only parameters with requires_grad=True.

    Returns:
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def param_generator(model: nn.Module) -> Generator[torch.Tensor, None, None]:
    """Yield all parameters of a model in consistent order.

    Args:
        model: Target PyTorch neural network.

    Yields:
        Parameter tensors.
    """
    yield from model.parameters()


def flatten_params(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a contiguous 1D tensor.

    Args:
        model: Target PyTorch neural network.

    Returns:
        1D tensor of shape (P,) where P is the total parameter count.
    """
    params = [p.detach().reshape(-1) for p in model.parameters()]
    if not params:
        return torch.empty(0)
    return torch.cat(params)


def unflatten_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    """Copy values from a flat 1D tensor back into model parameters in-place.

    Args:
        model: Target PyTorch neural network to update.
        flat_params: 1D tensor of shape (P,) containing updated parameter values.

    Raises:
        ValueError: If flat_params size does not match total parameter count.
    """
    total_params = count_params(model, trainable_only=False)
    if flat_params.numel() != total_params:
        raise ValueError(
            f"Dimension mismatch: flat_params has {flat_params.numel()} elements, "
            f"but model has {total_params} parameters."
        )

    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            chunk = flat_params[offset : offset + numel]
            p.copy_(chunk.view_as(p))
            offset += numel


def flatten_grads(model: nn.Module) -> torch.Tensor:
    """Extract and flatten first-order parameter gradients into a 1D tensor.

    Args:
        model: Target PyTorch neural network.

    Returns:
        1D tensor of shape (P,) containing flattened gradients.
    """
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().reshape(-1))
        else:
            grads.append(torch.zeros_like(p).reshape(-1))
    if not grads:
        return torch.empty(0)
    return torch.cat(grads)
