"""1D and 2D loss landscape grid evaluation with exact bit-identical weight restoration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from hesse.tracker import _compute_accuracy
from hesse.utils import flatten_params, unflatten_params

logger = logging.getLogger(__name__)


def _evaluate_on_data(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data: Any,
) -> Tuple[float, float | None]:
    """Compute average loss and classification accuracy on evaluation data."""
    # Handle direct (x, y) tuple
    if isinstance(data, (tuple, list)) and len(data) == 2 and isinstance(data[0], torch.Tensor):
        x, y = data[0], data[1]
        p_ref = next(model.parameters(), None)
        if p_ref is not None:
            x, y = x.to(device=p_ref.device), y.to(device=p_ref.device)
        preds = model(x)
        loss = loss_fn(preds, y).item()
        acc = _compute_accuracy(preds, y)
        return loss, acc

    # Handle DatasetBundle
    if hasattr(data, "full_batch") and callable(getattr(data, "full_batch")):
        x, y = data.full_batch()
        p_ref = next(model.parameters(), None)
        if p_ref is not None:
            x, y = x.to(device=p_ref.device), y.to(device=p_ref.device)
        preds = model(x)
        loss = loss_fn(preds, y).item()
        acc = _compute_accuracy(preds, y)
        return loss, acc

    # Handle DataLoader or iterable of batches
    total_samples = 0
    total_loss = 0.0
    correct_samples = 0
    has_accuracy = True

    p_ref = next(model.parameters(), None)
    device = p_ref.device if p_ref is not None else torch.device("cpu")

    for batch in data:
        bx, by = batch[0].to(device), batch[1].to(device)
        b_size = bx.size(0)
        preds = model(bx)
        loss_val = loss_fn(preds, by).item()
        total_loss += loss_val * b_size
        total_samples += b_size

        acc = _compute_accuracy(preds, by)
        if acc is not None:
            correct_samples += int(round(acc * b_size))
        else:
            has_accuracy = False

    mean_loss = total_loss / float(total_samples) if total_samples > 0 else 0.0
    mean_acc = (
        float(correct_samples) / float(total_samples)
        if (has_accuracy and total_samples > 0)
        else None
    )
    return mean_loss, mean_acc


def evaluate_1d(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data: Any,
    d: torch.Tensor,
    alphas: Sequence[float] | np.ndarray | torch.Tensor,
) -> Dict[str, Any]:
    """Evaluate loss and accuracy along a 1D direction slice theta(alpha) = theta* + alpha * d.

    Restores model parameters bit-identically upon completion.

    Args:
        model: Target PyTorch model.
        loss_fn: Loss function.
        data: Evaluation data.
        d: 1D perturbation direction vector of shape (P,).
        alphas: Sequence of alpha coordinates.

    Returns:
        Dictionary containing 'alphas', 'loss', and 'acc' arrays.
    """
    alphas_arr = np.array(alphas, dtype=np.float32)
    original_weights = [p.clone() for p in model.parameters()]
    flat_orig = flatten_params(model).clone()

    was_training = model.training
    model.eval()

    d_vec = d.to(device=flat_orig.device, dtype=flat_orig.dtype)
    losses: list[float] = []
    accuracies: list[float | None] = []

    try:
        with torch.no_grad():
            for alpha in alphas_arr:
                perturbed_theta = flat_orig + float(alpha) * d_vec
                unflatten_params(model, perturbed_theta)
                l_val, acc_val = _evaluate_on_data(model, loss_fn, data)
                losses.append(l_val)
                accuracies.append(acc_val)
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


def evaluate_2d(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data: Any,
    d1: torch.Tensor,
    d2: torch.Tensor,
    alphas: Sequence[float] | np.ndarray | torch.Tensor,
    betas: Sequence[float] | np.ndarray | torch.Tensor,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Evaluate loss surface over 2D grid theta(alpha, beta) = theta* + alpha*d1 + beta*d2.

    Guarantees bit-identical restoration of all model weights upon completion.

    Args:
        model: Target PyTorch model.
        loss_fn: Loss function.
        data: Evaluation data.
        d1: Horizontal perturbation direction vector of shape (P,).
        d2: Vertical perturbation direction vector of shape (P,).
        alphas: 1D sequence of horizontal coordinates.
        betas: 1D sequence of vertical coordinates.
        show_progress: Whether to render a tqdm progress bar across grid points.

    Returns:
        Dictionary containing 'alphas', 'betas', 2D 'loss' array (shape: n_betas, n_alphas),
        and optional 2D 'acc' array.
    """
    alphas_arr = np.array(alphas, dtype=np.float32)
    betas_arr = np.array(betas, dtype=np.float32)

    n_alphas = len(alphas_arr)
    n_betas = len(betas_arr)

    original_weights = [p.clone() for p in model.parameters()]
    flat_orig = flatten_params(model).clone()

    was_training = model.training
    model.eval()

    d1_vec = d1.to(device=flat_orig.device, dtype=flat_orig.dtype)
    d2_vec = d2.to(device=flat_orig.device, dtype=flat_orig.dtype)

    loss_grid = np.zeros((n_betas, n_alphas), dtype=np.float32)
    acc_grid = np.zeros((n_betas, n_alphas), dtype=np.float32)
    has_acc = True

    total_evals = n_alphas * n_betas
    pbar = tqdm(
        total=total_evals,
        desc=f"Evaluating landscape grid ({n_betas}x{n_alphas})",
        disable=not show_progress,
        leave=False,
    )

    try:
        with torch.no_grad():
            for i, beta in enumerate(betas_arr):
                for j, alpha in enumerate(alphas_arr):
                    perturbed_theta = flat_orig + float(alpha) * d1_vec + float(beta) * d2_vec
                    unflatten_params(model, perturbed_theta)
                    l_val, a_val = _evaluate_on_data(model, loss_fn, data)
                    loss_grid[i, j] = l_val
                    if a_val is not None:
                        acc_grid[i, j] = a_val
                    else:
                        has_acc = False
                    pbar.update(1)
    finally:
        pbar.close()
        with torch.no_grad():
            for p, orig in zip(model.parameters(), original_weights):
                p.copy_(orig)
            for p, orig in zip(model.parameters(), original_weights):
                assert torch.equal(p, orig), "Model weights were not restored bit-identically!"
        model.train(was_training)

    return {
        "alphas": alphas_arr,
        "betas": betas_arr,
        "loss": loss_grid,
        "acc": acc_grid if has_acc else None,
    }


def save_grid_npz(
    path: str | Path,
    alphas: np.ndarray,
    betas: np.ndarray,
    loss: np.ndarray,
    acc: np.ndarray | None = None,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Save 2D evaluation grid as compressed .npz archive."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta_str = json.dumps(metadata) if metadata is not None else "{}"
    save_dict: dict[str, Any] = {
        "alphas": alphas,
        "betas": betas,
        "loss": loss,
        "metadata": np.array(meta_str),
    }
    if acc is not None:
        save_dict["acc"] = acc

    np.savez_compressed(out_path, **save_dict)


def load_grid_npz(path: str | Path) -> Dict[str, Any]:
    """Load 2D evaluation grid from compressed .npz archive."""
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Landscape grid archive not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        meta_raw = data.get("metadata")
        meta_dict = json.loads(str(meta_raw)) if meta_raw is not None else {}
        return {
            "alphas": data["alphas"],
            "betas": data["betas"],
            "loss": data["loss"],
            "acc": data["acc"] if "acc" in data else None,
            "metadata": meta_dict,
        }
