"""Reproducible training harness with live curvature tracking and divergence guards."""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from hesse.config import RunConfig
from hesse.data import DatasetBundle, load_dataset
from hesse.models import get_model
from hesse.optim import get_optimizer
from hesse.tracker import Tracker
from hesse.utils import flatten_params, get_device, set_seed

logger = logging.getLogger(__name__)


def apply_dotted_overrides(cfg_dict: Dict[str, Any], overrides: list[str]) -> None:
    """Apply dotted key-value overrides into a nested configuration dictionary.

    Example:
        'optim.lr=0.02' -> cfg_dict['optim']['lr'] = 0.02

    Args:
        cfg_dict: Nested configuration dictionary to modify in-place.
        overrides: List of strings of the form 'path.to.key=value'.
    """
    import yaml

    for ov in overrides:
        if "=" not in ov:
            continue
        key_str, val_str = ov.split("=", 1)
        val = yaml.safe_load(val_str.strip())
        parts = key_str.strip().split(".")

        target = cfg_dict
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = val


def train(
    config: RunConfig,
    overrides: list[str] | None = None,
    resume_from: str | Path | None = None,
) -> Dict[str, Any]:
    """Execute neural network training with live curvature tracking telemetry.

    Orchestrates data loading, model initialization, optimization, step-by-step
    tracking, early stopping, divergence protection, and checkpoint persistence.

    Args:
        config: Top-level RunConfig specification.
        overrides: Optional list of dotted override strings (e.g. ['optim.lr=0.02']).
        resume_from: Optional path to a checkpoint (.pt) file to resume from.

    Returns:
        Summary dictionary containing terminal status and run statistics.
    """
    # 1. Apply dotted overrides if provided
    if overrides:
        cfg_dict = {
            "run_name": config.run_name,
            "epochs": config.epochs,
            "steps": config.steps,
            "seed": config.seed,
            "device": config.device,
            "checkpoint_interval": config.checkpoint_interval,
            "target_loss": config.target_loss,
            "divergence_threshold": config.divergence_threshold,
            "data": config.data.__dict__,
            "model": config.model.__dict__,
            "optim": config.optim.__dict__,
            "tracker": config.tracker.__dict__,
        }
        apply_dotted_overrides(cfg_dict, overrides)
        config = RunConfig.from_dict(cfg_dict)

    # 2. Setup run directory and persistence paths
    run_dir = Path("runs") / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    config.to_yaml(run_dir / "config.yaml")

    # 3. Master seed and device resolution
    set_seed(config.seed)
    device = get_device(config.device)

    # 4. Data pipeline
    dataset_bundle: DatasetBundle = load_dataset(config.data)
    is_full_batch = config.data.batch_size is None

    if is_full_batch:
        train_x, train_y = dataset_bundle.full_batch()
        train_x, train_y = train_x.to(device), train_y.to(device)
        eval_data = (train_x, train_y)
        steps_per_epoch = 1
    else:
        train_loader = dataset_bundle.train_loader
        if train_loader is None:
            raise RuntimeError("DataLoader was not initialized for mini-batch training.")
        # Fixed evaluation batch for curvature tracking: first batch or up to 500 samples
        train_x_all, train_y_all = dataset_bundle.full_batch()
        n_eval = min(500, train_x_all.size(0))
        eval_data = (train_x_all[:n_eval].to(device), train_y_all[:n_eval].to(device))
        steps_per_epoch = len(train_loader)

    # 5. Model initialization
    model = get_model(config.model).to(device)

    # 6. Objective function
    if config.data.loss_type == "mse":
        loss_fn = nn.MSELoss()
    elif config.data.loss_type == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss_type: {config.data.loss_type}")

    # 7. Optimizer
    optimizer = get_optimizer(model, config.optim)

    # 8. Checkpoint resume
    start_step = 0
    if resume_from is not None:
        ckpt_path = Path(resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt["step"] + 1

        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"])
        if "numpy_rng_state" in ckpt:
            np.random.set_state(ckpt["numpy_rng_state"])
        if "python_rng_state" in ckpt:
            random.setstate(ckpt["python_rng_state"])
        logger.info("Resumed run %s from step %d", config.run_name, start_step)

    # 9. Tracker harness
    tracker = Tracker(
        config=config.tracker,
        model=model,
        loss_fn=loss_fn,
        eval_data=eval_data,
        run_dir=run_dir,
        optimizer_name=config.optim.optimizer,
        lr=config.optim.lr,
        beta1=config.optim.beta1,
        seed=config.seed,
    )

    # 10. Step budget
    total_steps = config.steps if config.steps is not None else config.epochs * steps_per_epoch
    if start_step >= total_steps:
        logger.warning(
            "Start step %d exceeds or equals total steps %d; no steps executed.",
            start_step,
            total_steps,
        )
        return tracker.save_summary(status="already_completed")

    initial_loss: float | None = None
    data_iterator = None if is_full_batch else iter(train_loader)

    progress_bar = tqdm(
        range(start_step, total_steps),
        desc=f"hesse [{config.run_name}]",
        leave=True,
    )

    terminal_status = "completed"
    divergence_reason: str | None = None

    try:
        for step in progress_bar:
            step_t0 = time.perf_counter()

            # Acquire batch
            if is_full_batch:
                batch_x, batch_y = train_x, train_y
            else:
                try:
                    raw_batch = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_loader)
                    raw_batch = next(data_iterator)
                batch_x, batch_y = raw_batch[0].to(device), raw_batch[1].to(device)

            # Forward pass
            model.train()
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch_x)
            loss_tensor = loss_fn(preds, batch_y)
            loss_val = loss_tensor.item()

            if initial_loss is None:
                initial_loss = loss_val

            # --- Divergence Guard ---
            if (
                torch.isnan(loss_tensor)
                or torch.isinf(loss_tensor)
                or loss_val > config.divergence_threshold
                or (initial_loss > 0 and loss_val > 1e5 * initial_loss)
            ):
                terminal_status = "diverged"
                divergence_reason = f"Loss exploded to {loss_val} at step {step}"
                logger.warning("DIVERGENCE GUARD TRIGGERED at step %d: %s", step, divergence_reason)
                tracker.on_step_end(
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    batch=(batch_x, batch_y),
                    loss=loss_val,
                    update_norm=0.0,
                    train_step_time=time.perf_counter() - step_t0,
                )
                break

            # Backward pass
            loss_tensor.backward()

            # Record parameter state before update to compute update norm
            theta_pre = flatten_params(model).clone()
            optimizer.step()
            theta_post = flatten_params(model)
            update_norm = torch.linalg.vector_norm(theta_post - theta_pre).item()

            train_step_duration = time.perf_counter() - step_t0

            # Tracker telemetry hook
            log_record = tracker.on_step_end(
                step=step,
                model=model,
                optimizer=optimizer,
                batch=(batch_x, batch_y),
                loss=loss_val,
                update_norm=update_norm,
                train_step_time=train_step_duration,
            )

            # Update progress bar description with loss and sharpness
            if log_record is not None:
                lam_str = (
                    f"{log_record['sharpness']:.2f}"
                    if log_record.get("sharpness") is not None
                    else "N/A"
                )
                thresh_str = (
                    f"{log_record['stability_threshold']:.1f}"
                    if log_record.get("stability_threshold") is not None
                    else "N/A"
                )
                progress_bar.set_postfix(
                    loss=f"{loss_val:.4f}",
                    lam_max=lam_str,
                    thresh=thresh_str,
                )

            # Checkpoint saving
            if (
                config.checkpoint_interval is not None
                and (step + 1) % config.checkpoint_interval == 0
            ):
                ckpt_data = {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate(),
                    "loss": loss_val,
                }
                torch.save(ckpt_data, checkpoints_dir / f"checkpoint_step_{step + 1}.pt")

            # Early stopping check
            if config.target_loss is not None and loss_val <= config.target_loss:
                terminal_status = "early_stopped"
                logger.info(
                    "Early stopping target reached: loss %f <= %f", loss_val, config.target_loss
                )
                break

    finally:
        # Save final checkpoint
        final_ckpt = {
            "step": total_steps - 1 if terminal_status == "completed" else step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "loss": loss_val if "loss_val" in locals() else None,
        }
        torch.save(final_ckpt, checkpoints_dir / "checkpoint_final.pt")

        summary = tracker.save_summary(
            status=terminal_status,
            divergence_reason=divergence_reason,
            final_step=step if "step" in locals() else start_step,
        )
        tracker.close()

    return summary
