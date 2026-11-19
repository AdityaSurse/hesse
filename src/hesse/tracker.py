"""Live curvature telemetry tracker writing immediate JSONL metrics for real-time dashboards."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from hesse.config import TrackerConfig
from hesse.hessian import Curvature
from hesse.optim import get_stability_threshold
from hesse.utils import flatten_grads

logger = logging.getLogger(__name__)


def _compute_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float | None:
    """Compute classification accuracy between model logits and ground-truth targets."""
    if preds.ndim < 2:
        return None
    # 1D class indices
    if targets.ndim == 1:
        correct = (preds.argmax(dim=-1) == targets).sum().item()
        return float(correct) / float(targets.size(0))
    # One-hot encoded 2D targets
    if targets.ndim == 2 and targets.shape == preds.shape:
        correct = (preds.argmax(dim=-1) == targets.argmax(dim=-1)).sum().item()
        return float(correct) / float(targets.size(0))
    return None


class Tracker:
    """Tracks training metrics and live loss-landscape curvature every k steps."""

    def __init__(
        self,
        config: TrackerConfig,
        model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        eval_data: Any,
        run_dir: str | Path,
        optimizer_name: str = "gd",
        lr: float = 0.05,
        beta1: float = 0.9,
        seed: int = 42,
    ) -> None:
        """Initialize Curvature Tracker.

        Args:
            config: TrackerConfig specifying tracking frequencies and metric options.
            model: PyTorch model being trained.
            loss_fn: Objective loss function.
            eval_data: Fixed evaluation dataset or batch for comparable curvature metrics.
            run_dir: Run storage directory for persisting metrics and artifacts.
            optimizer_name: Name of active optimizer.
            lr: Optimizer base learning rate.
            beta1: First-moment momentum coefficient.
            seed: Master random seed for tracker RNG isolation.
        """
        self.config = config
        self.loss_fn = loss_fn
        self.eval_data = eval_data
        self.optimizer_name = optimizer_name.lower()
        self.lr = lr
        self.beta1 = beta1
        self.master_seed = seed

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file_path = self.run_dir / "metrics.jsonl"
        self._metrics_file = open(self.metrics_file_path, "a", encoding="utf-8")

        self.eigvecs_dir = self.run_dir / "eigvecs"
        if self.config.save_eigenvectors:
            self.eigvecs_dir.mkdir(parents=True, exist_ok=True)

        # Isolated Hessian curvature engine
        self.curv = Curvature(model=model, loss_fn=loss_fn, data=eval_data)

        self.history: List[Dict[str, Any]] = []
        self.stability_threshold = get_stability_threshold(self.optimizer_name, self.lr, self.beta1)
        self.cumulative_train_time = 0.0
        self.cumulative_curvature_time = 0.0

    def close(self) -> None:
        """Close the underlying metrics file stream."""
        if not self._metrics_file.closed:
            self._metrics_file.flush()
            self._metrics_file.close()

    def __enter__(self) -> Tracker:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def on_step_end(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optimizer,
        batch: Tuple[torch.Tensor, torch.Tensor],
        loss: float,
        update_norm: float | None = None,
        train_step_time: float = 0.0,
    ) -> Dict[str, Any] | None:
        """Hook called at the end of every training step.

        Evaluates loss and gradient norms on every step, and computes full matrix-free
        Hessian curvature metrics when step % track_interval == 0.

        Args:
            step: Current zero-indexed training step.
            model: Current model instance.
            optimizer: Active optimizer instance.
            batch: Tuple of (inputs, targets) from training stream.
            loss: Training loss float for the current batch.
            update_norm: Euclidean norm of parameter update vector.
            train_step_time: Wall-clock seconds spent on this training step.

        Returns:
            Metrics dictionary if logging occurred, else None.
        """
        self.cumulative_train_time += train_step_time

        # Only evaluate and log every track_interval steps (or step 0)
        if step % self.config.track_interval != 0:
            return None

        # 1. Compute training accuracy if applicable
        train_x, train_y = batch
        was_training = model.training
        model.eval()

        with torch.no_grad():
            preds = model(train_x)
            train_acc = _compute_accuracy(preds, train_y)

        # 2. Extract gradient norm from first-order parameter gradients
        flat_g = flatten_grads(model)
        grad_norm = torch.linalg.vector_norm(flat_g).item() if flat_g.numel() > 0 else 0.0

        # 3. Compute Curvature Metrics under isolated RNG
        curv_t0 = time.perf_counter()
        probe_seed = self.master_seed + 10007 + step

        sharpness: float | None = None
        topk_eigenvalues: List[float] | None = None
        trace_est: float | None = None
        trace_std_err: float | None = None
        precond_sharpness: float | None = None

        try:
            if self.config.compute_sharpness:
                sharpness = self.curv.sharpness(iters=60, tol=1e-4, seed=probe_seed)

            if self.config.compute_topk > 0:
                m_dim = max(20, self.config.compute_topk * 3)
                ritz_vals, ritz_vecs = self.curv.top_k(
                    k=self.config.compute_topk, m=m_dim, seed=probe_seed + 1
                )
                topk_eigenvalues = [float(ev.item()) for ev in ritz_vals]

                if self.config.save_eigenvectors:
                    eig_file = self.eigvecs_dir / f"eigvecs_step_{step}.pt"
                    torch.save(
                        {
                            "step": step,
                            "ritz_values": ritz_vals.cpu(),
                            "ritz_vectors": ritz_vecs.cpu(),
                        },
                        eig_file,
                    )

            if self.config.compute_trace:
                trace_est, trace_std_err = self.curv.trace(n_samples=40, seed=probe_seed + 2)

            if self.optimizer_name in {"adam", "adamw"}:
                precond_sharp, _ = self.curv.preconditioned_sharpness(
                    optimizer=optimizer,
                    iters=60,
                    tol=1e-4,
                    seed=probe_seed + 3,
                    lr=self.lr,
                    beta1=self.beta1,
                )
                precond_sharpness = precond_sharp

        finally:
            curv_duration = time.perf_counter() - curv_t0
            self.cumulative_curvature_time += curv_duration
            # Zero out any graph state from HVP and restore model training state
            model.zero_grad(set_to_none=True)
            model.train(was_training)

        # 4. Construct telemetry record
        record: Dict[str, Any] = {
            "step": int(step),
            "loss": float(loss),
            "train_acc": train_acc,
            "grad_norm": float(grad_norm),
            "update_norm": float(update_norm) if update_norm is not None else 0.0,
            "sharpness": float(sharpness) if sharpness is not None else None,
            "topk_eigenvalues": topk_eigenvalues,
            "trace": float(trace_est) if trace_est is not None else None,
            "trace_std_err": float(trace_std_err) if trace_std_err is not None else None,
            "stability_threshold": self.stability_threshold,
            "preconditioned_sharpness": precond_sharpness,
            "curvature_time": float(curv_duration),
            "train_time": float(self.cumulative_train_time),
        }

        # 5. Immediately persist to JSONL and flush
        line = json.dumps(record) + "\n"
        self._metrics_file.write(line)
        self._metrics_file.flush()

        self.history.append(record)
        return record

    def save_summary(self, status: str = "completed", **extra_fields: Any) -> Dict[str, Any]:
        """Compile and write summary.json for run post-mortem and dashboard ingestion.

        Args:
            status: Terminal execution state ('completed', 'diverged', 'early_stopped').
            **extra_fields: Arbitrary additional metadata fields.

        Returns:
            Dictionary written to summary.json.
        """
        final_record = self.history[-1] if self.history else {}
        summary: Dict[str, Any] = {
            "status": status,
            "total_steps": final_record.get("step", 0) + 1 if self.history else 0,
            "final_loss": final_record.get("loss"),
            "final_sharpness": final_record.get("sharpness"),
            "stability_threshold": self.stability_threshold,
            "total_train_time": float(self.cumulative_train_time),
            "total_curvature_time": float(self.cumulative_curvature_time),
            "metrics_logged": len(self.history),
        }
        summary.update(extra_fields)

        summary_path = self.run_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
