"""Publication-quality 3-panel loss, sharpness, and gradient trajectory plots."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

# Ensure writable cache directory for Matplotlib in sandboxed environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

from hesse.viz.style import (
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_PURPLE,
    save_figure,
    setup_style,
)


def plot_run(
    run_dir: str | Path,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generate a 3-panel figure showing loss, sharpness with threshold, and gradient norm.

    Args:
        run_dir: Path to directory containing metrics.jsonl.
        save_path: Optional output path for the saved figure. If None, saves
                   to run_dir / "curves.png".

    Returns:
        Rendered Matplotlib Figure instance.

    Raises:
        FileNotFoundError: If metrics.jsonl does not exist in run_dir.
        ValueError: If metrics.jsonl is empty.
    """
    setup_style()
    r_dir = Path(run_dir)
    metrics_file = r_dir / "metrics.jsonl"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    records: List[Dict[str, Any]] = []
    with open(metrics_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))

    if not records:
        raise ValueError(f"No metric records found in {metrics_file}")

    # Extract series data
    steps: List[int] = [r["step"] for r in records]
    losses: List[float] = [r["loss"] for r in records]
    grad_norms: List[float] = [r.get("grad_norm", 0.0) for r in records]

    sharpness_steps: List[int] = [r["step"] for r in records if r.get("sharpness") is not None]
    sharpness_vals: List[float] = [
        r["sharpness"] for r in records if r.get("sharpness") is not None
    ]

    threshold = next(
        (r["stability_threshold"] for r in records if r.get("stability_threshold") is not None),
        None,
    )

    fig, (ax_loss, ax_sharp, ax_grad) = plt.subplots(1, 3, figsize=(18, 5.2), dpi=200, sharex=False)

    # Panel 1: Loss on log scale
    ax_loss.semilogy(steps, losses, color=COLOR_BLUE, linewidth=2.0, marker="o", markersize=3.5)
    ax_loss.set_title(r"Empirical Loss $\mathcal{L}(\theta_t)$ (Log Scale)")
    ax_loss.set_xlabel("Training Step $t$")
    ax_loss.set_ylabel(r"Loss $\mathcal{L}(\theta)$")
    ax_loss.grid(True, which="both", linestyle="--", alpha=0.35)

    # Panel 2: Top Eigenvalue / Sharpness vs Stability Threshold
    if sharpness_vals:
        ax_sharp.plot(
            sharpness_steps,
            sharpness_vals,
            color=COLOR_GREEN,
            linewidth=2.0,
            marker="s",
            markersize=4.0,
            label=r"Sharpness $\lambda_{\max}(H)$",
        )
    if threshold is not None:
        ax_sharp.axhline(
            threshold,
            color=COLOR_ORANGE,
            linestyle="--",
            linewidth=1.8,
            label=f"Stability Threshold ({threshold:.1f})",
        )
    ax_sharp.set_title(r"Curvature Dynamics: Sharpness $\lambda_{\max}$")
    ax_sharp.set_xlabel("Training Step $t$")
    ax_sharp.set_ylabel(r"Eigenvalue $\lambda_{\max}$")
    ax_sharp.grid(True, linestyle="--", alpha=0.35)
    if sharpness_vals or threshold is not None:
        ax_sharp.legend(loc="best", framealpha=0.85)

    # Panel 3: Gradient Norm
    ax_grad.plot(steps, grad_norms, color=COLOR_PURPLE, linewidth=2.0, marker="^", markersize=3.5)
    ax_grad.set_title(r"First-Order Gradient Norm $\|\nabla \mathcal{L}\|_2$")
    ax_grad.set_xlabel("Training Step $t$")
    ax_grad.set_ylabel(r"$\|\nabla \mathcal{L}\|_2$")
    ax_grad.grid(True, linestyle="--", alpha=0.35)

    out_file = Path(save_path) if save_path is not None else r_dir / "curves.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_file)

    return fig
