"""Benchmark execution cost and scaling of matrix-free Hessian estimators."""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# Set writable cache directory for Matplotlib in sandboxed environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hesse.hessian import (
    HVPOperator,
    hutchinson_trace,
    hvp,
    lanczos,
    power_iteration,
    stochastic_lanczos_quadrature,
)
from hesse.utils import set_seed
from scripts._style import (
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_PINK,
    COLOR_PURPLE,
    COLOR_YELLOW,
    save_figure,
    setup_style,
)


def make_bench_model(target_params: int) -> nn.Module:
    """Construct an MLP whose parameter count closely approximates target_params.

    For input_dim d_in=32, output_dim d_out=16, and hidden_dim h:
    Total params = (32*h + h) + (h*16 + 16) = 49*h + 16.
    If target_params > 50_000, uses a 2-hidden layer network:
    Total params = 33*h + h^2 + h + 17*h + 16 = h^2 + 51*h + 16.
    """
    if target_params <= 50_000:
        h = max(2, int((target_params - 16) / 49))
        return nn.Sequential(
            nn.Linear(32, h),
            nn.Tanh(),
            nn.Linear(h, 16),
        )
    # Quadratic formula: h^2 + 51*h + (16 - target_params) = 0
    disc = 51**2 - 4 * (16 - target_params)
    h = max(2, int((-51 + math.sqrt(disc)) / 2))
    return nn.Sequential(
        nn.Linear(32, h),
        nn.Tanh(),
        nn.Linear(h, h),
        nn.Tanh(),
        nn.Linear(h, 16),
    )


def run_benchmark() -> None:
    """Benchmark all matrix-free curvature estimators from 1e3 to 1e6 parameters."""
    set_seed(42)
    setup_style()

    target_param_counts = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
    actual_param_counts: List[int] = []

    times: Dict[str, List[float]] = {
        "hvp": [],
        "power_iteration": [],
        "lanczos": [],
        "hutchinson": [],
        "slq": [],
    }

    batch_size = 16
    loss_fn = nn.MSELoss()

    print("Running Hessian estimators scaling benchmark...")
    for target in target_param_counts:
        model = make_bench_model(target)
        p_actual = sum(p.numel() for p in model.parameters())
        actual_param_counts.append(p_actual)
        print(f"  Target: {target:,} -> Actual parameters: {p_actual:,}")

        x = torch.randn(batch_size, 32)
        y = torch.randn(batch_size, 16)
        op = HVPOperator(model=model, loss_fn=loss_fn, data=(x, y))

        # Warmup
        v_warmup = torch.randn(p_actual)
        _ = hvp(model, loss_fn, (x, y), v_warmup)

        # 1. HVP (Single call)
        v = torch.randn(p_actual)
        t0 = time.perf_counter()
        _ = hvp(model, loss_fn, (x, y), v)
        t_hvp = time.perf_counter() - t0
        times["hvp"].append(t_hvp)

        # 2. Power Iteration (top-1 eigenvalue, 15 iterations)
        t0 = time.perf_counter()
        _ = power_iteration(op, iters=15, tol=1e-4, seed=42)
        t_pi = time.perf_counter() - t0
        times["power_iteration"].append(t_pi)

        # 3. Lanczos (top-5 eigenvalues, m=15 steps)
        t0 = time.perf_counter()
        _ = lanczos(op, k=5, m=15, seed=42)
        t_lz = time.perf_counter() - t0
        times["lanczos"].append(t_lz)

        # 4. Hutchinson Trace (15 Rademacher probes)
        t0 = time.perf_counter()
        _ = hutchinson_trace(op, n_samples=15, seed=42)
        t_ht = time.perf_counter() - t0
        times["hutchinson"].append(t_ht)

        # 5. SLQ Density (3 probes, m=10 steps)
        t0 = time.perf_counter()
        _ = stochastic_lanczos_quadrature(op, n_probes=3, m=10, sigma=0.2, n_grid=50, seed=42)
        t_slq = time.perf_counter() - t0
        times["slq"].append(t_slq)

    # Generate publication-quality log-log plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    params_arr = np.array(actual_param_counts)

    # Plot estimator timing curves
    ax.loglog(
        params_arr,
        times["hvp"],
        marker="o",
        color=COLOR_BLUE,
        linewidth=2.2,
        markersize=6,
        label=r"Exact HVP $Hv$ (Pearlmutter $2\times$ backprop)",
    )
    ax.loglog(
        params_arr,
        times["power_iteration"],
        marker="s",
        color=COLOR_GREEN,
        linewidth=2.2,
        markersize=6,
        label=r"Power Iteration (Top Eigenvalue $\lambda_{\max}$, 15 steps)",
    )
    ax.loglog(
        params_arr,
        times["lanczos"],
        marker="^",
        color=COLOR_PURPLE,
        linewidth=2.2,
        markersize=6,
        label=r"Lanczos Algorithm (Top-5 Ritz Spectrum, $m=15$)",
    )
    ax.loglog(
        params_arr,
        times["hutchinson"],
        marker="D",
        color=COLOR_ORANGE,
        linewidth=2.2,
        markersize=6,
        label=r"Hutchinson Trace $\mathrm{tr}(H)$ ($M=15$ Rademacher Probes)",
    )
    ax.loglog(
        params_arr,
        times["slq"],
        marker="v",
        color=COLOR_PINK,
        linewidth=2.2,
        markersize=6,
        label=r"SLQ Spectral Density $\rho(\lambda)$ ($M=3, m=10$)",
    )

    # Linear O(P) scaling reference line calibrated to HVP
    ref_scale = times["hvp"][0] / params_arr[0]
    ref_y = ref_scale * params_arr
    ax.loglog(
        params_arr,
        ref_y,
        linestyle="--",
        color=COLOR_YELLOW,
        linewidth=1.6,
        alpha=0.8,
        label=r"Ideal Linear Scaling $\mathcal{O}(P)$",
    )

    ax.set_title(
        r"Linear Scaling $\mathcal{O}(P)$: Matrix-Free Hessian from $10^3$ to $10^6$ Parameters",
        pad=14,
    )
    ax.set_xlabel(r"Total Model Parameters $P$ (Log Scale)", labelpad=8)
    ax.set_ylabel(r"Wall-Clock Execution Time (Seconds, Log Scale)", labelpad=8)

    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

    # Annotate O(P) memory efficiency
    ax.annotate(
        r"$\mathcal{O}(P)$ Memory: Never materializes full $P \times P$ Hessian"
        "\n"
        r"($10^6$ params $\to 4$ TB avoided; evaluated in milliseconds)",
        xy=(params_arr[-1], times["hvp"][-1]),
        xytext=(params_arr[-3] * 0.4, times["hvp"][-1] * 0.08),
        arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=1.4),
        fontsize=9,
        color=COLOR_BLUE,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#161B22", edgecolor=COLOR_BLUE, alpha=0.9),
    )

    output_path = Path("docs/figures/hessian_cost.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path)
    print(f"Benchmark figure successfully saved to {output_path}")


if __name__ == "__main__":
    run_benchmark()
