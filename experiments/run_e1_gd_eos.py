"""Experiment 1: Gradient Descent Edge of Stability (Reproduction).

Protocol:
  - Full-batch GD on tanh MLP with MSE loss on CIFAR-10 subset.
  - 4 learning rates on a log scale: eta in {0.01, 0.02, 0.05, 0.10}.
  - 3 seeds per rate (42, 43, 44).
  - Track lambda_max and loss every 2 steps.
  - Benchmarks against analytical stability threshold 2/eta.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure writable matplotlib cache in sandboxed environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import yaml

from experiments.common import (
    aggregate_seed_metrics,
    compute_run_metrics,
    interpolate_common_steps,
    save_experiment_json,
)
from experiments.generate_summary import generate_summary
from hesse.config import RunConfig
from hesse.train import train
from hesse.viz.style import (
    COLOR_ORANGE,
    save_figure,
    setup_style,
)


def run_e1() -> dict:
    """Execute Experiment 1 end-to-end across all learning rates and seeds."""
    t0 = time.perf_counter()
    config_file = Path("configs/experiments/e1_gd_eos.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    lrs = cfg_dict.get("learning_rates", [0.01, 0.02, 0.05, 0.10])
    seeds = cfg_dict.get("seeds", [42, 43, 44])
    base_raw = cfg_dict.get("base_config", {})

    print("=== Starting E1: GD Edge of Stability (4 LRs x 3 seeds) ===")
    runs_base_dir = Path("runs/experiments/e1_gd_eos")
    runs_base_dir.mkdir(parents=True, exist_ok=True)

    # Color shades for learning rates (blue gradient)
    lr_colors = {
        0.01: "#79C0FF",
        0.02: "#58A6FF",
        0.05: "#1F6FEB",
        0.10: "#0D419D",
    }

    all_sweep_data = {}

    for lr in lrs:
        thresh = 2.0 / lr
        print(f"\n--- Running GD with eta = {lr} (Threshold 2/eta = {thresh:.1f}) ---")
        seed_runs = []

        for seed in seeds:
            run_name = str(runs_base_dir / f"gd_lr_{lr}_s{seed}")
            if Path(run_name).exists():
                shutil.rmtree(run_name)

            run_cfg_dict = dict(base_raw)
            run_cfg_dict["run_name"] = run_name
            run_cfg_dict["seed"] = seed
            if "optim" not in run_cfg_dict:
                run_cfg_dict["optim"] = {}
            run_cfg_dict["optim"]["lr"] = lr
            run_cfg_dict["optim"]["optimizer"] = "gd"

            cfg = RunConfig.from_dict(run_cfg_dict)
            train(cfg)

            # Load metrics.jsonl
            metrics_file = Path(run_name) / "metrics.jsonl"
            records = []
            if metrics_file.exists():
                with open(metrics_file, "r", encoding="utf-8") as mf:
                    for line in mf:
                        if line.strip():
                            records.append(json.loads(line))

            run_m = compute_run_metrics(records, stability_threshold=thresh)
            seed_runs.append(run_m)

        agg = aggregate_seed_metrics(seed_runs)
        config_summary = {
            "config": {"lr": lr, "threshold": thresh, "optimizer": "gd", "loss_type": "mse"},
            "metrics": agg,
            "seed_runs": [
                {
                    "seed": s,
                    "final_loss": r["final_loss"],
                    "plateau_ratio": r["plateau_ratio"],
                    "time_to_threshold": r["time_to_threshold"],
                    "non_monotone_loss_fraction": r["non_monotone_loss_fraction"],
                }
                for s, r in zip(seeds, seed_runs)
            ],
            "raw_series": seed_runs,
        }
        all_sweep_data[f"lr_{lr}"] = config_summary

        mean_p_ratio = agg.get("plateau_ratio", {}).get("mean")
        t_reach = agg.get("time_to_threshold", {}).get("mean")
        non_mono = agg.get("non_monotone_loss_fraction", {}).get("mean")
        print(
            f"  eta={lr} Result: Plateau Ratio={mean_p_ratio:.3f} | "
            f"Time to Threshold={t_reach} | Non-monotone Frac={non_mono:.3f}"
        )

    runtime_sec = time.perf_counter() - t0
    print(f"\nCompleted E1 in {runtime_sec:.2f}s total runtime.")

    # Save JSON results
    payload = {
        "experiment_id": "e1_gd_eos",
        "title": cfg_dict.get("title", "Experiment 1: GD Edge of Stability"),
        "description": cfg_dict.get("description", ""),
        "runtime_seconds": float(runtime_sec),
        "protocol": {
            "seeds": seeds,
            "learning_rates": lrs,
            "plateau_window": "t >= t_reach if reached, else t >= 0.6 * t_final",
            "metric_definitions": {
                "plateau_ratio": "mean(lambda_max / (2/eta)) over plateau window",
                "time_to_threshold": "first step t where lambda_max >= 2/eta",
                "non_monotone_loss_fraction": "fraction of transitions where L_t > L_{t-1}",
            },
        },
        "configurations": all_sweep_data,
    }
    json_path = save_experiment_json("e1_gd_eos", payload)
    print(f"Saved E1 raw results to {json_path}")

    # Generate Publication Figure: docs/figures/e1_gd_eos_trajectory.png
    setup_style()
    fig, (ax_loss, ax_sharp) = plt.subplots(1, 2, figsize=(15, 6.0), dpi=300)

    for lr in lrs:
        key = f"lr_{lr}"
        raw_runs = all_sweep_data[key]["raw_series"]
        col = lr_colors.get(lr, "#58A6FF")
        thresh = 2.0 / lr

        # 1. Loss trajectories
        l_steps, l_mean, l_std = interpolate_common_steps(
            raw_runs, val_key="losses", step_key="steps"
        )
        if len(l_steps) > 0:
            ax_loss.semilogy(l_steps, l_mean, color=col, linewidth=2.0, label=f"$\\eta = {lr}$")
            ax_loss.fill_between(
                l_steps,
                np.maximum(1e-6, l_mean - l_std),
                l_mean + l_std,
                color=col,
                alpha=0.18,
            )

        # 2. Sharpness trajectories with threshold
        s_steps, s_mean, s_std = interpolate_common_steps(
            raw_runs, val_key="sharp_vals", step_key="sharp_steps"
        )
        if len(s_steps) > 0:
            ax_sharp.plot(
                s_steps,
                s_mean,
                color=col,
                linewidth=2.0,
                label=f"$\\eta = {lr}$ ($\\lambda_{{\\max}}$)",
            )
            ax_sharp.fill_between(
                s_steps,
                np.maximum(0, s_mean - s_std),
                s_mean + s_std,
                color=col,
                alpha=0.18,
            )
            ax_sharp.axhline(thresh, color=COLOR_ORANGE, linestyle="--", linewidth=1.5, alpha=0.85)

    ax_loss.set_title(r"Full-Batch GD: Empirical Loss $\mathcal{L}(\theta_t)$ (Log Scale)")
    ax_loss.set_xlabel("Optimization Step $t$")
    ax_loss.set_ylabel(r"Loss $\mathcal{L}$")
    ax_loss.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_loss.legend(loc="upper right", framealpha=0.85)

    ax_sharp.set_title(r"Curvature Trajectory: $\lambda_{\max}(H_t)$ Hovering near $2/\eta$")
    ax_sharp.set_xlabel("Optimization Step $t$")
    ax_sharp.set_ylabel(r"Top Hessian Eigenvalue $\lambda_{\max}$")
    ax_sharp.grid(True, linestyle="--", alpha=0.3)
    ax_sharp.legend(loc="upper right", framealpha=0.85, fontsize=8.5)

    fig_out = Path("docs/figures/e1_gd_eos_trajectory.png")
    save_figure(fig, fig_out)
    print(f"Saved E1 figure to {fig_out}")

    generate_summary()
    return payload


if __name__ == "__main__":
    run_e1()
