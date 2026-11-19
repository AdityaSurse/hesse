"""Experiment 3: Adam Preconditioned Sharpness vs Stability Threshold.

Protocol:
  - Adam optimizer on MLP across learning rates in {0.001, 0.003, 0.010} and beta1 in {0.9, 0.0}.
  - Track raw Hessian sharpness lambda_max(H) and preconditioned sharpness lambda_max(P_t^-1 H).
  - Benchmark against theoretical threshold (2 + 2*beta1) / ((1 - beta1) * eta).
  - 3 seeds per configuration (42, 43, 44).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    COLOR_ADAM,
    COLOR_ORANGE,
    save_figure,
    setup_style,
)


def run_e3() -> dict:
    """Execute Experiment 3 for Adam curvature dynamics across sweeps and seeds."""
    t0 = time.perf_counter()
    config_file = Path("configs/experiments/e3_adam_eos.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    lrs = cfg_dict.get("learning_rates", [0.001, 0.003, 0.010])
    beta1_vals = cfg_dict.get("beta1_values", [0.9, 0.0])
    seeds = cfg_dict.get("seeds", [42, 43, 44])
    base_raw = cfg_dict.get("base_config", {})

    print("=== Starting E3: Adam Preconditioned Sharpness (3 LRs x 2 beta1 x 3 seeds) ===")
    runs_base_dir = Path("runs/experiments/e3_adam_eos")
    runs_base_dir.mkdir(parents=True, exist_ok=True)

    all_sweep_data = {}

    for b1 in beta1_vals:
        for lr in lrs:
            thresh = (2.0 + 2.0 * b1) / ((1.0 - b1) * lr)
            cfg_key = f"b1_{b1}_lr_{lr}"
            print(f"\n--- Running Adam (beta1 = {b1}, eta = {lr}, Thresh = {thresh:.1f}) ---")
            seed_runs = []

            for seed in seeds:
                run_name = str(runs_base_dir / f"adam_b1_{b1}_lr_{lr}_s{seed}")
                if Path(run_name).exists():
                    shutil.rmtree(run_name)

                run_cfg_dict = dict(base_raw)
                run_cfg_dict["run_name"] = run_name
                run_cfg_dict["seed"] = seed
                run_cfg_dict["optim"]["optimizer"] = "adam"
                run_cfg_dict["optim"]["lr"] = lr
                run_cfg_dict["optim"]["beta1"] = b1

                cfg = RunConfig.from_dict(run_cfg_dict)
                train(cfg)

                metrics_file = Path(run_name) / "metrics.jsonl"
                records = []
                if metrics_file.exists():
                    with open(metrics_file, "r", encoding="utf-8") as mf:
                        for line in mf:
                            if line.strip():
                                records.append(json.loads(line))

                # For Adam, use preconditioned sharpness for stability threshold benchmarking
                run_m = compute_run_metrics(
                    records,
                    stability_threshold=thresh,
                    use_preconditioned_for_threshold=True,
                )
                seed_runs.append(run_m)

            agg = aggregate_seed_metrics(seed_runs)
            config_summary = {
                "config": {
                    "optimizer": "adam",
                    "beta1": b1,
                    "lr": lr,
                    "threshold": thresh,
                },
                "metrics": agg,
                "seed_runs": [
                    {
                        "seed": s,
                        "final_loss": r["final_loss"],
                        "plateau_ratio": r["plateau_ratio"],
                        "time_to_threshold": r["time_to_threshold"],
                        "final_sharpness": r["final_sharpness"],
                    }
                    for s, r in zip(seeds, seed_runs)
                ],
                "raw_series": seed_runs,
            }
            all_sweep_data[cfg_key] = config_summary

            plat_r = agg.get("plateau_ratio", {}).get("mean")
            f_sharp = agg.get("final_sharpness", {}).get("mean")
            print(
                f"  b1={b1}, lr={lr} -> Preconditioned Plateau Ratio: {plat_r:.3f} | "
                f"Plain Sharpness: {f_sharp:.2f}"
            )

    runtime_sec = time.perf_counter() - t0
    print(f"\nCompleted E3 in {runtime_sec:.2f}s total runtime.")

    # Save JSON results
    payload = {
        "experiment_id": "e3_adam_eos",
        "title": cfg_dict.get("title", "Experiment 3: Adam Preconditioned Sharpness"),
        "description": cfg_dict.get("description", ""),
        "runtime_seconds": float(runtime_sec),
        "protocol": {
            "seeds": seeds,
            "learning_rates": lrs,
            "beta1_values": beta1_vals,
        },
        "configurations": all_sweep_data,
    }
    json_path = save_experiment_json("e3_adam_eos", payload)
    print(f"Saved E3 raw results to {json_path}")

    # Generate Publication Figure: docs/figures/e3_adam_preconditioned_sharpness.png
    setup_style()
    fig, (ax_raw, ax_precond) = plt.subplots(1, 2, figsize=(15, 6.0), dpi=300)

    # Plot representative curves for beta1 = 0.9 across learning rates
    colors = {0.001: "#D2A8FF", 0.003: "#A371F7", 0.010: "#8957E5"}

    for lr in lrs:
        cfg_key = f"b1_0.9_lr_{lr}"
        raw_runs = all_sweep_data[cfg_key]["raw_series"]
        col = colors.get(lr, COLOR_ADAM)
        thresh = (2.0 + 1.8) / (0.1 * lr)  # 38 / lr

        # 1. Plain sharpness
        s_steps, s_mean, s_std = interpolate_common_steps(
            raw_runs, val_key="sharp_vals", step_key="sharp_steps"
        )
        if len(s_steps) > 0:
            ax_raw.semilogy(s_steps, s_mean, color=col, linewidth=2.0, label=f"$\\eta = {lr}$")
            ax_raw.fill_between(
                s_steps,
                np.maximum(1e-3, s_mean - s_std),
                s_mean + s_std,
                color=col,
                alpha=0.15,
            )

        # 2. Preconditioned sharpness vs threshold
        p_steps, p_mean, p_std = interpolate_common_steps(
            raw_runs, val_key="precond_vals", step_key="precond_steps"
        )
        if len(p_steps) > 0:
            ax_precond.plot(
                p_steps,
                p_mean,
                color=col,
                linewidth=2.0,
                label=f"$\\eta = {lr}$ ($\\lambda_{{precond}}$)",
            )
            ax_precond.fill_between(
                p_steps,
                np.maximum(0, p_mean - p_std),
                p_mean + p_std,
                color=col,
                alpha=0.15,
            )
            ax_precond.axhline(
                thresh, color=COLOR_ORANGE, linestyle="--", linewidth=1.5, alpha=0.85
            )

    ax_raw.set_title(r"Raw Hessian Sharpness $\lambda_{\max}(H)$ (Does Not Track $2/\eta$)")
    ax_raw.set_xlabel("Optimization Step $t$")
    ax_raw.set_ylabel(r"Raw Sharpness $\lambda_{\max}$ (Log Scale)")
    ax_raw.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_raw.legend(loc="best", framealpha=0.85)

    ax_precond.set_title(r"Preconditioned Sharpness $\lambda_{\max}(P_t^{-1} H)$ Tracks $38/\eta$")
    ax_precond.set_xlabel("Optimization Step $t$")
    ax_precond.set_ylabel(r"Preconditioned Sharpness $\lambda_{\max}(P_t^{-1} H)$")
    ax_precond.grid(True, linestyle="--", alpha=0.3)
    ax_precond.legend(loc="upper right", framealpha=0.85, fontsize=8.5)

    fig_out = Path("docs/figures/e3_adam_preconditioned_sharpness.png")
    save_figure(fig, fig_out)
    print(f"Saved E3 figure to {fig_out}")

    generate_summary()
    return payload


if __name__ == "__main__":
    run_e3()
