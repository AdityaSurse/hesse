"""Experiment 6: Objective Geometry: MSE vs Cross-Entropy.

Protocol:
  - Compare one-hot MSE loss vs categorical Cross-Entropy on tanh MLP.
  - Full-batch GD with fixed eta = 0.04 (stability threshold 2/eta = 50.0).
  - 3 seeds per objective (42, 43, 44).
  - Track lambda_max and trace over 60 optimization steps.
  - Evaluates persistent hovering (MSE) vs late-stage curvature flattening (Cross-Entropy).
"""

from __future__ import annotations

import copy
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
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_ORANGE,
    save_figure,
    setup_style,
)


def run_e6() -> dict:
    """Execute Experiment 6 comparing MSE and Cross-Entropy curvature trajectories."""
    t0 = time.perf_counter()
    config_file = Path("configs/experiments/e6_objectives.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    loss_types = cfg_dict.get("loss_types", ["mse", "cross_entropy"])
    seeds = cfg_dict.get("seeds", [42, 43, 44])
    base_raw = cfg_dict.get("base_config", {})

    print("=== Starting E6: MSE vs Cross-Entropy Curvature Dynamics (2 Objectives x 3 Seeds) ===")
    runs_base_dir = Path("runs/experiments/e6_objectives")
    runs_base_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "mse": COLOR_BLUE,  # #58A6FF
        "cross_entropy": COLOR_GREEN,  # #3FB950
    }
    all_sweep_data = {}
    thresh = 50.0  # 2 / 0.04

    for lt in loss_types:
        print(f"\n--- Running Objective: {lt.upper()} (eta = 0.04, Thresh = {thresh:.1f}) ---")
        seed_runs = []

        for seed in seeds:
            run_name = str(runs_base_dir / f"obj_{lt}_s{seed}")
            if Path(run_name).exists():
                shutil.rmtree(run_name)

            run_cfg_dict = copy.deepcopy(base_raw)
            run_cfg_dict["run_name"] = run_name
            run_cfg_dict["seed"] = seed
            run_cfg_dict["data"]["loss_type"] = lt
            run_cfg_dict["optim"]["lr"] = 0.04

            cfg = RunConfig.from_dict(run_cfg_dict)
            train(cfg)

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
            "config": {
                "loss_type": lt,
                "optimizer": "gd",
                "lr": 0.04,
                "threshold": thresh,
            },
            "metrics": agg,
            "seed_runs": [
                {
                    "seed": s,
                    "final_loss": r["final_loss"],
                    "plateau_ratio": r["plateau_ratio"],
                    "final_sharpness": r["final_sharpness"],
                    "max_sharpness": r["max_sharpness"],
                    "non_monotone_loss_fraction": r["non_monotone_loss_fraction"],
                }
                for s, r in zip(seeds, seed_runs)
            ],
            "raw_series": seed_runs,
        }
        all_sweep_data[lt] = config_summary

        plat_r = agg.get("plateau_ratio", {}).get("mean")
        f_sharp = agg.get("final_sharpness", {}).get("mean")
        max_s = agg.get("max_sharpness", {}).get("mean")
        print(
            f"  {lt.upper()} -> Peak Sharpness: {max_s:.2f} | "
            f"Final Sharpness: {f_sharp:.2f} | Plateau Ratio: {plat_r:.3f}"
        )

    runtime_sec = time.perf_counter() - t0
    print(f"\nCompleted E6 in {runtime_sec:.2f}s total runtime.")

    # Save JSON results
    payload = {
        "experiment_id": "e6_objectives",
        "title": cfg_dict.get("title", "Experiment 6: Objective Geometry"),
        "description": cfg_dict.get("description", ""),
        "runtime_seconds": float(runtime_sec),
        "protocol": {
            "seeds": seeds,
            "loss_types": loss_types,
            "learning_rate": 0.04,
            "stability_threshold": thresh,
        },
        "configurations": all_sweep_data,
    }
    json_path = save_experiment_json("e6_objectives", payload)
    print(f"Saved E6 raw results to {json_path}")

    # Generate Publication Figure: docs/figures/e6_loss_function_spectral_density.png
    setup_style()
    fig, (ax_loss, ax_sharp) = plt.subplots(1, 2, figsize=(15, 6.0), dpi=300)

    for lt in loss_types:
        raw_runs = all_sweep_data[lt]["raw_series"]
        col = colors.get(lt, COLOR_BLUE)
        label_str = "MSE (One-Hot)" if lt == "mse" else "Categorical Cross-Entropy"

        # 1. Loss trajectories
        l_steps, l_mean, l_std = interpolate_common_steps(
            raw_runs, val_key="losses", step_key="steps"
        )
        if len(l_steps) > 0:
            ax_loss.semilogy(l_steps, l_mean, color=col, linewidth=2.0, label=label_str)
            ax_loss.fill_between(
                l_steps,
                np.maximum(1e-6, l_mean - l_std),
                l_mean + l_std,
                color=col,
                alpha=0.15,
            )

        # 2. Sharpness trajectories
        s_steps, s_mean, s_std = interpolate_common_steps(
            raw_runs, val_key="sharp_vals", step_key="sharp_steps"
        )
        if len(s_steps) > 0:
            ax_sharp.plot(s_steps, s_mean, color=col, linewidth=2.0, label=label_str)
            ax_sharp.fill_between(
                s_steps,
                np.maximum(0, s_mean - s_std),
                s_mean + s_std,
                color=col,
                alpha=0.15,
            )

    ax_sharp.axhline(
        thresh,
        color=COLOR_ORANGE,
        linestyle="--",
        linewidth=1.8,
        label=f"Stability Threshold $2/\\eta = {thresh:.1f}$",
    )

    ax_loss.set_title(r"Loss Decay Comparison: MSE vs Cross-Entropy (Log Scale)")
    ax_loss.set_xlabel("Optimization Step $t$")
    ax_loss.set_ylabel(r"Loss $\mathcal{L}$")
    ax_loss.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_loss.legend(loc="upper right", framealpha=0.85)

    ax_sharp.set_title(r"Persistent Hovering (MSE) vs Late-Time Flattening (Cross-Entropy)")
    ax_sharp.set_xlabel("Optimization Step $t$")
    ax_sharp.set_ylabel(r"Sharpness $\lambda_{\max}(H_t)$")
    ax_sharp.grid(True, linestyle="--", alpha=0.3)
    ax_sharp.legend(loc="upper left", framealpha=0.85)

    fig_out = Path("docs/figures/e6_loss_function_spectral_density.png")
    save_figure(fig, fig_out)
    print(f"Saved E6 figure to {fig_out}")

    generate_summary()
    return payload


if __name__ == "__main__":
    run_e6()
