"""Experiment 4: Lion Sign-Update Dynamics and Curvature Plateau (Open Question).

Protocol:
  - Lion optimizer (beta1=0.9, beta2=0.99) on CIFAR-10 subset across 3 learning rates:
    eta in {0.0001, 0.0003, 0.0010}.
  - Track plain sharpness lambda_max(H) and sign-directional curvature u_t^T H u_t / ||u_t||^2.
  - Test whether a stable plateau exists and whether it scales inversely with learning rate (1/eta).
  - 3 seeds per rate (42, 43, 44).
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
    COLOR_LION,
    save_figure,
    setup_style,
)


def run_e4() -> dict:
    """Execute Experiment 4 investigating Lion curvature dynamics and open questions."""
    t0 = time.perf_counter()
    config_file = Path("configs/experiments/e4_lion.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    lrs = cfg_dict.get("learning_rates", [0.0001, 0.0003, 0.0010])
    seeds = cfg_dict.get("seeds", [42, 43, 44])
    base_raw = cfg_dict.get("base_config", {})

    print("=== Starting E4: Lion Sign-Update Dynamics (3 LRs x 3 seeds) ===")
    runs_base_dir = Path("runs/experiments/e4_lion")
    runs_base_dir.mkdir(parents=True, exist_ok=True)

    colors = {0.0001: "#F778BA", 0.0003: "#DB61A2", 0.0010: "#B83A82"}
    all_sweep_data = {}

    for lr in lrs:
        cfg_key = f"lr_{lr}"
        print(f"\n--- Running Lion with eta = {lr} ---")
        seed_runs = []

        for seed in seeds:
            run_name = str(runs_base_dir / f"lion_lr_{lr}_s{seed}")
            if Path(run_name).exists():
                shutil.rmtree(run_name)

            run_cfg_dict = dict(base_raw)
            run_cfg_dict["run_name"] = run_name
            run_cfg_dict["seed"] = seed
            run_cfg_dict["optim"]["optimizer"] = "lion"
            run_cfg_dict["optim"]["lr"] = lr

            cfg = RunConfig.from_dict(run_cfg_dict)
            train(cfg)

            metrics_file = Path(run_name) / "metrics.jsonl"
            records = []
            if metrics_file.exists():
                with open(metrics_file, "r", encoding="utf-8") as mf:
                    for line in mf:
                        if line.strip():
                            records.append(json.loads(line))

            # Lion has no 2/lr threshold; stability_threshold is None
            run_m = compute_run_metrics(records, stability_threshold=None)
            seed_runs.append(run_m)

        agg = aggregate_seed_metrics(seed_runs)
        config_summary = {
            "config": {
                "optimizer": "lion",
                "lr": lr,
                "beta1": 0.9,
                "beta2": 0.99,
            },
            "metrics": agg,
            "seed_runs": [
                {
                    "seed": s,
                    "final_loss": r["final_loss"],
                    "mean_sign_curvature": r["mean_sign_curvature"],
                    "final_sharpness": r["final_sharpness"],
                    "non_monotone_loss_fraction": r["non_monotone_loss_fraction"],
                }
                for s, r in zip(seeds, seed_runs)
            ],
            "raw_series": seed_runs,
        }
        all_sweep_data[cfg_key] = config_summary

        sign_curv_m = agg.get("mean_sign_curvature", {}).get("mean")
        raw_sharp_m = agg.get("final_sharpness", {}).get("mean")
        non_mono = agg.get("non_monotone_loss_fraction", {}).get("mean")
        print(
            f"  eta={lr} -> Mean Sign Curvature: {sign_curv_m:.3f} | "
            f"Plain Sharpness: {raw_sharp_m:.2f} | Non-monotone: {non_mono:.3f}"
        )

    runtime_sec = time.perf_counter() - t0
    print(f"\nCompleted E4 in {runtime_sec:.2f}s total runtime.")

    # Save JSON results
    payload = {
        "experiment_id": "e4_lion",
        "title": cfg_dict.get("title", "Experiment 4: Lion Sign-Update Dynamics"),
        "description": cfg_dict.get("description", ""),
        "runtime_seconds": float(runtime_sec),
        "protocol": {
            "seeds": seeds,
            "learning_rates": lrs,
            "notes": "Sign-directional curvature evaluated along u_t = sign(c_t).",
        },
        "configurations": all_sweep_data,
    }
    json_path = save_experiment_json("e4_lion", payload)
    print(f"Saved E4 raw results to {json_path}")

    # Generate Publication Figure: docs/figures/e4_lion_sign_dynamics.png
    setup_style()
    fig, (ax_sign, ax_raw) = plt.subplots(1, 2, figsize=(15, 6.0), dpi=300)

    for lr in lrs:
        cfg_key = f"lr_{lr}"
        raw_runs = all_sweep_data[cfg_key]["raw_series"]
        col = colors.get(lr, COLOR_LION)

        # 1. Sign-directional curvature
        u_steps, u_mean, u_std = interpolate_common_steps(
            raw_runs, val_key="sign_vals", step_key="sign_steps"
        )
        if len(u_steps) > 0:
            ax_sign.plot(u_steps, u_mean, color=col, linewidth=2.0, label=f"$\\eta = {lr}$")
            ax_sign.fill_between(u_steps, u_mean - u_std, u_mean + u_std, color=col, alpha=0.15)

        # 2. Raw sharpness
        s_steps, s_mean, s_std = interpolate_common_steps(
            raw_runs, val_key="sharp_vals", step_key="sharp_steps"
        )
        if len(s_steps) > 0:
            ax_raw.plot(s_steps, s_mean, color=col, linewidth=2.0, label=f"$\\eta = {lr}$")
            ax_raw.fill_between(s_steps, s_mean - s_std, s_mean + s_std, color=col, alpha=0.15)

    ax_sign.set_title(r"Lion Sign-Directional Curvature $\frac{u_t^T H u_t}{\|u_t\|_2^2}$")
    ax_sign.set_xlabel("Optimization Step $t$")
    ax_sign.set_ylabel(r"Sign Directional Curvature $\kappa_{\mathrm{sign}}$")
    ax_sign.grid(True, linestyle="--", alpha=0.3)
    ax_sign.legend(loc="upper right", framealpha=0.85)

    ax_raw.set_title(r"Lion Unconstrained Hessian Sharpness $\lambda_{\max}(H)$")
    ax_raw.set_xlabel("Optimization Step $t$")
    ax_raw.set_ylabel(r"Top Eigenvalue $\lambda_{\max}$")
    ax_raw.grid(True, linestyle="--", alpha=0.3)
    ax_raw.legend(loc="upper left", framealpha=0.85)

    fig_out = Path("docs/figures/e4_lion_sign_dynamics.png")
    save_figure(fig, fig_out)
    print(f"Saved E4 figure to {fig_out}")

    generate_summary()
    return payload


if __name__ == "__main__":
    run_e4()
