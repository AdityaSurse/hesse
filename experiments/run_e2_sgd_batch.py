"""Experiment 2: SGD Batch Size Scaling and Curvature Noise.

Protocol:
  - Tanh MLP with MSE loss on CIFAR-10 subset (500 samples).
  - Fixed learning rate eta = 0.05 (threshold 2/eta = 40.0).
  - Minibatch sizes B in {32, 128, 512, full (500)}.
  - 3 seeds per batch size (42, 43, 44).
  - Curvature evaluated on a FIXED full evaluation batch across all batch sizes.
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
    COLOR_ORANGE,
    save_figure,
    setup_style,
)


def run_e2() -> dict:
    """Execute Experiment 2 across batch sizes and seeds."""
    t0 = time.perf_counter()
    config_file = Path("configs/experiments/e2_sgd_batch.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    batch_sizes = cfg_dict.get("batch_sizes", [32, 128, 512, None])
    seeds = cfg_dict.get("seeds", [42, 43, 44])
    base_raw = cfg_dict.get("base_config", {})

    print("=== Starting E2: SGD Batch Size Scaling (4 batch sizes x 3 seeds) ===")
    runs_base_dir = Path("runs/experiments/e2_sgd_batch")
    runs_base_dir.mkdir(parents=True, exist_ok=True)

    bs_colors = {
        32: "#D29922",  # Yellow/amber (noisy)
        128: "#3FB950",  # Light green
        512: "#2EA043",  # Deep green
        None: "#58A6FF",  # Blue (full-batch)
    }

    all_sweep_data = {}
    thresh = 40.0  # 2 / 0.05

    for bs in batch_sizes:
        bs_label = "full" if bs is None else str(bs)
        print(f"\n--- Running SGD with Batch Size B = {bs_label} ---")
        seed_runs = []

        for seed in seeds:
            run_name = str(runs_base_dir / f"sgd_bs_{bs_label}_s{seed}")
            if Path(run_name).exists():
                shutil.rmtree(run_name)

            run_cfg_dict = dict(base_raw)
            run_cfg_dict["run_name"] = run_name
            run_cfg_dict["seed"] = seed
            run_cfg_dict["data"]["batch_size"] = bs
            run_cfg_dict["optim"]["optimizer"] = "sgd" if bs is not None else "gd"
            run_cfg_dict["optim"]["lr"] = 0.05

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
            "config": {"batch_size": bs, "lr": 0.05, "threshold": thresh, "optimizer": "sgd"},
            "metrics": agg,
            "seed_runs": [
                {
                    "seed": s,
                    "final_loss": r["final_loss"],
                    "plateau_ratio": r["plateau_ratio"],
                    "time_to_threshold": r["time_to_threshold"],
                    "non_monotone_loss_fraction": r["non_monotone_loss_fraction"],
                    "final_sharpness": r["final_sharpness"],
                }
                for s, r in zip(seeds, seed_runs)
            ],
            "raw_series": seed_runs,
        }
        all_sweep_data[f"bs_{bs_label}"] = config_summary

        mean_p_ratio = agg.get("plateau_ratio", {}).get("mean")
        t_reach = agg.get("time_to_threshold", {}).get("mean")
        final_s = agg.get("final_sharpness", {}).get("mean")
        print(
            f"  B={bs_label} Result: Plateau Ratio={mean_p_ratio if mean_p_ratio else 'N/A'} | "
            f"Final Sharpness={final_s:.2f} | Time to Thresh={t_reach}"
        )

    runtime_sec = time.perf_counter() - t0
    print(f"\nCompleted E2 in {runtime_sec:.2f}s total runtime.")

    # Save JSON results
    payload = {
        "experiment_id": "e2_sgd_batch",
        "title": cfg_dict.get("title", "Experiment 2: SGD Batch Size Scaling"),
        "description": cfg_dict.get("description", ""),
        "runtime_seconds": float(runtime_sec),
        "protocol": {
            "seeds": seeds,
            "batch_sizes": [32, 128, 512, "full"],
            "learning_rate": 0.05,
            "stability_threshold": thresh,
        },
        "configurations": all_sweep_data,
    }
    json_path = save_experiment_json("e2_sgd_batch", payload)
    print(f"Saved E2 raw results to {json_path}")

    # Generate Publication Figure: docs/figures/e2_sgd_batch_size_scaling.png
    setup_style()
    fig, (ax_loss, ax_sharp) = plt.subplots(1, 2, figsize=(15, 6.0), dpi=300)

    for bs in batch_sizes:
        bs_label = "full" if bs is None else str(bs)
        key = f"bs_{bs_label}"
        raw_runs = all_sweep_data[key]["raw_series"]
        col = bs_colors.get(bs, "#3FB950")
        name_str = f"Batch B = {bs_label.capitalize()}"

        # Loss trajectories
        l_steps, l_mean, l_std = interpolate_common_steps(
            raw_runs, val_key="losses", step_key="steps"
        )
        if len(l_steps) > 0:
            ax_loss.semilogy(l_steps, l_mean, color=col, linewidth=2.0, label=name_str)
            ax_loss.fill_between(
                l_steps,
                np.maximum(1e-6, l_mean - l_std),
                l_mean + l_std,
                color=col,
                alpha=0.15,
            )

        # Sharpness trajectories
        s_steps, s_mean, s_std = interpolate_common_steps(
            raw_runs, val_key="sharp_vals", step_key="sharp_steps"
        )
        if len(s_steps) > 0:
            ax_sharp.plot(s_steps, s_mean, color=col, linewidth=2.0, label=name_str)
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
        label=f"Full-Batch Threshold $2/\\eta = {thresh:.1f}$",
    )

    ax_loss.set_title(r"SGD Loss Decay across Minibatch Sizes $B$ (Log Scale)")
    ax_loss.set_xlabel("Optimization Step $t$")
    ax_loss.set_ylabel(r"Loss $\mathcal{L}$")
    ax_loss.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_loss.legend(loc="upper right", framealpha=0.85)

    ax_sharp.set_title(r"Curvature Dissipation: Small Batches Suppress $\lambda_{\max} < 2/\eta$")
    ax_sharp.set_xlabel("Optimization Step $t$")
    ax_sharp.set_ylabel(r"Top Hessian Eigenvalue $\lambda_{\max}$")
    ax_sharp.grid(True, linestyle="--", alpha=0.3)
    ax_sharp.legend(loc="upper left", framealpha=0.85, fontsize=8.5)

    fig_out = Path("docs/figures/e2_sgd_batch_size_scaling.png")
    save_figure(fig, fig_out)
    print(f"Saved E2 figure to {fig_out}")

    generate_summary()
    return payload


if __name__ == "__main__":
    run_e2()
