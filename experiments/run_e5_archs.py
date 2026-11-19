"""Experiment 5: Architectural Curvature Comparison (MLP vs CNN vs ViT).

Protocol:
  - Three standardized architectures: MLP, SmallCNN, TinyTransformer.
  - Two optimizers: Full-batch GD (eta=0.04, threshold 2/eta=50.0) and Adam (eta=0.001).
  - Track lambda_max, top-5 eigenvalues, and trace.
  - Measure spectral gap (lambda_1 - lambda_2) to evaluate outlier dominance vs continuous bulk.
  - 3 seeds per architecture (42, 43, 44).
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
    COLOR_ORANGE,
    save_figure,
    setup_style,
)


def run_e5() -> dict:
    """Execute Experiment 5 comparing MLP, CNN, and Transformer curvature dynamics."""
    t0 = time.perf_counter()
    config_file = Path("configs/experiments/e5_archs.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    architectures = cfg_dict.get("architectures", {})
    optimizers = cfg_dict.get("optimizers", [{"optimizer": "gd", "lr": 0.04}])
    seeds = cfg_dict.get("seeds", [42, 43, 44])
    base_raw = cfg_dict.get("base_config", {})

    print("=== Starting E5: Architectural Curvature Comparison (3 Archs x 2 Optims x 3 Seeds) ===")
    runs_base_dir = Path("runs/experiments/e5_archs")
    runs_base_dir.mkdir(parents=True, exist_ok=True)

    arch_colors = {
        "mlp": "#58A6FF",  # Blue
        "cnn": "#3FB950",  # Green
        "transformer": "#A371F7",  # Purple
    }

    all_sweep_data = {}

    for opt_cfg in optimizers:
        opt_name = opt_cfg["optimizer"]
        lr = opt_cfg["lr"]
        thresh = 2.0 / lr if opt_name == "gd" else 38000.0

        for arch_name, arch_spec in architectures.items():
            cfg_key = f"{arch_name}_{opt_name}"
            msg = f"\n--- Architecture: {arch_name.upper()} with {opt_name.upper()} (eta={lr}) ---"
            print(msg)
            seed_runs = []

            for seed in seeds:
                run_name = str(runs_base_dir / f"{arch_name}_{opt_name}_s{seed}")
                if Path(run_name).exists():
                    shutil.rmtree(run_name)

                run_cfg_dict = copy.deepcopy(base_raw)
                run_cfg_dict["run_name"] = run_name
                run_cfg_dict["seed"] = seed
                run_cfg_dict["model"] = dict(arch_spec)
                run_cfg_dict["optim"]["optimizer"] = opt_name
                run_cfg_dict["optim"]["lr"] = lr
                if opt_name == "adam":
                    run_cfg_dict["data"]["batch_size"] = 128
                else:
                    run_cfg_dict["data"]["batch_size"] = None

                cfg = RunConfig.from_dict(run_cfg_dict)
                train(cfg)

                metrics_file = Path(run_name) / "metrics.jsonl"
                records = []
                if metrics_file.exists():
                    with open(metrics_file, "r", encoding="utf-8") as mf:
                        for line in mf:
                            if line.strip():
                                records.append(json.loads(line))

                run_m = compute_run_metrics(
                    records,
                    stability_threshold=thresh,
                    use_preconditioned_for_threshold=(opt_name == "adam"),
                )
                seed_runs.append(run_m)

            agg = aggregate_seed_metrics(seed_runs)
            config_summary = {
                "config": {
                    "architecture": arch_name,
                    "optimizer": opt_name,
                    "lr": lr,
                    "threshold": thresh,
                },
                "metrics": agg,
                "seed_runs": [
                    {
                        "seed": s,
                        "final_loss": r["final_loss"],
                        "plateau_ratio": r["plateau_ratio"],
                        "mean_spectral_gap": r["mean_spectral_gap"],
                        "mean_trace": r["mean_trace"],
                    }
                    for s, r in zip(seeds, seed_runs)
                ],
                "raw_series": seed_runs,
            }
            all_sweep_data[cfg_key] = config_summary

            plat_r = agg.get("plateau_ratio", {}).get("mean")
            gap_m = agg.get("mean_spectral_gap", {}).get("mean")
            trace_m = agg.get("mean_trace", {}).get("mean")
            print(
                f"  {arch_name.upper()}-{opt_name.upper()} -> Plateau Ratio: {plat_r:.3f} | "
                f"Spectral Gap: {gap_m:.2f} | Trace: {trace_m:.1f}"
            )

    runtime_sec = time.perf_counter() - t0
    print(f"\nCompleted E5 in {runtime_sec:.2f}s total runtime.")

    # Save JSON results
    payload = {
        "experiment_id": "e5_archs",
        "title": cfg_dict.get("title", "Experiment 5: Architectural Comparison"),
        "description": cfg_dict.get("description", ""),
        "runtime_seconds": float(runtime_sec),
        "protocol": {
            "seeds": seeds,
            "architectures": list(architectures.keys()),
            "optimizers": optimizers,
        },
        "configurations": all_sweep_data,
    }
    json_path = save_experiment_json("e5_archs", payload)
    print(f"Saved E5 raw results to {json_path}")

    # Generate Publication Figure: docs/figures/e5_architecture_curvature_comparison.png
    setup_style()
    fig, (ax_gd, ax_gap) = plt.subplots(1, 2, figsize=(15, 6.0), dpi=300)
    thresh_gd = 50.0  # 2 / 0.04

    for arch_name in architectures.keys():
        cfg_key = f"{arch_name}_gd"
        raw_runs = all_sweep_data[cfg_key]["raw_series"]
        col = arch_colors.get(arch_name, "#58A6FF")
        label_str = arch_name.upper()

        # 1. GD Sharpness curves
        s_steps, s_mean, s_std = interpolate_common_steps(
            raw_runs, val_key="sharp_vals", step_key="sharp_steps"
        )
        if len(s_steps) > 0:
            ax_gd.plot(s_steps, s_mean, color=col, linewidth=2.0, label=label_str)
            ax_gd.fill_between(
                s_steps,
                np.maximum(0, s_mean - s_std),
                s_mean + s_std,
                color=col,
                alpha=0.15,
            )

    ax_gd.axhline(
        thresh_gd,
        color=COLOR_ORANGE,
        linestyle="--",
        linewidth=1.8,
        label=f"Linear Stability Limit $2/\\eta = {thresh_gd:.1f}$",
    )

    ax_gd.set_title(r"Full-Batch GD Curvature Trajectories ($\eta = 0.04$)")
    ax_gd.set_xlabel("Optimization Step $t$")
    ax_gd.set_ylabel(r"Sharpness $\lambda_{\max}(H_t)$")
    ax_gd.grid(True, linestyle="--", alpha=0.3)
    ax_gd.legend(loc="upper left", framealpha=0.85)

    # 2. Spectral Gap comparison across architectures under GD
    arch_labels = [a.upper() for a in architectures.keys()]
    gaps_mean = [
        all_sweep_data[f"{a}_gd"]["metrics"]["mean_spectral_gap"]["mean"]
        for a in architectures.keys()
    ]
    gaps_std = [
        all_sweep_data[f"{a}_gd"]["metrics"]["mean_spectral_gap"]["std"]
        for a in architectures.keys()
    ]
    bar_cols = [arch_colors[a] for a in architectures.keys()]

    bars = ax_gap.bar(
        arch_labels,
        gaps_mean,
        yerr=gaps_std,
        capsize=6,
        color=bar_cols,
        alpha=0.85,
        edgecolor="#E6EDF3",
        linewidth=1.2,
    )
    ax_gap.set_title(r"Outlier Spectral Gap $\lambda_1 - \lambda_2$ across Architectures")
    ax_gap.set_ylabel(r"Mean Spectral Gap $\lambda_1 - \lambda_2$")
    ax_gap.grid(True, axis="y", linestyle="--", alpha=0.3)

    for bar, val in zip(bars, gaps_mean):
        ax_gap.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() * 1.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            color="#E6EDF3",
            fontweight="bold",
            fontsize=10,
        )

    fig_out = Path("docs/figures/e5_architecture_curvature_comparison.png")
    save_figure(fig, fig_out)
    print(f"Saved E5 figure to {fig_out}")

    generate_summary()
    return payload


if __name__ == "__main__":
    run_e5()
