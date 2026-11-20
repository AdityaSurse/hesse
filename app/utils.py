"""Helper utilities, loaders, and Plotly builders for the hesse Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import yaml

from hesse.viz.style import (
    COLOR_ADAM,
    COLOR_BLUE,
    COLOR_GD,
    COLOR_LION,
    COLOR_PURPLE,
    COLOR_SGD,
    COLOR_THRESHOLD,
)


def get_available_runs(base_dir: str | Path = "runs") -> List[str]:
    """Scan base_dir for all runs containing metrics.jsonl or config.yaml.

    Returns relative string paths sorted by directory modification time (newest first).
    """
    b_path = Path(base_dir)
    if not b_path.exists():
        return []

    runs: List[Path] = []
    # Search top-level runs and nested experiment directories
    for p in b_path.rglob("*"):
        if p.is_dir():
            if (p / "metrics.jsonl").exists() or (p / "config.yaml").exists():
                runs.append(p)

    # Sort by mtime descending
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p.relative_to(b_path)) for p in runs]


def load_run_data(run_path_str: str, base_dir: str | Path = "runs") -> Dict[str, Any]:
    """Safely load run configuration, metrics, summary, landscape, and density artifacts.

    Guarantees no unhandled exceptions on partial or empty runs.
    """
    p = Path(run_path_str)
    if not p.is_absolute() and not p.exists():
        p = Path(base_dir) / p

    if not p.exists():
        return {
            "exists": False,
            "path": p,
            "config": {},
            "records": [],
            "summary": {},
            "landscape_grids": {},
            "density_steps": {},
        }

    config: Dict[str, Any] = {}
    config_file = p / "config.yaml"
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            config = {}

    summary: Dict[str, Any] = {}
    summary_file = p / "summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                summary = json.load(f) or {}
        except Exception:
            summary = {}

    records: List[Dict[str, Any]] = []
    metrics_file = p / "metrics.jsonl"
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        try:
                            records.append(json.loads(s))
                        except Exception:
                            continue
        except Exception:
            records = []

    # Landscape grids
    landscape_grids: Dict[str, Path] = {}
    landscape_dir = p / "landscape"
    if landscape_dir.exists():
        for npz_file in sorted(landscape_dir.glob("grid_*.npz")):
            basis = npz_file.stem.replace("grid_", "")
            landscape_grids[basis] = npz_file

    # Density files
    density_steps: Dict[int, Path] = {}
    density_dir = p / "density"
    if density_dir.exists():
        for npz_file in sorted(density_dir.glob("density_step_*.npz")):
            try:
                step_str = npz_file.stem.replace("density_step_", "")
                step_val = int(step_str)
                density_steps[step_val] = npz_file
            except ValueError:
                continue

    return {
        "exists": True,
        "path": p,
        "name": config.get("run_name", p.name),
        "config": config,
        "records": records,
        "summary": summary,
        "landscape_grids": landscape_grids,
        "density_steps": density_steps,
    }


def get_optimizer_color(opt_name: str | None) -> str:
    """Return canonical hex color string matching project palette for given optimizer."""
    if not opt_name:
        return "#E6EDF3"
    op = opt_name.lower().strip()
    if op == "gd":
        return COLOR_GD
    if op == "sgd":
        return COLOR_SGD
    if op in {"adam", "adamw"}:
        return COLOR_ADAM
    if op == "lion":
        return COLOR_LION
    return COLOR_BLUE


def build_comparison_charts(
    selected_runs: List[str], base_dir: str | Path = "runs"
) -> Tuple[go.Figure, go.Figure, List[Dict[str, Any]]]:
    """Build normalized sharpness comparison (lambda_max / threshold) and loss curves."""
    fig_sharp = go.Figure()
    fig_loss = go.Figure()
    table_rows: List[Dict[str, Any]] = []

    # Horizontal boundary line for sharpness ratio = 1.0 (Edge of Stability)
    fig_sharp.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[1.0, 1.0],
            mode="lines",
            name="Stability Boundary (λ_max = Threshold)",
            line=dict(color=COLOR_THRESHOLD, width=2.0, dash="dash"),
        )
    )

    max_step_global = 10

    for run_key in selected_runs:
        data = load_run_data(run_key, base_dir=base_dir)
        records = data["records"]
        config = data["config"]
        if not records:
            continue

        opt_name = config.get("optim", {}).get("optimizer", "gd")
        lr = config.get("optim", {}).get("lr", 0.0)
        col = get_optimizer_color(opt_name)

        steps = [r["step"] for r in records]
        losses = [r["loss"] for r in records]
        max_step_global = max(max_step_global, max(steps) if steps else 10)

        # Sharpness steps and ratios
        sharp_steps = []
        ratios = []
        for r in records:
            s_val = r.get("sharpness")
            thresh = r.get("stability_threshold")
            if s_val is not None and thresh and thresh > 0:
                sharp_steps.append(r["step"])
                ratios.append(float(s_val) / float(thresh))

        label_name = f"{data['name']} ({opt_name.upper()}, η={lr})"

        if sharp_steps:
            fig_sharp.add_trace(
                go.Scatter(
                    x=sharp_steps,
                    y=ratios,
                    mode="lines+markers",
                    name=label_name,
                    line=dict(color=col, width=2.2),
                    marker=dict(size=4.5, color=col),
                )
            )

        if steps:
            fig_loss.add_trace(
                go.Scatter(
                    x=steps,
                    y=losses,
                    mode="lines",
                    name=label_name,
                    line=dict(color=col, width=2.0),
                )
            )

        # Metrics for summary table
        final_loss = losses[-1] if losses else None
        peak_sharp = max([r["sharpness"] for r in records if r.get("sharpness")], default=None)
        avg_ratio = np.mean(ratios) if ratios else None

        table_rows.append(
            {
                "Run": data["name"],
                "Optimizer": opt_name.upper(),
                "Learning Rate": lr,
                "Steps": steps[-1] if steps else 0,
                "Final Loss": f"{final_loss:.4f}" if final_loss is not None else "N/A",
                "Peak Sharpness": f"{peak_sharp:.2f}" if peak_sharp is not None else "N/A",
                "Mean Plateau Ratio": f"{avg_ratio:.3f}" if avg_ratio is not None else "N/A",
            }
        )

    # Update sharp chart range
    fig_sharp.update_xaxes(
        title_text="Optimization Step t",
        range=[0, max_step_global * 1.02],
        gridcolor="#30363D",
        zerolinecolor="#30363D",
    )
    fig_sharp.update_yaxes(
        title_text="Normalized Curvature (λ_max ÷ Stability Threshold)",
        gridcolor="#30363D",
        zerolinecolor="#30363D",
    )
    fig_sharp.update_layout(
        title=dict(
            text="Normalized Curvature Trajectories vs Edge-of-Stability Limit",
            font=dict(color="#E6EDF3", size=15),
        ),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        font=dict(color="#E6EDF3"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(22, 27, 34, 0.8)",
        ),
        margin=dict(l=60, r=30, b=50, t=60),
        height=480,
    )

    fig_loss.update_xaxes(
        title_text="Optimization Step t",
        gridcolor="#30363D",
        zerolinecolor="#30363D",
    )
    fig_loss.update_yaxes(
        title_text="Empirical Loss L(θ) (log scale)",
        type="log",
        gridcolor="#30363D",
        zerolinecolor="#30363D",
    )
    fig_loss.update_layout(
        title=dict(
            text="Empirical Loss Descent Trajectories",
            font=dict(color="#E6EDF3", size=15),
        ),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        font=dict(color="#E6EDF3"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(22, 27, 34, 0.8)",
        ),
        margin=dict(l=60, r=30, b=50, t=60),
        height=420,
    )

    return fig_sharp, fig_loss, table_rows


def build_density_plot(
    grid: np.ndarray, density: np.ndarray, step: int, log_scale: bool = True
) -> go.Figure:
    """Build Plotly spectral density figure with log-scaled density axis."""
    fig = go.Figure()

    dens_clipped = np.clip(density, 1e-6, None) if log_scale else density

    fig.add_trace(
        go.Scatter(
            x=grid,
            y=dens_clipped,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(163, 113, 247, 0.25)",
            line=dict(color=COLOR_PURPLE, width=2.5),
            name=f"Spectral Density ρ(λ) at step {step}",
        )
    )

    fig.update_xaxes(
        title_text="Eigenvalue λ",
        gridcolor="#30363D",
        zerolinecolor="#30363D",
    )
    fig.update_yaxes(
        title_text="Eigenvalue Density ρ(λ)" + (" (Log Scale)" if log_scale else ""),
        type="log" if log_scale else "linear",
        gridcolor="#30363D",
        zerolinecolor="#30363D",
    )

    fig.update_layout(
        title=dict(
            text=f"Stochastic Lanczos Quadrature (SLQ) Spectral Density at Step {step}",
            font=dict(color="#E6EDF3", size=15),
        ),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        font=dict(color="#E6EDF3"),
        margin=dict(l=60, r=30, b=50, t=60),
        height=480,
    )
    return fig
