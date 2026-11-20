"""Self-contained static HTML report generator for hesse curvature tracking runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hesse.landscape.grid import load_grid_npz
from hesse.viz.style import (
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_PINK,
    COLOR_PURPLE,
    COLOR_YELLOW,
)


def load_run_data(run_dir: str | Path) -> Dict[str, Any]:
    """Load all metrics, configurations, summaries, and landscape data from a run directory."""
    r_dir = Path(run_dir)
    if not r_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {r_dir}")

    config: Dict[str, Any] = {}
    config_file = r_dir / "config.yaml"
    if config_file.exists():
        import yaml

        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    summary: Dict[str, Any] = {}
    summary_file = r_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f) or {}

    records: List[Dict[str, Any]] = []
    metrics_file = r_dir / "metrics.jsonl"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    records.append(json.loads(s))

    landscape_grids: Dict[str, Dict[str, Any]] = {}
    landscape_dir = r_dir / "landscape"
    if landscape_dir.exists():
        for npz_file in sorted(landscape_dir.glob("grid_*.npz")):
            basis_name = npz_file.stem.replace("grid_", "")
            try:
                landscape_grids[basis_name] = load_grid_npz(npz_file)
            except Exception:
                pass

    density_data: Dict[int, Dict[str, np.ndarray]] = {}
    density_dir = r_dir / "density"
    if density_dir.exists():
        for npz_file in sorted(density_dir.glob("density_step_*.npz")):
            try:
                with np.load(npz_file, allow_pickle=True) as data:
                    step_val = int(data.get("step", 0))
                    density_data[step_val] = {
                        "grid": data["grid"],
                        "density": data["density"],
                    }
            except Exception:
                pass

    return {
        "run_dir": r_dir,
        "run_name": config.get("run_name", r_dir.name),
        "config": config,
        "summary": summary,
        "records": records,
        "landscape_grids": landscape_grids,
        "density_data": density_data,
    }


def build_live_run_fig(
    records: List[Dict[str, Any]],
    threshold: float | None = None,
    title: str = "Live Loss & Curvature Trajectory",
) -> go.Figure:
    """Create a 2-panel Plotly figure with shared x-axis showing Sharpness and Log Loss."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=(
            "Maximum Hessian Eigenvalue λ_max (Sharpness)",
            "Empirical Loss L(θ) (Log Scale)",
        ),
    )

    steps = [r["step"] for r in records]
    losses = [r["loss"] for r in records]
    sharpness_steps = [r["step"] for r in records if r.get("sharpness") is not None]
    sharpness_vals = [r["sharpness"] for r in records if r.get("sharpness") is not None]

    if threshold is None:
        threshold = next(
            (r["stability_threshold"] for r in records if r.get("stability_threshold") is not None),
            None,
        )

    # 1. Sharpness trace
    if sharpness_vals:
        fig.add_trace(
            go.Scatter(
                x=sharpness_steps,
                y=sharpness_vals,
                mode="lines+markers",
                name="Sharpness λ_max",
                line=dict(color=COLOR_GREEN, width=2.5),
                marker=dict(size=5, color=COLOR_GREEN),
            ),
            row=1,
            col=1,
        )

    # Threshold dashed line
    if threshold is not None and steps:
        max_step = max(steps)
        fig.add_trace(
            go.Scatter(
                x=[0, max_step],
                y=[threshold, threshold],
                mode="lines",
                name=f"Stability Limit (2/η = {threshold:.1f})",
                line=dict(color=COLOR_ORANGE, width=2.0, dash="dash"),
            ),
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=max_step * 0.98,
            y=threshold,
            text=f"2/η = {threshold:.1f}",
            showarrow=False,
            font=dict(color=COLOR_ORANGE, size=11, family="sans-serif"),
            xanchor="right",
            yanchor="bottom",
            row=1,
            col=1,
        )

    # 2. Loss trace
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=losses,
            mode="lines+markers",
            name="Loss L",
            line=dict(color=COLOR_BLUE, width=2.5),
            marker=dict(size=4, color=COLOR_BLUE),
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="Sharpness λ_max",
        gridcolor="#30363D",
        zerolinecolor="#30363D",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Loss L (log scale)",
        type="log",
        gridcolor="#30363D",
        zerolinecolor="#30363D",
        row=2,
        col=1,
    )
    fig.update_xaxes(
        title_text="Optimization Step t",
        gridcolor="#30363D",
        zerolinecolor="#30363D",
        row=2,
        col=1,
    )
    fig.update_xaxes(
        gridcolor="#30363D",
        zerolinecolor="#30363D",
        row=1,
        col=1,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#E6EDF3", size=16)),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        font=dict(color="#E6EDF3"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(22, 27, 34, 0.8)",
            bordercolor="#30363D",
            borderwidth=1,
        ),
        margin=dict(l=60, r=30, b=50, t=80),
        height=620,
    )
    return fig


def build_topk_fig(records: List[Dict[str, Any]]) -> go.Figure | None:
    """Build Plotly chart showing top-k eigenvalue trajectories over steps."""
    steps_with_topk = [r["step"] for r in records if r.get("topk_eigenvalues")]
    if not steps_with_topk:
        return None

    all_topk = [r["topk_eigenvalues"] for r in records if r.get("topk_eigenvalues")]
    k = len(all_topk[0])

    palette = [
        COLOR_BLUE,
        COLOR_GREEN,
        COLOR_PURPLE,
        COLOR_YELLOW,
        COLOR_PINK,
        "#38BEC9",
        "#F56565",
    ]

    fig = go.Figure()
    for i in range(k):
        vals = [row[i] if i < len(row) else None for row in all_topk]
        col = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=steps_with_topk,
                y=vals,
                mode="lines+markers",
                name=f"λ_{i + 1}",
                line=dict(color=col, width=2.0),
                marker=dict(size=4, color=col),
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Top-{k} Eigenvalue Trajectories (Lanczos Ritz Values)",
            font=dict(color="#E6EDF3", size=15),
        ),
        xaxis=dict(title="Optimization Step t", gridcolor="#30363D", zerolinecolor="#30363D"),
        yaxis=dict(title="Eigenvalue λ_i", gridcolor="#30363D", zerolinecolor="#30363D"),
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
        margin=dict(l=60, r=30, b=50, t=70),
        height=450,
    )
    return fig


def build_landscape_3d_fig(
    alphas: np.ndarray,
    betas: np.ndarray,
    loss_grid: np.ndarray,
    trajectory_coords: tuple[Any, Any] | None = None,
    title: str = "3D Interactive Loss Surface",
) -> go.Figure:
    """Create interactive 3D loss surface in Plotly."""
    A, B = np.meshgrid(alphas, betas)

    surface = go.Surface(
        x=A,
        y=B,
        z=loss_grid,
        colorscale="Viridis",
        colorbar=dict(title="Loss L", len=0.75, tickfont=dict(color="#E6EDF3")),
        opacity=0.92,
        name="Loss Surface",
    )

    data: list[Any] = [surface]
    if trajectory_coords is not None and len(trajectory_coords[0]) > 0:
        traj_a = list(trajectory_coords[0])
        traj_b = list(trajectory_coords[1])
        # Sample loss at trajectory coordinates
        traj_z = []
        for a, b in zip(traj_a, traj_b):
            i = int(np.clip(np.argmin(np.abs(betas - b)), 0, len(betas) - 1))
            j = int(np.clip(np.argmin(np.abs(alphas - a)), 0, len(alphas) - 1))
            traj_z.append(float(loss_grid[i, j]) * 1.01)

        data.append(
            go.Scatter3d(
                x=traj_a,
                y=traj_b,
                z=traj_z,
                mode="lines+markers",
                line=dict(color=COLOR_ORANGE, width=5),
                marker=dict(size=4, color=COLOR_ORANGE),
                name="Optimizer Path",
            )
        )

    fig = go.Figure(data=data)
    fig.update_layout(
        title=dict(text=title, font=dict(color="#E6EDF3", size=15)),
        paper_bgcolor="#0E1117",
        scene=dict(
            xaxis=dict(
                title="Direction 1 (α)",
                backgroundcolor="#161B22",
                gridcolor="#30363D",
                color="#E6EDF3",
            ),
            yaxis=dict(
                title="Direction 2 (β)",
                backgroundcolor="#161B22",
                gridcolor="#30363D",
                color="#E6EDF3",
            ),
            zaxis=dict(
                title="Loss L",
                backgroundcolor="#161B22",
                gridcolor="#30363D",
                color="#E6EDF3",
            ),
        ),
        margin=dict(l=20, r=20, b=20, t=50),
        height=550,
    )
    return fig


def build_landscape_contour_fig(
    alphas: np.ndarray,
    betas: np.ndarray,
    loss_grid: np.ndarray,
    trajectory_coords: tuple[Any, Any] | None = None,
    title: str = "2D Loss Landscape Contours",
) -> go.Figure:
    """Create interactive 2D loss contour in Plotly."""
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=alphas,
            y=betas,
            z=loss_grid,
            colorscale="Viridis",
            contours=dict(
                coloring="heatmap",
                showlabels=True,
                labelfont=dict(size=9, color="#E6EDF3"),
            ),
            colorbar=dict(title="Loss L", len=0.8, tickfont=dict(color="#E6EDF3")),
            name="Contour",
        )
    )

    if trajectory_coords is not None and len(trajectory_coords[0]) > 0:
        fig.add_trace(
            go.Scatter(
                x=trajectory_coords[0],
                y=trajectory_coords[1],
                mode="lines+markers",
                name="Trajectory",
                line=dict(color=COLOR_ORANGE, width=2.5),
                marker=dict(size=5, color=COLOR_ORANGE),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[trajectory_coords[0][0]],
                y=[trajectory_coords[1][0]],
                mode="markers",
                name="Start (θ_0)",
                marker=dict(size=10, color=COLOR_GREEN, symbol="circle"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[trajectory_coords[0][-1]],
                y=[trajectory_coords[1][-1]],
                mode="markers",
                name="Final (θ*)",
                marker=dict(size=12, color=COLOR_ORANGE, symbol="star"),
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#E6EDF3", size=15)),
        xaxis=dict(title="Direction 1 (α)", gridcolor="#30363D", zerolinecolor="#30363D"),
        yaxis=dict(title="Direction 2 (β)", gridcolor="#30363D", zerolinecolor="#30363D"),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        font=dict(color="#E6EDF3"),
        margin=dict(l=60, r=30, b=50, t=60),
        height=520,
    )
    return fig


def generate_report(run_dir: str | Path, out_path: str | Path | None = None) -> Path:
    """Generate a single self-contained HTML report file with inline Plotly JS.

    Args:
        run_dir: Path to directory containing run metrics and artifacts.
        out_path: Destination HTML file path. If None, defaults to `run_dir / "report.html"`.

    Returns:
        Path to the saved report.html file.
    """
    data = load_run_data(run_dir)
    r_dir = data["run_dir"]
    records = data["records"]
    config = data["config"]
    grids = data["landscape_grids"]

    out_file = Path(out_path) if out_path else (r_dir / "report.html")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Compute KPI statistics
    total_steps = len(records)
    final_loss = records[-1]["loss"] if records else 0.0
    final_step = records[-1]["step"] if records else 0
    sharpness_vals = [r["sharpness"] for r in records if r.get("sharpness") is not None]
    max_sharp = max(sharpness_vals) if sharpness_vals else 0.0
    curr_sharp = sharpness_vals[-1] if sharpness_vals else 0.0

    threshold = next(
        (r["stability_threshold"] for r in records if r.get("stability_threshold") is not None),
        None,
    )
    plateau_pct = (curr_sharp / threshold * 100.0) if (threshold and threshold > 0) else 0.0

    total_curv_time = sum(r.get("curvature_time", 0.0) for r in records)
    total_train_time = sum(r.get("train_time", 0.0) for r in records) + total_curv_time
    curv_overhead_pct = (
        (total_curv_time / total_train_time * 100.0) if total_train_time > 0 else 0.0
    )

    opt_name = (
        config.get("optim", {}).get("optimizer", "unknown").upper() if "optim" in config else "N/A"
    )
    arch_name = (
        config.get("model", {}).get("architecture", "unknown").upper()
        if "model" in config
        else "N/A"
    )
    lr_val = config.get("optim", {}).get("lr", "N/A")

    # Generate Plotly charts
    charts_html: list[str] = []

    if records:
        title_text = f"Run: {data['run_name']} — Curvature & Loss Dynamics"
        fig_live = build_live_run_fig(records, threshold=threshold, title=title_text)
        include_js = True if len(charts_html) == 0 else "cdn"
        charts_html.append(fig_live.to_html(full_html=False, include_plotlyjs=include_js))

        fig_topk = build_topk_fig(records)
        if fig_topk is not None:
            charts_html.append(fig_topk.to_html(full_html=False, include_plotlyjs=False))
    else:
        charts_html.append(
            "<div class='alert warning'>No training metrics recorded in this run directory.</div>"
        )

    # Landscape charts
    landscape_html: list[str] = []
    if grids:
        for basis, g_data in grids.items():
            fig_3d = build_landscape_3d_fig(
                g_data["alphas"],
                g_data["betas"],
                g_data["loss"],
                title=f"3D Loss Landscape ({basis.capitalize()} Basis)",
            )
            fig_cnt = build_landscape_contour_fig(
                g_data["alphas"],
                g_data["betas"],
                g_data["loss"],
                title=f"2D Loss Contours ({basis.capitalize()} Basis)",
            )
            landscape_html.append(
                f"<h3 style='color:#58A6FF; margin-top:30px;'>Basis: {basis.upper()}</h3>"
            )
            landscape_html.append(fig_3d.to_html(full_html=False, include_plotlyjs=False))
            landscape_html.append(fig_cnt.to_html(full_html=False, include_plotlyjs=False))
    else:
        landscape_html.append(
            "<div class='alert info'>No 2D/3D landscape grid archives found in this run directory. "
            f"Generate them via <code>hesse landscape --run {r_dir}</code>.</div>"
        )

    color_threshold = "var(--orange)" if plateau_pct >= 90 else "var(--green)"
    thresh_display = f"{threshold:.1f}" if threshold is not None else "N/A"

    # HTML Template
    template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>hesse Report: {data["run_name"]}</title>
  <style>
    :root {{
      --bg: #0E1117;
      --card-bg: #161B22;
      --border: #30363D;
      --text: #E6EDF3;
      --text-muted: #8B949E;
      --accent: #58A6FF;
      --green: #3FB950;
      --orange: #F0883E;
      --purple: #A371F7;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding: 30px 40px;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 20px;
      margin-bottom: 25px;
    }}
    .title h1 {{
      font-size: 26px;
      font-weight: 700;
      color: var(--accent);
      letter-spacing: -0.5px;
    }}
    .title p {{
      color: var(--text-muted);
      font-size: 14px;
      margin-top: 4px;
    }}
    .badges {{
      display: flex;
      gap: 10px;
    }}
    .badge {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
    }}
    .badge.opt {{ color: var(--purple); }}
    .badge.arch {{ color: var(--green); }}
    .badge.lr {{ color: var(--accent); }}

    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 30px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }}
    .kpi-card .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
    }}
    .kpi-card .value {{
      font-size: 24px;
      font-weight: 700;
      margin-top: 6px;
      color: var(--text);
    }}
    .kpi-card .sub {{
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 4px;
    }}

    .section {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 30px;
    }}
    .section-title {{
      font-size: 18px;
      font-weight: 600;
      margin-bottom: 16px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 10px;
      color: var(--text);
    }}
    .alert {{
      padding: 14px 18px;
      border-radius: 6px;
      font-size: 14px;
      margin: 10px 0;
    }}
    .alert.info {{
      background: rgba(88, 166, 255, 0.1);
      border: 1px solid var(--accent);
      color: var(--accent);
    }}
    .alert.warning {{
      background: rgba(240, 136, 62, 0.1);
      border: 1px solid var(--orange);
      color: var(--orange);
    }}
    code {{
      background: rgba(255, 255, 255, 0.1);
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 13px;
    }}
    footer {{
      text-align: center;
      color: var(--text-muted);
      font-size: 13px;
      margin-top: 40px;
      border-top: 1px solid var(--border);
      padding-top: 20px;
    }}
  </style>
</head>
<body>

  <div class="header">
    <div class="title">
      <h1>hesse Telemetry Report</h1>
      <p>Run Directory: <code>{r_dir}</code></p>
    </div>
    <div class="badges">
      <div class="badge arch">{arch_name}</div>
      <div class="badge opt">{opt_name}</div>
      <div class="badge lr">η = {lr_val}</div>
    </div>
  </div>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="label">Total Steps</div>
      <div class="value">{final_step}</div>
      <div class="sub">{total_steps} telemetry records</div>
    </div>
    <div class="kpi-card">
      <div class="label">Final Loss</div>
      <div class="value" style="color:var(--accent);">{final_loss:.4f}</div>
      <div class="sub">Empirical loss</div>
    </div>
    <div class="kpi-card">
      <div class="label">Current λ_max</div>
      <div class="value" style="color:var(--green);">{curr_sharp:.2f}</div>
      <div class="sub">Peak: {max_sharp:.2f}</div>
    </div>
    <div class="kpi-card">
      <div class="label">λ_max ÷ (2/η)</div>
      <div class="value" style="color:{color_threshold};">{plateau_pct:.1f}%</div>
      <div class="sub">Threshold: {thresh_display}</div>
    </div>
    <div class="kpi-card">
      <div class="label">Curvature Overhead</div>
      <div class="value">{curv_overhead_pct:.1f}%</div>
      <div class="sub">{total_curv_time:.2f}s curvature time</div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">1. Curvature & Optimization Dynamics (Live Run View)</div>
    {"".join(charts_html)}
  </div>

  <div class="section">
    <div class="section-title">2. Loss Landscape Slices & Geometry</div>
    {"".join(landscape_html)}
  </div>

  <footer>
    Generated by <b>hesse</b> — Live Curvature Tracking & Diagnostics
  </footer>

</body>
</html>"""

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(template)

    print(f"Report successfully generated at: {out_file}")
    return out_file
