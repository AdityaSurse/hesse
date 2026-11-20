"""hesse: Real-Time Curvature Tracking & Loss-Landscape Streamlit Dashboard."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import streamlit as st
import yaml

from app.utils import (
    build_comparison_charts,
    build_density_plot,
    get_available_runs,
    load_run_data,
)
from hesse.landscape.grid import load_grid_npz
from hesse.report import (
    build_landscape_3d_fig,
    build_landscape_contour_fig,
    build_live_run_fig,
    build_topk_fig,
    generate_report,
)

ROOT_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------
# Page Configuration & Global Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="hesse — Live Curvature Telemetry",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp {
        background-color: #0E1117;
        color: #E6EDF3;
      }
      div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 12px 18px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
      }
      div[data-testid="stMetricLabel"] {
        color: #8B949E;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }
      div[data-testid="stMetricValue"] {
        color: #E6EDF3;
        font-size: 24px;
        font-weight: 700;
      }
      .css-1544g2n, .e1fqkh3o4 {
        padding-top: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Sidebar: Navigation & Start a Run Subprocess Form
# ---------------------------------------------------------
st.sidebar.title("hesse")
st.sidebar.caption("Curvature Tracking & Edge-of-Stability Diagnostics")

query_page = st.query_params.get("page", None)
page_options = ["Live Run", "Compare Runs", "Spectrum", "Landscape", "Experiments"]
default_idx = page_options.index(query_page) if query_page in page_options else 0

page = st.sidebar.radio(
    "Navigation",
    page_options,
    index=default_idx,
)

st.sidebar.markdown("---")

# "Start a Run" Subprocess Launcher
with st.sidebar.expander("🚀 Start a New Run", expanded=False):
    st.markdown("Configure and launch a background training run.")
    run_name_input = st.text_input("Run Name", value=f"run_gui_{int(time.time()) % 10000:04d}")
    arch_input = st.selectbox("Model Architecture", ["mlp", "cnn", "transformer"], index=0)
    dataset_input = st.selectbox("Dataset", ["cifar10", "mnist", "synthetic"], index=0)
    opt_input = st.selectbox("Optimizer", ["gd", "sgd", "adam", "lion"], index=0)
    lr_input = st.number_input(
        "Learning Rate (η)", value=0.04, min_value=0.0001, max_value=1.0, format="%.4f"
    )
    steps_input = st.number_input("Steps", value=30, min_value=5, max_value=500, step=5)
    interval_input = st.number_input(
        "Curvature Track Interval (k)", value=3, min_value=1, max_value=50, step=1
    )
    batch_choice = st.selectbox(
        "Batch Mode", ["Full Batch", "Minibatch (128)", "Minibatch (64)"], index=0
    )

    track_topk = st.checkbox("Compute Top-5 Eigenvalues", value=True)
    track_trace = st.checkbox("Compute Trace (Hutchinson)", value=True)

    if st.button("Launch Training", type="primary", use_container_width=True):
        if batch_choice == "Full Batch":
            batch_size_val = None
        else:
            batch_size_val = 128 if "128" in batch_choice else 64
        run_dir_out = Path("runs") / run_name_input

        cfg_dict: Dict[str, Any] = {
            "run_name": run_name_input,
            "epochs": 100,
            "steps": int(steps_input),
            "device": "cpu",
            "seed": 42,
            "data": {
                "dataset": dataset_input,
                "subset_size": 200,
                "batch_size": batch_size_val,
                "loss_type": "cross_entropy",
                "val_size": 50,
                "data_dir": "data",
            },
            "model": {
                "architecture": arch_input,
                "input_dim": 3072,
                "hidden_dims": [128, 64] if arch_input == "mlp" else None,
                "channels": [16, 32] if arch_input == "cnn" else None,
                "patch_size": 8 if arch_input == "transformer" else None,
                "embed_dim": 32 if arch_input == "transformer" else None,
                "num_heads": 2 if arch_input == "transformer" else None,
                "num_layers": 2 if arch_input == "transformer" else None,
                "num_classes": 10,
                "activation": "relu",
                "dropout": 0.0,
            },
            "optim": {
                "optimizer": opt_input,
                "lr": float(lr_input),
                "momentum": 0.0,
                "weight_decay": 0.0,
            },
            "tracker": {
                "track_interval": int(interval_input),
                "compute_sharpness": True,
                "compute_trace": track_trace,
                "compute_topk": 5 if track_topk else 0,
                "compute_density": False,
                "log_dir": str(run_dir_out),
            },
        }
        # Clean null values
        cfg_dict["model"] = {k: v for k, v in cfg_dict["model"].items() if v is not None}

        cfg_save_path = Path("configs") / "runs" / f"{run_name_input}.yaml"
        cfg_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_save_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_dict, f)

        # Launch subprocess
        proc = subprocess.Popen(
            [sys.executable, "-m", "hesse.cli", "train", "--config", str(cfg_save_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        st.session_state["active_proc"] = proc
        st.session_state["active_proc_name"] = run_name_input
        st.success(f"Training started for '{run_name_input}' (PID {proc.pid})")

# Process status indicator if active
if "active_proc" in st.session_state:
    p = st.session_state["active_proc"]
    ret = p.poll()
    if ret is None:
        st.sidebar.info(f"⏳ Training in progress: `{st.session_state.get('active_proc_name')}`")
    else:
        st.sidebar.success(f"✅ Run finished with exit code {ret}!")


# ---------------------------------------------------------
# PAGE 1: LIVE RUN
# ---------------------------------------------------------
if page == "Live Run":
    st.title("📈 Live Curvature Dynamics & Telemetry")
    st.markdown(
        "Monitor loss and matrix-free curvature dynamics in real time while training executes."
    )

    all_runs = get_available_runs()
    if not all_runs:
        st.warning("No runs found in `runs/`. Launch a run from the sidebar or via `hesse train`.")
        st.stop()

    c_run, c_refresh = st.columns([3, 1])
    with c_run:
        selected_run = st.selectbox("Select Run Directory", all_runs, index=0)
    with c_refresh:
        auto_refresh = st.checkbox("Auto-refresh (3s)", value=False)

    if auto_refresh:
        time.sleep(3.0)
        st.rerun()

    run_data = load_run_data(selected_run)
    records = run_data["records"]
    config = run_data["config"]

    if not records:
        st.info(f"ℹ️ Run `{selected_run}` exists, but no telemetry records logged yet.")
        st.stop()

    # KPI Metric Tiles
    final_rec = records[-1]
    step = final_rec["step"]
    loss = final_rec["loss"]
    sharpness = final_rec.get("sharpness")
    thresh = final_rec.get("stability_threshold")

    sharp_str = f"{sharpness:.2f}" if sharpness is not None else "N/A"
    plateau_ratio = (
        (sharpness / thresh) if (sharpness is not None and thresh and thresh > 0) else None
    )
    ratio_str = f"{plateau_ratio * 100:.1f}%" if plateau_ratio is not None else "N/A"

    total_curv_time = sum(r.get("curvature_time", 0.0) for r in records)
    total_train_time = sum(r.get("train_time", 0.0) for r in records) + total_curv_time
    curv_pct = (total_curv_time / total_train_time * 100.0) if total_train_time > 0 else 0.0

    delta_label = (
        f"{(plateau_ratio - 1.0) * 100:.1f}% from EoS" if plateau_ratio is not None else None
    )
    delta_col = "inverse" if plateau_ratio and plateau_ratio > 1.0 else "normal"

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Current Step", f"{step}")
    kpi2.metric("Loss L(θ)", f"{loss:.4f}")
    kpi3.metric("Sharpness λ_max", sharp_str)
    kpi4.metric("λ_max ÷ (2/η)", ratio_str, delta=delta_label, delta_color=delta_col)
    kpi5.metric("Curvature Overhead", f"{curv_pct:.1f}%", f"{total_curv_time:.2f}s total")

    st.markdown("### Telemetry Trajectory")
    opt_name = config.get("optim", {}).get("optimizer", "gd").upper()
    chart_title = f"{run_data['name']} — {opt_name} Curvature & Empirical Loss Descent"
    fig_live = build_live_run_fig(records, threshold=thresh, title=chart_title)
    st.plotly_chart(fig_live, use_container_width=True)

    # Static HTML Report Export
    col_exp1, col_exp2 = st.columns([2, 1])
    with col_exp1:
        st.caption("Export a standalone, self-contained HTML report with embedded Plotly charts.")
    with col_exp2:
        if st.button("📄 Generate Static HTML Report", use_container_width=True):
            report_path = Path("runs") / selected_run / "report.html"
            out_file = generate_report(run_data["path"], out_path=report_path)
            st.success(f"Report generated at `{out_file}`!")
            with open(out_file, "r", encoding="utf-8") as f:
                st.download_button(
                    "⬇️ Download HTML Report",
                    data=f.read(),
                    file_name=f"{run_data['name']}_report.html",
                    mime="text/html",
                    use_container_width=True,
                )


# ---------------------------------------------------------
# PAGE 2: COMPARE RUNS
# ---------------------------------------------------------
elif page == "Compare Runs":
    st.title("⚖️ Comparative Run Geometry")
    st.markdown(
        "Overlay normalized curvature $\\lambda_{\\max} / (2/\\eta)$ across multiple runs. "
        "The canonical optimizer palette is applied: **GD (Blue)**, **SGD (Green)**, "
        "**Adam (Purple)**, and **Lion (Pink)**."
    )

    all_runs = get_available_runs()
    default_selection = [r for r in all_runs if "e1" in r or "e2" in r or "gd" in r][:4]
    if not default_selection:
        default_selection = all_runs[:3]

    selected_runs = st.multiselect(
        "Choose Runs to Compare",
        all_runs,
        default=default_selection,
    )

    if not selected_runs:
        st.info("Select at least one run above to generate comparison curves.")
        st.stop()

    fig_sharp, fig_loss, table_rows = build_comparison_charts(selected_runs)

    st.plotly_chart(fig_sharp, use_container_width=True)
    st.plotly_chart(fig_loss, use_container_width=True)

    if table_rows:
        st.markdown("### Comparative Performance Summary")
        st.dataframe(table_rows, use_container_width=True)


# ---------------------------------------------------------
# PAGE 3: SPECTRUM
# ---------------------------------------------------------
elif page == "Spectrum":
    st.title("🔬 Eigenvalue Spectral Dynamics & SLQ Density")
    st.markdown(
        "Track individual top-$k$ Lanczos Ritz values and examine continuous eigenvalue "
        "spectral density $\\rho(\\lambda)$ computed via Stochastic Lanczos Quadrature (SLQ)."
    )

    all_runs = get_available_runs()
    selected_run = st.selectbox("Select Run", all_runs, index=0)
    run_data = load_run_data(selected_run)
    records = run_data["records"]

    if not records:
        st.warning(f"Run `{selected_run}` contains no metric records.")
        st.stop()

    # 1. Top-k trajectories
    fig_topk = build_topk_fig(records)
    if fig_topk is not None:
        st.markdown("### Top-$k$ Eigenvalue Evolution Over Time")
        st.plotly_chart(fig_topk, use_container_width=True)
    else:
        st.info("No top-$k$ eigenvalues logged for this run (ensure `compute_topk > 0`).")

    # 2. SLQ Spectral Density
    st.markdown("### Continuous Spectral Density $\\rho(\\lambda)$ (SLQ)")
    density_steps = run_data["density_steps"]

    if density_steps:
        sorted_steps = sorted(density_steps.keys())
        chosen_step = st.select_slider(
            "Select Training Step to Inspect Spectral Density",
            options=sorted_steps,
            value=sorted_steps[-1],
        )

        npz_file = density_steps[chosen_step]
        with np.load(npz_file, allow_pickle=True) as d:
            grid = d["grid"]
            density = d["density"]

        log_axis = st.checkbox("Log-Scaled Density Axis", value=True)
        fig_dens = build_density_plot(grid, density, step=chosen_step, log_scale=log_axis)
        st.plotly_chart(fig_dens, use_container_width=True)
    else:
        st.info(
            "No precomputed SLQ density archives found in `density/` for this run. "
            "Runs configured with `compute_density: true` automatically record density steps."
        )


# ---------------------------------------------------------
# PAGE 4: LANDSCAPE
# ---------------------------------------------------------
elif page == "Landscape":
    st.title("🏔️ Loss Landscape Topography")
    st.markdown(
        "Explore interactive 3D loss surface meshes and 2D contour slices evaluated along "
        "filter-normalized Random directions, top Hessian eigenvectors, or trajectory PCA."
    )

    all_runs = get_available_runs()
    selected_run = st.selectbox("Select Run", all_runs, index=0)
    run_data = load_run_data(selected_run)
    grids = run_data["landscape_grids"]

    if not grids:
        st.warning(
            f"No landscape grid archives (`grid_*.npz`) found for `{selected_run}`. "
            f"Generate one via CLI: `hesse landscape --run runs/{selected_run}`."
        )
        st.stop()

    basis_options = list(grids.keys())
    chosen_basis = st.radio(
        "Direction Basis",
        basis_options,
        format_func=lambda b: f"{b.upper()} Directions",
        horizontal=True,
    )

    npz_path = grids[chosen_basis]
    grid_data = load_grid_npz(npz_path)
    alphas = grid_data["alphas"]
    betas = grid_data["betas"]
    loss_grid = grid_data["loss"]

    # Trajectory overlay
    traj_coords = None

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Grid Size:** {len(alphas)} × {len(betas)}")
    with c2:
        st.markdown(f"**Loss Range:** [{np.min(loss_grid):.3f}, {np.max(loss_grid):.3f}]")

    # Interactive 3D Surface
    fig_3d = build_landscape_3d_fig(
        alphas,
        betas,
        loss_grid,
        trajectory_coords=traj_coords,
        title=f"3D Loss Landscape ({chosen_basis.upper()} Basis)",
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # 2D Contour
    fig_contour = build_landscape_contour_fig(
        alphas,
        betas,
        loss_grid,
        trajectory_coords=traj_coords,
        title=f"2D Loss Contours ({chosen_basis.upper()} Basis)",
    )
    st.plotly_chart(fig_contour, use_container_width=True)


# ---------------------------------------------------------
# PAGE 5: EXPERIMENTS
# ---------------------------------------------------------
elif page == "Experiments":
    st.title("📊 Empirical Research Benchmarks (E1–E6)")
    st.markdown(
        "Empirical study results comparing Edge-of-Stability dynamics across optimizers, "
        "batch sizes, architectures, and loss objectives."
    )

    # Render summary.md if exists
    summary_path = ROOT_DIR / "experiments" / "results" / "summary.md"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_content = f.read()

        tab_summary, tab_gallery = st.tabs(["📋 Summary Report & Tables", "🖼️ Publication Figures"])

        with tab_summary:
            st.markdown(summary_content)

        with tab_gallery:
            figures = [
                (
                    "E1: GD Edge of Stability Reproduction",
                    "docs/figures/e1_gd_eos_trajectory.png",
                    "Dual-axis trajectory showing progressive sharpening and hovering at 2/η.",
                ),
                (
                    "E2: SGD Batch Size Scaling & Curvature Noise",
                    "docs/figures/e2_sgd_batch_size_scaling.png",
                    "Transition from noise dissipation to deterministic 2/η boundary hovering.",
                ),
                (
                    "E3: Adam Preconditioned Sharpness vs 38/η",
                    "docs/figures/e3_adam_preconditioned_sharpness.png",
                    "Raw vs preconditioned sharpness confirming the analytical 38/η bound.",
                ),
                (
                    "E4: Lion Sign-Update Dynamics",
                    "docs/figures/e4_lion_sign_dynamics.png",
                    "Sign-aware directional curvature shows coordinate stability.",
                ),
                (
                    "E5: Architecture Comparison (MLP vs CNN vs ViT)",
                    "docs/figures/e5_architecture_curvature_comparison.png",
                    "Outlier spectral gaps and curvature trajectories across architectures.",
                ),
                (
                    "E6: Objective Geometry (MSE vs Cross-Entropy)",
                    "docs/figures/e6_loss_function_spectral_density.png",
                    "Persistent hovering under MSE vs late-time flattening under Cross-Entropy.",
                ),
            ]

            for title_fig, path_rel, caption in figures:
                img_path = ROOT_DIR / path_rel
                if img_path.exists():
                    st.markdown(f"#### {title_fig}")
                    st.image(str(img_path), caption=caption, use_container_width=True)
                    st.markdown("---")
    else:
        st.info("Run `make reproduce` to generate all experimental summaries and figures.")
