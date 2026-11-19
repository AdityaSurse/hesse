"""Publication-quality 2D contour maps, 3D interactive Plotly meshes, and comparisons."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Sequence, Tuple

# Set writable cache directory for Matplotlib in sandboxed environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from hesse.viz.style import (
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_YELLOW,
    save_figure,
    setup_style,
)

logger = logging.getLogger(__name__)


def _sample_grid_loss(
    alphas: np.ndarray,
    betas: np.ndarray,
    loss_grid: np.ndarray,
    a: float,
    b: float,
) -> float:
    """Sample scalar loss value from 2D grid at given coordinates (a, b)."""
    i = int(np.clip(np.argmin(np.abs(betas - b)), 0, len(betas) - 1))
    j = int(np.clip(np.argmin(np.abs(alphas - a)), 0, len(alphas) - 1))
    return float(loss_grid[i, j])


def plot_landscape_contour(
    alphas: np.ndarray,
    betas: np.ndarray,
    loss_grid: np.ndarray,
    trajectory_coords: Tuple[Sequence[float], Sequence[float]] | None = None,
    save_path: str | Path = "landscape_contour.png",
    title: str | None = None,
    levels: int = 35,
    cmap: str = "magma",
) -> plt.Figure:
    """Render a publication-quality 2D contour map with optional trajectory overlay.

    Args:
        alphas: 1D array of horizontal perturbation coordinates.
        betas: 1D array of vertical perturbation coordinates.
        loss_grid: 2D numpy array of evaluated losses (shape: n_betas, n_alphas).
        trajectory_coords: Optional tuple of (traj_alphas, traj_betas) path coordinates.
        save_path: Destination PNG image file path.
        title: Plot title stating the key insight.
        levels: Number of contour levels.
        cmap: Matplotlib colormap name ('magma', 'viridis', etc.).

    Returns:
        Rendered Matplotlib Figure instance.
    """
    setup_style()
    A, B = np.meshgrid(alphas, betas)

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=300)

    # Determine normalization: log-scale if strictly positive, else linear
    min_val = float(np.min(loss_grid))
    max_val = float(np.max(loss_grid))
    if max_val - min_val < 1e-7:
        max_val = min_val + 1.0

    if min_val > 1e-6:
        norm = mcolors.LogNorm(vmin=min_val, vmax=max_val)
        level_bounds = np.geomspace(min_val, max_val, levels)
    else:
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
        level_bounds = np.linspace(min_val, max_val, levels)

    # Filled contour
    cf = ax.contourf(A, B, loss_grid, levels=level_bounds, cmap=cmap, norm=norm, alpha=0.92)
    # Line contour overlay
    cs = ax.contour(
        A,
        B,
        loss_grid,
        levels=level_bounds[:: max(1, len(level_bounds) // 10)],
        colors="#E6EDF3",
        alpha=0.35,
        linewidths=0.6,
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f", colors="#E6EDF3")

    # Colorbar
    cbar = fig.colorbar(cf, ax=ax, pad=0.03, aspect=24)
    cbar.set_label(r"Empirical Loss $\mathcal{L}(\theta^* + \alpha d_1 + \beta d_2)$", labelpad=10)

    # Trajectory overlay
    if trajectory_coords is not None and len(trajectory_coords[0]) > 0:
        traj_a = np.array(trajectory_coords[0])
        traj_b = np.array(trajectory_coords[1])

        # Plot path line
        ax.plot(
            traj_a,
            traj_b,
            color=COLOR_YELLOW,
            linewidth=2.2,
            linestyle="-",
            label="Optimization Trajectory",
            zorder=4,
        )

        # Plot directional arrows along trajectory
        if len(traj_a) > 1:
            for k in range(len(traj_a) - 1):
                ax.annotate(
                    "",
                    xy=(traj_a[k + 1], traj_b[k + 1]),
                    xytext=(traj_a[k], traj_b[k]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=COLOR_YELLOW,
                        lw=1.6,
                        mutation_scale=12,
                    ),
                    zorder=5,
                )

        # Start marker
        ax.scatter(
            [traj_a[0]],
            [traj_b[0]],
            color=COLOR_GREEN,
            s=80,
            marker="o",
            edgecolors="#FFFFFF",
            linewidth=1.4,
            label=r"Start $\theta_0$",
            zorder=6,
        )
        # End marker
        ax.scatter(
            [traj_a[-1]],
            [traj_b[-1]],
            color=COLOR_ORANGE,
            s=100,
            marker="*",
            edgecolors="#FFFFFF",
            linewidth=1.4,
            label=r"Final $\theta^*$ (0, 0)",
            zorder=6,
        )
        ax.legend(loc="upper right", framealpha=0.88, fontsize=9.5)

    plot_title = (
        title
        if title is not None
        else r"Loss Landscape Surface $\mathcal{L}(\theta^* + \alpha d_1 + \beta d_2)$"
    )
    ax.set_title(plot_title, pad=12)
    ax.set_xlabel(r"Normalized Direction 1 ($\alpha$)", labelpad=8)
    ax.set_ylabel(r"Normalized Direction 2 ($\beta$)", labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.25)

    save_figure(fig, save_path, dpi=300)
    return fig


def plot_landscape_surface_3d(
    alphas: np.ndarray,
    betas: np.ndarray,
    loss_grid: np.ndarray,
    trajectory_coords: Tuple[Sequence[float], Sequence[float]] | None = None,
    save_path: str | Path = "landscape_3d.html",
    title: str | None = None,
) -> Any:
    """Render an interactive 3D surface mesh with Plotly, saved as a standalone HTML file.

    Args:
        alphas: 1D array of horizontal perturbation coordinates.
        betas: 1D array of vertical perturbation coordinates.
        loss_grid: 2D numpy array of evaluated losses.
        trajectory_coords: Optional tuple of (traj_alphas, traj_betas) path coordinates.
        save_path: Destination HTML file path.
        title: Plot title.

    Returns:
        Plotly Figure instance.
    """
    import plotly.graph_objects as go

    A, B = np.meshgrid(alphas, betas)

    surface = go.Surface(
        x=A,
        y=B,
        z=loss_grid,
        colorscale="Viridis",
        colorbar=dict(title=r"Loss L(θ)", len=0.75),
        opacity=0.92,
        name="Loss Surface",
    )

    data: list[Any] = [surface]

    # Overlay 3D trajectory
    if trajectory_coords is not None and len(trajectory_coords[0]) > 0:
        traj_a = list(trajectory_coords[0])
        traj_b = list(trajectory_coords[1])
        traj_z = [
            _sample_grid_loss(alphas, betas, loss_grid, a, b) * 1.01 for a, b in zip(traj_a, traj_b)
        ]

        traj_line = go.Scatter3d(
            x=traj_a,
            y=traj_b,
            z=traj_z,
            mode="lines+markers",
            line=dict(color="#F0883E", width=5),
            marker=dict(size=4, color="#F0883E"),
            name="Optimizer Path",
        )
        data.append(traj_line)

    fig = go.Figure(data=data)

    plot_title = title if title is not None else "3D Interactive Loss Landscape Surface"

    fig.update_layout(
        title=dict(text=plot_title, font=dict(color="#E6EDF3", size=14)),
        paper_bgcolor="#0E1117",
        scene=dict(
            xaxis=dict(
                title="Direction α",
                backgroundcolor="#161B22",
                gridcolor="#30363D",
                color="#E6EDF3",
            ),
            yaxis=dict(
                title="Direction β",
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
    )

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_file))
    return fig


def plot_side_by_side(
    grid_rand: dict[str, Any],
    grid_hess: dict[str, Any],
    save_path: str | Path = "landscape_comparison.png",
    title: str | None = None,
    cmap: str = "magma",
) -> plt.Figure:
    """Render side-by-side 2D contour comparison of random directions vs Hessian eigenvectors.

    Uses a shared color scale for fair comparative visualization across bases.

    Args:
        grid_rand: Grid dictionary from random filter-normalized directions.
        grid_hess: Grid dictionary from top Hessian eigenvectors.
        save_path: Destination PNG image file path.
        title: Overall super-title.
        cmap: Colormap name.

    Returns:
        Rendered Matplotlib Figure instance.
    """
    setup_style()

    alphas_r, betas_r, loss_r = grid_rand["alphas"], grid_rand["betas"], grid_rand["loss"]
    alphas_h, betas_h, loss_h = grid_hess["alphas"], grid_hess["betas"], grid_hess["loss"]

    Ar, Br = np.meshgrid(alphas_r, betas_r)
    Ah, Bh = np.meshgrid(alphas_h, betas_h)

    # Shared scale bounds
    vmin = min(float(np.min(loss_r)), float(np.min(loss_h)))
    vmax = max(float(np.max(loss_r)), float(np.max(loss_h)))
    if vmax - vmin < 1e-7:
        vmax = vmin + 1.0

    if vmin > 1e-6:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        levels = np.geomspace(vmin, vmax, 30)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        levels = np.linspace(vmin, vmax, 30)

    fig, (ax_r, ax_h) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

    # Left: Random filter-normalized directions
    ax_r.contourf(Ar, Br, loss_r, levels=levels, cmap=cmap, norm=norm, alpha=0.92)
    ax_r.contour(
        Ar,
        Br,
        loss_r,
        levels=levels[::3],
        colors="#E6EDF3",
        alpha=0.3,
        linewidths=0.6,
    )
    ax_r.scatter([0], [0], color=COLOR_ORANGE, marker="*", s=100, label=r"$\theta^*$ (0, 0)")
    ax_r.set_title("Random Filter-Normalized Directions\n(Obscures Curvature Anisotropy)", pad=10)
    ax_r.set_xlabel(r"Random Direction $d_1$", labelpad=8)
    ax_r.set_ylabel(r"Random Direction $d_2$", labelpad=8)
    ax_r.grid(True, linestyle="--", alpha=0.25)
    ax_r.legend(loc="upper right", framealpha=0.85)

    # Right: Hessian eigenvectors
    cf_h = ax_h.contourf(Ah, Bh, loss_h, levels=levels, cmap=cmap, norm=norm, alpha=0.92)
    ax_h.contour(
        Ah,
        Bh,
        loss_h,
        levels=levels[::3],
        colors="#E6EDF3",
        alpha=0.3,
        linewidths=0.6,
    )
    ax_h.scatter([0], [0], color=COLOR_ORANGE, marker="*", s=100, label=r"$\theta^*$ (0, 0)")
    ax_h.set_title("Top Hessian Eigenvectors\n(Exposes True Sharp Ravine along $q_1$)", pad=10)
    ax_h.set_xlabel(r"Sharp Eigenvector $q_1$ ($\lambda_{\max}$)", labelpad=8)
    ax_h.set_ylabel(r"Secondary Eigenvector $q_2$", labelpad=8)
    ax_h.grid(True, linestyle="--", alpha=0.25)
    ax_h.legend(loc="upper right", framealpha=0.85)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(cf_h, cax=cbar_ax)
    cbar.set_label(r"Loss $\mathcal{L}(\theta)$ (Log Scale)", labelpad=10)

    super_title = (
        title
        if title is not None
        else "Landscape Slicing: Projecting on Hessian Eigenvectors Exposes True Sharpness"
    )
    fig.suptitle(super_title, y=0.98, fontsize=13, fontweight="bold", color="#E6EDF3")

    save_figure(fig, save_path, dpi=300)
    return fig
