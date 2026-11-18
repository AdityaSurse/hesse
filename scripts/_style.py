"""Shared styling definitions for dark research-lab publication figures."""

from __future__ import annotations

import os
from pathlib import Path

# Set writable cache directory for Matplotlib in sandboxed environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Research lab dark theme palette
BG_COLOR = "#0E1117"
AXIS_BG = "#161B22"
TEXT_COLOR = "#E6EDF3"
GRID_COLOR = "#30363D"
GRID_ALPHA = 0.6

# Semantic color constants
COLOR_BLUE = "#58A6FF"  # Primary curves, empirical estimations
COLOR_ORANGE = "#F0883E"  # Critical stability threshold boundaries (dashed)
COLOR_GREEN = "#3FB950"  # Stable regime, convergence trajectories
COLOR_PINK = "#DB61A2"  # Unstable regime, divergence trajectories
COLOR_PURPLE = "#A371F7"  # Secondary distributions, spectral histograms
COLOR_YELLOW = "#D29922"  # Highlights, markers, outlier annotations


def setup_style() -> None:
    """Apply global matplotlib style parameters for dark-themed figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": BG_COLOR,
            "axes.facecolor": AXIS_BG,
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.labelsize": 11,
            "text.color": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "grid.color": GRID_COLOR,
            "grid.alpha": GRID_ALPHA,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "mathtext.fontset": "cm",
            "legend.facecolor": AXIS_BG,
            "legend.edgecolor": GRID_COLOR,
            "legend.fontsize": 10,
        }
    )


def save_figure(fig: plt.Figure, output_path: str | Path, dpi: int = 200) -> None:
    """Save figure with dark background and tight bounding box.

    Args:
        fig: Target matplotlib Figure.
        output_path: File destination.
        dpi: Resolution (default 200 for ~2000px wide).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=dpi,
        facecolor=BG_COLOR,
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)
