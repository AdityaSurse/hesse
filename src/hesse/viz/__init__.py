"""Visualization modules for telemetry trajectories, spectra, and loss landscapes."""

from __future__ import annotations

from hesse.viz.curves import plot_run
from hesse.viz.landscape_plots import (
    plot_landscape_contour,
    plot_landscape_surface_3d,
    plot_side_by_side,
)
from hesse.viz.style import (
    BG_COLOR,
    COLOR_BLUE,
    COLOR_GD,
    COLOR_GREEN,
    COLOR_LION,
    COLOR_ORANGE,
    COLOR_PINK,
    COLOR_PURPLE,
    COLOR_SGD,
    COLOR_THRESHOLD,
    COLOR_YELLOW,
    save_figure,
    setup_style,
)

__all__ = [
    "BG_COLOR",
    "COLOR_ADAM",
    "COLOR_BLUE",
    "COLOR_GD",
    "COLOR_GREEN",
    "COLOR_LION",
    "COLOR_ORANGE",
    "COLOR_PINK",
    "COLOR_PURPLE",
    "COLOR_SGD",
    "COLOR_THRESHOLD",
    "COLOR_YELLOW",
    "plot_landscape_contour",
    "plot_landscape_surface_3d",
    "plot_run",
    "plot_side_by_side",
    "save_figure",
    "setup_style",
]
