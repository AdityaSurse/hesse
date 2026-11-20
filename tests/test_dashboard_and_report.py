"""Unit and integration tests for hesse report generation and dashboard utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.utils import (
    build_comparison_charts,
    build_density_plot,
    get_available_runs,
    get_optimizer_color,
    load_run_data,
)
from hesse.cli import main
from hesse.report import (
    build_landscape_3d_fig,
    build_landscape_contour_fig,
    build_live_run_fig,
    build_topk_fig,
    generate_report,
)
from hesse.viz.style import COLOR_ADAM, COLOR_GD, COLOR_LION, COLOR_SGD


def test_cli_report_subcommand(tmp_path: Path) -> None:
    """Test 'hesse report' CLI command generates a self-contained HTML file."""
    out_html = tmp_path / "custom_report.html"
    exit_code = main(["report", "--run", "runs/gd_mlp_cifar5k", "--out", str(out_html)])
    assert exit_code == 0
    assert out_html.exists()
    content = out_html.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "hesse Telemetry Report" in content
    assert "plotly" in content.lower()


def test_generate_report_empty_or_partial_run(tmp_path: Path) -> None:
    """Verify generate_report handles runs without metrics gracefully without errors."""
    empty_run = tmp_path / "empty_run"
    empty_run.mkdir()
    out_html = empty_run / "report.html"

    report_path = generate_report(empty_run, out_path=out_html)
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "No training metrics recorded" in content


def test_get_available_runs() -> None:
    """Verify get_available_runs discovers existing directories in runs/."""
    runs = get_available_runs("runs")
    assert isinstance(runs, list)
    assert len(runs) > 0
    # Must contain at least one of our standard runs
    assert any("gd_mlp_cifar5k" in r or "e1" in r for r in runs)


def test_load_run_data_nonexistent(tmp_path: Path) -> None:
    """Verify load_run_data handles non-existent paths safely."""
    data = load_run_data(tmp_path / "does_not_exist")
    assert data["exists"] is False
    assert data["records"] == []


def test_get_optimizer_color() -> None:
    """Verify optimizer color mapping matches canonical project palette."""
    assert get_optimizer_color("gd") == COLOR_GD
    assert get_optimizer_color("SGD") == COLOR_SGD
    assert get_optimizer_color("Adam") == COLOR_ADAM
    assert get_optimizer_color("adamw") == COLOR_ADAM
    assert get_optimizer_color("lion") == COLOR_LION
    assert get_optimizer_color(None) == "#E6EDF3"


def test_build_live_run_fig() -> None:
    """Verify live run figure builds with subplots and dashed threshold line."""
    records = [
        {"step": 0, "loss": 2.5, "sharpness": 10.0, "stability_threshold": 40.0},
        {"step": 5, "loss": 1.8, "sharpness": 25.0, "stability_threshold": 40.0},
        {"step": 10, "loss": 1.2, "sharpness": 38.0, "stability_threshold": 40.0},
    ]
    fig = build_live_run_fig(records, threshold=40.0, title="Test Fig")
    assert fig is not None
    # 2 rows of subplots
    assert len(fig.data) >= 3  # sharpness, threshold, loss


def test_build_topk_fig() -> None:
    """Verify top-k eigenvalues plot handles multi-step records."""
    records = [
        {"step": 0, "topk_eigenvalues": [12.0, 8.0, 5.0]},
        {"step": 5, "topk_eigenvalues": [15.0, 9.0, 6.0]},
    ]
    fig = build_topk_fig(records)
    assert fig is not None
    assert len(fig.data) == 3

    # None if no topk
    assert build_topk_fig([{"step": 0, "loss": 1.0}]) is None


def test_build_landscape_plotly_figures() -> None:
    """Verify 3D surface and 2D contour figures compile cleanly."""
    alphas = np.linspace(-1.0, 1.0, 11)
    betas = np.linspace(-1.0, 1.0, 11)
    A, B = np.meshgrid(alphas, betas)
    loss = A**2 + B**2

    fig_3d = build_landscape_3d_fig(alphas, betas, loss, trajectory_coords=([0.0], [0.0]))
    assert fig_3d is not None

    fig_cnt = build_landscape_contour_fig(alphas, betas, loss, trajectory_coords=([0.0], [0.0]))
    assert fig_cnt is not None


def test_build_comparison_charts() -> None:
    """Verify comparative run overlays and table generation."""
    runs = get_available_runs("runs")
    sample_runs = runs[:2] if len(runs) >= 2 else runs
    fig_sharp, fig_loss, table_rows = build_comparison_charts(sample_runs)

    assert fig_sharp is not None
    assert fig_loss is not None
    assert isinstance(table_rows, list)


def test_build_density_plot() -> None:
    """Verify SLQ density figure builds with log-scale support."""
    grid = np.linspace(-2.0, 10.0, 50)
    density = np.exp(-(grid**2) / 2.0)
    fig = build_density_plot(grid, density, step=20, log_scale=True)
    assert fig is not None
    assert fig.layout.yaxis.type == "log"
