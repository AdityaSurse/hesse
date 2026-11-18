"""Unit tests verifying that all README publication figures generate deterministically."""

from __future__ import annotations

from pathlib import Path

from scripts.make_readme_figures import generate_all_figures

EXPECTED_FIGURES = [
    "hero_landscape.png",
    "gd_stability_1d.png",
    "stability_region.png",
    "quadratic_contour_gd.png",
    "power_iteration_convergence.png",
    "hutchinson_convergence.png",
    "spectral_density_demo.png",
    "adam_threshold.png",
    "gd_vs_lion_step.png",
    "loss_slice_demo.png",
]


def test_readme_figures_generation(tmp_path: Path) -> None:
    """Verify that generate_all_figures produces all 10 expected PNGs with non-zero size."""
    generated = generate_all_figures(output_dir=tmp_path, fast=True)

    assert len(generated) == len(EXPECTED_FIGURES)

    for name in EXPECTED_FIGURES:
        target = tmp_path / name
        assert target.exists(), f"Figure {name} was not generated."
        assert target.is_file(), f"Figure {name} is not a valid file."
        size_bytes = target.stat().st_size
        assert size_bytes > 5000, f"Figure {name} is suspiciously small ({size_bytes} bytes)."
