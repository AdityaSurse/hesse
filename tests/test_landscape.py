"""Unit tests for loss-landscape evaluation, slicing, and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from hesse.cli import main
from hesse.config import ModelConfig, RunConfig
from hesse.landscape import (
    create_hessian_directions,
    create_pca_directions,
    create_random_direction,
    evaluate_1d,
    evaluate_2d,
    interpolate_1d,
    load_grid_npz,
    project_trajectory_2d,
    save_grid_npz,
)
from hesse.models import get_model
from hesse.train import train
from hesse.utils import flatten_params, unflatten_params
from hesse.viz import (
    plot_landscape_contour,
    plot_landscape_surface_3d,
    plot_side_by_side,
)


class Quadratic2DModel(nn.Module):
    """Minimal 2-parameter model computing 1/2 * theta^T * A * theta."""

    def __init__(self, A: torch.Tensor) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))
        self.register_buffer("A", A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns scalar loss wrapped in 1D tensor
        val = 0.5 * torch.sum(self.theta * (self.A @ self.theta))
        return val.unsqueeze(0)


def test_quadratic_eigenvector_landscape() -> None:
    """Verify that on a 2D quadratic 1/2 theta^T A theta with A = diag(lambda1, lambda2),

    the grid along the eigenvectors reproduces 1/2 (lambda1 alpha^2 + lambda2 beta^2) within 1e-5.
    """
    lambda1 = 8.0
    lambda2 = 2.0
    A = torch.tensor([[lambda1, 0.0], [0.0, lambda2]], dtype=torch.float32)

    model = Quadratic2DModel(A)
    loss_fn = lambda preds, target: preds.squeeze()  # noqa: E731
    dummy_data = (torch.zeros(1, 2), torch.zeros(1))

    # Compute top-2 Hessian eigenvectors
    q1, q2 = create_hessian_directions(model, loss_fn, dummy_data, seed=42)

    alphas = np.linspace(-1.0, 1.0, 15)
    betas = np.linspace(-1.0, 1.0, 15)

    res = evaluate_2d(
        model=model,
        loss_fn=loss_fn,
        data=dummy_data,
        d1=q1,
        d2=q2,
        alphas=alphas,
        betas=betas,
        show_progress=False,
    )

    A_grid, B_grid = np.meshgrid(alphas, betas)
    expected_loss = 0.5 * (lambda1 * (A_grid**2) + lambda2 * (B_grid**2))

    max_abs_diff = float(np.max(np.abs(res["loss"] - expected_loss)))
    assert max_abs_diff < 1e-5, f"Quadratic grid max diff {max_abs_diff} exceeds 1e-5"


def test_filter_normalized_directions() -> None:
    """Verify filter-normalized directions match per-filter Frobenius norms of theta to 1e-6

    and zero out 1D parameters/biases (Li et al., 2018).
    """
    cfg = ModelConfig(
        architecture="mlp",
        hidden_dims=[32, 16],
        input_dim=64,
        num_classes=10,
    )
    model = get_model(cfg, seed=42)
    d = create_random_direction(model, seed=42, filter_normalized=True)

    dummy_model = get_model(cfg, seed=42)
    unflatten_params(dummy_model, d)

    max_norm_diff = 0.0
    for (name, p), (_, d_p) in zip(model.named_parameters(), dummy_model.named_parameters()):
        if p.ndim >= 2:
            for j in range(p.shape[0]):
                w_norm = torch.linalg.norm(p[j]).item()
                d_norm = torch.linalg.norm(d_p[j]).item()
                diff = abs(w_norm - d_norm)
                max_norm_diff = max(max_norm_diff, diff)
                assert diff < 1e-6, f"Norm mismatch on {name} slice {j}: {w_norm} vs {d_norm}"
        else:
            # 1D biases and vectors must be strictly 0.0
            assert torch.all(d_p == 0.0), f"1D parameter {name} was not zeroed out"

    assert max_norm_diff < 1e-6


def test_model_weights_bit_identical_restoration() -> None:
    """Assert that model weights remain bit-identical before and after 1D, 2D, and

    interpolation evaluations.
    """
    cfg = ModelConfig(
        architecture="mlp",
        hidden_dims=[20, 10],
        input_dim=32,
        num_classes=5,
    )
    model = get_model(cfg, seed=123)
    initial_params = [p.clone() for p in model.parameters()]

    d1 = create_random_direction(model, seed=10)
    d2 = create_random_direction(model, seed=11)
    data = (torch.randn(8, 32), torch.randint(0, 5, (8,)))
    loss_fn = nn.CrossEntropyLoss()

    # 1. Test evaluate_1d
    evaluate_1d(model, loss_fn, data, d1, [-0.5, 0.0, 0.5])
    for p, orig in zip(model.parameters(), initial_params):
        assert torch.equal(p, orig), "Weights altered after evaluate_1d"

    # 2. Test evaluate_2d
    evaluate_2d(
        model,
        loss_fn,
        data,
        d1,
        d2,
        [-0.2, 0.0, 0.2],
        [-0.2, 0.0, 0.2],
        show_progress=False,
    )
    for p, orig in zip(model.parameters(), initial_params):
        assert torch.equal(p, orig), "Weights altered after evaluate_2d"

    # 3. Test interpolate_1d
    th_init = d1.clone()
    th_final = d2.clone()
    interpolate_1d(model, loss_fn, data, th_init, th_final, [0.0, 0.5, 1.0])
    for p, orig in zip(model.parameters(), initial_params):
        assert torch.equal(p, orig), "Weights altered after interpolate_1d"


def test_trajectory_projection_origin() -> None:
    """Verify that projecting an optimization trajectory centers the final checkpoint

    exactly at (0.0, 0.0).
    """
    p_dim = 40
    ckpts = [torch.randn(p_dim) for _ in range(6)]
    d1 = torch.randn(p_dim)
    d2 = torch.randn(p_dim)

    # Default reference (last checkpoint)
    alphas, betas = project_trajectory_2d(ckpts, d1, d2)
    assert len(alphas) == 6
    assert len(betas) == 6
    assert abs(alphas[-1]) < 1e-6
    assert abs(betas[-1]) < 1e-6

    # Explicit reference
    alphas_ref, betas_ref = project_trajectory_2d(ckpts, d1, d2, reference_theta=ckpts[-1])
    assert abs(alphas_ref[-1]) < 1e-6
    assert abs(betas_ref[-1]) < 1e-6


def test_pca_directions() -> None:
    """Verify that PCA directions are orthonormal and span trajectory variations."""
    p_dim = 25
    ckpts = [torch.randn(p_dim) for _ in range(5)]
    u1, u2 = create_pca_directions(ckpts)

    assert u1.shape == (p_dim,)
    assert u2.shape == (p_dim,)

    norm_u1 = torch.linalg.vector_norm(u1).item()
    norm_u2 = torch.linalg.vector_norm(u2).item()
    assert abs(norm_u1 - 1.0) < 1e-5
    assert abs(norm_u2 - 1.0) < 1e-5

    dot_prod = torch.dot(u1, u2).item()
    assert abs(dot_prod) < 1e-5


def test_interpolate_1d_endpoints() -> None:
    """Verify that 1D linear interpolation reproduces endpoint evaluations accurately."""
    cfg = ModelConfig(
        architecture="mlp",
        hidden_dims=[16],
        input_dim=16,
        num_classes=2,
    )
    model = get_model(cfg, seed=77)
    loss_fn = nn.CrossEntropyLoss()
    data = (torch.randn(6, 16), torch.randint(0, 2, (6,)))

    th_init = flatten_params(model).clone()
    th_final = th_init + 0.1 * torch.randn_like(th_init)

    # Evaluate at alpha = 0.0 and alpha = 1.0
    res = interpolate_1d(model, loss_fn, data, th_init, th_final, alphas=[0.0, 1.0])

    # Direct evaluation at th_init
    unflatten_params(model, th_init)
    l_init = loss_fn(model(data[0]), data[1]).item()

    # Direct evaluation at th_final
    unflatten_params(model, th_final)
    l_final = loss_fn(model(data[0]), data[1]).item()

    # Restore
    unflatten_params(model, th_init)

    assert abs(res["loss"][0] - l_init) < 1e-5
    assert abs(res["loss"][1] - l_final) < 1e-5


def test_grid_npz_roundtrip(tmp_path: Path) -> None:
    """Verify that saving and loading landscape grids via compressed .npz preserves all data."""
    alphas = np.linspace(-1.5, 1.5, 9, dtype=np.float32)
    betas = np.linspace(-1.5, 1.5, 9, dtype=np.float32)
    loss = np.random.randn(9, 9).astype(np.float32)
    acc = np.random.rand(9, 9).astype(np.float32)
    meta = {"method": "hessian", "span": 1.5, "checkpoint": "step_100.pt"}

    out_file = tmp_path / "grid_test.npz"
    save_grid_npz(out_file, alphas, betas, loss, acc, meta)

    loaded = load_grid_npz(out_file)
    assert np.allclose(loaded["alphas"], alphas)
    assert np.allclose(loaded["betas"], betas)
    assert np.allclose(loaded["loss"], loss)
    assert loaded["acc"] is not None
    assert np.allclose(loaded["acc"], acc)
    assert loaded["metadata"] == meta


def test_visualizations(tmp_path: Path) -> None:
    """Verify that 2D contour, 3D Plotly surface, and side-by-side comparison plots render."""
    alphas = np.linspace(-1.0, 1.0, 11)
    betas = np.linspace(-1.0, 1.0, 11)
    loss_r = np.outer(np.linspace(1, 2, 11), np.linspace(1, 2, 11))
    loss_h = np.outer(np.linspace(0.5, 3, 11), np.linspace(0.5, 3, 11))
    traj: tuple[list[float], list[float]] = ([-0.5, -0.2, 0.0], [0.4, 0.1, 0.0])

    # 1. 2D contour
    png_path = tmp_path / "contour.png"
    plot_landscape_contour(alphas, betas, loss_r, trajectory_coords=traj, save_path=png_path)
    assert png_path.is_file()
    assert png_path.stat().st_size > 0

    # 2. 3D surface HTML
    html_path = tmp_path / "surface.html"
    plot_landscape_surface_3d(alphas, betas, loss_r, trajectory_coords=traj, save_path=html_path)
    assert html_path.is_file()
    assert html_path.stat().st_size > 0

    # 3. Side-by-side comparison
    grid_r: dict[str, Any] = {"alphas": alphas, "betas": betas, "loss": loss_r}
    grid_h: dict[str, Any] = {"alphas": alphas, "betas": betas, "loss": loss_h}
    comp_path = tmp_path / "side_by_side.png"
    plot_side_by_side(grid_r, grid_h, save_path=comp_path)
    assert comp_path.is_file()
    assert comp_path.stat().st_size > 0


def test_cli_landscape_integration(tmp_path: Path) -> None:
    """Verify that `hesse landscape` subcommand executes end-to-end on a trained run."""
    run_dir = tmp_path / "test_run"
    cfg = RunConfig(
        run_name=str(run_dir),
        steps=2,
        checkpoint_interval=1,
    )
    cfg.data.dataset = "cifar10"
    cfg.data.subset_size = 20
    cfg.tracker.track_interval = 1

    train(cfg)
    assert (run_dir / "checkpoints" / "checkpoint_final.pt").is_file()

    # Run CLI landscape command
    ret = main(
        [
            "landscape",
            "--run",
            str(run_dir),
            "--directions",
            "random",
            "--res",
            "7",
            "--span",
            "0.5",
        ]
    )
    assert ret == 0

    landscape_dir = run_dir / "landscape"
    assert (landscape_dir / "grid_random.npz").is_file()
    assert (landscape_dir / "landscape_contour_random.png").is_file()
    assert (landscape_dir / "landscape_3d_random.html").is_file()
