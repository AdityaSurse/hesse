"""End-to-end smoke tests and integration pipeline tests for hesse.

Verifies the full research cycle:
train -> track -> landscape -> report -> CLI.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hesse.cli import main
from hesse.config import DataConfig, ModelConfig, OptimConfig, RunConfig, TrackerConfig
from hesse.data import (
    _create_synthetic_classification,
    _select_balanced_subset,
    get_cifar10_subset,
    get_mnist_subset,
    get_synthetic_regression,
    load_dataset,
)
from hesse.landscape.directions import (
    create_pca_directions,
    get_directions,
)
from hesse.landscape.grid import evaluate_1d, evaluate_2d
from hesse.report import generate_report
from hesse.train import train


def test_end_to_end_train_track_landscape_report(tmp_path: Path):
    """Smoke test: train tiny model, track curvature, compute landscape, export report."""
    run_dir = tmp_path / "smoke_run"
    cfg = RunConfig(
        run_name=str(run_dir),
        data=DataConfig(
            dataset="synthetic",
            input_dim=16,
            subset_size=32,
            batch_size=32,
            loss_type="mse",
        ),
        model=ModelConfig(
            architecture="mlp",
            input_dim=16,
            hidden_dims=[8],
            num_classes=1,
            activation="relu",
        ),
        optim=OptimConfig(
            optimizer="gd",
            lr=0.001,
        ),
        tracker=TrackerConfig(
            track_interval=2,
            compute_topk=2,
            compute_trace=True,
            trace_samples=5,
            save_eigenvectors=True,
        ),
        steps=6,
        checkpoint_interval=2,
        seed=42,
    )

    # 1. Train & Track
    summary = train(cfg)
    assert summary["status"] == "completed"
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint_final.pt").is_file()
    assert (run_dir / "eigvecs" / "eigvecs_step_2.pt").is_file()

    # 2. Landscape Slicing
    from hesse.models import get_model

    bundle = load_dataset(cfg.data)
    model = get_model(cfg.model)
    ckpt = torch.load(run_dir / "checkpoints" / "checkpoint_final.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # Directions: Random, Hessian, PCA
    d1_rand, d2_rand = get_directions("random", model=model, seed=123)
    p_dim = sum(p.numel() for p in model.parameters())
    assert d1_rand.shape == (p_dim,)

    eval_data = bundle.full_batch()
    d1_hess, d2_hess = get_directions(
        "hessian",
        model=model,
        loss_fn=nn.MSELoss(),
        data=eval_data,
        seed=123,
    )
    assert d1_hess.shape == (p_dim,)

    ckpts = [
        torch.load(run_dir / "checkpoints" / "checkpoint_step_2.pt", weights_only=False)[
            "model_state"
        ],
        torch.load(run_dir / "checkpoints" / "checkpoint_step_4.pt", weights_only=False)[
            "model_state"
        ],
    ]
    # Flatten checkpoints for PCA
    flat_ckpts = [
        torch.cat([v.reshape(-1) for v in c.values()]).to(dtype=torch.float32) for c in ckpts
    ]
    d1_pca, d2_pca = get_directions("pca", model=model, checkpoints=flat_ckpts)
    assert d1_pca.shape == (p_dim,)

    # 1D & 2D Grid evaluations
    alphas_1d = np.linspace(-0.5, 0.5, 5)
    res_1d = evaluate_1d(model, nn.MSELoss(), bundle, d1_rand, alphas=alphas_1d)
    assert len(res_1d["loss"]) == 5

    # Test evaluate_1d with a DataLoader to exercise data loader batching
    tensor_ds = TensorDataset(eval_data[0], eval_data[1])
    loader = DataLoader(tensor_ds, batch_size=8)
    res_loader = evaluate_1d(model, nn.MSELoss(), loader, d1_rand, alphas=alphas_1d)
    assert len(res_loader["loss"]) == 5

    alphas = torch.linspace(-0.5, 0.5, 3)
    betas = torch.linspace(-0.5, 0.5, 3)
    grid_out = evaluate_2d(
        model,
        nn.MSELoss(),
        eval_data,
        d1_rand,
        d2_rand,
        alphas=alphas,
        betas=betas,
        show_progress=False,
    )
    assert grid_out["loss"].shape == (3, 3)

    # Save landscape npz so report generator finds it
    landscape_dir = run_dir / "landscape"
    landscape_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        landscape_dir / "grid_random.npz",
        loss=grid_out["loss"],
        alphas=alphas.numpy(),
        betas=betas.numpy(),
        trajectory_alphas=np.array([0.0, 0.1]),
        trajectory_betas=np.array([0.0, 0.1]),
    )

    # 3. Report Generation
    report_path = run_dir / "report.html"
    out = generate_report(run_dir=run_dir, out_path=report_path)
    assert out.is_file()
    content = out.read_text(encoding="utf-8")
    assert "hesse Telemetry Report" in content
    assert "kpi-card" in content


def test_train_adam_and_lion_tracking(tmp_path: Path):
    """Test training loops with Adam and Lion optimizers to exercise tracking hooks."""
    # Adam
    run_adam = tmp_path / "adam_run"
    cfg_adam = RunConfig(
        run_name=str(run_adam),
        data=DataConfig(
            dataset="synthetic",
            input_dim=8,
            subset_size=16,
            batch_size=8,
            loss_type="mse",
        ),
        model=ModelConfig(architecture="mlp", input_dim=8, hidden_dims=[4], num_classes=1),
        optim=OptimConfig(optimizer="adam", lr=0.001, beta1=0.9, beta2=0.999),
        tracker=TrackerConfig(track_interval=2),
        steps=4,
    )
    res_adam = train(cfg_adam)
    assert res_adam["status"] == "completed"

    # Lion
    run_lion = tmp_path / "lion_run"
    cfg_lion = RunConfig(
        run_name=str(run_lion),
        data=DataConfig(
            dataset="synthetic",
            input_dim=8,
            subset_size=16,
            batch_size=8,
            loss_type="mse",
        ),
        model=ModelConfig(architecture="mlp", input_dim=8, hidden_dims=[4], num_classes=1),
        optim=OptimConfig(optimizer="lion", lr=0.0003, beta1=0.9, beta2=0.99),
        tracker=TrackerConfig(track_interval=2),
        steps=4,
    )
    res_lion = train(cfg_lion)
    assert res_lion["status"] == "completed"


def test_train_error_and_edge_branches(tmp_path: Path):
    """Test error handling in train(): missing checkpoint and unsupported loss."""
    cfg = RunConfig(
        run_name=str(tmp_path / "err_run"),
        data=DataConfig(dataset="synthetic", input_dim=4, subset_size=16, loss_type="mse"),
        model=ModelConfig(architecture="mlp", input_dim=4, num_classes=1, hidden_dims=[4]),
        optim=OptimConfig(optimizer="gd", lr=0.01),
        tracker=TrackerConfig(track_interval=1),
        steps=2,
    )

    with pytest.raises(FileNotFoundError):
        train(cfg, resume_from=tmp_path / "nonexistent.pt")


def test_cli_landscape_subcommand(tmp_path: Path, monkeypatch):
    """Test CLI landscape invocation with random and hessian directions."""
    run_dir = tmp_path / "cli_run"
    cfg = RunConfig(
        run_name=str(run_dir),
        data=DataConfig(
            dataset="synthetic",
            input_dim=8,
            subset_size=16,
            batch_size=16,
            loss_type="mse",
        ),
        model=ModelConfig(architecture="mlp", input_dim=8, num_classes=1, hidden_dims=[4]),
        optim=OptimConfig(optimizer="gd", lr=0.01),
        tracker=TrackerConfig(track_interval=1),
        steps=2,
        checkpoint_interval=1,
    )
    train(cfg)

    # Run CLI landscape with random
    cmd_random = [
        "hesse",
        "landscape",
        "--run",
        str(run_dir),
        "--directions",
        "random",
        "--grid-size",
        "3",
    ]
    monkeypatch.setattr("sys.argv", cmd_random)
    ret = main()
    assert ret == 0

    # Run CLI landscape with hessian
    cmd_hessian = [
        "hesse",
        "landscape",
        "--run",
        str(run_dir),
        "--directions",
        "hessian",
        "--grid-size",
        "3",
    ]
    monkeypatch.setattr("sys.argv", cmd_hessian)
    ret_hessian = main()
    assert ret_hessian == 0


def test_directions_and_pca_edges():
    """Test PCA edge branches (rank=1) and invalid get_directions calls."""
    p_dim = 10
    v = torch.randn(p_dim)
    # Collinear checkpoints -> rank 1
    ckpts = [v, 2.0 * v, 3.0 * v]
    u1, u2 = create_pca_directions(ckpts)
    assert u1.shape == (p_dim,)
    assert u2.shape == (p_dim,)
    assert abs(torch.dot(u1, u2).item()) < 1e-4

    with pytest.raises(ValueError, match="at least 2 checkpoints"):
        create_pca_directions([v])

    model = nn.Linear(4, 2)
    with pytest.raises(ValueError, match="require both loss_fn and data"):
        get_directions("hessian", model=model)

    with pytest.raises(ValueError, match="require at least 2 checkpoints"):
        get_directions("pca", model=model, checkpoints=[])

    with pytest.raises(ValueError, match="Unknown direction method"):
        get_directions("unknown_method", model=model)


def test_stratified_subset_and_data_utilities():
    """Test stratified sampling and data utilities directly."""
    targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]).numpy()
    # Test _select_balanced_subset
    idx_bal = _select_balanced_subset(targets, subset_size=6, num_classes=3, seed=42)
    assert len(idx_bal) == 6

    # Test synthetic classification generator
    x_syn, y_syn = _create_synthetic_classification(10, (4,), 2, seed=42)
    assert x_syn.shape == (10, 4)
    assert y_syn.shape == (10,)

    # Test get_cifar10_subset and get_mnist_subset synthetic fallback
    cifar_bundle = get_cifar10_subset(subset_size=20, val_size=10, seed=42)
    assert cifar_bundle.train_x.shape[0] == 20
    assert cifar_bundle.val_x.shape[0] == 10

    mnist_bundle = get_mnist_subset(subset_size=20, val_size=10, seed=42)
    assert mnist_bundle.train_x.shape[0] == 20
    assert mnist_bundle.val_x.shape[0] == 10

    reg_bundle = get_synthetic_regression(num_samples=20, input_dim=4, noise_std=0.01)
    assert reg_bundle.train_x.shape == (20, 4)
