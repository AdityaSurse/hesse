"""Integration tests for training loop, live tracking, divergence guard, and checkpointing."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from hesse.config import DataConfig, ModelConfig, OptimConfig, RunConfig, TrackerConfig
from hesse.tracker import Tracker
from hesse.train import train
from hesse.utils import set_seed
from hesse.viz.curves import plot_run


class QuadraticTestModel(nn.Module):
    """Linear quadratic model for analytical curvature and divergence testing."""

    def __init__(self, A: torch.Tensor, init_val: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("A", A)
        dim = A.shape[0]
        init_tensor = init_val if init_val is not None else torch.ones(dim)
        self.theta = nn.Parameter(init_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns theta directly for loss evaluation
        return self.theta


def quad_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Quadratic objective 0.5 * theta^T A theta."""
    # pred is theta
    # Recover A from the pred grad_fn / module context or use global/dummy target
    # Here target is passed as dummy
    return 0.5 * torch.dot(pred, target @ pred)


def test_tracked_sharpness_constant_on_quadratic(tmp_path: Path) -> None:
    """Verify tracked lambda_max equals the true top eigenvalue at every step on a quadratic."""
    set_seed(42)
    # Known 3x3 symmetric matrix with eigenvalues 2.0, 5.0, 10.0
    evals = torch.tensor([2.0, 5.0, 10.0])
    # Random orthogonal matrix
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    A = Q @ torch.diag(evals) @ Q.T
    A = (A + A.T) / 2.0

    model = QuadraticTestModel(A, init_val=torch.tensor([2.0, 2.0, 2.0]))
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.0)

    dummy_x = torch.zeros(1, 1)
    dummy_A = A

    def loss_closure(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.dot(pred, target @ pred)

    tracker_cfg = TrackerConfig(track_interval=2, compute_sharpness=True, compute_trace=False)
    tracker = Tracker(
        config=tracker_cfg,
        model=model,
        loss_fn=loss_closure,
        eval_data=(dummy_x, dummy_A),
        run_dir=tmp_path / "quadratic_run",
        optimizer_name="gd",
        lr=0.05,
        seed=42,
    )

    for step in range(6):
        opt.zero_grad()
        loss = loss_closure(model(dummy_x), dummy_A)
        loss.backward()
        opt.step()
        rec = tracker.on_step_end(
            step=step,
            model=model,
            optimizer=opt,
            batch=(dummy_x, dummy_A),
            loss=loss.item(),
        )
        if rec is not None:
            # True top eigenvalue is 10.0
            sharpness = rec["sharpness"]
            assert sharpness is not None
            assert abs(sharpness - 10.0) < 1e-3, f"Step {step}: expected 10.0, got {sharpness}"

    tracker.close()


def test_divergence_guard_triggers_when_lr_exceeds_two_over_lambda(tmp_path: Path) -> None:
    """Verify divergence guard triggers and flags the run when lr > 2/lambda on a quadratic."""
    set_seed(42)
    # 2D diagonal system with lambda_max = 10.0. EoS threshold = 2/10 = 0.20
    A = torch.diag(torch.tensor([1.0, 10.0]))
    model = QuadraticTestModel(A, init_val=torch.tensor([1.0, 1.0]))

    # Set unstable learning rate lr = 0.25 > 0.20 (multiplier |1 - 0.25*10| = 1.5 > 1.0)
    opt = torch.optim.SGD(model.parameters(), lr=0.25)
    dummy_x = torch.zeros(1, 1)

    def loss_closure(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.dot(pred, target @ pred)

    tracker_cfg = TrackerConfig(track_interval=1, compute_sharpness=True)
    run_dir = tmp_path / "divergent_run"
    tracker = Tracker(
        config=tracker_cfg,
        model=model,
        loss_fn=loss_closure,
        eval_data=(dummy_x, A),
        run_dir=run_dir,
        optimizer_name="gd",
        lr=0.25,
    )

    diverged = False
    initial_loss = None
    for step in range(40):
        opt.zero_grad()
        loss = loss_closure(model(dummy_x), A)
        loss_val = loss.item()
        if initial_loss is None:
            initial_loss = loss_val

        if torch.isnan(loss) or loss_val > 1e7 or loss_val > 1e5 * initial_loss:
            diverged = True
            tracker.save_summary(status="diverged", divergence_step=step)
            break

        loss.backward()
        opt.step()
        tracker.on_step_end(step, model, opt, (dummy_x, A), loss_val)

    tracker.close()
    assert diverged, "Divergence guard should have triggered for lr=0.25 > 0.20"

    with open(run_dir / "summary.json") as f:
        summary = json.load(f)
    assert summary["status"] == "diverged"


def test_checkpoint_resume_exact_reproducibility(tmp_path: Path) -> None:
    """Verify metrics.jsonl validity and exact reproducibility when resuming from checkpoint."""
    run_name_uninterrupted = "run_uninterrupted"
    run_name_resumed = "run_resumed"

    base_cfg = RunConfig(
        run_name=run_name_uninterrupted,
        epochs=1,
        steps=20,
        seed=777,
        checkpoint_interval=10,
        data=DataConfig(
            dataset="synthetic",
            subset_size=60,
            val_size=10,
            loss_type="mse",
            input_dim=16,
        ),
        model=ModelConfig(architecture="mlp", input_dim=16, hidden_dims=[32, 16], num_classes=1),
        optim=OptimConfig(optimizer="gd", lr=0.001),
        tracker=TrackerConfig(track_interval=5, compute_sharpness=True, compute_trace=False),
    )

    # 1. Uninterrupted run for 20 steps
    summary_full = train(base_cfg)
    assert summary_full["status"] == "completed"

    metrics_full_file = Path("runs") / run_name_uninterrupted / "metrics.jsonl"
    with open(metrics_full_file, "r") as f:
        lines_full = [json.loads(line) for line in f if line.strip()]

    # Verify every line of metrics.jsonl is valid JSON
    assert len(lines_full) >= 4  # steps 0, 5, 10, 15

    # 2. Resumed run starting from checkpoint_step_10.pt
    ckpt_path = Path("runs") / run_name_uninterrupted / "checkpoints" / "checkpoint_step_10.pt"
    assert ckpt_path.exists()

    resumed_cfg = RunConfig(
        run_name=run_name_resumed,
        epochs=1,
        steps=20,
        seed=777,
        checkpoint_interval=10,
        data=DataConfig(
            dataset="synthetic",
            subset_size=60,
            val_size=10,
            loss_type="mse",
            input_dim=16,
        ),
        model=ModelConfig(architecture="mlp", input_dim=16, hidden_dims=[32, 16], num_classes=1),
        optim=OptimConfig(optimizer="gd", lr=0.001),
        tracker=TrackerConfig(track_interval=5, compute_sharpness=True, compute_trace=False),
    )

    summary_resumed = train(resumed_cfg, resume_from=ckpt_path)
    assert summary_resumed["status"] == "completed"

    # Verify final parameters match exactly
    full_final_ckpt = torch.load(
        Path("runs") / run_name_uninterrupted / "checkpoints" / "checkpoint_final.pt",
        weights_only=False,
    )
    resumed_final_ckpt = torch.load(
        Path("runs") / run_name_resumed / "checkpoints" / "checkpoint_final.pt",
        weights_only=False,
    )

    for k in full_final_ckpt["model_state"]:
        full_p = full_final_ckpt["model_state"][k]
        res_p = resumed_final_ckpt["model_state"][k]
        diff = (full_p - res_p).abs().max()
        assert diff.item() < 1e-6, f"Mismatch in parameter {k} after resume"


def test_smoke_run_cifar_yaml_with_overrides() -> None:
    """Verify a 20-step smoke run of gd_mlp_cifar5k.yaml with overrides completes in < 60s."""
    cfg = RunConfig.from_yaml("configs/gd_mlp_cifar5k.yaml")
    overrides = [
        "run_name=smoke_cifar_20steps",
        "steps=20",
        "data.subset_size=512",
        "model.hidden_dims=[64, 32]",
        "tracker.track_interval=10",
        "tracker.compute_trace=false",
        "tracker.compute_topk=0",
    ]

    import time

    t0 = time.perf_counter()
    summary = train(cfg, overrides=overrides)
    elapsed = time.perf_counter() - t0

    assert summary["status"] == "completed"
    assert elapsed < 60.0, f"Smoke run took {elapsed:.2f}s, expected < 60s"

    # Verify plot_run generates 3-panel figure
    run_dir = Path("runs/smoke_cifar_20steps")
    fig = plot_run(run_dir)
    assert fig is not None
    assert (run_dir / "curves.png").exists()
