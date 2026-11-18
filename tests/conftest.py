"""Pytest configuration and shared fixtures for hesse test suite."""

from __future__ import annotations

import pytest
import torch.nn as nn

from hesse.config import DataConfig, ModelConfig, OptimConfig, RunConfig, TrackerConfig
from hesse.utils import set_seed


@pytest.fixture(autouse=True)
def reset_seed() -> None:
    """Ensure every test runs with a clean, deterministic seed."""
    set_seed(42)


@pytest.fixture
def sample_run_config() -> RunConfig:
    """Fixture returning a standard RunConfig for testing."""
    return RunConfig(
        run_name="test_experiment",
        epochs=5,
        seed=1337,
        data=DataConfig(dataset="synthetic", subset_size=40, val_size=10, loss_type="mse"),
        model=ModelConfig(architecture="mlp", input_dim=16, hidden_dims=[32, 16], num_classes=2),
        optim=OptimConfig(optimizer="gd", lr=0.01),
        tracker=TrackerConfig(track_interval=2, compute_sharpness=True),
    )


@pytest.fixture
def small_mlp() -> nn.Module:
    """Fixture returning a tiny MLP for fast unit tests."""
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )
