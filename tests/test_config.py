"""Unit tests for configuration schemas, defaults, and serialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from hesse.config import (
    DataConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
)


def test_default_config_instantiation() -> None:
    """Verify that all configuration dataclasses instantiate with sensible defaults."""
    cfg = RunConfig()
    assert cfg.run_name == "hesse_run"
    assert cfg.epochs == 100
    assert cfg.data.dataset == "cifar10"
    assert cfg.data.subset_size == 5000
    assert cfg.model.architecture == "mlp"
    assert cfg.optim.optimizer == "gd"
    assert cfg.optim.lr == 0.05
    assert cfg.tracker.track_interval == 10


def test_data_config_validation() -> None:
    """Verify validation logic in DataConfig."""
    with pytest.raises(ValueError, match="Invalid dataset"):
        DataConfig(dataset="imagenet")

    with pytest.raises(ValueError, match="subset_size must be positive"):
        DataConfig(subset_size=0)

    with pytest.raises(ValueError, match="val_size cannot be negative"):
        DataConfig(val_size=-5)

    with pytest.raises(ValueError, match="Invalid loss_type"):
        DataConfig(loss_type="hinge")

    with pytest.raises(ValueError, match="batch_size must be positive"):
        DataConfig(batch_size=-1)


def test_model_config_validation() -> None:
    """Verify validation logic in ModelConfig."""
    with pytest.raises(ValueError, match="Invalid architecture"):
        ModelConfig(architecture="resnet")

    with pytest.raises(ValueError, match="Invalid activation"):
        ModelConfig(activation="sigmoid")

    with pytest.raises(ValueError, match="num_classes must be positive"):
        ModelConfig(num_classes=0)

    with pytest.raises(ValueError, match="dropout must be in"):
        ModelConfig(dropout=1.5)

    with pytest.raises(ValueError, match="hidden_dims must be non-empty"):
        ModelConfig(architecture="mlp", hidden_dims=[])

    with pytest.raises(ValueError, match="channels must be non-empty"):
        ModelConfig(architecture="cnn", channels=[-4])

    with pytest.raises(ValueError, match="divisible by num_heads"):
        ModelConfig(architecture="transformer", embed_dim=33, num_heads=4)


def test_optim_config_validation() -> None:
    """Verify validation logic in OptimConfig."""
    with pytest.raises(ValueError, match="Invalid optimizer"):
        OptimConfig(optimizer="adagrad")

    with pytest.raises(ValueError, match="Learning rate lr must be strictly positive"):
        OptimConfig(lr=0.0)

    with pytest.raises(ValueError, match="momentum must be in"):
        OptimConfig(momentum=1.2)

    with pytest.raises(ValueError, match="beta1 must be in"):
        OptimConfig(beta1=-0.1)

    with pytest.raises(ValueError, match="beta2 must be in"):
        OptimConfig(beta2=1.0)

    with pytest.raises(ValueError, match="eps must be strictly positive"):
        OptimConfig(eps=-1e-8)


def test_yaml_serialization_round_trip(tmp_path: Path, sample_run_config: RunConfig) -> None:
    """Verify that RunConfig can serialize to and deserialize from YAML without loss."""
    yaml_file = tmp_path / "test_config.yaml"
    sample_run_config.to_yaml(yaml_file)
    assert yaml_file.exists()

    loaded_cfg = RunConfig.from_yaml(yaml_file)
    assert loaded_cfg.run_name == sample_run_config.run_name
    assert loaded_cfg.epochs == sample_run_config.epochs
    assert loaded_cfg.seed == sample_run_config.seed
    assert loaded_cfg.data.dataset == sample_run_config.data.dataset
    assert loaded_cfg.data.subset_size == sample_run_config.data.subset_size
    assert loaded_cfg.model.architecture == sample_run_config.model.architecture
    assert loaded_cfg.optim.optimizer == sample_run_config.optim.optimizer
    assert loaded_cfg.tracker.track_interval == sample_run_config.tracker.track_interval


def test_yaml_missing_file_raises() -> None:
    """Verify that from_yaml raises FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError):
        RunConfig.from_yaml("non_existent_config.yaml")


def test_yaml_invalid_content_raises(tmp_path: Path) -> None:
    """Verify that from_yaml raises ValueError if YAML does not contain a mapping."""
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must define a mapping"):
        RunConfig.from_yaml(bad_file)
