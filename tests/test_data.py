"""Unit tests for dataset loaders, deterministic sampling, and class balancing."""

from __future__ import annotations

import torch

from hesse.config import DataConfig
from hesse.data import (
    get_cifar10_subset,
    get_mnist_subset,
    get_synthetic_regression,
    load_dataset,
)


def test_cifar10_subset_shapes_and_balance() -> None:
    """Verify CIFAR-10 subset shapes, full-batch output, and exact class balance."""
    subset_size = 50
    val_size = 20
    bundle = get_cifar10_subset(
        subset_size=subset_size,
        val_size=val_size,
        loss_type="mse",
        download=False,  # Use fast synthetic fallback for unit test speed
        seed=42,
    )

    # Verify tensor shapes
    assert bundle.train_x.shape == (subset_size, 3, 32, 32)
    assert bundle.train_y.shape == (subset_size, 10)  # MSE one-hot
    assert bundle.val_x is not None and bundle.val_x.shape == (val_size, 3, 32, 32)
    assert bundle.val_y is not None and bundle.val_y.shape == (val_size, 10)

    # Verify full_batch helper
    fx, fy = bundle.full_batch()
    assert torch.equal(fx, bundle.train_x)
    assert torch.equal(fy, bundle.train_y)

    # Verify class balance: exactly 5 samples per class (50 // 10)
    class_counts = bundle.train_y.sum(dim=0)
    assert torch.all(class_counts == (subset_size // 10))


def test_cifar10_cross_entropy_labels() -> None:
    """Verify that cross_entropy mode returns integer class index targets."""
    bundle = get_cifar10_subset(
        subset_size=30,
        loss_type="cross_entropy",
        download=False,
        seed=100,
    )
    assert bundle.train_y.shape == (30,)
    assert bundle.train_y.dtype == torch.int64
    assert bundle.train_y.min() >= 0
    assert bundle.train_y.max() < 10


def test_sampling_determinism() -> None:
    """Verify that identical seeds yield identical subsets and different seeds differ."""
    b1 = get_cifar10_subset(subset_size=40, seed=123, download=False)
    b2 = get_cifar10_subset(subset_size=40, seed=123, download=False)
    b3 = get_cifar10_subset(subset_size=40, seed=456, download=False)

    assert torch.equal(b1.train_x, b2.train_x)
    assert torch.equal(b1.train_y, b2.train_y)
    assert not torch.equal(b1.train_x, b3.train_x)


def test_mnist_subset_shapes_and_balance() -> None:
    """Verify MNIST subset loader shapes, single channel, and balanced classes."""
    subset_size = 40
    val_size = 10
    bundle = get_mnist_subset(
        subset_size=subset_size,
        val_size=val_size,
        loss_type="cross_entropy",
        download=False,
        seed=77,
    )

    assert bundle.train_x.shape == (subset_size, 1, 28, 28)
    assert bundle.train_y.shape == (subset_size,)
    assert bundle.num_classes == 10

    # Class balance check
    counts = torch.bincount(bundle.train_y, minlength=10)
    assert torch.all(counts == (subset_size // 10))


def test_synthetic_regression() -> None:
    """Verify synthetic quadratic regression shapes and numeric properties."""
    num_samples = 60
    input_dim = 16
    val_size = 15
    bundle = get_synthetic_regression(
        num_samples=num_samples,
        input_dim=input_dim,
        val_size=val_size,
        seed=42,
    )

    assert bundle.train_x.shape == (num_samples, input_dim)
    assert bundle.train_y.shape == (num_samples, 1)
    assert bundle.val_x is not None and bundle.val_x.shape == (val_size, input_dim)
    assert bundle.val_y is not None and bundle.val_y.shape == (val_size, 1)
    assert not torch.isnan(bundle.train_x).any()
    assert not torch.isnan(bundle.train_y).any()


def test_load_dataset_dispatch() -> None:
    """Verify load_dataset dispatches according to DataConfig."""
    cfg = DataConfig(dataset="synthetic", subset_size=20, val_size=5, input_dim=8)
    bundle = load_dataset(cfg)
    assert bundle.train_x.shape == (20, 8)
