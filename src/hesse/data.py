"""Dataset loaders, deterministic subset sampling, and full-batch tensor utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from hesse.config import DataConfig
from hesse.utils import set_seed

logger = logging.getLogger(__name__)


@dataclass
class DatasetBundle:
    """Container holding full-batch tensors and optional PyTorch DataLoaders.

    Attributes:
        train_x: Training input tensor of shape (N, *input_shape).
        train_y: Training target tensor of shape (N, num_classes) for MSE
                 or (N,) for cross_entropy.
        val_x: Validation input tensor, or None.
        val_y: Validation target tensor, or None.
        train_loader: DataLoader for mini-batch or full-batch iteration.
        val_loader: DataLoader for validation set iteration, or None.
        num_classes: Number of distinct classes or regression targets.
        input_shape: Shape tuple of an individual input sample (e.g. (3, 32, 32)).
    """

    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor | None = None
    val_y: torch.Tensor | None = None
    train_loader: DataLoader | None = None
    val_loader: DataLoader | None = None
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (3, 32, 32)

    def full_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the complete training set directly as a tensor tuple (X, y)."""
        return self.train_x, self.train_y


def _to_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer class indices to float one-hot vectors for MSE loss.

    Args:
        labels: 1D tensor of integer class labels.
        num_classes: Total number of target classes.

    Returns:
        Float tensor of shape (N, num_classes) with one-hot encodings.
    """
    labels_int = labels.to(torch.int64)
    return torch.nn.functional.one_hot(labels_int, num_classes=num_classes).to(torch.float32)


def _select_balanced_subset(
    targets: np.ndarray,
    subset_size: int,
    num_classes: int,
    seed: int = 42,
) -> np.ndarray:
    """Select a deterministic, class-balanced subset of indices.

    Args:
        targets: Array of integer class labels for the full dataset.
        subset_size: Total number of samples to select.
        num_classes: Number of distinct target classes.
        seed: Random seed for deterministic selection.

    Returns:
        1D array of selected indices.
    """
    rng = np.random.RandomState(seed)
    base_per_class = subset_size // num_classes
    remainder = subset_size % num_classes

    selected: list[int] = []
    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        n_samples = base_per_class + (1 if c < remainder else 0)
        n_samples = min(n_samples, len(class_indices))
        chosen = rng.choice(class_indices, size=n_samples, replace=False)
        selected.extend(chosen.tolist())

    rng.shuffle(selected)
    return np.array(selected, dtype=np.int64)


def _create_synthetic_classification(
    num_samples: int,
    input_shape: Tuple[int, ...],
    num_classes: int,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic synthetic image-like data for offline fallback or testing.

    Args:
        num_samples: Number of samples to generate.
        input_shape: Dimensions of each sample (e.g. (3, 32, 32)).
        num_classes: Number of discrete classes.
        seed: Random seed.

    Returns:
        Tuple of (X, y_labels) tensors.
    """
    rng = np.random.RandomState(seed)
    x_np = rng.randn(num_samples, *input_shape).astype(np.float32)
    # Scale to typical image normalization range [0, 1]
    x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)

    # Generate balanced class labels
    labels_list = []
    base = num_samples // num_classes
    rem = num_samples % num_classes
    for c in range(num_classes):
        count = base + (1 if c < rem else 0)
        labels_list.extend([c] * count)
    rng.shuffle(labels_list)

    x_tensor = torch.from_numpy(x_np)
    y_tensor = torch.tensor(labels_list, dtype=torch.int64)
    return x_tensor, y_tensor


def get_cifar10_subset(
    subset_size: int = 5000,
    val_size: int = 1000,
    batch_size: int | None = None,
    loss_type: str = "mse",
    data_dir: str = "data",
    seed: int = 42,
    download: bool = True,
) -> DatasetBundle:
    """Load a deterministic, class-balanced subset of CIFAR-10.

    Args:
        subset_size: Number of training examples to select (default 5000).
        val_size: Number of validation samples to select (default 1000).
        batch_size: Batch size for DataLoader. If None, full-batch mode is used.
        loss_type: 'mse' (one-hot targets) or 'cross_entropy' (class indices).
        data_dir: Path to directory caching downloaded datasets.
        seed: Random seed for deterministic sample selection.
        download: Whether to attempt downloading CIFAR-10 via torchvision.

    Returns:
        DatasetBundle containing full-batch tensors and DataLoaders.
    """
    set_seed(seed)
    num_classes = 10
    input_shape = (3, 32, 32)
    cache_path = Path(data_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    loaded = False
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor

    if download:
        try:
            import torchvision  # type: ignore
            import torchvision.transforms as transforms  # type: ignore

            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            raw_train = torchvision.datasets.CIFAR10(
                root=str(cache_path), train=True, download=True, transform=transform
            )
            raw_test = torchvision.datasets.CIFAR10(
                root=str(cache_path), train=False, download=True, transform=transform
            )

            train_targets = np.array(raw_train.targets)
            train_idx = _select_balanced_subset(train_targets, subset_size, num_classes, seed=seed)

            test_targets = np.array(raw_test.targets)
            val_idx = _select_balanced_subset(test_targets, val_size, num_classes, seed=seed + 1)

            # Stack images into tensors
            train_imgs = [raw_train[i][0] for i in train_idx]
            train_labels = [raw_train[i][1] for i in train_idx]
            val_imgs = [raw_test[i][0] for i in val_idx]
            val_labels = [raw_test[i][1] for i in val_idx]

            train_x = torch.stack(train_imgs)
            train_y = torch.tensor(train_labels, dtype=torch.int64)
            val_x = torch.stack(val_imgs)
            val_y = torch.tensor(val_labels, dtype=torch.int64)
            loaded = True
        except Exception as exc:
            logger.warning(
                f"Could not load CIFAR-10 via torchvision ({exc}). "
                "Falling back to deterministic synthetic dataset."
            )

    if not loaded:
        train_x, train_y = _create_synthetic_classification(
            subset_size, input_shape, num_classes, seed=seed
        )
        val_x, val_y = _create_synthetic_classification(
            val_size, input_shape, num_classes, seed=seed + 1
        )

    # Format targets for loss type
    if loss_type == "mse":
        train_y_out = _to_one_hot(train_y, num_classes)
        val_y_out = _to_one_hot(val_y, num_classes)
    else:
        train_y_out = train_y
        val_y_out = val_y

    effective_batch_size = subset_size if batch_size is None else batch_size
    train_ds = TensorDataset(train_x, train_y_out)
    val_ds = TensorDataset(val_x, val_y_out)

    train_loader = DataLoader(train_ds, batch_size=effective_batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=effective_batch_size, shuffle=False)

    return DatasetBundle(
        train_x=train_x,
        train_y=train_y_out,
        val_x=val_x,
        val_y=val_y_out,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        input_shape=input_shape,
    )


def get_mnist_subset(
    subset_size: int = 5000,
    val_size: int = 1000,
    batch_size: int | None = None,
    loss_type: str = "mse",
    data_dir: str = "data",
    seed: int = 42,
    download: bool = True,
) -> DatasetBundle:
    """Load a deterministic, class-balanced subset of MNIST.

    Args:
        subset_size: Number of training examples to select.
        val_size: Number of validation samples to select.
        batch_size: Batch size for DataLoader. If None, full-batch mode is used.
        loss_type: 'mse' (one-hot targets) or 'cross_entropy' (class indices).
        data_dir: Path to directory caching downloaded datasets.
        seed: Random seed for deterministic sample selection.
        download: Whether to attempt downloading MNIST via torchvision.

    Returns:
        DatasetBundle containing full-batch tensors and DataLoaders.
    """
    set_seed(seed)
    num_classes = 10
    input_shape = (1, 28, 28)
    cache_path = Path(data_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    loaded = False
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor

    if download:
        try:
            import torchvision  # type: ignore
            import torchvision.transforms as transforms  # type: ignore

            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )

            raw_train = torchvision.datasets.MNIST(
                root=str(cache_path), train=True, download=True, transform=transform
            )
            raw_test = torchvision.datasets.MNIST(
                root=str(cache_path), train=False, download=True, transform=transform
            )

            train_targets = np.array(raw_train.targets)
            train_idx = _select_balanced_subset(train_targets, subset_size, num_classes, seed=seed)

            test_targets = np.array(raw_test.targets)
            val_idx = _select_balanced_subset(test_targets, val_size, num_classes, seed=seed + 1)

            train_imgs = [raw_train[i][0] for i in train_idx]
            train_labels = [int(raw_train[i][1]) for i in train_idx]
            val_imgs = [raw_test[i][0] for i in val_idx]
            val_labels = [int(raw_test[i][1]) for i in val_idx]

            train_x = torch.stack(train_imgs)
            train_y = torch.tensor(train_labels, dtype=torch.int64)
            val_x = torch.stack(val_imgs)
            val_y = torch.tensor(val_labels, dtype=torch.int64)
            loaded = True
        except Exception as exc:
            logger.warning(
                f"Could not load MNIST via torchvision ({exc}). "
                "Falling back to deterministic synthetic dataset."
            )

    if not loaded:
        train_x, train_y = _create_synthetic_classification(
            subset_size, input_shape, num_classes, seed=seed
        )
        val_x, val_y = _create_synthetic_classification(
            val_size, input_shape, num_classes, seed=seed + 1
        )

    if loss_type == "mse":
        train_y_out = _to_one_hot(train_y, num_classes)
        val_y_out = _to_one_hot(val_y, num_classes)
    else:
        train_y_out = train_y
        val_y_out = val_y

    effective_batch_size = subset_size if batch_size is None else batch_size
    train_ds = TensorDataset(train_x, train_y_out)
    val_ds = TensorDataset(val_x, val_y_out)

    train_loader = DataLoader(train_ds, batch_size=effective_batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=effective_batch_size, shuffle=False)

    return DatasetBundle(
        train_x=train_x,
        train_y=train_y_out,
        val_x=val_x,
        val_y=val_y_out,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        input_shape=input_shape,
    )


def get_synthetic_regression(
    num_samples: int = 1000,
    input_dim: int = 64,
    val_size: int = 200,
    batch_size: int | None = None,
    seed: int = 42,
    noise_std: float = 0.05,
) -> DatasetBundle:
    """Generate a synthetic quadratic regression problem with controllable condition number.

    The target objective follows a non-linear quadratic function with additive Gaussian noise:
    y = 0.5 * x^T A x + w^T x + noise

    Args:
        num_samples: Number of training samples.
        input_dim: Input feature dimension.
        val_size: Number of validation samples.
        batch_size: Batch size for DataLoader. If None, full-batch mode is used.
        seed: Random seed for generation.
        noise_std: Standard deviation of additive Gaussian observation noise.

    Returns:
        DatasetBundle containing regression tensors and DataLoaders.
    """
    set_seed(seed)
    rng = np.random.RandomState(seed)

    # Generate well-defined ill-conditioned positive semi-definite matrix A
    eigenvalues = np.logspace(0, 2, num=input_dim).astype(np.float32)  # condition number = 100
    q, _ = np.linalg.qr(rng.randn(input_dim, input_dim))
    a_mat = (q @ np.diag(eigenvalues) @ q.T).astype(np.float32)
    w_vec = rng.randn(input_dim, 1).astype(np.float32)

    def _eval(x: np.ndarray) -> np.ndarray:
        # 0.5 * sum_i (x_i (A x)_i) + x w
        ax = x @ a_mat
        quad = 0.5 * np.sum(x * ax, axis=1, keepdims=True)
        lin = x @ w_vec
        noise = rng.randn(x.shape[0], 1).astype(np.float32) * noise_std
        return (quad + lin + noise).astype(np.float32)

    train_x_np = rng.randn(num_samples, input_dim).astype(np.float32)
    train_y_np = _eval(train_x_np)

    val_x_np = rng.randn(val_size, input_dim).astype(np.float32)
    val_y_np = _eval(val_x_np)

    train_x = torch.from_numpy(train_x_np)
    train_y = torch.from_numpy(train_y_np)
    val_x = torch.from_numpy(val_x_np)
    val_y = torch.from_numpy(val_y_np)

    effective_batch_size = num_samples if batch_size is None else batch_size
    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)

    train_loader = DataLoader(train_ds, batch_size=effective_batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=effective_batch_size, shuffle=False)

    return DatasetBundle(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=1,
        input_shape=(input_dim,),
    )


def load_dataset(config: DataConfig) -> DatasetBundle:
    """Load dataset bundle corresponding to the given DataConfig.

    Args:
        config: DataConfig specifying dataset, subset sizes, loss formulation, and seed.

    Returns:
        Configured DatasetBundle instance.
    """
    if config.dataset == "cifar10":
        return get_cifar10_subset(
            subset_size=config.subset_size,
            val_size=config.val_size,
            batch_size=config.batch_size,
            loss_type=config.loss_type,
            data_dir=config.data_dir,
            seed=config.seed,
        )
    if config.dataset == "mnist":
        return get_mnist_subset(
            subset_size=config.subset_size,
            val_size=config.val_size,
            batch_size=config.batch_size,
            loss_type=config.loss_type,
            data_dir=config.data_dir,
            seed=config.seed,
        )
    if config.dataset == "synthetic":
        dim = config.input_dim if config.input_dim is not None else 64
        return get_synthetic_regression(
            num_samples=config.subset_size,
            input_dim=dim,
            val_size=config.val_size,
            batch_size=config.batch_size,
            seed=config.seed,
        )
    raise ValueError(f"Unknown dataset '{config.dataset}' in configuration.")
