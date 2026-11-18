"""Configuration dataclasses and validation for the hesse package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing.

    Attributes:
        dataset: Dataset identifier ('cifar10', 'mnist', 'synthetic').
        subset_size: Number of training examples to select (e.g. 5000).
        batch_size: Batch size for mini-batch training. None implies full-batch.
        loss_type: Objective loss formulation ('mse' or 'cross_entropy').
        val_size: Number of validation samples.
        data_dir: Path to storage directory for caching datasets.
        seed: Random seed for deterministic sample selection and split.
        input_dim: Input dimensionality for synthetic datasets.
    """

    dataset: str = "cifar10"
    subset_size: int = 5000
    batch_size: int | None = None
    loss_type: str = "mse"
    val_size: int = 1000
    data_dir: str = "data"
    seed: int = 42
    input_dim: int | None = None

    def __post_init__(self) -> None:
        """Validate data configuration parameters."""
        valid_datasets = {"cifar10", "mnist", "synthetic"}
        if self.dataset not in valid_datasets:
            raise ValueError(
                f"Invalid dataset '{self.dataset}'. Expected one of {sorted(valid_datasets)}."
            )
        if self.subset_size <= 0:
            raise ValueError(f"subset_size must be positive, got {self.subset_size}.")
        if self.val_size < 0:
            raise ValueError(f"val_size cannot be negative, got {self.val_size}.")
        valid_losses = {"mse", "cross_entropy"}
        if self.loss_type not in valid_losses:
            raise ValueError(
                f"Invalid loss_type '{self.loss_type}'. Expected one of {sorted(valid_losses)}."
            )
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive when specified, got {self.batch_size}.")


@dataclass
class ModelConfig:
    """Configuration for neural network architectures.

    Attributes:
        architecture: Model architecture ('mlp', 'cnn', 'transformer').
        input_dim: Flattened input feature dimension (e.g. 3072 for CIFAR-10).
        hidden_dims: List of hidden layer dimensions for MLP.
        num_classes: Number of prediction targets or output logits.
        activation: Non-linear activation function ('relu', 'tanh', 'gelu').
        use_batch_norm: Whether to apply batch normalization (default False).
        dropout: Dropout probability.
        channels: List of channel dimensions for CNN layers.
        kernel_size: Convolution kernel size for CNN layers.
        embed_dim: Embedding dimension for Transformer models.
        num_heads: Number of attention heads for Transformer models.
        num_layers: Number of transformer encoder blocks.
        patch_size: 2D patch size for vision sequence tokenization.
    """

    architecture: str = "mlp"
    input_dim: int = 3072
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    num_classes: int = 10
    activation: str = "relu"
    use_batch_norm: bool = False
    dropout: float = 0.0
    channels: list[int] = field(default_factory=lambda: [32, 64])
    kernel_size: int = 3
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    patch_size: int = 4

    def __post_init__(self) -> None:
        """Validate model configuration parameters."""
        valid_archs = {"mlp", "cnn", "transformer"}
        if self.architecture not in valid_archs:
            raise ValueError(
                f"Invalid architecture '{self.architecture}'. "
                f"Expected one of {sorted(valid_archs)}."
            )
        valid_activations = {"relu", "tanh", "gelu"}
        if self.activation not in valid_activations:
            raise ValueError(
                f"Invalid activation '{self.activation}'. "
                f"Expected one of {sorted(valid_activations)}."
            )
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}.")
        if self.architecture == "mlp":
            if not self.hidden_dims or any(d <= 0 for d in self.hidden_dims):
                raise ValueError(
                    f"hidden_dims must be non-empty positive integers, got {self.hidden_dims}."
                )
        elif self.architecture == "cnn":
            if not self.channels or any(c <= 0 for c in self.channels):
                raise ValueError(
                    f"channels must be non-empty positive integers, got {self.channels}."
                )
            if self.kernel_size <= 0:
                raise ValueError(f"kernel_size must be positive, got {self.kernel_size}.")
        elif self.architecture == "transformer":
            if self.embed_dim <= 0 or self.num_heads <= 0 or self.num_layers <= 0:
                raise ValueError("embed_dim, num_heads, and num_layers must be positive.")
            if self.embed_dim % self.num_heads != 0:
                raise ValueError(
                    f"embed_dim ({self.embed_dim}) must be divisible "
                    f"by num_heads ({self.num_heads})."
                )


@dataclass
class OptimConfig:
    """Configuration for optimization algorithms.

    Attributes:
        optimizer: Optimizer identifier ('gd', 'sgd', 'adam', 'lion').
        lr: Base learning rate.
        momentum: Momentum factor for SGD.
        beta1: First moment coefficient for Adam and Lion.
        beta2: Second moment coefficient for Adam and Lion.
        weight_decay: L2 penalty coefficient.
        eps: Epsilon denominator term for numerical stability in Adam.
    """

    optimizer: str = "gd"
    lr: float = 0.05
    momentum: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        """Validate optimizer configuration parameters."""
        valid_opts = {"gd", "sgd", "adam", "lion"}
        if self.optimizer not in valid_opts:
            raise ValueError(
                f"Invalid optimizer '{self.optimizer}'. Expected one of {sorted(valid_opts)}."
            )
        if self.lr <= 0.0:
            raise ValueError(f"Learning rate lr must be strictly positive, got {self.lr}.")
        if not (0.0 <= self.momentum < 1.0):
            raise ValueError(f"momentum must be in [0.0, 1.0), got {self.momentum}.")
        if not (0.0 <= self.beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0.0, 1.0), got {self.beta1}.")
        if not (0.0 <= self.beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0.0, 1.0), got {self.beta2}.")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay cannot be negative, got {self.weight_decay}.")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be strictly positive, got {self.eps}.")


@dataclass
class TrackerConfig:
    """Configuration for live curvature tracking telemetry.

    Attributes:
        track_interval: Step frequency for evaluating curvature metrics (k steps).
        compute_sharpness: Whether to calculate top eigenvalue lambda_max.
        compute_trace: Whether to evaluate Hutchinson's trace estimator.
        compute_topk: Number of top eigenvalues to extract via Lanczos.
        compute_density: Whether to estimate spectral density via SLQ.
        log_dir: Directory where run artifacts and metric logs are persisted.
    """

    track_interval: int = 10
    compute_sharpness: bool = True
    compute_trace: bool = True
    compute_topk: int = 5
    compute_density: bool = False
    log_dir: str = "runs"

    def __post_init__(self) -> None:
        """Validate tracker configuration parameters."""
        if self.track_interval <= 0:
            raise ValueError(f"track_interval must be positive integer, got {self.track_interval}.")
        if self.compute_topk < 0:
            raise ValueError(f"compute_topk cannot be negative, got {self.compute_topk}.")


@dataclass
class RunConfig:
    """Top-level configuration orchestrating complete training and curvature tracking runs.

    Attributes:
        run_name: Human-readable identifier for the experimental execution.
        epochs: Number of training epochs.
        steps: Total step budget (overrides epochs when set).
        seed: Master random seed for deterministic reproduction.
        device: Target execution device ('auto', 'cpu', 'cuda', 'mps').
        data: Data configuration.
        model: Model architecture configuration.
        optim: Optimizer configuration.
        tracker: Telemetry tracking configuration.
    """

    run_name: str = "hesse_run"
    epochs: int = 100
    steps: int | None = None
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    def __post_init__(self) -> None:
        """Validate run configuration parameters."""
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}.")
        if self.steps is not None and self.steps <= 0:
            raise ValueError(f"steps must be positive when set, got {self.steps}.")
        valid_devices = {"auto", "cpu", "cuda", "mps"}
        if self.device not in valid_devices:
            raise ValueError(
                f"Invalid device '{self.device}'. Expected one of {sorted(valid_devices)}."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        """Construct a RunConfig instance from a nested dictionary.

        Args:
            data: Raw dictionary loaded from configuration sources.

        Returns:
            Validated RunConfig instance.
        """
        raw = dict(data)
        data_cfg = DataConfig(**raw.pop("data", {}))
        model_cfg = ModelConfig(**raw.pop("model", {}))
        optim_cfg = OptimConfig(**raw.pop("optim", {}))
        tracker_cfg = TrackerConfig(**raw.pop("tracker", {}))
        return cls(
            data=data_cfg,
            model=model_cfg,
            optim=optim_cfg,
            tracker=tracker_cfg,
            **raw,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        """Load and validate a RunConfig from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated RunConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If parsing fails or parameters are invalid.
        """
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f)

        if not isinstance(raw_dict, dict):
            raise ValueError(f"YAML configuration at {cfg_path} must define a mapping.")

        return cls.from_dict(raw_dict)

    def to_yaml(self, path: str | Path) -> None:
        """Serialize configuration to a YAML file.

        Args:
            path: Target file path to write YAML.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
