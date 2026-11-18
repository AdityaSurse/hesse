"""Neural architectures with deterministic initialization and configurable curvature properties."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
import torch.nn as nn

from hesse.config import ModelConfig
from hesse.utils import set_seed


def get_activation(name: str) -> nn.Module:
    """Return non-linear activation module by identifier.

    Args:
        name: Activation name ('relu', 'tanh', 'gelu').

    Returns:
        PyTorch activation module instance.

    Raises:
        ValueError: If activation name is not recognized.
    """
    act = name.lower()
    if act == "relu":
        return nn.ReLU()
    if act == "tanh":
        return nn.Tanh()
    if act == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'. Expected 'relu', 'tanh', or 'gelu'.")


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable hidden layer widths and activations.

    Attributes:
        input_dim: Total flattened input features.
        hidden_dims: Sequence of hidden layer dimensions.
        num_classes: Dimension of final output logits.
        activation_name: Name of activation function applied to hidden layers.
    """

    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dims: Sequence[int] = (512, 256),
        num_classes: int = 10,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.num_classes = num_classes
        self.activation_name = activation

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(get_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass flattening arbitrary multi-dimensional inputs to (B, input_dim).

        Args:
            x: Input tensor of shape (B, ...).

        Returns:
            Output logits of shape (B, num_classes).
        """
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        return self.network(x)


class SmallCNN(nn.Module):
    """Small Convolutional Neural Network with 2-3 conv layers and no BatchNorm by default.

    BatchNorm is disabled by default because its filter scale invariance distorts the true
    Hessian eigenspectrum unless explicit filter-normalization is applied.

    Attributes:
        in_channels: Input image channel count (e.g. 3 for RGB, 1 for grayscale).
        channels: Sequence of output channels for convolutional stages.
        num_classes: Number of prediction classes.
        use_batch_norm: Whether to insert BatchNorm2d layers (default False).
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64),
        num_classes: int = 10,
        activation: str = "relu",
        use_batch_norm: bool = False,
        dropout: float = 0.0,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channels = list(channels)
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm

        conv_blocks: list[nn.Module] = []
        prev_c = in_channels
        padding = kernel_size // 2

        for c in self.channels:
            conv_blocks.append(
                nn.Conv2d(
                    prev_c,
                    c,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=not use_batch_norm,
                )
            )
            if use_batch_norm:
                conv_blocks.append(nn.BatchNorm2d(c))
            conv_blocks.append(get_activation(activation))
            conv_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if dropout > 0.0:
                conv_blocks.append(nn.Dropout2d(dropout))
            prev_c = c

        self.features = nn.Sequential(*conv_blocks)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Linear(prev_c * 2 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional feature extractor and classifier.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Output logits of shape (B, num_classes).
        """
        feats = self.features(x)
        pooled = self.pool(feats)
        flat = pooled.reshape(pooled.size(0), -1)
        return self.classifier(flat)


class TinyTransformer(nn.Module):
    """Miniature Vision / Sequence Transformer for edge-of-stability sequence experiments.

    Supports image patch tokenization or direct sequence embedding.

    Attributes:
        embed_dim: Token embedding dimensionality.
        num_heads: Number of multi-head self-attention heads.
        num_layers: Number of TransformerEncoderLayer blocks.
        num_classes: Output prediction dimension.
        patch_size: Square patch spatial dimensions for 2D inputs.
    """

    def __init__(
        self,
        in_channels: int = 3,
        input_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Number of patches along height and width
        num_patches_1d = input_size // patch_size
        self.num_patches = num_patches_1d * num_patches_1d
        patch_dim = in_channels * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize CLS token with truncated normal-like values
        nn.init.normal_(self.cls_token, std=0.02)

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Slice 2D image into flattened patch tokens.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Patches of shape (B, num_patches, patch_dim).
        """
        b, c, h, w = x.shape
        p = self.patch_size
        # Reshape to (B, C, H/P, P, W/P, P) -> (B, (H/P)*(W/P), C*P*P)
        patches = (
            x.unfold(2, p, p)
            .unfold(3, p, p)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .view(b, -1, c * p * p)
        )
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass tokenizing input and routing through Transformer encoder.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, SeqLen, Dim).

        Returns:
            Output logits of shape (B, num_classes).
        """
        b = x.size(0)
        if x.dim() == 4:
            tokens = self.patch_embed(self._extract_patches(x))
        elif x.dim() == 3:
            tokens = x if x.size(-1) == self.embed_dim else self.patch_embed(x)
        else:
            # Flatten 2D input into sequence
            dim = int(math.isqrt(x.size(-1) // 3)) or 32
            x_reshaped = x.view(b, 3, dim, dim)
            tokens = self.patch_embed(self._extract_patches(x_reshaped))

        # Prepend CLS token and add positional embedding
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x_tok = torch.cat([cls_tokens, tokens], dim=1)

        pos_embed = self.pos_embedding
        if pos_embed.size(1) != x_tok.size(1):
            # Interpolate or truncate positional embeddings if sequence length varies
            pos_embed = pos_embed[:, : x_tok.size(1), :]

        x_tok = x_tok + pos_embed
        encoded = self.transformer(x_tok)
        cls_out = self.norm(encoded[:, 0])
        return self.head(cls_out)


# Architecture Registry Mapping
MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "mlp": MLP,
    "cnn": SmallCNN,
    "transformer": TinyTransformer,
}


def get_model(config: ModelConfig, seed: int = 42) -> nn.Module:
    """Instantiate a neural network model with deterministic weight initialization.

    Args:
        config: ModelConfig specifying architecture, dimensions, and activation.
        seed: Random seed applied immediately prior to model construction.

    Returns:
        Deterministically initialized PyTorch nn.Module.

    Raises:
        ValueError: If architecture is not in MODEL_REGISTRY.
    """
    set_seed(seed)
    arch = config.architecture.lower()
    if arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{config.architecture}'. "
            f"Expected one of {sorted(MODEL_REGISTRY.keys())}."
        )

    if arch == "mlp":
        return MLP(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            num_classes=config.num_classes,
            activation=config.activation,
            dropout=config.dropout,
        )

    if arch == "cnn":
        # Determine in_channels from input_dim (e.g. 3072 -> 3 for CIFAR, 784 -> 1 for MNIST)
        in_c = 1 if config.input_dim == 784 else 3
        return SmallCNN(
            in_channels=in_c,
            channels=config.channels,
            num_classes=config.num_classes,
            activation=config.activation,
            use_batch_norm=config.use_batch_norm,
            dropout=config.dropout,
            kernel_size=config.kernel_size,
        )

    if arch == "transformer":
        in_c = 1 if config.input_dim == 784 else 3
        side = int(math.isqrt(config.input_dim // in_c)) if config.input_dim > 0 else 32
        return TinyTransformer(
            in_channels=in_c,
            input_size=side,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

    raise ValueError(f"Architecture '{arch}' not properly dispatched.")
