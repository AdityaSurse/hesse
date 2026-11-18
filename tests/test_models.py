"""Unit tests for neural network architectures, initialization determinism, and output shapes."""

from __future__ import annotations

import pytest
import torch

from hesse.config import ModelConfig
from hesse.models import (
    MLP,
    MODEL_REGISTRY,
    SmallCNN,
    TinyTransformer,
    get_activation,
    get_model,
)
from hesse.utils import count_params, flatten_params


def test_activation_factory() -> None:
    """Verify that get_activation returns proper modules or raises ValueError."""
    assert isinstance(get_activation("relu"), torch.nn.ReLU)
    assert isinstance(get_activation("tanh"), torch.nn.Tanh)
    assert isinstance(get_activation("gelu"), torch.nn.GELU)

    with pytest.raises(ValueError, match="Unsupported activation"):
        get_activation("invalid_act")


def test_mlp_forward_and_shapes() -> None:
    """Verify forward pass output dimensions for MLP under 2D and 4D inputs."""
    model = MLP(input_dim=12, hidden_dims=[24, 16], num_classes=3, activation="relu")
    # Linear(12, 24): 12*24 + 24 = 312
    # Linear(24, 16): 24*16 + 16 = 400
    # Linear(16, 3): 16*3 + 3 = 51
    # Total = 763
    assert count_params(model) == 763

    # 2D input (B, input_dim)
    x2d = torch.randn(4, 12)
    out2d = model(x2d)
    assert out2d.shape == (4, 3)

    # 4D input (B, C, H, W) where C*H*W = 12
    x4d = torch.randn(4, 3, 2, 2)
    out4d = model(x4d)
    assert out4d.shape == (4, 3)


def test_mlp_activations() -> None:
    """Verify MLP instantiation across all supported activations."""
    for act in ["relu", "tanh", "gelu"]:
        m = MLP(input_dim=8, hidden_dims=[16], num_classes=2, activation=act)
        out = m(torch.randn(2, 8))
        assert out.shape == (2, 2)


def test_cnn_forward_and_shapes() -> None:
    """Verify SmallCNN forward pass and absence of BatchNorm layers by default."""
    model = SmallCNN(in_channels=3, channels=[8, 16], num_classes=5, use_batch_norm=False)
    # Check that no BatchNorm modules exist
    bn_modules = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert len(bn_modules) == 0

    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 5)


def test_transformer_forward_and_shapes() -> None:
    """Verify TinyTransformer tokenization and forward pass shapes."""
    model = TinyTransformer(
        in_channels=3,
        input_size=16,
        patch_size=4,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        num_classes=4,
    )
    x = torch.randn(2, 3, 16, 16)
    out = model(x)
    assert out.shape == (2, 4)


def test_deterministic_initialization() -> None:
    """Verify that models initialized with identical seeds have identical parameters."""
    cfg = ModelConfig(architecture="mlp", input_dim=16, hidden_dims=[32], num_classes=2)

    m1 = get_model(cfg, seed=999)
    m2 = get_model(cfg, seed=999)
    m3 = get_model(cfg, seed=888)

    p1 = flatten_params(m1)
    p2 = flatten_params(m2)
    p3 = flatten_params(m3)

    assert torch.equal(p1, p2), "Models with the same seed must have identical parameters."
    assert not torch.equal(p1, p3), "Models with different seeds must have distinct parameters."


def test_get_model_dispatch() -> None:
    """Verify get_model factory dispatches to all registry entries correctly."""
    for arch in MODEL_REGISTRY:
        if arch == "transformer":
            cfg = ModelConfig(
                architecture=arch,
                input_dim=3072,
                embed_dim=32,
                num_heads=2,
                num_layers=1,
                num_classes=5,
            )
        else:
            cfg = ModelConfig(architecture=arch, input_dim=3072, num_classes=5)
        m = get_model(cfg, seed=42)
        assert isinstance(m, torch.nn.Module)

    cfg_invalid = ModelConfig(architecture="mlp")
    cfg_invalid.architecture = "unsupported_arch"
    with pytest.raises(ValueError, match="Unknown architecture"):
        get_model(cfg_invalid)
