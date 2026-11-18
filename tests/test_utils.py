"""Unit tests for utility functions, seeding, and parameter manipulation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hesse.utils import (
    count_params,
    flatten_grads,
    flatten_params,
    get_device,
    set_seed,
    unflatten_params,
)


def test_set_seed_determinism() -> None:
    """Verify that set_seed guarantees deterministic random generation across libraries."""
    set_seed(12345)
    t1 = torch.randn(5, 5)
    set_seed(12345)
    t2 = torch.randn(5, 5)
    assert torch.equal(t1, t2)


def test_get_device() -> None:
    """Verify device resolution for auto, cpu, and explicit overrides."""
    dev_cpu = get_device("cpu")
    assert dev_cpu.type == "cpu"

    dev_auto = get_device("auto")
    assert dev_auto.type in {"cpu", "cuda", "mps"}


def test_count_params(small_mlp: nn.Module) -> None:
    """Verify parameter counting on trainable and frozen parameters."""
    # Linear(8, 16): 8*16 + 16 = 144
    # Linear(16, 4): 16*4 + 4 = 68
    # Total = 212
    assert count_params(small_mlp) == 212

    # Freeze first layer: 144 parameters frozen, 68 trainable
    for p in small_mlp[0].parameters():
        p.requires_grad = False

    assert count_params(small_mlp, trainable_only=True) == 68
    assert count_params(small_mlp, trainable_only=False) == 212


def test_flatten_unflatten_round_trip(small_mlp: nn.Module) -> None:
    """Verify that flattening and unflattening parameters preserves values exactly."""
    flat = flatten_params(small_mlp)
    assert flat.dim() == 1
    assert flat.numel() == count_params(small_mlp, trainable_only=False)

    # Mutate flat vector with deterministic random values
    new_values = torch.randn_like(flat)
    unflatten_params(small_mlp, new_values)

    # Re-flatten and assert values match mutated vector
    reflat = flatten_params(small_mlp)
    assert torch.allclose(reflat, new_values, atol=1e-7)


def test_unflatten_dimension_mismatch(small_mlp: nn.Module) -> None:
    """Verify that unflatten_params raises ValueError on mismatched tensor sizes."""
    wrong_size_tensor = torch.zeros(10)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        unflatten_params(small_mlp, wrong_size_tensor)


def test_flatten_grads(small_mlp: nn.Module) -> None:
    """Verify gradient flattening after a forward and backward pass."""
    x = torch.randn(2, 8)
    out = small_mlp(x).sum()
    out.backward()

    flat_g = flatten_grads(small_mlp)
    assert flat_g.dim() == 1
    assert flat_g.numel() == count_params(small_mlp, trainable_only=False)
    assert not torch.all(flat_g == 0)
