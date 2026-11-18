"""Hessian-Vector Product (HVP) operators using Pearlmutter's trick and torch.func."""

from __future__ import annotations

import logging
from typing import Any, Callable, Generator, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.func import functional_call, grad, jvp
from torch.utils.data import DataLoader

from hesse.utils import count_params

logger = logging.getLogger(__name__)


def _unflatten_to_shapes(
    flat_vec: torch.Tensor,
    reference_tensors: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    """Unflatten a 1D tensor into a list of tensors matching reference shapes.

    Args:
        flat_vec: 1D tensor of shape (P,).
        reference_tensors: Sequence of tensors whose shapes define the partitioning.

    Returns:
        List of view tensors matching reference tensor shapes.
    """
    unflat = []
    offset = 0
    for p in reference_tensors:
        numel = p.numel()
        unflat.append(flat_vec[offset : offset + numel].view_as(p))
        offset += numel
    return unflat


def _iter_batches(
    data: Any,
    chunk_size: int | None = None,
    device: torch.device | None = None,
) -> Generator[Tuple[torch.Tensor, torch.Tensor, float], None, None]:
    """Yield batches or chunks with appropriate loss-averaging weights.

    Args:
        data: Data bundle, DataLoader, or (x, y) tuple.
        chunk_size: Optional chunk size for sub-batching full batches.
        device: Target compute device.

    Yields:
        Tuples of (batch_x, batch_y, weight) where weight sums to 1.0 over all chunks.
    """
    # Case 1: DatasetBundle with full_batch method
    if hasattr(data, "full_batch") and callable(getattr(data, "full_batch")):
        x, y = data.full_batch()
        if device is not None:
            x, y = x.to(device), y.to(device)
        total_samples = x.size(0)
        if chunk_size is None or chunk_size >= total_samples:
            yield x, y, 1.0
        else:
            for start in range(0, total_samples, chunk_size):
                end = min(start + chunk_size, total_samples)
                w = (end - start) / float(total_samples)
                yield x[start:end], y[start:end], w
        return

    # Case 2: Direct (x, y) tuple
    if isinstance(data, (tuple, list)) and len(data) == 2 and isinstance(data[0], torch.Tensor):
        x, y = data[0], data[1]
        if device is not None:
            x, y = x.to(device), y.to(device)
        total_samples = x.size(0)
        if chunk_size is None or chunk_size >= total_samples:
            yield x, y, 1.0
        else:
            for start in range(0, total_samples, chunk_size):
                end = min(start + chunk_size, total_samples)
                w = (end - start) / float(total_samples)
                yield x[start:end], y[start:end], w
        return

    # Case 3: DataLoader or iterable of batches
    if isinstance(data, DataLoader) or hasattr(data, "__iter__"):
        batches = list(data)
        total_samples = sum(b[0].size(0) for b in batches)
        for b in batches:
            bx, by = b[0], b[1]
            if device is not None:
                bx, by = bx.to(device), by.to(device)
            w = bx.size(0) / float(total_samples) if total_samples > 0 else 1.0 / len(batches)
            yield bx, by, w
        return

    raise ValueError(f"Unsupported data format for HVP computation: {type(data)}")


def hvp(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data: Any,
    v: torch.Tensor,
    use_torch_func: bool = False,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Compute exact Hessian-vector product Hv without forming the Hessian matrix.

    Supports Pearlmutter's double backward pass and an alternative torch.func path
    using the forward-over-reverse Jacobian-Vector Product (jvp of grad).

    Args:
        model: Target PyTorch model.
        loss_fn: Loss function mapping (predictions, targets) to a scalar loss.
        data: Evaluation dataset as (x, y), DataLoader, or DatasetBundle.
        v: 1D probe tensor of shape (P,) matching trainable model parameters.
        use_torch_func: If True, use torch.func (jvp of grad) instead of double backward.
        chunk_size: Optional chunk size for sub-batching to bound peak memory.

    Returns:
        1D tensor of shape (P,) representing Hv.

    Raises:
        ValueError: If probe vector v shape does not match model parameter count.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in trainable_params)
    if v.numel() != total_params:
        raise ValueError(
            f"Dimension mismatch: probe vector v has {v.numel()} elements, "
            f"but model has {total_params} trainable parameters."
        )

    # Ensure v is on the same device and dtype as parameters
    p_ref = trainable_params[0]
    v_aligned = v.to(device=p_ref.device, dtype=p_ref.dtype).contiguous()
    v_unflat = _unflatten_to_shapes(v_aligned, trainable_params)

    was_training = model.training
    model.eval()

    total_hvp = torch.zeros_like(v_aligned)

    try:
        if not use_torch_func:
            # --- Pearlmutter Double Backward Path ---
            for x_chunk, y_chunk, weight in _iter_batches(
                data, chunk_size=chunk_size, device=p_ref.device
            ):
                model.zero_grad(set_to_none=True)
                outputs = model(x_chunk)
                loss = loss_fn(outputs, y_chunk)

                # Step 1: First-order gradients with graph retention
                grads = torch.autograd.grad(
                    loss,
                    trainable_params,
                    create_graph=True,
                    retain_graph=True,
                )

                # Step 2: Inner product of gradient with probe vector v
                grad_v_prod = sum(torch.sum(g * vi) for g, vi in zip(grads, v_unflat))

                # Step 3: Exact second backward pass
                hvp_tensors = torch.autograd.grad(
                    grad_v_prod,
                    trainable_params,
                    retain_graph=False,
                )

                flat_chunk_hvp = torch.cat([h.contiguous().view(-1) for h in hvp_tensors])
                total_hvp += weight * flat_chunk_hvp

        else:
            # --- torch.func (jvp of grad) Path ---
            named_params = {name: p for name, p in model.named_parameters() if p.requires_grad}
            v_dict = {}
            for (name, p), v_tensor in zip(named_params.items(), v_unflat):
                v_dict[name] = v_tensor

            for x_chunk, y_chunk, weight in _iter_batches(
                data, chunk_size=chunk_size, device=p_ref.device
            ):

                def loss_closure(p_dict: dict[str, torch.Tensor]) -> torch.Tensor:
                    preds = functional_call(model, p_dict, (x_chunk,))
                    # Handle MSELoss with reduction='none'.mean() to prevent ZeroTensor issue in jvp
                    if isinstance(loss_fn, nn.MSELoss) and loss_fn.reduction == "mean":
                        return nn.functional.mse_loss(preds, y_chunk, reduction="none").mean()
                    return loss_fn(preds, y_chunk)

                _, hvp_dict = jvp(grad(loss_closure), (named_params,), (v_dict,))
                flat_chunk_hvp = torch.cat(
                    [hvp_dict[name].contiguous().view(-1) for name in named_params]
                )
                total_hvp += weight * flat_chunk_hvp

    finally:
        model.train(was_training)

    return total_hvp


class HVPOperator:
    """Linear operator computing Hessian-vector products without materializing H.

    Exposes matvec(v), dim, and an internal counter of HVP evaluations.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        data: Any | None = None,
        dim: int | None = None,
        use_torch_func: bool = False,
        chunk_size: int | None = None,
        matvec_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the Hessian-Vector Product Operator.

        Args:
            model: PyTorch model whose loss curvature is tracked.
            loss_fn: Loss function.
            data: Evaluation data.
            dim: Dimension of operator if model is not provided.
            use_torch_func: If True, uses torch.func for HVP.
            chunk_size: Optional batch chunk size to limit memory.
            matvec_fn: Optional custom matrix-vector product function.
            device: Compute device.
            dtype: Compute data type.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.data = data
        self.use_torch_func = use_torch_func
        self.chunk_size = chunk_size
        self._custom_matvec = matvec_fn
        self.n_calls = 0

        if model is not None:
            self.dim = count_params(model, trainable_only=True)
            p_sample = next((p for p in model.parameters() if p.requires_grad), None)
            self.device = p_sample.device if p_sample is not None else torch.device("cpu")
            self.dtype = p_sample.dtype if p_sample is not None else torch.float32
        elif dim is not None:
            self.dim = dim
            self.device = device or torch.device("cpu")
            self.dtype = dtype or torch.float32
        else:
            raise ValueError("Either model or dim must be provided to HVPOperator.")

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        """Compute the matrix-vector product H * v and increment call counter.

        Args:
            v: 1D probe tensor of shape (dim,).

        Returns:
            1D result tensor of shape (dim,).
        """
        self.n_calls += 1
        if self._custom_matvec is not None:
            return self._custom_matvec(v)

        if self.model is None or self.loss_fn is None or self.data is None:
            raise RuntimeError("HVPOperator requires model, loss_fn, and data for HVP evaluation.")

        return hvp(
            model=self.model,
            loss_fn=self.loss_fn,
            data=self.data,
            v=v,
            use_torch_func=self.use_torch_func,
            chunk_size=self.chunk_size,
        )

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        """Alias for matvec(v)."""
        return self.matvec(v)

    def reset_counter(self) -> None:
        """Reset the HVP evaluation counter to zero."""
        self.n_calls = 0

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor) -> HVPOperator:
        """Create an HVPOperator backed by an explicit 2D matrix.

        Args:
            matrix: 2D square tensor representing A.

        Returns:
            HVPOperator whose matvec(v) evaluates A @ v.
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Expected square 2D matrix, got shape {matrix.shape}")

        def _matvec(v: torch.Tensor) -> torch.Tensor:
            return matrix @ v

        return cls(
            dim=matrix.shape[0],
            matvec_fn=_matvec,
            device=matrix.device,
            dtype=matrix.dtype,
        )

    @classmethod
    def from_callable(
        cls,
        dim: int,
        fn: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> HVPOperator:
        """Create an HVPOperator backed by an arbitrary matvec callable.

        Args:
            dim: Dimension of the linear operator.
            fn: Callable computing A * v.
            device: Compute device.
            dtype: Compute data type.

        Returns:
            HVPOperator wrapping fn.
        """
        return cls(
            dim=dim,
            matvec_fn=fn,
            device=device,
            dtype=dtype,
        )
