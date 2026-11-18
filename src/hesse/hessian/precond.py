"""Preconditioned sharpness and stability thresholds for adaptive optimizers (Adam)."""

from __future__ import annotations

import logging
from typing import Iterator

import torch
import torch.nn as nn

from hesse.hessian.hvp import HVPOperator
from hesse.hessian.power import power_iteration

logger = logging.getLogger(__name__)


class PreconditionedResult:
    """Result container for preconditioned sharpness with 2-tuple unpacking.

    Attributes:
        sharpness: Top eigenvalue lambda_max(P^{-1/2} H P^{-1/2}).
        threshold: Theoretical stability threshold (2 + 2*beta1) / ((1 - beta1) * lr).
        eigenvector: Corresponding unit eigenvector in the transformed coordinate space.
        preconditioner: Diagonal scaling tensor P of shape (dim,).
    """

    def __init__(
        self,
        sharpness: float,
        threshold: float,
        eigenvector: torch.Tensor,
        preconditioner: torch.Tensor,
    ) -> None:
        self.sharpness = float(sharpness)
        self.threshold = float(threshold)
        self.eigenvector = eigenvector
        self.preconditioner = preconditioner

    def __iter__(self) -> Iterator[float]:
        """Allow 2-tuple unpacking: (sharpness, threshold)."""
        yield self.sharpness
        yield self.threshold

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> float:
        return (self.sharpness, self.threshold)[idx]

    def __repr__(self) -> str:
        return (
            f"PreconditionedResult(sharpness={self.sharpness:.4f}, threshold={self.threshold:.4f})"
        )


def extract_adam_preconditioner(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> torch.Tensor:
    """Extract diagonal Adam preconditioner P = diag(sqrt(v_hat) + eps) across all parameters.

    If optimizer state is not yet populated for a parameter (e.g. prior to step 1),
    falls back to an identity diagonal slice (1.0).

    Args:
        model: Target PyTorch neural network.
        optimizer: Adam optimizer instance with state dict.

    Returns:
        1D tensor of shape (P,) representing the diagonal entries of P.
    """
    p_entries: list[torch.Tensor] = []

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        return torch.empty(0)

    device = trainable_params[0].device
    dtype = trainable_params[0].dtype

    for p in trainable_params:
        if optimizer is not None and p in optimizer.state and "exp_avg_sq" in optimizer.state[p]:
            state = optimizer.state[p]
            v_t = state["exp_avg_sq"].to(device=device, dtype=dtype)
            step_val = state.get("step", 1)
            t = step_val.item() if isinstance(step_val, torch.Tensor) else float(step_val)

            # Match parameter group hyperparameters
            group = next(
                (g for g in optimizer.param_groups if any(p is param for param in g["params"])),
                optimizer.param_groups[0],
            )
            beta2 = group["betas"][1]
            eps = group["eps"]

            bias_correction = 1.0 - (beta2**t)
            if bias_correction > 0:
                v_hat = v_t / bias_correction
            else:
                v_hat = v_t

            diag_p = torch.sqrt(torch.clamp(v_hat, min=0.0)) + eps
            p_entries.append(diag_p.view(-1))
        else:
            # Identity fallback when uninitialized
            p_entries.append(torch.ones(p.numel(), device=device, dtype=dtype))

    return torch.cat(p_entries)


def preconditioned_sharpness(
    op: HVPOperator,
    optimizer: torch.optim.Optimizer | None = None,
    preconditioner: torch.Tensor | None = None,
    iters: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
    lr: float | None = None,
    beta1: float | None = None,
) -> PreconditionedResult:
    """Compute preconditioned sharpness lambda_max(P^{-1/2} H P^{-1/2}) and stability threshold.

    Solves the generalized eigenvalue problem H v = lambda P v in symmetric coordinate
    space u = P^{1/2} v, evaluating (P^{-1/2} H P^{-1/2}) u without matrix inversion.

    Args:
        op: Hessian-vector product operator.
        optimizer: PyTorch Adam/AdamW optimizer instance.
        preconditioner: Optional explicit 1D diagonal tensor P of shape (dim,).
        iters: Maximum iterations for power iteration.
        tol: Convergence tolerance for power iteration.
        seed: Random seed for power iteration probe.
        lr: Learning rate override (reads from optimizer if None).
        beta1: First-moment momentum beta1 override (reads from optimizer if None).

    Returns:
        PreconditionedResult holding sharpness, threshold, and transformed eigenvector.
    """
    # 1. Resolve diagonal preconditioner P
    if preconditioner is not None:
        p_diag = preconditioner.to(device=op.device, dtype=op.dtype)
    elif op.model is not None:
        p_diag = extract_adam_preconditioner(op.model, optimizer)
    else:
        # Fallback to identity operator
        p_diag = torch.ones(op.dim, device=op.device, dtype=op.dtype)

    if p_diag.numel() != op.dim:
        raise ValueError(
            f"Preconditioner size ({p_diag.numel()}) does not match operator dimension ({op.dim})."
        )

    # 2. Compute P^{-1/2} element-wise
    p_inv_sqrt = 1.0 / torch.sqrt(torch.clamp(p_diag, min=1e-16))

    # 3. Define symmetric preconditioned operator u -> P^{-1/2} * H * (P^{-1/2} * u)
    def _preconditioned_matvec(u: torch.Tensor) -> torch.Tensor:
        w = p_inv_sqrt * u
        hw = op.matvec(w)
        return p_inv_sqrt * hw

    precond_op = HVPOperator.from_callable(
        dim=op.dim,
        fn=_preconditioned_matvec,
        device=op.device,
        dtype=op.dtype,
    )

    # 4. Run power iteration on preconditioned operator
    power_res = power_iteration(precond_op, iters=iters, tol=tol, seed=seed)
    sharpness = power_res.eigenvalue

    # 5. Compute theoretical stability threshold: (2 + 2*beta1) / ((1 - beta1) * lr)
    if optimizer is not None:
        group = optimizer.param_groups[0]
        actual_lr = lr if lr is not None else float(group.get("lr", 1e-3))
        actual_beta1 = beta1 if beta1 is not None else float(group.get("betas", (0.9, 0.999))[0])
    else:
        actual_lr = lr if lr is not None else 1e-3
        actual_beta1 = beta1 if beta1 is not None else 0.9

    if actual_lr <= 0 or actual_beta1 >= 1.0:
        threshold = float("inf")
    else:
        threshold = (2.0 + 2.0 * actual_beta1) / ((1.0 - actual_beta1) * actual_lr)

    return PreconditionedResult(
        sharpness=sharpness,
        threshold=threshold,
        eigenvector=power_res.eigenvector,
        preconditioner=p_diag,
    )
