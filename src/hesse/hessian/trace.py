"""Hutchinson randomized trace estimator using Rademacher probe vectors."""

from __future__ import annotations

import logging
import math
from typing import Iterator, List

import torch

from hesse.hessian.hvp import HVPOperator

logger = logging.getLogger(__name__)


class TraceResult:
    """Result container for Hutchinson trace estimation with 2-tuple unpacking.

    Attributes:
        estimate: Monte Carlo estimate of tr(H).
        std_err: Empirical standard error of the mean estimate: s / sqrt(M).
        samples: List of individual probe trace estimates z_j^T H z_j.
    """

    def __init__(
        self,
        estimate: float,
        std_err: float,
        samples: List[float],
    ) -> None:
        self.estimate = float(estimate)
        self.std_err = float(std_err)
        self.samples = samples

    def __iter__(self) -> Iterator[float]:
        """Allow 2-tuple unpacking: (estimate, std_err)."""
        yield self.estimate
        yield self.std_err

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> float:
        return (self.estimate, self.std_err)[idx]

    def __repr__(self) -> str:
        return f"TraceResult(estimate={self.estimate:.4f}, std_err={self.std_err:.4f})"


def hutchinson_trace(
    op: HVPOperator,
    n_samples: int = 100,
    seed: int = 42,
) -> TraceResult:
    """Estimate the Hessian trace tr(H) using Hutchinson's Rademacher Monte Carlo estimator.

    Each probe draws z_i in {-1, +1} i.i.d., computes H z, and evaluates the quadratic
    form z^T H z. Rademacher probes attain minimal variance among all isotropic distributions.

    Args:
        op: Hessian-vector product operator.
        n_samples: Number of random Rademacher probe evaluations (M).
        seed: Random seed for probe generation.

    Returns:
        TraceResult containing trace estimate, standard error, and individual samples.

    Raises:
        ValueError: If n_samples is not positive.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive integer, got {n_samples}")

    gen = torch.Generator(device=op.device)
    gen.manual_seed(seed)

    samples: List[float] = []

    for _ in range(n_samples):
        # Sample Rademacher probe in {-1, +1}^P
        # torch.randint(0, 2) produces 0 or 1; multiply by 2 and subtract 1
        z = (
            torch.randint(
                0,
                2,
                (op.dim,),
                generator=gen,
                device=op.device,
                dtype=torch.int64,
            ).to(op.dtype)
            * 2.0
            - 1.0
        )

        hz = op.matvec(z)
        sample = torch.dot(z, hz).item()
        samples.append(sample)

    estimate = sum(samples) / float(n_samples)

    if n_samples > 1:
        variance = sum((s - estimate) ** 2 for s in samples) / float(n_samples - 1)
        std_err = math.sqrt(variance / float(n_samples))
    else:
        std_err = 0.0

    return TraceResult(
        estimate=estimate,
        std_err=std_err,
        samples=samples,
    )
