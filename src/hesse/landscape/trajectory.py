"""Trajectory projection and checkpoint extraction for loss-landscape overlays."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def project_trajectory_2d(
    checkpoints: Sequence[torch.Tensor] | torch.Tensor,
    d1: torch.Tensor,
    d2: torch.Tensor,
    reference_theta: torch.Tensor | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project parameter trajectory checkpoints onto 2D coordinate directions (d1, d2).

    Coordinates are given by:
        alpha_t = <theta_t - theta*, d1> / ||d1||^2
        beta_t = <theta_t - theta*, d2> / ||d2||^2
    where theta* is the reference point (defaults to the final checkpoint theta_final).
    At theta_final, delta = 0, so (alpha_final, beta_final) = (0.0, 0.0).

    Args:
        checkpoints: Sequence of 1D parameter vectors along optimization trajectory.
        d1: Horizontal projection direction vector of shape (P,).
        d2: Vertical projection direction vector of shape (P,).
        reference_theta: Optional origin parameter vector (defaults to checkpoints[-1]).

    Returns:
        Tuple of (alphas, betas) numpy arrays representing 2D trajectory coordinates.
    """
    flat_ckpts = [ckpt.detach().reshape(-1) for ckpt in checkpoints]
    ref = reference_theta.detach().reshape(-1) if reference_theta is not None else flat_ckpts[-1]

    d1_vec = d1.to(device=ref.device, dtype=ref.dtype).reshape(-1)
    d2_vec = d2.to(device=ref.device, dtype=ref.dtype).reshape(-1)

    norm1_sq = torch.dot(d1_vec, d1_vec).item()
    norm2_sq = torch.dot(d2_vec, d2_vec).item()

    if norm1_sq == 0.0:
        norm1_sq = 1.0
    if norm2_sq == 0.0:
        norm2_sq = 1.0

    alphas: List[float] = []
    betas: List[float] = []

    for theta in flat_ckpts:
        diff = theta - ref
        alpha_val = torch.dot(diff, d1_vec).item() / norm1_sq
        beta_val = torch.dot(diff, d2_vec).item() / norm2_sq
        alphas.append(alpha_val)
        betas.append(beta_val)

    return np.array(alphas, dtype=np.float32), np.array(betas, dtype=np.float32)


def load_checkpoints_from_run(run_dir: str | Path) -> List[torch.Tensor]:
    """Load parameter checkpoints in chronological step order from a run directory.

    Args:
        run_dir: Path to directory containing a 'checkpoints' subfolder.

    Returns:
        List of 1D flattened parameter tensors ordered by training step.

    Raises:
        FileNotFoundError: If the checkpoints directory does not exist or contains no checkpoints.
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found at {ckpt_dir}")

    # Discover periodic and final checkpoints
    step_files: list[tuple[int, Path]] = []
    final_file: Path | None = None

    for pt_file in ckpt_dir.glob("*.pt"):
        if pt_file.name == "checkpoint_final.pt":
            final_file = pt_file
            continue
        match = re.search(r"checkpoint_step_(\d+)\.pt", pt_file.name)
        if match:
            step_files.append((int(match.group(1)), pt_file))

    step_files.sort(key=lambda item: item[0])

    files_to_load = [f for _, f in step_files]
    if final_file is not None:
        files_to_load.append(final_file)

    if not files_to_load:
        raise FileNotFoundError(f"No checkpoint .pt files found in {ckpt_dir}")

    tensors: List[torch.Tensor] = []
    for f in files_to_load:
        ckpt_data = torch.load(f, map_location="cpu", weights_only=False)
        if isinstance(ckpt_data, dict) and "model_state" in ckpt_data:
            state_dict = ckpt_data["model_state"]
            flat = torch.cat([v.detach().reshape(-1) for v in state_dict.values()])
            tensors.append(flat)
        elif isinstance(ckpt_data, torch.Tensor):
            tensors.append(ckpt_data.detach().reshape(-1))
        elif isinstance(ckpt_data, dict):
            # Assume dict of parameters
            flat = torch.cat([v.detach().reshape(-1) for v in ckpt_data.values()])
            tensors.append(flat)

    return tensors
