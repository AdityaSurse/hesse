"""Shared experimental protocol utilities, metrics computation, and publication plotting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def compute_run_metrics(
    records: Sequence[Dict[str, Any]],
    stability_threshold: float | None = None,
    use_preconditioned_for_threshold: bool = False,
) -> Dict[str, Any]:
    """Compute scientific metrics from a single run's logged telemetry stream.

    Metrics include:
      1. time_to_threshold: First discrete training step where sharpness >= threshold.
      2. plateau_window: Steps t >= time_to_threshold (or t >= 0.6 * max_step if unreached).
      3. plateau_ratio: Mean of (sharpness / threshold) over the plateau window.
      4. non_monotone_loss_fraction: Fraction of step transitions where L_t > L_{t-1}.

    Args:
        records: List of telemetry dictionaries from metrics.jsonl.
        stability_threshold: Analytical linear stability threshold (e.g. 2/eta).
        use_preconditioned_for_threshold: If True (Adam), benchmarks preconditioned
            sharpness against the threshold.

    Returns:
        Dictionary of scalar metrics and series extracted from the run.
    """
    if not records:
        return {}

    steps = [int(r["step"]) for r in records]
    losses = [float(r["loss"]) for r in records]
    max_step = max(steps)

    # Sharpness series
    sharp_records = [r for r in records if r.get("sharpness") is not None]
    sharp_steps = [int(r["step"]) for r in sharp_records]
    sharp_vals = [float(r["sharpness"]) for r in sharp_records]

    # Preconditioned sharpness series (Adam)
    precond_records = [r for r in records if r.get("preconditioned_sharpness") is not None]
    precond_steps = [int(r["step"]) for r in precond_records]
    precond_vals = [float(r["preconditioned_sharpness"]) for r in precond_records]

    # Sign curvature series (Lion)
    sign_records = [r for r in records if r.get("sign_curvature") is not None]
    sign_steps = [int(r["step"]) for r in sign_records]
    sign_vals = [float(r["sign_curvature"]) for r in sign_records]

    # Top-k eigenvalues & trace
    topk_vals = [r["topk_eigenvalues"] for r in records if r.get("topk_eigenvalues") is not None]
    traces = [float(r["trace"]) for r in records if r.get("trace") is not None]

    # 1. Time to threshold
    effective_thresh = stability_threshold
    if effective_thresh is None and len(records) > 0 and records[0].get("stability_threshold"):
        effective_thresh = float(records[0]["stability_threshold"])

    time_to_thresh: int | None = None
    if effective_thresh is not None:
        target_steps = precond_steps if use_preconditioned_for_threshold else sharp_steps
        target_vals = precond_vals if use_preconditioned_for_threshold else sharp_vals
        for s, v in zip(target_steps, target_vals):
            if v >= effective_thresh:
                time_to_thresh = s
                break

    # 2. Plateau Window
    if time_to_thresh is not None:
        plateau_cutoff = time_to_thresh
    else:
        plateau_cutoff = int(0.6 * max_step)

    # 3. Plateau curvature metrics
    plateau_sharpness_pts = [v for s, v in zip(sharp_steps, sharp_vals) if s >= plateau_cutoff]
    if plateau_sharpness_pts:
        mean_plateau_sharpness = float(np.mean(plateau_sharpness_pts))
        std_plateau_sharpness = float(np.std(plateau_sharpness_pts))
    else:
        mean_plateau_sharpness = sharp_vals[-1] if sharp_vals else 0.0
        std_plateau_sharpness = 0.0

    plateau_ratio: float | None = None
    if effective_thresh is not None and effective_thresh > 0:
        if use_preconditioned_for_threshold and precond_vals:
            plateau_precond_pts = [
                v for s, v in zip(precond_steps, precond_vals) if s >= plateau_cutoff
            ]
            if plateau_precond_pts:
                plateau_ratio = float(np.mean(plateau_precond_pts)) / effective_thresh
        elif plateau_sharpness_pts:
            plateau_ratio = mean_plateau_sharpness / effective_thresh

    # 4. Non-monotonic loss transitions
    non_mono_count = 0
    total_transitions = max(1, len(losses) - 1)
    for i in range(1, len(losses)):
        if losses[i] > losses[i - 1]:
            non_mono_count += 1
    non_monotone_frac = float(non_mono_count) / float(total_transitions)

    # 5. Sign curvature metrics (Lion)
    mean_sign_curv: float | None = None
    if sign_vals:
        plateau_sign_pts = [v for s, v in zip(sign_steps, sign_vals) if s >= plateau_cutoff]
        mean_sign_curv = (
            float(np.mean(plateau_sign_pts)) if plateau_sign_pts else float(np.mean(sign_vals))
        )

    # 6. Spectral gap (lambda_1 - lambda_2) if top-k available
    spectral_gaps: List[float] = []
    if topk_vals:
        for evs in topk_vals:
            if len(evs) >= 2:
                spectral_gaps.append(float(evs[0] - evs[1]))

    return {
        "steps": steps,
        "losses": losses,
        "sharp_steps": sharp_steps,
        "sharp_vals": sharp_vals,
        "precond_steps": precond_steps,
        "precond_vals": precond_vals,
        "sign_steps": sign_steps,
        "sign_vals": sign_vals,
        "time_to_threshold": time_to_thresh,
        "plateau_cutoff_step": plateau_cutoff,
        "plateau_sharpness_mean": mean_plateau_sharpness,
        "plateau_sharpness_std": std_plateau_sharpness,
        "plateau_ratio": plateau_ratio,
        "non_monotone_loss_fraction": non_monotone_frac,
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "final_sharpness": sharp_vals[-1] if sharp_vals else None,
        "max_sharpness": max(sharp_vals) if sharp_vals else None,
        "stability_threshold": effective_thresh,
        "mean_sign_curvature": mean_sign_curv,
        "mean_spectral_gap": float(np.mean(spectral_gaps)) if spectral_gaps else None,
        "mean_trace": float(np.mean(traces)) if traces else None,
    }


def aggregate_seed_metrics(seed_metrics_list: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metric dictionaries across multiple seeds into mean and std summaries.

    Args:
        seed_metrics_list: List of metric dictionaries from compute_run_metrics.

    Returns:
        Aggregated dictionary containing mean, std, and raw values for each metric.
    """
    if not seed_metrics_list:
        return {}

    metric_keys = [
        "time_to_threshold",
        "plateau_sharpness_mean",
        "plateau_ratio",
        "non_monotone_loss_fraction",
        "final_loss",
        "min_loss",
        "final_sharpness",
        "max_sharpness",
        "mean_sign_curvature",
        "mean_spectral_gap",
        "mean_trace",
    ]

    agg: Dict[str, Any] = {}
    for key in metric_keys:
        raw_vals = [
            m[key] for m in seed_metrics_list if m.get(key) is not None and not np.isnan(m[key])
        ]
        if raw_vals:
            # Check if all values are numeric
            num_vals = [float(v) for v in raw_vals]
            agg[key] = {
                "mean": float(np.mean(num_vals)),
                "std": float(np.std(num_vals, ddof=1)) if len(num_vals) > 1 else 0.0,
                "raw": num_vals,
                "n_valid": len(num_vals),
            }
        else:
            agg[key] = {"mean": None, "std": None, "raw": [], "n_valid": 0}

    return agg


def interpolate_common_steps(
    runs: Sequence[Dict[str, Any]],
    val_key: str = "sharp_vals",
    step_key: str = "sharp_steps",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute interpolated mean and standard deviation curves across multiple runs.

    Args:
        runs: List of run metric dictionaries.
        val_key: Key in dictionary corresponding to the metric array.
        step_key: Key in dictionary corresponding to the step array.

    Returns:
        Tuple of (common_steps, mean_series, std_series).
    """
    valid_runs = [r for r in runs if r.get(step_key) and r.get(val_key)]
    if not valid_runs:
        return np.array([]), np.array([]), np.array([])

    all_steps = sorted(list({s for r in valid_runs for s in r[step_key]}))
    common_steps = np.array(all_steps, dtype=np.float32)

    interpolated_matrix = []
    for r in valid_runs:
        x = np.array(r[step_key], dtype=np.float32)
        y = np.array(r[val_key], dtype=np.float32)
        if len(x) == 1:
            y_interp = np.full_like(common_steps, y[0])
        else:
            y_interp = np.interp(common_steps, x, y)
        interpolated_matrix.append(y_interp)

    mat = np.array(interpolated_matrix)
    mean_series = np.mean(mat, axis=0)
    std_series = np.std(mat, axis=0, ddof=1) if len(valid_runs) > 1 else np.zeros_like(mean_series)

    return common_steps, mean_series, std_series


def save_experiment_json(
    exp_id: str,
    payload: Dict[str, Any],
    results_dir: str | Path = "experiments/results",
) -> Path:
    """Save raw experimental results dictionary to a nicely formatted JSON file.

    Args:
        exp_id: Experiment identifier string (e.g. 'e1_gd_eos').
        payload: Full experimental data dictionary.
        results_dir: Destination results directory path.

    Returns:
        Path to saved JSON file.
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{exp_id}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_file
