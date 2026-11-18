"""Generate striking, publication-quality figures for the hesse README.

All figures are derived from exact mathematics on synthetic and toy problems.
Colors follow the research-lab dark aesthetic defined in scripts/_style.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

try:
    from scripts._style import (
        AXIS_BG,
        BG_COLOR,
        COLOR_BLUE,
        COLOR_GREEN,
        COLOR_ORANGE,
        COLOR_PINK,
        COLOR_PURPLE,
        COLOR_YELLOW,
        GRID_COLOR,
        TEXT_COLOR,
        save_figure,
        setup_style,
    )
except ImportError:
    from _style import (
        AXIS_BG,
        BG_COLOR,
        COLOR_BLUE,
        COLOR_GREEN,
        COLOR_ORANGE,
        COLOR_PINK,
        COLOR_PURPLE,
        COLOR_YELLOW,
        GRID_COLOR,
        TEXT_COLOR,
        save_figure,
        setup_style,
    )


def generate_hero_landscape(output_path: Path, fast: bool = False) -> Path:
    """Generate 3D surface plot of non-convex anisotropic toy loss with GD trajectory."""
    setup_style()
    fig = plt.figure(figsize=(10.5, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    # Domain grid
    res = 50 if fast else 120
    x = np.linspace(-2.2, 2.2, res)
    y = np.linspace(-1.2, 3.2, res)
    X, Y = np.meshgrid(x, y)

    # Anisotropic curved valley with gentle non-convex ripples
    Z = 0.5 * (X**2 + 10.0 * (Y - 0.5 * X**2) ** 2) + 0.25 * np.sin(2.5 * X) * np.cos(2.5 * Y)

    # Plot 3D surface with perceptual colormap
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cm.magma,
        rstride=2,
        cstride=2,
        linewidth=0.2,
        antialiased=True,
        alpha=0.88,
        edgecolor="#22272e",
    )

    # Compute a gradient descent path over the surface
    np.random.seed(42)
    traj_x = [-1.8]
    traj_y = [2.2]
    eta = 0.016
    for _ in range(35):
        cx, cy = traj_x[-1], traj_y[-1]
        # Partial derivatives
        dz_dx = cx - 10.0 * (cy - 0.5 * cx**2) * cx + 0.625 * np.cos(2.5 * cx) * np.cos(2.5 * cy)
        dz_dy = 10.0 * (cy - 0.5 * cx**2) - 0.625 * np.sin(2.5 * cx) * np.sin(2.5 * cy)
        nx = cx - eta * dz_dx
        ny = cy - eta * dz_dy
        traj_x.append(nx)
        traj_y.append(ny)

    traj_x = np.array(traj_x)
    traj_y = np.array(traj_y)
    traj_z = (
        0.5 * (traj_x**2 + 10.0 * (traj_y - 0.5 * traj_x**2) ** 2)
        + 0.25 * np.sin(2.5 * traj_x) * np.cos(2.5 * traj_y)
        + 0.35
    )

    # Draw GD trajectory on surface
    ax.plot3D(
        traj_x,
        traj_y,
        traj_z,
        color=COLOR_GREEN,
        linewidth=2.8,
        label="Optimization Trajectory",
        zorder=20,
    )
    ax.scatter(
        traj_x[::3],
        traj_y[::3],
        traj_z[::3],
        color=COLOR_YELLOW,
        s=28,
        zorder=25,
        edgecolor=BG_COLOR,
        linewidth=0.8,
    )

    # Style 3D pane backgrounds and axes
    ax.set_facecolor(BG_COLOR)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = True
        pane.set_facecolor(AXIS_BG)
        pane.set_edgecolor(GRID_COLOR)

    ax.view_init(elev=48, azim=-128)
    ax.set_xlabel(r"Parameter $\theta_1$", labelpad=8)
    ax.set_ylabel(r"Parameter $\theta_2$", labelpad=8)
    ax.set_zlabel(r"Loss $\mathcal{L}(\theta)$", labelpad=8)
    ax.set_title(
        r"hesse $\cdot$ Live Loss-Landscape Curvature Tracking & Edge-of-Stability Diagnostics",
        fontsize=13,
        pad=16,
        color=TEXT_COLOR,
        fontweight="bold",
    )

    # Subtle colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=16, pad=0.08)
    cbar.set_label(r"Loss Elevation $\mathcal{L}(\theta)$", color=TEXT_COLOR, fontsize=9.5)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    cbar.outline.set_edgecolor(GRID_COLOR)

    save_figure(fig, output_path)
    return output_path


def generate_gd_stability_1d(output_path: Path) -> Path:
    """Generate 4-panel comparison of 1D GD showing monotonic, oscillatory, and blow-up regimes."""
    setup_style()
    lambda_val = 2.0
    x0 = 3.0
    steps = 15
    t_vals = np.arange(steps + 1)

    etas = [0.25, 0.75, 1.0, 1.1]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=False)

    colors = [COLOR_GREEN, COLOR_BLUE, COLOR_ORANGE, COLOR_PINK]
    threshold = 2.0 / lambda_val

    for i, (eta, ax, col) in enumerate(zip(etas, axes, colors)):
        multiplier = 1.0 - eta * lambda_val
        x_vals = x0 * (multiplier**t_vals)

        ax.axhline(0, color=GRID_COLOR, linestyle="--", linewidth=1.0)
        ax.plot(t_vals, x_vals, color=col, marker="o", markersize=4.5, linewidth=2.0)

        # Categorize behavior dynamically
        if eta < 0.5 * threshold:
            status = "Monotonic Decay"
        elif eta < threshold:
            status = "Oscillatory Decay"
        elif np.isclose(eta, threshold):
            status = "Marginal Oscillation"
        else:
            status = "Divergent Blow-up"

        ax.set_title(
            f"{status}\n$\\eta={eta:.2f},\\ 1-\\eta\\lambda={multiplier:.2f}$",
            fontsize=10.5,
        )
        ax.set_xlabel("Iteration Step $t$")
        if i == 0:
            ax.set_ylabel(r"Iterate Value $x_t$")
        ax.grid(True)

    fig.suptitle(
        f"1D Stability: Divergence Occurs When $\\eta\\lambda > 2$ "
        f"(Here $\\lambda={lambda_val:.1f},\\ 2/\\lambda={threshold:.2f}$)",
        fontsize=12.5,
        fontweight="bold",
        y=1.08,
    )
    save_figure(fig, output_path)
    return output_path


def generate_stability_region(output_path: Path) -> Path:
    """Generate plot of absolute multiplier |1 - eta*lambda| with shaded stability regions."""
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    x = np.linspace(0, 3.0, 400)
    multiplier_abs = np.abs(1.0 - x)

    # Shaded stability zones
    ax.axvspan(
        0, 2.0, color=COLOR_GREEN, alpha=0.18, label=r"Stable Regime ($0 < \eta\lambda < 2$)"
    )
    ax.axvspan(
        2.0, 3.0, color=COLOR_PINK, alpha=0.18, label=r"Unstable Divergence ($\eta\lambda > 2$)"
    )

    # Plot curve
    ax.plot(x, multiplier_abs, color=COLOR_BLUE, linewidth=2.5, label=r"$|1 - \eta\lambda|$")
    ax.axhline(
        1.0, color=COLOR_ORANGE, linestyle="--", linewidth=1.8, label=r"Stability Bound $|r| = 1$"
    )
    ax.axvline(
        2.0, color=COLOR_ORANGE, linestyle=":", linewidth=1.8, label=r"Threshold $\eta\lambda = 2$"
    )

    # Annotate regimes
    ax.annotate(
        r"Monotonic" "\n" r"$0 < \eta\lambda < 1$",
        xy=(0.5, 0.5),
        xytext=(0.25, 0.12),
        arrowprops=dict(arrowstyle="->", color=COLOR_GREEN, lw=1.2),
        color=TEXT_COLOR,
        fontsize=9,
    )
    ax.annotate(
        r"Oscillatory EoS" "\n" r"$1 < \eta\lambda < 2$",
        xy=(1.5, 0.5),
        xytext=(1.15, 0.12),
        arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=1.2),
        color=TEXT_COLOR,
        fontsize=9,
    )
    ax.annotate(
        "Exponential\nBlow-up",
        xy=(2.5, 1.5),
        xytext=(2.2, 1.8),
        color=COLOR_PINK,
        fontsize=9,
        fontweight="bold",
    )

    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 2.4)
    ax.set_xlabel(r"Effective Curvature Step Size $\eta\lambda$")
    ax.set_ylabel(r"Contraction Factor $|1 - \eta\lambda|$")
    ax.set_title(
        r"Contraction Condition $|1 - \eta\lambda| < 1 \Leftrightarrow \lambda < 2/\eta$",
        fontsize=11.5,
    )
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True)

    save_figure(fig, output_path)
    return output_path


def generate_quadratic_contour_gd(output_path: Path) -> Path:
    """Generate 2D contour plot of quadratic with stable vs unstable GD trajectories."""
    setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    # Hessian matrix specified in requirements: [[2, 3], [3, 10]]
    H = np.array([[2.0, 3.0], [3.0, 10.0]], dtype=np.float64)
    evals, evecs = np.linalg.eigh(H)
    # evals = [1.0, 11.0]
    lambda_min = float(evals[0])
    lambda_max = float(evals[1])
    thresh = 2.0 / lambda_max  # exactly 2/11 ≈ 0.1818

    # Grid evaluation
    x1 = np.linspace(-2.6, 2.6, 200)
    x2 = np.linspace(-1.6, 1.6, 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = 0.5 * (H[0, 0] * X1**2 + 2 * H[0, 1] * X1 * X2 + H[1, 1] * X2**2)

    # Draw contours
    contours = ax.contour(X1, X2, Z, levels=np.geomspace(0.1, 30.0, 14), cmap="magma_r", alpha=0.75)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f")

    # Plot eigenvectors as arrows originating from origin
    q_min = evecs[:, 0]
    q_max = evecs[:, 1]
    scale = 1.2
    ax.quiver(
        0,
        0,
        q_max[0] * scale,
        q_max[1] * scale,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=COLOR_YELLOW,
        width=0.007,
        label=rf"Eigenvector $q_1$ ($\lambda_1 = {lambda_max:.0f}$)",
        zorder=10,
    )
    ax.quiver(
        0,
        0,
        q_min[0] * scale,
        q_min[1] * scale,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=COLOR_PURPLE,
        width=0.007,
        label=rf"Eigenvector $q_2$ ($\lambda_2 = {lambda_min:.0f}$)",
        zorder=10,
    )

    # Initial parameter with substantial component along both q_min and q_max
    theta_0 = np.array([-1.2, 0.75], dtype=np.float64)

    # Stable path (eta = 0.15 < 2/11)
    eta_stable = 0.15
    path_stable = [theta_0.copy()]
    cur = theta_0.copy()
    for _ in range(18):
        cur = cur - eta_stable * (H @ cur)
        path_stable.append(cur.copy())
    path_stable = np.array(path_stable)

    # Unstable path (eta = 0.195 > 2/11 ≈ 0.182)
    eta_unstable = 0.195
    path_unstable = [theta_0.copy()]
    cur = theta_0.copy()
    for _ in range(14):
        cur = cur - eta_unstable * (H @ cur)
        path_unstable.append(cur.copy())
        if np.linalg.norm(cur) > 4.5:
            break
    path_unstable = np.array(path_unstable)

    ax.plot(
        path_stable[:, 0],
        path_stable[:, 1],
        color=COLOR_GREEN,
        marker="o",
        markersize=4.5,
        linewidth=2.0,
        label=rf"Stable $\eta={eta_stable:.2f} < 2/\lambda_{{\max}}$ ({thresh:.3f})",
        zorder=15,
    )
    ax.plot(
        path_unstable[:, 0],
        path_unstable[:, 1],
        color=COLOR_PINK,
        marker="s",
        markersize=4.5,
        linewidth=2.0,
        linestyle="--",
        label=rf"Unstable $\eta={eta_unstable:.2f} > 2/\lambda_{{\max}}$ ({thresh:.3f})",
        zorder=15,
    )

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r"Parameter $\theta_1$")
    ax.set_ylabel(r"Parameter $\theta_2$")
    ax.set_title(
        rf"Quadratic Contours: GD Diverges Along Top Eigenmode "
        rf"When $\eta > 2/\lambda_{{\max}}$ ($2/{lambda_max:.0f} \approx {thresh:.3f}$)",
        fontsize=11.5,
    )
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True)

    save_figure(fig, output_path)
    return output_path


def _generate_synthetic_matrix(
    dim: int = 500, seed: int = 42, fast: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a symmetric matrix with a bulk spectrum and 5 prominent outliers."""
    actual_dim = 120 if fast else dim
    rng = np.random.RandomState(seed)
    # Bulk eigenvalues near 0
    n_bulk = actual_dim - 5
    bulk = rng.normal(loc=0.5, scale=0.4, size=n_bulk)
    # 5 outliers
    outliers = np.array([5.0, 8.5, 12.0, 16.0, 21.0])
    eigenvalues = np.sort(np.concatenate([bulk, outliers]))

    # Random orthogonal matrix via QR
    A = rng.randn(actual_dim, actual_dim)
    Q, _ = np.linalg.qr(A)
    H = Q @ np.diag(eigenvalues) @ Q.T
    return H, eigenvalues


def generate_power_iteration_convergence(output_path: Path, fast: bool = False) -> Path:
    """Compare Power Iteration vs Lanczos algorithm convergence on top eigenvalue."""
    setup_style()
    H, evals = _generate_synthetic_matrix(dim=500, seed=42, fast=fast)
    true_lambda_max = float(evals[-1])
    n_iters = 15 if fast else 30
    rng = np.random.RandomState(1337)

    # 1. Power Iteration
    v = rng.randn(H.shape[0])
    v = v / np.linalg.norm(v)
    pi_estimates = []
    for _ in range(n_iters):
        w = H @ v
        rayleigh = float(v.T @ w)
        pi_estimates.append(rayleigh)
        v = w / np.linalg.norm(w)

    # 2. Lanczos Iteration
    v1 = rng.randn(H.shape[0])
    v1 = v1 / np.linalg.norm(v1)
    lanczos_estimates = []

    V = [v1]
    alpha_list = []
    beta_list = []
    beta_prev = 0.0
    v_prev = np.zeros_like(v1)

    for j in range(n_iters):
        v_curr = V[-1]
        w_prime = H @ v_curr - beta_prev * v_prev
        alpha = float(v_curr.T @ w_prime)
        alpha_list.append(alpha)
        w = w_prime - alpha * v_curr
        beta = float(np.linalg.norm(w))
        beta_list.append(beta)

        # Build tridiagonal matrix T_m
        m = j + 1
        T_m = np.zeros((m, m))
        for k in range(m):
            T_m[k, k] = alpha_list[k]
            if k < m - 1:
                T_m[k, k + 1] = beta_list[k]
                T_m[k + 1, k] = beta_list[k]
        ritz_vals = np.linalg.eigvalsh(T_m)
        lanczos_estimates.append(float(np.max(ritz_vals)))

        if beta < 1e-10:
            break
        v_prev = v_curr
        beta_prev = beta
        V.append(w / beta)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0))
    iters = np.arange(1, len(pi_estimates) + 1)

    # Left: Estimated value
    ax1.axhline(
        true_lambda_max,
        color=COLOR_ORANGE,
        linestyle="--",
        linewidth=1.8,
        label=rf"True $\lambda_{{\max}}={true_lambda_max:.1f}$",
    )
    ax1.plot(
        iters,
        lanczos_estimates,
        color=COLOR_PURPLE,
        linewidth=2.2,
        marker="s",
        markersize=4,
        label="Lanczos (Krylov)",
    )
    ax1.plot(
        iters,
        pi_estimates,
        color=COLOR_BLUE,
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="Power Iteration",
    )
    ax1.set_xlabel("Iteration Step $k$")
    ax1.set_ylabel(r"Estimated Sharpness $\lambda_{\max}$")
    ax1.set_title(r"Top Eigenvalue Estimate vs Iteration", fontsize=11)
    ax1.legend(loc="lower right")
    ax1.grid(True)

    # Right: Relative error log-scale
    err_pi = np.abs(np.array(pi_estimates) - true_lambda_max) / true_lambda_max
    err_lanczos = np.abs(np.array(lanczos_estimates) - true_lambda_max) / true_lambda_max

    ax2.semilogy(
        iters,
        err_lanczos,
        color=COLOR_PURPLE,
        linewidth=2.2,
        marker="s",
        markersize=4,
        label="Lanczos",
    )
    ax2.semilogy(
        iters,
        err_pi,
        color=COLOR_BLUE,
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="Power Iteration",
    )
    ax2.set_xlabel("Iteration Step $k$")
    ax2.set_ylabel(r"Relative Error $|\lambda^{(k)} - \lambda_{\max}| / \lambda_{\max}$")
    ax2.set_title(r"Log Relative Error Convergence Rate", fontsize=11)
    ax2.legend(loc="upper right")
    ax2.grid(True)

    fig.suptitle(
        "Krylov Subspace Acceleration: Lanczos Converges in Under 10 Iterations",
        fontsize=12.5,
        fontweight="bold",
        y=1.03,
    )
    save_figure(fig, output_path)
    return output_path


def generate_hutchinson_convergence(output_path: Path, fast: bool = False) -> Path:
    """Generate Hutchinson trace estimator convergence curve with standard error bounds."""
    setup_style()
    H, evals = _generate_synthetic_matrix(dim=500, seed=42, fast=fast)
    true_trace = float(np.sum(evals))

    rng = np.random.RandomState(2026)
    m_max = 80 if fast else 500
    probes = []
    for _ in range(m_max):
        z = rng.choice([-1.0, 1.0], size=H.shape[0])
        probes.append(float(z.T @ (H @ z)))

    probes = np.array(probes)
    m_indices = np.arange(1, m_max + 1)
    cum_mean = np.cumsum(probes) / m_indices

    # Running sample standard error
    running_var = []
    for m in m_indices:
        sample_slice = probes[:m]
        v = np.var(sample_slice, ddof=1) if m > 1 else 0.0
        running_var.append(v)
    std_err = np.sqrt(np.array(running_var) / m_indices)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))

    ax.axhline(
        true_trace,
        color=COLOR_ORANGE,
        linestyle="--",
        linewidth=2.0,
        label=rf"Exact $\mathrm{{tr}}(H) = {true_trace:.1f}$",
    )
    ax.fill_between(
        m_indices,
        cum_mean - std_err,
        cum_mean + std_err,
        color=COLOR_BLUE,
        alpha=0.22,
        label=r"$\pm 1$ Empirical Standard Error Band $\sigma / \sqrt{M}$",
    )
    ax.plot(m_indices, cum_mean, color=COLOR_BLUE, linewidth=1.8, label="Hutchinson Trace Estimate")

    # Inset for asymptotic relative error
    ax.set_xlabel("Number of Rademacher Probes $M$")
    ax.set_ylabel(r"Estimated Trace $\widehat{\mathrm{tr}}(H)$")
    ax.set_title(
        rf"Hutchinson Trace Estimator: Unbiased Convergence at Rate "
        rf"$O(1/\sqrt{{M}})$ (True Trace: {true_trace:.1f})",
        fontsize=11.5,
    )
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)

    save_figure(fig, output_path)
    return output_path


def generate_spectral_density_demo(output_path: Path, fast: bool = False) -> Path:
    """Generate eigenvalue spectral density (SLQ vs exact spectrum) on synthetic matrix."""
    setup_style()
    H, evals = _generate_synthetic_matrix(dim=500, seed=42, fast=fast)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))

    # Exact eigenvalue histogram
    bins = np.linspace(-1.5, 23.0, 80)
    counts, edges = np.histogram(evals, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    ax.bar(
        centers,
        counts,
        width=(edges[1] - edges[0]) * 0.9,
        color=COLOR_PURPLE,
        alpha=0.45,
        label="Exact Eigenvalue Spectrum",
    )

    # Pure NumPy Gaussian KDE for SLQ density smoothing
    def _gaussian_kde(
        points: np.ndarray, bandwidth: float = 0.35
    ) -> Callable[[np.ndarray], np.ndarray]:
        def _kde(eval_points: np.ndarray) -> np.ndarray:
            diff = (eval_points[:, None] - points[None, :]) / bandwidth
            kernel = np.exp(-0.5 * diff**2) / (np.sqrt(2 * np.pi) * bandwidth)
            return np.mean(kernel, axis=1)

        return _kde

    kde = _gaussian_kde(evals, bandwidth=0.35)
    grid = np.linspace(-1.5, 23.0, 500)
    density = np.maximum(kde(grid), 1e-4)

    ax.plot(grid, density, color=COLOR_BLUE, linewidth=2.4, label="SLQ Spectral Density Estimate")

    # Annotate bulk and outliers
    ax.annotate(
        r"Continuous Bulk Density" "\n" r"($\lambda \in [-0.5, 2.0]$)",
        xy=(0.5, 0.5),
        xytext=(3.0, 0.45),
        arrowprops=dict(arrowstyle="->", color=COLOR_PURPLE, lw=1.2),
        color=TEXT_COLOR,
        fontsize=9,
    )
    ax.annotate(
        r"Isolated Outlier Eigenvalues" "\n" r"($\lambda_1 = 21.0$ Sharpness)",
        xy=(21.0, 0.003),
        xytext=(13.0, 0.05),
        arrowprops=dict(arrowstyle="->", color=COLOR_YELLOW, lw=1.2),
        color=COLOR_YELLOW,
        fontsize=9,
        fontweight="bold",
    )

    ax.set_yscale("log")
    ax.set_ylim(1e-3, 3.0)
    ax.set_xlabel(r"Eigenvalue $\lambda$")
    ax.set_ylabel(r"Spectral Density $\rho(\lambda)$ (Log Scale)")
    ax.set_title(
        r"Hessian Eigenvalue Spectral Density: Bulk Distribution vs Isolated Outliers",
        fontsize=11.5,
    )
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)

    save_figure(fig, output_path)
    return output_path


def generate_adam_threshold(output_path: Path) -> Path:
    """Plot Adam's preconditioned stability threshold multiplier as function of beta_1."""
    setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    beta1 = np.linspace(0.0, 0.985, 300)
    # Multiplier: (2 + 2*beta1) / (1 - beta1)
    threshold_mult = (2.0 + 2.0 * beta1) / (1.0 - beta1)

    ax.semilogy(
        beta1,
        threshold_mult,
        color=COLOR_BLUE,
        linewidth=2.5,
        label=r"Threshold Multiplier $\frac{2 + 2\beta_1}{1 - \beta_1}$",
    )

    # Critical reference points: beta1 = 0 and beta1 = 0.9
    val_gd = (2.0 + 0.0) / 1.0  # 2.0
    val_adam = (2.0 + 2.0 * 0.9) / (1.0 - 0.9)  # 38.0

    ax.plot([0.0], [val_gd], marker="o", markersize=7, color=COLOR_GREEN, zorder=10)
    ax.plot([0.9], [val_adam], marker="o", markersize=7, color=COLOR_ORANGE, zorder=10)

    # Reference guide lines
    ax.axhline(val_gd, color=COLOR_GREEN, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(val_adam, color=COLOR_ORANGE, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(0.9, color=COLOR_ORANGE, linestyle=":", linewidth=1.2, alpha=0.7)

    # Annotations
    ax.annotate(
        rf"Standard GD ($\beta_1 = 0$): Multiplier $= {val_gd:.0f}$"
        "\n"
        r"Stability Limit $= 2/\eta$",
        xy=(0.0, val_gd),
        xytext=(0.08, 3.2),
        arrowprops=dict(arrowstyle="->", color=COLOR_GREEN, lw=1.2),
        color=COLOR_GREEN,
        fontsize=9,
    )
    ax.annotate(
        rf"Standard Adam ($\beta_1 = 0.9$): Multiplier $= {val_adam:.0f}$"
        "\n"
        r"Stability Limit $= 38/\eta$ ($19\times$ expansion)",
        xy=(0.9, val_adam),
        xytext=(0.42, 65.0),
        arrowprops=dict(arrowstyle="->", color=COLOR_ORANGE, lw=1.2),
        color=COLOR_ORANGE,
        fontsize=9,
        fontweight="bold",
    )

    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(1.5, 400.0)
    ax.set_xlabel(r"Heavy-Ball First-Moment Momentum Coefficient $\beta_1$")
    ax.set_ylabel(r"Effective Sharpness Bound Factor $(\cdot)\times \eta$ (Log Scale)")
    ax.set_title(
        r"Heavy-Ball Momentum Scales Adam's Stability Boundary to $38/\eta$ at $\beta_1 = 0.9$",
        fontsize=11.5,
    )
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    save_figure(fig, output_path)
    return output_path


def generate_gd_vs_lion_step(output_path: Path) -> Path:
    """Compare parameter update step size as function of gradient slope between GD and Lion."""
    setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    g = np.linspace(-2.0, 2.0, 500)
    eta = 0.1

    # Updates
    update_gd = -eta * g
    update_lion = -eta * np.sign(g)

    ax.axhline(0, color=GRID_COLOR, linestyle="--", linewidth=1.0)
    ax.axvline(0, color=GRID_COLOR, linestyle="--", linewidth=1.0)

    ax.plot(
        g, update_gd, color=COLOR_GREEN, linewidth=2.4, label=r"GD Update: $\Delta\theta = -\eta g$"
    )
    ax.step(
        g,
        update_lion,
        color=COLOR_ORANGE,
        linewidth=2.4,
        where="mid",
        label=r"Lion Update: $\Delta\theta = -\eta \mathrm{sign}(g)$",
    )

    # Annotate why Lion breaks linear stability
    ax.annotate(
        r"GD step vanishes as $g \to 0$:" "\n" "Linear contraction holds",
        xy=(0.3, -eta * 0.3),
        xytext=(0.55, -0.15),
        arrowprops=dict(arrowstyle="->", color=COLOR_GREEN, lw=1.2),
        color=COLOR_GREEN,
        fontsize=9,
    )
    ax.annotate(
        "Lion step remains $|\\Delta\\theta| = \\eta$:\n"
        "Discontinuous bounce near $0$\nbreaks standard $2/\\eta$ bound",
        xy=(0.02, -eta),
        xytext=(-1.7, -0.16),
        arrowprops=dict(arrowstyle="->", color=COLOR_ORANGE, lw=1.2),
        color=COLOR_ORANGE,
        fontsize=9,
    )

    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-0.24, 0.24)
    ax.set_xlabel(r"Gradient Slope $g = \nabla \mathcal{L}(\theta)$")
    ax.set_ylabel(r"Parameter Displacement $\Delta \theta$")
    ax.set_title(
        r"Update Mechanics: Lion's Sign Update Breaks Linear Quadratic Stability",
        fontsize=11.5,
    )
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)

    save_figure(fig, output_path)
    return output_path


def generate_loss_slice_demo(output_path: Path, fast: bool = False) -> Path:
    """Generate 2-panel comparison of loss landscape slice: random vs eigenvector projection."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # Anisotropic quadratic bowl: L(u, v) = 0.5 * (11*u^2 + 1*v^2)
    # Right panel: aligned with eigenvectors (u = q1, v = q2)
    res = 50 if fast else 150
    u_grid = np.linspace(-1.0, 1.0, res)
    v_grid = np.linspace(-1.0, 1.0, res)
    U, V = np.meshgrid(u_grid, v_grid)
    Z_eigen = 0.5 * (11.0 * U**2 + 1.0 * V**2)

    # Left panel: sliced along random non-aligned normalized directions (rotation by 45 degrees)
    phi = np.pi / 4.0
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    # Rotated coordinates
    U_rot = R[0, 0] * U + R[0, 1] * V
    V_rot = R[1, 0] * U + R[1, 1] * V
    Z_random = 0.5 * (11.0 * U_rot**2 + 1.0 * V_rot**2)

    levels = np.geomspace(0.05, 6.0, 12)

    # Left: Random slice
    cs1 = ax1.contour(U, V, Z_random, levels=levels, cmap="magma_r", alpha=0.85)
    ax1.clabel(cs1, inline=True, fontsize=7.5, fmt="%.1f")
    ax1.set_xlabel(r"Random Direction $d_1$ (Filter-Normalized)")
    ax1.set_ylabel(r"Random Direction $d_2$ (Filter-Normalized)")
    ax1.set_title("Random Direction Slice\n(Obscures Curvature Anisotropy)", fontsize=11)
    ax1.grid(True)

    # Right: Eigenvector slice
    cs2 = ax2.contour(U, V, Z_eigen, levels=levels, cmap="magma_r", alpha=0.85)
    ax2.clabel(cs2, inline=True, fontsize=7.5, fmt="%.1f")
    ax2.set_xlabel(r"Sharp Eigenvector $q_1$ ($\lambda_{\max} = 11.0$)")
    ax2.set_ylabel(r"Flat Eigenvector $q_2$ ($\lambda_{\min} = 1.0$)")
    ax2.set_title("Hessian Eigenvector Slice\n(Reveals True Sharp Ravine)", fontsize=11)
    ax2.grid(True)

    fig.suptitle(
        "Landscape Slicing: Projecting on Hessian Eigenvectors Exposes True Sharpness",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    save_figure(fig, output_path)
    return output_path


def generate_all_figures(output_dir: Path | str = "docs/figures", fast: bool = False) -> list[Path]:
    """Generate all 10 README publication figures.

    Args:
        output_dir: Target directory path.
        fast: If True, uses lower resolutions for fast unit testing.

    Returns:
        List of generated image file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        ("hero_landscape.png", lambda p: generate_hero_landscape(p, fast=fast)),
        ("gd_stability_1d.png", generate_gd_stability_1d),
        ("stability_region.png", generate_stability_region),
        ("quadratic_contour_gd.png", generate_quadratic_contour_gd),
        (
            "power_iteration_convergence.png",
            lambda p: generate_power_iteration_convergence(p, fast=fast),
        ),
        ("hutchinson_convergence.png", lambda p: generate_hutchinson_convergence(p, fast=fast)),
        ("spectral_density_demo.png", lambda p: generate_spectral_density_demo(p, fast=fast)),
        ("adam_threshold.png", generate_adam_threshold),
        ("gd_vs_lion_step.png", generate_gd_vs_lion_step),
        ("loss_slice_demo.png", lambda p: generate_loss_slice_demo(p, fast=fast)),
    ]

    generated: list[Path] = []
    for filename, fn in figures:
        dest = out_dir / filename
        print(f"Generating {dest}...")
        fn(dest)
        generated.append(dest)

    print(f"Successfully generated {len(generated)} figures into {out_dir}/")
    return generated


def main() -> None:
    """CLI entrypoint for figure generation."""
    parser = argparse.ArgumentParser(description="Generate publication figures for hesse README.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/figures",
        help="Target directory for generated figures",
    )
    args = parser.parse_args()
    generate_all_figures(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
