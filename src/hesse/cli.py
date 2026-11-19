"""Command-line interface (CLI) for hesse."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser with subcommands.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="hesse",
        description="hesse: Live Loss-Landscape Curvature Tracking & Edge-of-Stability Diagnostics",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    # Subcommand: train
    train_parser = subparsers.add_parser(
        "train",
        help="Train a neural network with live curvature tracking",
    )
    train_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    train_parser.add_argument(
        "--override",
        "-o",
        nargs="+",
        action="extend",
        default=[],
        help="Dotted configuration overrides (e.g. --override optim.lr=0.02 steps=20)",
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (.pt) file to resume training from",
    )

    # Subcommand: landscape
    landscape_parser = subparsers.add_parser(
        "landscape",
        help="Compute and render 1D/2D/3D loss landscape slices",
    )
    landscape_parser.add_argument(
        "--run",
        "-r",
        type=str,
        default=None,
        help="Path to run directory containing config.yaml and checkpoints",
    )
    landscape_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint tensor file",
    )
    landscape_parser.add_argument(
        "--directions",
        "--basis",
        dest="directions",
        type=str,
        choices=["random", "hessian", "pca"],
        default="random",
        help="Direction basis generation method: 'random', 'hessian', or 'pca'",
    )
    landscape_parser.add_argument(
        "--res",
        "--grid-size",
        dest="res",
        type=int,
        default=41,
        help="Grid resolution per axis (default: 41)",
    )
    landscape_parser.add_argument(
        "--span",
        "--range",
        dest="span",
        type=float,
        default=1.0,
        help="Coordinate perturbation span [-span, span] (default: 1.0)",
    )
    landscape_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Target output figure path",
    )

    # Subcommand: report
    report_parser = subparsers.add_parser(
        "report",
        help="Compile experimental figures and markdown telemetry reports",
    )
    report_parser.add_argument(
        "--run-dir",
        type=str,
        default="runs",
        help="Directory containing run logs and telemetry artifacts",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint.

    Args:
        argv: Command-line arguments sequence, or None for sys.argv[1:].

    Returns:
        Exit code integer (0 for success).
    """
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.subcommand == "train":
        from hesse.config import RunConfig
        from hesse.train import train

        cfg = RunConfig.from_yaml(args.config)
        summary = train(cfg, overrides=args.override, resume_from=args.resume)
        status = summary.get("status", "unknown")
        total_steps = summary.get("total_steps", 0)
        print(f"Run '{cfg.run_name}' terminated with status '{status}' ({total_steps} steps).")
        return 0

    if args.subcommand == "landscape":
        from pathlib import Path

        import numpy as np
        import torch

        from hesse.config import RunConfig
        from hesse.data import load_dataset
        from hesse.landscape import (
            evaluate_2d,
            get_directions,
            load_checkpoints_from_run,
            project_trajectory_2d,
            save_grid_npz,
        )
        from hesse.models import get_model
        from hesse.utils import flatten_params, get_device, unflatten_params
        from hesse.viz import plot_landscape_contour, plot_landscape_surface_3d

        # 1. Resolve run_dir and ckpt_path
        run_dir: Path | None = Path(args.run) if args.run else None
        ckpt_path: Path | None = Path(args.checkpoint) if args.checkpoint else None

        if run_dir is None and ckpt_path is not None:
            if (ckpt_path.parent / "config.yaml").is_file():
                run_dir = ckpt_path.parent
            elif (ckpt_path.parent.parent / "config.yaml").is_file():
                run_dir = ckpt_path.parent.parent
            else:
                run_dir = ckpt_path.parent

        if run_dir is None:
            default_run = Path("runs/gd_mlp_cifar5k")
            if default_run.exists():
                run_dir = default_run
            else:
                runs_root = Path("runs")
                subdirs = (
                    [d for d in runs_root.iterdir() if d.is_dir()] if runs_root.exists() else []
                )
                if subdirs:
                    run_dir = subdirs[0]
                else:
                    raise FileNotFoundError(
                        "No run directory found. Provide --run or run a training job first."
                    )

        if ckpt_path is None:
            final_ckpt = run_dir / "checkpoints" / "checkpoint_final.pt"
            if final_ckpt.is_file():
                ckpt_path = final_ckpt
            else:
                ckpt_dir = run_dir / "checkpoints"
                ckpt_candidates = (
                    list(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else list(run_dir.glob("*.pt"))
                )
                if ckpt_candidates:
                    ckpt_candidates.sort(key=lambda p: (0 if "final" in p.name else 1, p.name))
                    ckpt_path = ckpt_candidates[0]
                else:
                    raise FileNotFoundError(
                        f"No checkpoint .pt found in {run_dir} or {run_dir / 'checkpoints'}."
                    )

        # 2. Resolve configuration
        cfg_file = run_dir / "config.yaml"
        if cfg_file.is_file():
            cfg = RunConfig.from_yaml(cfg_file)
        elif Path("configs/default.yaml").is_file():
            cfg = RunConfig.from_yaml("configs/default.yaml")
        else:
            cfg = RunConfig()

        # 3. Model setup
        device = get_device()
        model = get_model(cfg.model, seed=cfg.seed)
        ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt_data, dict) and "model_state" in ckpt_data:
            model.load_state_dict(ckpt_data["model_state"])
        elif isinstance(ckpt_data, dict) and any(k.startswith("layers.") for k in ckpt_data.keys()):
            model.load_state_dict(ckpt_data)
        elif isinstance(ckpt_data, torch.Tensor):
            unflatten_params(model, ckpt_data)
        elif isinstance(ckpt_data, dict) and "state_dict" in ckpt_data:
            model.load_state_dict(ckpt_data["state_dict"])
        else:
            try:
                model.load_state_dict(ckpt_data)
            except Exception as e:
                raise RuntimeError(f"Could not load state dict from {ckpt_path}: {e}")

        model.to(device)
        model.eval()

        # 4. Data setup
        bundle = load_dataset(cfg.data)
        if hasattr(bundle, "full_batch"):
            data = bundle.full_batch()
        elif hasattr(bundle, "train_loader") and bundle.train_loader is not None:
            data = next(iter(bundle.train_loader))
        else:
            data = bundle
        if cfg.data.loss_type == "mse":
            loss_fn = torch.nn.MSELoss()
        elif cfg.data.loss_type == "cross_entropy":
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {cfg.data.loss_type}")

        # 5. Checkpoints / Trajectory
        ckpts = None
        try:
            ckpts = load_checkpoints_from_run(run_dir)
        except Exception:
            ckpts = None

        # 6. Directions
        d1, d2 = get_directions(
            method=args.directions,
            model=model,
            loss_fn=loss_fn,
            data=data,
            checkpoints=ckpts,
            seed=cfg.seed,
        )

        # 7. Evaluate 2D grid
        alphas = np.linspace(-args.span, args.span, args.res)
        betas = np.linspace(-args.span, args.span, args.res)
        grid_result = evaluate_2d(
            model=model,
            loss_fn=loss_fn,
            data=data,
            d1=d1,
            d2=d2,
            alphas=alphas,
            betas=betas,
            show_progress=True,
        )

        # 8. Save grid .npz
        landscape_dir = run_dir / "landscape"
        landscape_dir.mkdir(parents=True, exist_ok=True)
        grid_npz_path = landscape_dir / f"grid_{args.directions}.npz"
        save_grid_npz(
            grid_npz_path,
            alphas=alphas,
            betas=betas,
            loss=grid_result["loss"],
            acc=grid_result.get("acc"),
            metadata={
                "directions": args.directions,
                "span": args.span,
                "res": args.res,
                "checkpoint": str(ckpt_path),
            },
        )
        print(f"Saved landscape grid archive to {grid_npz_path}")

        # 9. Trajectory projection
        traj_coords = None
        if ckpts is not None and len(ckpts) > 0:
            flat_final = flatten_params(model).detach()
            traj_coords = project_trajectory_2d(
                ckpts,
                d1=d1,
                d2=d2,
                reference_theta=flat_final,
            )

        # 10. Plots
        contour_out = (
            Path(args.output)
            if args.output
            else (landscape_dir / f"landscape_contour_{args.directions}.png")
        )
        plot_landscape_contour(
            alphas=alphas,
            betas=betas,
            loss_grid=grid_result["loss"],
            trajectory_coords=traj_coords,
            save_path=contour_out,
            title=f"Loss Landscape ({args.directions.capitalize()} Directions)",
        )
        print(f"Saved landscape contour plot to {contour_out}")

        surface_out = (
            contour_out.with_suffix(".html")
            if args.output
            else (landscape_dir / f"landscape_3d_{args.directions}.html")
        )
        plot_landscape_surface_3d(
            alphas=alphas,
            betas=betas,
            loss_grid=grid_result["loss"],
            trajectory_coords=traj_coords,
            save_path=surface_out,
            title=f"3D Interactive Loss Surface ({args.directions.capitalize()})",
        )
        print(f"Saved 3D interactive surface to {surface_out}")
        return 0

    if args.subcommand in {"report"}:
        print(f"Command '{args.subcommand}' not implemented yet.")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
