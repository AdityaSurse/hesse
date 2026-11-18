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

    # Subcommand: landscape
    landscape_parser = subparsers.add_parser(
        "landscape",
        help="Compute and render 1D/2D/3D loss landscape slices",
    )
    landscape_parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint tensor file",
    )
    landscape_parser.add_argument(
        "--output",
        type=str,
        default="landscape.png",
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

    if args.subcommand in {"train", "landscape", "report"}:
        print(f"Command '{args.subcommand}' not implemented yet.")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
