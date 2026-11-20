"""Unit tests for the command-line interface entrypoints."""

from __future__ import annotations

import pytest

from hesse.cli import main


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify that calling CLI without arguments displays usage help."""
    ret = main([])
    assert ret == 0
    captured = capsys.readouterr()
    assert "usage: hesse" in captured.out


def test_cli_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify that report subcommand generates report, and train executes."""
    ret = main(["report", "--run", "runs/default_run"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Generated static HTML report" in captured.out

    # Test train subcommand with 1-step small subset override
    ret_train = main(
        [
            "train",
            "--config",
            "configs/default.yaml",
            "--override",
            "steps=1",
            "data.subset_size=20",
            "tracker.track_interval=1",
        ]
    )
    assert ret_train == 0
    captured_train = capsys.readouterr()
    assert "terminated with status 'completed'" in captured_train.out
