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
    """Verify that train, landscape, and report subcommands execute cleanly."""
    for cmd in ["train", "landscape", "report"]:
        ret = main([cmd])
        assert ret == 0
        captured = capsys.readouterr()
        assert f"Command '{cmd}' not implemented yet." in captured.out
