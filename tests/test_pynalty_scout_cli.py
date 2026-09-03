"""Tests for the Pynalty and Scout dual GUI/CLI entry points.

Both `vaila.pynalty` and `vaila.scout_vaila` are interactive (pygame /
Tkinter) tools, so these tests stay on the argument-parsing and
error-before-GUI-opens paths only -- no display/mainloop is exercised.

Update Date: 02 September 2026
Version: 0.3.119
"""

from __future__ import annotations

from pathlib import Path

from vaila import pynalty, scout_vaila


def test_pynalty_build_parser_flags() -> None:
    parser = pynalty.build_parser()
    args = parser.parse_args(["-i", "video.mp4", "-o", "out_dir", "-c", "cfg.toml"])
    assert args.input == "video.mp4"
    assert args.output == "out_dir"
    assert args.config == "cfg.toml"
    assert args.gui is False

    args_gui = parser.parse_args(["--gui"])
    assert args_gui.gui is True


def test_pynalty_main_reports_missing_video_without_opening_gui(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "does_not_exist.mp4"
    rc = pynalty.main(["-i", str(missing)])
    assert rc == 1
    assert "not found" in capsys.readouterr().out.lower()


def test_scout_build_parser_flags() -> None:
    parser = scout_vaila.build_parser()
    args = parser.parse_args(["-c", "cfg.toml"])
    assert args.config == "cfg.toml"
    assert args.gui is False


def test_scout_main_reports_missing_config_without_opening_gui(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing_scout_config.toml"
    rc = scout_vaila.main(["-c", str(missing)])
    assert rc == 1
    out = capsys.readouterr().out.lower()
    assert "not found" in out or "invalid" in out
