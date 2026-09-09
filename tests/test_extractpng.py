"""Tests for vaila.extractpng command builders and helpers (no GUI)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vaila import extractpng as ep


def test_build_extract_png_command_hwaccel_before_input() -> None:
    cmd = ep.build_extract_png_command(
        "/data/clip.mp4",
        "/out/%09d.png",
        width=640,
        height=360,
        hwaccel=True,
    )
    assert cmd[0] == "ffmpeg"
    assert "-hwaccel" in cmd
    assert cmd.index("-hwaccel") < cmd.index("-i")
    assert "hevc_cuvid" not in cmd
    assert cmd[-1] == "/out/%09d.png"


def test_build_extract_png_command_software_has_no_hwaccel() -> None:
    cmd = ep.build_extract_png_command("a.mp4", "b/%09d.png", width=100, height=100, hwaccel=False)
    assert "-hwaccel" not in cmd
    assert cmd.index("-i") < len(cmd) - 1


def test_build_select_frame_command_output_last() -> None:
    cmd = ep.build_select_frame_command("v.mp4", 7, "out/frame_007.png")
    assert cmd[-1] == "out/frame_007.png"
    assert "-i" in cmd
    assert cmd.index("-i") < cmd.index("-vf")


def test_build_png_to_video_codec_264_and_265() -> None:
    c264 = ep.build_png_to_video_command("d/%09d.png", "o.mp4", fps=30.0, codec="264")
    assert "libx264" in c264
    c265 = ep.build_png_to_video_command("d/%09d.png", "o.mp4", fps=60.0, codec="265")
    assert "libx265" in c265
    assert c265[-1] == "o.mp4"


def test_parse_frame_list() -> None:
    assert ep.parse_frame_list("0,3,5") == [0, 3, 5]
    assert ep.parse_frame_list("5 3 5 0") == [0, 3, 5]
    with pytest.raises(ValueError):
        ep.parse_frame_list("-1,2")


def test_list_videos_in_dir(tmp_path: Path) -> None:
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "b.AVI").write_bytes(b"x")
    (tmp_path / "note.txt").write_text("nope")
    names = [p.name for p in ep.list_videos_in_dir(tmp_path)]
    assert names == ["a.mp4", "b.AVI"]


def test_png_dirs_to_process(tmp_path: Path) -> None:
    seq = tmp_path / "seq"
    seq.mkdir()
    (seq / "000000001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    out = tmp_path / "vaila_png2videos_ignore"
    out.mkdir()
    dirs = ep._png_dirs_to_process(tmp_path, exclude=out)
    assert dirs == [seq]


def test_build_cli_argv_extract() -> None:
    argv = ep.build_cli_argv("extract", input_path="/videos", pattern="%07d.png")
    assert argv[:3] == ["uv", "run", "vaila/extractpng.py"]
    assert "extract" in argv
    assert "-i" in argv
    assert "/videos" in argv
    assert "--pattern" in argv


def test_help_exits_zero() -> None:
    assert ep.main(["--help"]) == 0


def test_get_cuda_status() -> None:
    status = ep.get_cuda_status()
    assert "has_nvidia" in status
    assert "device_name" in status
    assert "ffmpeg_cuda" in status
    assert "ffmpeg_nvenc" in status
    assert status["recommended_hwaccel"] in ("cuda", "auto")


def test_build_extract_png_command_cuda_and_fast_compression() -> None:
    cmd = ep.build_extract_png_command(
        "/data/clip.mp4",
        "/out/%09d.png",
        width=1920,
        height=1080,
        orig_width=1920,
        orig_height=1080,
        hwaccel="cuda",
        compression_level=1,
    )
    assert "-hwaccel" in cmd
    assert cmd[cmd.index("-hwaccel") + 1] == "cuda"
    assert cmd.index("-hwaccel") < cmd.index("-i")
    # Redundant scaling skipped when dimensions match native video
    assert "-vf" not in cmd
    assert "-compression_level" in cmd
    assert cmd[cmd.index("-compression_level") + 1] == "1"
    assert "-pred" in cmd
    assert cmd[cmd.index("-pred") + 1] == "none"


def test_build_extract_png_command_with_rescaling() -> None:
    cmd = ep.build_extract_png_command(
        "/data/clip.mp4",
        "/out/%09d.png",
        width=1280,
        height=720,
        orig_width=1920,
        orig_height=1080,
        hwaccel="cuda",
        compression_level=6,
    )
    assert "-vf" in cmd
    assert "scale=1280:720:flags=lanczos" in cmd[cmd.index("-vf") + 1]
    assert cmd[cmd.index("-compression_level") + 1] == "6"


def test_build_select_frame_command_cuda() -> None:
    cmd = ep.build_select_frame_command("v.mp4", 12, "out/frame_012.png", hwaccel="cuda")
    assert "-hwaccel" in cmd
    assert cmd[cmd.index("-hwaccel") + 1] == "cuda"
    assert cmd.index("-hwaccel") < cmd.index("-i")


def test_build_png_to_video_nvenc() -> None:
    cmd_h264 = ep.build_png_to_video_command(
        "d/%09d.png", "out.mp4", fps=30.0, codec="264", hwaccel="cuda"
    )
    assert "h264_nvenc" in cmd_h264

    cmd_hevc = ep.build_png_to_video_command(
        "d/%09d.png", "out.mp4", fps=30.0, codec="265", hwaccel="cuda"
    )
    assert "hevc_nvenc" in cmd_hevc


def test_build_cli_argv_options() -> None:
    argv = ep.build_cli_argv(
        "extract",
        input_path="/videos",
        output_path="/out",
        hwaccel="cuda",
        compression=1,
        workers=3,
    )
    assert "--hwaccel" in argv
    assert "cuda" in argv
    assert "--workers" in argv
    assert "3" in argv


def test_extractpng_cli_argparser() -> None:
    parser = ep.build_arg_parser()
    args = parser.parse_args(
        ["extract", "-i", "/videos", "--hwaccel", "cuda", "-c", "1", "-j", "4"]
    )
    assert args.command == "extract"
    assert args.hwaccel == "cuda"
    assert args.compression == 1
    assert args.workers == 4


def test_extractpng_gui_builds() -> None:
    import tkinter as tk

    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("No display available for Tkinter GUI test")

    try:
        app = ep.ExtractPngApp(root)
        assert "NVIDIA GPU" in app.root.title() or "Video ↔ PNG" in app.root.title()
        app.root.destroy()
    finally:
        root.destroy()
