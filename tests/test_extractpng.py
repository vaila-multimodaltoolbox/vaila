"""Tests for vaila.extractpng.

Update Date: 02 October 2026
Version: 0.4.7
"""

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


def test_list_pngs_sorted_natural_order(tmp_path: Path) -> None:
    for name in ("b2.png", "b10.png", "a.png"):
        (tmp_path / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "note.txt").write_text("skip")
    assert [p.name for p in ep.list_pngs_sorted(tmp_path)] == ["a.png", "b2.png", "b10.png"]


def test_pattern_covers_contiguous_sequence(tmp_path: Path) -> None:
    for i in range(3):
        (tmp_path / f"{i:09d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    assert ep._pattern_covers_pngs(tmp_path, "%09d.png") is True


def test_pattern_rejects_arbitrary_names_and_gaps(tmp_path: Path) -> None:
    loose = tmp_path / "loose"
    loose.mkdir()
    (loose / "frame.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (loose / "img_2.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    assert ep._pattern_covers_pngs(loose, "%09d.png") is False

    gap = tmp_path / "gap"
    gap.mkdir()
    (gap / "000000000.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (gap / "000000002.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    assert ep._pattern_covers_pngs(gap, "%09d.png") is False


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


def test_sample_random_frame_indices_spacing() -> None:
    total_frames = 5000
    num_frames = 20
    indices = ep.sample_random_frame_indices(total_frames, num_frames=num_frames, seed=42)

    assert len(indices) == num_frames
    assert sorted(indices) == indices
    assert len(set(indices)) == num_frames

    # Verify spacing: consecutive frames must be well-spaced
    diffs = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
    min_spacing = min(diffs)
    # L = 5000 / 20 = 250, margin = 250 * 0.15 = 37, minimum distance >= 74
    assert min_spacing >= 50
    assert indices[0] >= 0
    assert indices[-1] < total_frames


def test_sample_random_frame_indices_small_video() -> None:
    # When total frames <= requested frames, return all frames
    assert ep.sample_random_frame_indices(10, num_frames=20) == list(range(10))
    assert ep.sample_random_frame_indices(0, num_frames=20) == []
    assert ep.sample_random_frame_indices(10, num_frames=0) == []


def test_sample_random_frame_indices_seed() -> None:
    f1 = ep.sample_random_frame_indices(20000, num_frames=20, seed=12345)
    f2 = ep.sample_random_frame_indices(20000, num_frames=20, seed=12345)
    f3 = ep.sample_random_frame_indices(20000, num_frames=20, seed=99999)
    assert f1 == f2
    assert f1 != f3


def test_list_videos_in_dir_recursive(tmp_path: Path) -> None:
    sub1 = tmp_path / "sub1"
    sub2 = tmp_path / "sub2" / "nested"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)

    (tmp_path / "root.mp4").write_bytes(b"x")
    (sub1 / "clip1.avi").write_bytes(b"x")
    (sub2 / "clip2.MP4").write_bytes(b"x")
    (sub1 / "text.txt").write_text("skip")

    non_rec = ep.list_videos_in_dir(tmp_path, recursive=False)
    assert [p.name for p in non_rec] == ["root.mp4"]

    rec = ep.list_videos_in_dir(tmp_path, recursive=True)
    assert sorted(p.name for p in rec) == ["clip1.avi", "clip2.MP4", "root.mp4"]


def test_build_fast_frame_command() -> None:
    cmd = ep.build_fast_frame_command("video.mp4", 12.3456, "out/frame_100.png", hwaccel="cuda")
    assert cmd[0] == "ffmpeg"
    assert "-ss" in cmd
    assert cmd[cmd.index("-ss") + 1] == "12.3456"
    assert cmd.index("-ss") < cmd.index("-i")
    assert "-hwaccel" in cmd
    assert cmd[cmd.index("-hwaccel") + 1] == "cuda"
    assert cmd[-1] == "out/frame_100.png"


def test_sample_cli_argparser() -> None:
    parser = ep.build_arg_parser()
    args = parser.parse_args(
        ["sample", "-i", "/videos", "-n", "25", "--no-recursive", "--flat", "--seed", "77"]
    )
    assert args.command == "sample"
    assert args.num_frames == 25
    assert args.recursive is False
    assert args.flat is True
    assert args.seed == 77


def test_build_cli_argv_sample() -> None:
    argv = ep.build_cli_argv(
        "sample",
        input_path="/videos",
        output_path="/out",
        num_frames=20,
        recursive=True,
        flat=False,
    )
    assert "sample" in argv
    assert "-i" in argv
    assert "/videos" in argv
    assert "-o" in argv
    assert "/out" in argv


@pytest.mark.parametrize("quality,gop", [(0, 1), (52, 1), (18, 0), (18, 1.5)])
def test_invalid_creation_options(quality, gop):
    with pytest.raises(ValueError):
        ep.build_png_to_video_command("%09d.png", "out.mp4", fps=30, quality=quality, gop=gop)


def test_creation_cli_controls():
    argv = ep.build_cli_argv("create", input_path="images", quality=12, gop=3)
    args = ep.build_arg_parser().parse_args(argv[3:])
    assert (args.quality, args.gop, args.hwaccel) == (12, 3, "auto")


def test_nvenc_fallback_preserves_family_and_options(tmp_path, monkeypatch):
    import subprocess

    from PIL import Image

    source = tmp_path / "images"
    source.mkdir()
    for name in ("frame 10.png", "frame 2.png"):
        Image.new("RGB", (64, 48)).save(source / name)
    monkeypatch.setattr(ep, "get_cuda_status", lambda: {"device_name": "test"})
    monkeypatch.setattr(
        ep.os, "link", lambda *args: (_ for _ in ()).throw(OSError("copy fallback"))
    )
    calls = []

    def run(cmd):
        calls.append(cmd)
        pattern = Path(cmd[cmd.index("-i") + 1])
        assert len(list(pattern.parent.glob("*.png"))) == 2
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(ep, "_run_ffmpeg", run)
    dest = ep.create_video_from_png(source, codec="265_nvenc", quality=13, gop=2, hwaccel="cuda")
    assert "hevc_nvenc" in calls[0] and "libx265" in calls[1]
    assert calls[0][calls[0].index("-cq") + 1] == "13"
    assert calls[1][calls[1].index("-crf") + 1] == "13"
    assert all(cmd[cmd.index("-g") + 1] == "2" for cmd in calls)
    assert not list(dest.glob("vaila_png_sequence_*"))


def test_mixed_dimensions_rejected(tmp_path):
    from PIL import Image

    Image.new("RGB", (64, 48)).save(tmp_path / "1.png")
    Image.new("RGB", (66, 48)).save(tmp_path / "2.png")
    with pytest.raises(ValueError, match="dimensions differ"):
        ep.create_video_from_png(tmp_path, hwaccel="cpu")


@pytest.mark.parametrize(
    "names",
    [
        ["frame 10.png", "frame 2.png", "frame 1.png"],
        ["000000000.png", "000000002.png"],
        ["single.png"],
    ],
)
@pytest.mark.parametrize(
    "codec,hwaccel", [("264", "cpu"), ("265", "cpu"), ("264", "cuda"), ("265", "cuda")]
)
def test_encoded_sequence_exact_frames(tmp_path, names, codec, hwaccel):
    import json
    import shutil
    import subprocess

    import numpy as np
    from PIL import Image

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg required")
    if hwaccel == "cuda" and not ep.get_cuda_status()["has_nvidia"]:
        pytest.skip("NVIDIA required")
    source = tmp_path / "input"
    source.mkdir()
    ordered = sorted(names, key=ep._natural_sort_key)
    for index, name in enumerate(ordered):
        Image.new("RGB", (64, 48), (40 + index * 70,) * 3).save(source / name)
    dest = ep.create_video_from_png(source, fps=25, codec=codec, hwaccel=hwaccel)
    video = dest / "input.mp4"
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_streams",
                "-of",
                "json",
                str(video),
            ]
        )
    )
    assert len(probe["frames"]) == len(names)
    assert all(frame["key_frame"] == 1 for frame in probe["frames"])
    stream = probe["streams"][0]
    assert (stream["width"], stream["height"], stream["avg_frame_rate"]) == (64, 48, "25/1")
    decoded = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    )
    frames = np.frombuffer(decoded, dtype=np.uint8).reshape(len(names), 48, 64, 3)
    for index, frame in enumerate(frames):
        assert np.abs(frame.astype(float) - (40 + index * 70)).mean() < 4
