"""Tests for numberframes Adjust real FPS (parse + remux)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from vaila import numberframes


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("30", 30.0),
        ("29.97", 29.97),
        ("30000/1001", 30000 / 1001),
        (" 60/1 ", 60.0),
        ("25,0", 25.0),
    ],
)
def test_parse_fps_value_ok(text: str, expected: float) -> None:
    assert numberframes.parse_fps_value(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "  ", "abc", "30/0", "-1", "0", "20000"])
def test_parse_fps_value_invalid(text: str) -> None:
    with pytest.raises(ValueError):
        numberframes.parse_fps_value(text)


def test_fps_rates_match_tolerance() -> None:
    assert numberframes.fps_rates_match(29.97, 29.97003)
    assert numberframes.fps_rates_match(30.0, 30.04)
    assert not numberframes.fps_rates_match(25.0, 30.0)


def test_fps_to_rational() -> None:
    assert numberframes.fps_to_rational(30.0) == (30, 1)
    assert numberframes.fps_to_rational("30000/1001") == (30000, 1001)
    num, den = numberframes.fps_to_rational(30000 / 1001)
    assert num / den == pytest.approx(30000 / 1001)


def test_build_fpsfix_copy_cmd_uses_setts(tmp_path: Path) -> None:
    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    src.write_bytes(b"")  # path only; command builder does not open the file
    cmd = numberframes.build_fpsfix_copy_cmd(
        src, out, 30.0, comment="vaila numberframes fpsfix: test"
    )
    assert "-bsf:v" in cmd
    bsf = cmd[cmd.index("-bsf:v") + 1]
    assert "setts=" in bsf
    assert "-video_track_timescale" in cmd
    assert "-c" in cmd and "copy" in cmd
    assert str(out) == cmd[-1]


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg required",
)
def test_adjust_video_fps_copy_retags_synthetic_mp4(tmp_path: Path) -> None:
    src = tmp_path / "src_60fps.mp4"
    # 30 frames at 60 fps → 0.5 s CFR test pattern
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=60",
            "-frames:v",
            "30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(src),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    meta_in = numberframes.get_precise_video_metadata(src)
    assert meta_in is not None
    assert meta_in["nb_frames"] == 30
    assert numberframes.fps_rates_match(float(meta_in["fps"]), 60.0)

    out = tmp_path / "out_30fps.mp4"
    row = numberframes.adjust_video_fps(src, 30.0, out)
    assert row["ok"] is True, row.get("error")
    assert row["method"] in {"copy", "reencode"}
    assert out.is_file()
    assert numberframes.fps_rates_match(float(row["fps_out"]), 30.0)
    if row["method"] == "copy":
        assert row["frames_out"] == 30


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg required",
)
def test_adjust_videos_fps_batch_writes_summary(tmp_path: Path) -> None:
    for name, rate in (("a.mp4", 60), ("b.mp4", 60)):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=size=160x120:rate={rate}",
                "-frames:v",
                "30",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(tmp_path / name),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    out_dir = numberframes.adjust_videos_fps(
        directory_path=str(tmp_path),
        real_fps=30.0,
        show_gui=False,
    )
    assert out_dir is not None
    assert out_dir.is_dir()
    summary = out_dir / "fpsfix_summary.csv"
    assert summary.is_file()
    text = summary.read_text(encoding="utf-8")
    assert "a.mp4" in text and "b.mp4" in text
    assert (out_dir / "a.mp4").is_file()
    assert (out_dir / "b.mp4").is_file()
