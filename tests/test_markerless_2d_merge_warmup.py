"""Unit tests for merge-style FFmpeg warmup helpers (markerless_2d_analysis)."""

from __future__ import annotations

from vaila.markerless_2d_analysis import build_merge_warmup_ffmpeg_argv, merge_warmup_video_filter


def test_merge_warmup_video_filter_full_reverse_or_unknown_duration() -> None:
    assert merge_warmup_video_filter(36.08, 100) == "reverse"
    assert merge_warmup_video_filter(None, 50) == "reverse"


def test_merge_warmup_video_filter_partial_trim_reverse() -> None:
    vf = merge_warmup_video_filter(36.08, 50)
    assert vf == "trim=start=0:end=18.040000,reverse,setpts=PTS-STARTPTS"


def test_build_merge_warmup_ffmpeg_argv() -> None:
    argv = build_merge_warmup_ffmpeg_argv("/tmp/in.mp4", "reverse")
    assert argv[:4] == ["ffmpeg", "-y", "-i", "/tmp/in.mp4"]
    assert argv[-5:] == ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    assert argv[argv.index("-vf") + 1] == "reverse"
