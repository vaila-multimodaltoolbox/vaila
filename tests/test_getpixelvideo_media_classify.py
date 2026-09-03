"""Tests for getpixelvideo auto media-type detection (no video/PNG chooser)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vaila import getpixelvideo as gpv


def test_classify_media_path_video_file(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    path, source, start = gpv.classify_media_path(str(video))
    assert path == str(video.resolve())
    assert source == "video"
    assert start == 0


def test_classify_media_path_lone_png_is_single(tmp_path: Path) -> None:
    png = tmp_path / "solo.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    path, source, start = gpv.classify_media_path(str(png))
    assert path == str(png.resolve())
    assert source == "single_png"
    assert start == 0


def test_classify_media_path_png_among_siblings_is_sequence(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    a = frames / "0001.png"
    b = frames / "0002.png"
    a.write_bytes(b"\x89PNG\r\n\x1a\n")
    b.write_bytes(b"\x89PNG\r\n\x1a\n")
    path, source, start = gpv.classify_media_path(str(a))
    assert path == str(frames.resolve())
    assert source == "png_sequence"
    assert start == 0


def test_classify_media_path_selected_png_keeps_start_index(tmp_path: Path) -> None:
    """Selecting a non-first PNG in a folder must open that frame, not alpha-first."""
    frames = tmp_path / "kikiv1"
    frames.mkdir()
    first = frames / "a_frame.png"
    second = frames / "z_frame.png"
    first.write_bytes(b"\x89PNG\r\n\x1a\n")
    second.write_bytes(b"\x89PNG\r\n\x1a\n")
    path, source, start = gpv.classify_media_path(str(second))
    assert path == str(frames.resolve())
    assert source == "png_sequence"
    assert start == 1


def test_png_sequence_frame_source_honours_start_index(tmp_path: Path) -> None:
    frames = tmp_path / "seq"
    frames.mkdir()
    # Minimal valid 1x1 PNGs via OpenCV so read() succeeds.
    import cv2

    for name in ("000.png", "001.png", "002.png"):
        cv2.imwrite(str(frames / name), np.zeros((2, 2, 3), dtype=np.uint8))
    src = gpv.PngSequenceFrameSource(str(frames), start_index=2)
    assert src.get(cv2.CAP_PROP_POS_FRAMES) == 2
    ok, _frame = src.read()
    assert ok
    assert src.get(cv2.CAP_PROP_POS_FRAMES) == 3


def test_classify_media_path_directory_is_sequence(tmp_path: Path) -> None:
    frames = tmp_path / "seq"
    frames.mkdir()
    (frames / "f0.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    path, source, start = gpv.classify_media_path(str(frames))
    assert path == str(frames.resolve())
    assert source == "png_sequence"
    assert start == 0


def test_classify_media_path_nested_png_dir(tmp_path: Path) -> None:
    root = tmp_path / "run"
    nested = root / "png"
    nested.mkdir(parents=True)
    (nested / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    path, source, start = gpv.classify_media_path(str(root))
    assert path == str(nested.resolve())
    assert source == "png_sequence"
    assert start == 0


def test_classify_media_path_missing_returns_none(tmp_path: Path) -> None:
    path, source, start = gpv.classify_media_path(str(tmp_path / "missing.mp4"))
    assert path is None
    assert source is None
    assert start == 0
