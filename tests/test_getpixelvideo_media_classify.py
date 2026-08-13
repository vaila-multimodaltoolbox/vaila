"""Tests for getpixelvideo auto media-type detection (no video/PNG chooser)."""

from __future__ import annotations

from pathlib import Path

from vaila import getpixelvideo as gpv


def test_classify_media_path_video_file(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    path, source = gpv.classify_media_path(str(video))
    assert path == str(video.resolve())
    assert source == "video"


def test_classify_media_path_lone_png_is_single(tmp_path: Path) -> None:
    png = tmp_path / "solo.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    path, source = gpv.classify_media_path(str(png))
    assert path == str(png.resolve())
    assert source == "single_png"


def test_classify_media_path_png_among_siblings_is_sequence(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    a = frames / "0001.png"
    b = frames / "0002.png"
    a.write_bytes(b"\x89PNG\r\n\x1a\n")
    b.write_bytes(b"\x89PNG\r\n\x1a\n")
    path, source = gpv.classify_media_path(str(a))
    assert path == str(frames.resolve())
    assert source == "png_sequence"


def test_classify_media_path_directory_is_sequence(tmp_path: Path) -> None:
    frames = tmp_path / "seq"
    frames.mkdir()
    (frames / "f0.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    path, source = gpv.classify_media_path(str(frames))
    assert path == str(frames.resolve())
    assert source == "png_sequence"


def test_classify_media_path_nested_png_dir(tmp_path: Path) -> None:
    root = tmp_path / "run"
    nested = root / "png"
    nested.mkdir(parents=True)
    (nested / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    path, source = gpv.classify_media_path(str(root))
    assert path == str(nested.resolve())
    assert source == "png_sequence"


def test_classify_media_path_missing_returns_none(tmp_path: Path) -> None:
    path, source = gpv.classify_media_path(str(tmp_path / "missing.mp4"))
    assert path is None
    assert source is None
