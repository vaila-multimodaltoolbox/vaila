"""Tests for GetPixelVideo ↔ Scout clock sync file."""

from __future__ import annotations

import json
from pathlib import Path

from vaila.scout_getpixel_sync import read_sync_state, sync_path, write_sync_state


def test_write_read_roundtrip(tmp_path: Path, monkeypatch):
    path = tmp_path / "scout_getpixel_sync.json"
    monkeypatch.setenv("VAILA_SCOUT_GETPIXEL_SYNC", str(path))
    assert sync_path() == path
    write_sync_state(
        time_s=12.5,
        frame=300,
        fps=24.0,
        paused=False,
        video_path="/tmp/demo.mp4",
    )
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["time_s"] == 12.5
    assert data["frame"] == 300
    assert data["playing"] is True
    assert data["paused"] is False
    state = read_sync_state()
    assert state is not None
    assert state["time_s"] == 12.5
    assert state["video_path"].endswith("demo.mp4")


def test_read_missing_returns_none(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VAILA_SCOUT_GETPIXEL_SYNC", str(tmp_path / "missing.json"))
    assert read_sync_state() is None
