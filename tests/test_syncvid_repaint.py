"""The syncvid window must show what it draws after a file dialog recreates it.

Runs only on an X display with ``xwd`` and ImageMagick ``convert``. The window content is captured
from the X server and compared with the surface pygame drew: a blank window shows whatever is behind
it, so the pixels would not match.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from vaila import syncvid

pytestmark = pytest.mark.skipif(
    not os.environ.get("DISPLAY") or shutil.which("xwd") is None or shutil.which("convert") is None,
    reason="needs an X display plus xwd and ImageMagick convert",
)


def _write_clip(path: Path, colour: tuple[int, int, int]) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
    for _ in range(10):
        writer.write(np.full((180, 320, 3), colour, np.uint8))
    writer.release()


def _capture_window(window_id: int, out: Path) -> np.ndarray:
    from PIL import Image

    xwd = out.with_suffix(".xwd")
    subprocess.run(["xwd", "-silent", "-id", str(window_id), "-out", str(xwd)], check=True)
    subprocess.run(["convert", str(xwd), str(out)], check=True)
    return np.asarray(Image.open(out).convert("RGB")).astype(int)


def test_window_shows_the_drawn_frame_after_loading_a_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pygame

    _write_clip(tmp_path / "cam_a.mp4", (200, 40, 40))
    _write_clip(tmp_path / "cam_b.mp4", (40, 200, 40))
    infos = syncvid.probe_video_directory(tmp_path)
    player = syncvid.PygameSyncPlayer(infos)

    def frames(count: int) -> None:
        for _ in range(count):
            for event in pygame.event.get():
                player._handle_event(event)
            player._draw()
            pygame.time.wait(20)

    try:
        frames(10)
        checkpoint = tmp_path / "resume.json"
        syncvid.write_checkpoint_file(player._checkpoint_state(completed=False), checkpoint)
        monkeypatch.setattr(syncvid, "_portable_path_dialog", lambda *a, **k: str(checkpoint))
        player._load_checkpoint()
        frames(30)

        window_id = int(pygame.display.get_wm_info()["window"])
        captured = _capture_window(window_id, tmp_path / "window.png")
        drawn = np.transpose(pygame.surfarray.array3d(player.screen), (1, 0, 2)).astype(int)
        assert captured.shape[:2] == drawn.shape[:2], "the window has the wrong size"

        # the same pixels up to the compression of the capture
        matches = np.all(np.abs(captured - drawn) <= 24, axis=2).mean()
        assert matches > 0.95, f"only {matches:.0%} of the window matches the drawn frame"
    finally:
        player._release_capture()
        pygame.display.quit()
        pygame.font.quit()
