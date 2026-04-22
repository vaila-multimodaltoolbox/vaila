"""Tests for vaila_stroboscopic modes (pose/motion/stack).

These tests avoid large media assets by generating a tiny synthetic MP4.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _write_synthetic_video(
    path: Path, *, w: int = 160, h: int = 120, n: int = 60, fps: int = 30
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    assert vw.isOpened()
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x = 10 + i * 2
        y = 40
        cv2.rectangle(frame, (x, y), (x + 20, y + 20), (255, 255, 255), -1)
        vw.write(frame)
    vw.release()


def test_mode_stack_writes_output(tmp_path: Path) -> None:
    from vaila.vaila_stroboscopic import generate_stack_multishot

    vid = tmp_path / "move.mp4"
    _write_synthetic_video(vid)
    out = tmp_path / "out.png"
    ok = generate_stack_multishot(vid, output_path=out, frame_interval=5, stack_op="max")
    assert ok is True
    assert out.is_file()
    img = cv2.imread(str(out))
    assert img is not None
    assert int(img.max()) > 0


def test_mode_motion_writes_output(tmp_path: Path) -> None:
    from vaila.vaila_stroboscopic import generate_motion_stroboscopic

    vid = tmp_path / "move.mp4"
    _write_synthetic_video(vid)
    out = tmp_path / "out.png"
    ok = generate_motion_stroboscopic(
        vid,
        output_path=out,
        threshold=10,
        blend_ratio=1.0,
        blur_size=3,
        open_kernel_size=3,
        frame_interval=1,
        stable_background=True,
        background_samples=8,
    )
    assert ok is True
    assert out.is_file()
    img = cv2.imread(str(out))
    assert img is not None
    assert int(img.max()) > 0


def test_pose_autodetect_csv_fuzzy_match(tmp_path: Path) -> None:
    from vaila.vaila_stroboscopic import generate_stroboscopic_image

    vid = tmp_path / "clip.mp4"
    _write_synthetic_video(vid, n=10)

    # Create a CSV that matches the fuzzy glob but not the explicit candidates.
    csv_path = tmp_path / f"{vid.stem}_something.csv"
    # vailá pixel format: frame,p1_x,p1_y,...,p33_x,p33_y (values can be NaN).
    rows: list[dict[str, float]] = []
    for fi in range(10):
        r: dict[str, float] = {"frame": float(fi)}
        for p in range(1, 34):
            r[f"p{p}_x"] = float("nan")
            r[f"p{p}_y"] = float("nan")
        # Provide a few normalized points so the code detects normalization and draws something.
        r["p12_x"] = 0.45 + 0.01 * fi  # left_shoulder (approx)
        r["p12_y"] = 0.40
        r["p13_x"] = 0.55 + 0.01 * fi  # right_shoulder (approx)
        r["p13_y"] = 0.40
        r["p24_x"] = 0.47 + 0.01 * fi  # left_hip (approx)
        r["p24_y"] = 0.60
        r["p25_x"] = 0.53 + 0.01 * fi  # right_hip (approx)
        r["p25_y"] = 0.60
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    out = tmp_path / "pose.png"
    ok = generate_stroboscopic_image(str(vid), csv_path=None, output_path=out, strobe_interval=2)
    assert ok is True
    assert out.is_file()
