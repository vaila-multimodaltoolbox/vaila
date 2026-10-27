"""Tests for the ball detector dataset, its metrics and the review CSV (no GPU, no weights).

The training call itself needs CUDA and the Ultralytics weights, so it is not run here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vaila import freekiki_ball_train as bt


def _write_review(path: Path, rows: list[str]) -> Path:
    path.write_text("video,frame,status,x1,y1,x2,y2\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_review_csv_reads_ball_and_none_rows(tmp_path: Path) -> None:
    path = _write_review(
        tmp_path / "review.csv",
        ["a.mp4,0,ball,10,20,30,40", "a.mp4,1,none,,,,"],
    )
    rows = bt.read_review_csv(path)
    assert rows[0] == {
        "video": "a.mp4",
        "frame": 0,
        "status": "ball",
        "box": [10.0, 20.0, 30.0, 40.0],
    }
    assert rows[1]["status"] == "none" and rows[1]["box"] is None


def test_review_csv_rejects_unknown_status(tmp_path: Path) -> None:
    path = _write_review(tmp_path / "bad.csv", ["a.mp4,0,maybe,1,2,3,4"])
    with pytest.raises(ValueError):
        bt.read_review_csv(path)


def test_split_is_stable_and_keeps_both_sides() -> None:
    names = [f"match_{i:02d}.mp4" for i in range(20)]
    first = bt.split_by_video(names, val_fraction=0.3, seed=1)
    again = bt.split_by_video(names, val_fraction=0.3, seed=1)
    assert first == again
    assert set(first.values()) == {"train", "val"}


def test_split_needs_two_videos() -> None:
    with pytest.raises(ValueError):
        bt.split_by_video(["only.mp4"])


def test_yolo_line_is_normalised_and_clipped() -> None:
    # Box 10..30 x 20..60 on a 100 x 200 image: centre (0.2, 0.2), size (0.2, 0.2).
    assert bt.yolo_ball_line([10, 20, 30, 60], 100, 200) == (
        "0 0.200000 0.200000 0.200000 0.200000"
    )
    # A box that runs past the border is clipped to the image.
    line = bt.yolo_ball_line([-10, 0, 50, 100], 100, 200)
    cx, cy, w, h = (float(v) for v in line.split()[1:])
    assert (cx, w) == pytest.approx((0.25, 0.5))
    assert (cy, h) == pytest.approx((0.25, 0.5))


def test_degenerate_box_is_rejected() -> None:
    with pytest.raises(ValueError):
        bt.yolo_ball_line([5, 5, 5, 9], 100, 100)


def test_metrics_count_hits_misses_and_false_positives() -> None:
    ref = {
        ("v", 0): (10.0, 10.0),  # hit, 1 px off
        ("v", 1): None,  # no ball, none predicted
        ("v", 2): (40.0, 40.0),  # ball, none predicted
        ("v", 3): None,  # no ball, but predicted
        ("v", 4): (50.0, 50.0),  # hit, exactly 10 px off (the limit)
    }
    pred = {
        ("v", 0): (11.0, 10.0),
        ("v", 3): (5.0, 5.0),
        ("v", 4): (60.0, 50.0),
    }
    got = bt.ball_metrics(pred, ref, match_px=10.0)
    assert (got["tp"], got["fn"], got["fp"], got["tn"]) == (2, 1, 1, 1)
    assert got["recall"] == pytest.approx(2 / 3)
    assert got["precision"] == pytest.approx(2 / 3)
    assert got["median_err_px"] == pytest.approx(5.5)


def test_metrics_with_no_hits_report_nan() -> None:
    got = bt.ball_metrics({}, {("v", 0): (1.0, 1.0)})
    assert got["recall"] == 0.0
    assert np.isnan(got["precision"]) and np.isnan(got["median_err_px"])


def _make_video(path: Path, n_frames: int = 6, size: tuple[int, int] = (64, 48)) -> Path:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), 40 + i, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_build_dataset_writes_images_labels_and_yaml(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    names = [f"match_{i}" for i in range(4)]
    seed = next(
        s
        for s in range(100)
        if set(bt.split_by_video([f"{n}.mp4" for n in names], val_fraction=0.5, seed=s).values())
        == {"train", "val"}
    )
    videos = {}
    labels = []
    for name in names:
        video = _make_video(tmp_path / f"{name}.mp4")
        videos[f"{name}.mp4"] = video
        labels.append(
            {"video": f"{name}.mp4", "frame": 1, "status": "ball", "box": [10, 10, 14, 14]}
        )
        labels.append({"video": f"{name}.mp4", "frame": 3, "status": "none", "box": None})

    out = tmp_path / "ball_ds"
    summary = bt.build_yolo_dataset(labels, videos, out, val_fraction=0.5, seed=seed)

    assert summary["positive"] == 4 and summary["negative"] == 4
    assert summary["train"] + summary["val"] == 8
    assert "names:\n  0: ball" in (out / "data.yaml").read_text(encoding="utf-8")
    images = sorted(out.rglob("*.jpg"))
    labels_txt = sorted(out.rglob("*.txt"))
    assert len(images) == 8 and len(labels_txt) == 8
    positives = [p for p in labels_txt if p.read_text(encoding="utf-8").strip()]
    assert len(positives) == 4
    assert positives[0].read_text(encoding="utf-8").startswith("0 ")


def test_build_dataset_rejects_a_video_without_a_file(tmp_path: Path) -> None:
    labels = [
        {"video": "x.mp4", "frame": 0, "status": "none", "box": None},
        {"video": "y.mp4", "frame": 0, "status": "none", "box": None},
    ]
    video_y = _make_video(tmp_path / "y.mp4")
    with pytest.raises(KeyError):
        bt.build_yolo_dataset(labels, {"y.mp4": video_y}, tmp_path / "ds", val_fraction=0.5)
