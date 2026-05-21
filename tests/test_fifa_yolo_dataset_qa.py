"""Tests for :mod:`vaila.fifa_yolo_dataset_qa`."""

from __future__ import annotations

from pathlib import Path

import pytest

import vaila.fifa_dataset_builder as fdb
import vaila.fifa_yolo_dataset_qa as qa


def _yolo_pose_line(
    cls: int,
    bbox: tuple[float, float, float, float],
    kps: list[tuple[float, float, int]],
) -> str:
    cx, cy, w, h = bbox
    kp_str = " ".join(f"{x:.6f} {y:.6f} {v}" for x, y, v in kps)
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kp_str}"


def _write_rgb_jpeg(path: Path, w: int = 64, h: int = 48) -> None:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np
    except ModuleNotFoundError:
        pytest.skip("opencv-python required")
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (40, 80, 120)
    cv2.imwrite(str(path), img)


def _make_roboflow_tree(tmp_path: Path) -> Path:
    root = tmp_path / "roboflow_ds"
    root.mkdir(parents=True, exist_ok=True)
    kps: list[tuple[float, float, int]] = [(0.0, 0.0, 0)] * fdb.NUM_KEYPOINTS
    for i in range(8):
        kps[i] = (0.1 + 0.03 * i, 0.2 + 0.02 * i, 2)
    line = _yolo_pose_line(0, (0.5, 0.5, 0.9, 0.85), kps)
    flip_s = ", ".join(str(i) for i in fdb.CANONICAL_FLIP_IDX_32)
    (root / "data.yaml").write_text(
        f"kpt_shape: [{fdb.NUM_KEYPOINTS}, 3]\n"
        f"flip_idx: [{flip_s}]\n"
        "nc: 1\n"
        "names: ['pitch']\n"
        "train: train/images\n"
        "val: valid/images\n",
        encoding="utf-8",
    )
    for split in ("train", "valid", "test"):
        img = root / split / "images" / "frame1.jpg"
        lbl = root / split / "labels" / "frame1.txt"
        _write_rgb_jpeg(img)
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text(line + "\n", encoding="utf-8")
    return root


def test_detect_roboflow_layout(tmp_path: Path):
    root = _make_roboflow_tree(tmp_path)
    layout = qa.detect_roboflow_layout(root)
    assert layout is not None
    assert len(layout.splits) == 3
    names = {s.roboflow_name for s in layout.splits}
    assert names == {"train", "valid", "test"}


def test_audit_schema_ok(tmp_path: Path):
    root = _make_roboflow_tree(tmp_path)
    report = qa.audit_roboflow_dataset(root)
    assert report.schema_ok
    assert report.counts["pairs_ok"] == 3
    assert report.counts["pairs_bad"] == 0


def test_export_images_with_labels(tmp_path: Path):
    root = _make_roboflow_tree(tmp_path)
    stats = qa.export_images_with_labels(root, clean=True)
    assert stats["written"] == 3
    assert (root / "images_with_labels" / "train" / "frame1.jpg").is_file()
    assert (root / "images_with_labels" / "valid" / "frame1.jpg").is_file()


def test_audit_bad_field_count(tmp_path: Path):
    root = _make_roboflow_tree(tmp_path)
    bad = root / "train" / "labels" / "bad.txt"
    bad.write_text("0 0.5 0.5 1 1\n", encoding="utf-8")
    _write_rgb_jpeg(root / "train" / "images" / "bad.jpg")
    report = qa.audit_roboflow_dataset(root)
    assert not report.schema_ok
    assert any(i.stem == "bad" for i in report.issues)
