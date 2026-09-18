"""Tests for vaila.ai_tracker_train and getpixelvideo marker→detect helpers."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vaila.ai_tracker_train import (
    TrainConfig,
    _infer_variant_from_path,
    _parse_yolo_boxes,
    collect_crop_samples,
    run_training,
)
from vaila.getpixelvideo import (
    _ensure_marker_slot_names_list,
    markers_to_detection_bboxes,
    write_ai_tracker_train_manifest,
)


def test_ensure_marker_slot_names_grows_and_keeps_custom() -> None:
    names: list[str] = ["person"]
    coords = {0: [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]}
    _ensure_marker_slot_names_list(names, coords, None, None)
    assert names[0] == "person"
    assert names[1] == "p1"
    assert names[2] == "p2"
    assert len(names) == 3


def test_markers_to_detection_bboxes_uses_slot_labels() -> None:
    coords = {
        0: [(100.0, 200.0), (300.0, 400.0)],
        1: [(110.0, 210.0), None],
    }
    names = ["person", "sports ball"]
    boxes = markers_to_detection_bboxes(coords, None, 1920, 1080, names, box_wh_px=(80, 120))
    assert 0 in boxes and len(boxes[0]) == 2
    assert boxes[0][0]["label"] == "person"
    assert boxes[0][1]["label"] == "sports ball"
    assert boxes[0][0]["w"] == 80
    assert 1 in boxes and len(boxes[1]) == 1


def test_write_ai_tracker_train_manifest(tmp_path: Path) -> None:
    path = write_ai_tracker_train_manifest(
        str(tmp_path),
        task="detect",
        classes=["person", "sports ball"],
        source_video="/tmp/video.mp4",
        marker_slot_names=["person", "sports ball"],
    )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["task"] == "detect"
    assert payload["classes"] == ["person", "sports ball"]


def test_parse_yolo_boxes_detect_line(tmp_path: Path) -> None:
    lab = tmp_path / "a.txt"
    lab.write_text("0 0.5 0.5 0.1 0.2\n", encoding="utf-8")
    boxes = _parse_yolo_boxes(lab, 1000, 1000)
    assert len(boxes) == 1
    cls, x, y, w, h = boxes[0]
    assert cls == 0
    assert w == 100
    assert h == 200


def test_infer_variant_from_path() -> None:
    assert _infer_variant_from_path(Path("resnet50_imagenet.pth")) == "resnet50"
    assert (
        _infer_variant_from_path(Path("mobilenet_v3_small_finetuned.pth")) == "mobilenet_v3_small"
    )


def _make_mini_detect_dataset(root: Path) -> Path:
    """Write a tiny YOLO detect dataset with 2 classes."""
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[40:120, 60:140] = (40, 180, 40)
        img[130:200, 180:260] = (180, 40, 40)
        img_path = root / split / "images" / f"{split}_0.jpg"
        cv2.imwrite(str(img_path), img)
        # two boxes: class 0 and 1
        (root / split / "labels" / f"{split}_0.txt").write_text(
            "0 0.3125 0.3333 0.25 0.3333\n1 0.6875 0.6875 0.25 0.2917\n",
            encoding="utf-8",
        )
    (root / "classes.txt").write_text("person\nsports ball\n", encoding="utf-8")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: .\n"
        f"train: {root / 'train' / 'images'}\n"
        f"val: {root / 'val' / 'images'}\n"
        f"nc: 2\n"
        f"names: ['person', 'sports ball']\n",
        encoding="utf-8",
    )
    write_ai_tracker_train_manifest(str(root), task="detect", classes=["person", "sports ball"])
    return yaml_path


def test_collect_crop_samples(tmp_path: Path) -> None:
    yaml_path = _make_mini_detect_dataset(tmp_path)
    samples, names = collect_crop_samples(yaml_path, max_samples=100)
    assert len(samples) >= 2
    assert "person" in names


def test_discriminator_offline_dry_run(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from vaila.tracking.ai_tracker import _ai_tracker_resnet_local_path, _is_valid_weight_file

    weights = _ai_tracker_resnet_local_path("resnet50")
    if not _is_valid_weight_file(weights):
        pytest.skip("No local resnet50_imagenet.pth under ai_tracker/")
    yaml_path = _make_mini_detect_dataset(tmp_path)
    cfg = TrainConfig(
        data_yaml=yaml_path,
        weights=weights,
        mode="discriminator",
        dry_run=True,
        profile="pytest_tmp",
        max_samples=50,
    )
    results = run_training(cfg)
    assert results["discriminator"] is not None


def test_backbone_dry_run(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from vaila.tracking.ai_tracker import _ai_tracker_resnet_local_path, _is_valid_weight_file

    weights = _ai_tracker_resnet_local_path("resnet50")
    if not _is_valid_weight_file(weights):
        pytest.skip("No local resnet50_imagenet.pth under ai_tracker/")
    yaml_path = _make_mini_detect_dataset(tmp_path)
    cfg = TrainConfig(
        data_yaml=yaml_path,
        weights=weights,
        mode="backbone",
        epochs=1,
        batch=2,
        dry_run=True,
        max_samples=20,
    )
    results = run_training(cfg)
    assert results["backbone"] is not None
