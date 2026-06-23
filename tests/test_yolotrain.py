"""Tests for vaila.yolotrain helpers (no Tk root required).

Focused on the three blocker fixes from the multi-model review of PR #550:

1. Sequential frame extraction with no train/val leakage.
2. Split sizing always assigns >=1 to train and val, with no train row
   silently duplicated into val.
3. ``_resolve_yaml_path`` resolves ``../`` relatives, list-valued entries,
   and ``./`` prefixes correctly (the old ``lstrip("./")`` was wrong).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

# Heavy GUI/torch import: skip the whole module if env can't load it.
yolotrain = pytest.importorskip("vaila.yolotrain")
_resolve_yaml_path = yolotrain.YOLOTrainApp._resolve_yaml_path
_point_to_yolo_label = yolotrain.YOLOTrainApp._point_to_yolo_label
_resolve_training_config = yolotrain._resolve_training_config


# ---------------------------------------------------------------------------
# _resolve_yaml_path
# ---------------------------------------------------------------------------


def test_resolve_yaml_path_handles_dot_slash_prefix(tmp_path: Path) -> None:
    yaml_data = {"path": str(tmp_path), "train": "./train/images"}
    out = _resolve_yaml_path(yaml_data, str(tmp_path), "train")
    assert out == os.path.normpath(str(tmp_path / "train" / "images"))


def test_resolve_yaml_path_preserves_parent_relative(tmp_path: Path) -> None:
    """Old lstrip("./") corrupted ``../images/train`` -> ``images/train``."""
    base = tmp_path / "datasets" / "soccer"
    yaml_data = {"path": str(base), "train": "../images/train"}
    out = _resolve_yaml_path(yaml_data, str(base), "train")
    assert out == os.path.normpath(str(tmp_path / "datasets" / "images" / "train"))


def test_resolve_yaml_path_preserves_dot_prefixed_dirname(tmp_path: Path) -> None:
    """Old lstrip("./") also corrupted ``.datasets/x`` -> ``datasets/x``."""
    yaml_data = {"path": str(tmp_path), "train": ".datasets/train"}
    out = _resolve_yaml_path(yaml_data, str(tmp_path), "train")
    assert out == os.path.normpath(str(tmp_path / ".datasets" / "train"))


def test_resolve_yaml_path_returns_absolute_unchanged(tmp_path: Path) -> None:
    abs_target = str((tmp_path / "abs" / "train").resolve())
    yaml_data = {"path": "/ignored", "train": abs_target}
    assert _resolve_yaml_path(yaml_data, str(tmp_path), "train") == abs_target


def test_resolve_yaml_path_accepts_list_train_entry(tmp_path: Path) -> None:
    """Ultralytics allows train: [dir1, dir2]; the resolver picks the first."""
    yaml_data = {
        "path": str(tmp_path),
        "train": ["./train/images", "./train2/images"],
    }
    out = _resolve_yaml_path(yaml_data, str(tmp_path), "train")
    assert out == os.path.normpath(str(tmp_path / "train" / "images"))


def test_resolve_yaml_path_empty_returns_empty(tmp_path: Path) -> None:
    assert _resolve_yaml_path({"train": ""}, str(tmp_path), "train") == ""


# ---------------------------------------------------------------------------
# Task/model auto-detection for getpixelvideo/SAM datasets
# ---------------------------------------------------------------------------


def test_training_config_pose_yaml_uses_yolo26_pose_and_yaml_relative_path(tmp_path: Path) -> None:
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "val" / "images").mkdir(parents=True)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        "path: .\ntrain: train/images\nval: val/images\nnames: [field]\nkpt_shape: [32, 3]\n",
        encoding="utf-8",
    )

    cfg = _resolve_training_config(str(yaml_path), task="auto", model="auto", device="cpu")

    assert cfg.dataset_dir == str(tmp_path.resolve())
    assert cfg.task == "pose"
    assert cfg.model_name == "yolo26m-pose.pt"
    assert cfg.train_path == str(tmp_path / "train" / "images")
    assert cfg.val_path == str(tmp_path / "val" / "images")


def test_training_config_fifa_pose_layout_uses_yolo26_pose(tmp_path: Path) -> None:
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "val").mkdir(parents=True)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        f"path: {tmp_path}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: football_pitch\n"
        "kpt_shape: [32, 3]\n",
        encoding="utf-8",
    )

    cfg = _resolve_training_config(str(yaml_path), task="auto", model="auto", device="cpu")

    assert cfg.task == "pose"
    assert cfg.model_name == "yolo26m-pose.pt"
    assert cfg.train_path == str(tmp_path / "images" / "train")
    assert cfg.val_path == str(tmp_path / "images" / "val")


def test_training_config_pose_yaml_rejects_detect_model(tmp_path: Path) -> None:
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "val" / "images").mkdir(parents=True)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        "path: .\ntrain: train/images\nval: val/images\nnames: [object]\nkpt_shape: [62, 3]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Model/task mismatch.*yolo26x-pose"):
        _resolve_training_config(str(yaml_path), task="auto", model="yolo26x.pt", device="cpu")


def test_training_config_segment_label_uses_yolo26_seg(tmp_path: Path) -> None:
    (tmp_path / "train" / "images").mkdir(parents=True)
    labels = tmp_path / "train" / "labels"
    labels.mkdir(parents=True)
    (tmp_path / "val" / "images").mkdir(parents=True)
    (labels / "frame_000001.txt").write_text(
        "0 0.10 0.10 0.20 0.10 0.20 0.20 0.10 0.20\n",
        encoding="utf-8",
    )
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        "path: .\ntrain: train/images\nval: val/images\nnames: [object]\n",
        encoding="utf-8",
    )

    cfg = _resolve_training_config(str(yaml_path), task="auto", model="auto", device="cpu")

    assert cfg.task == "segment"
    assert cfg.model_name == "yolo26m-seg.pt"


# ---------------------------------------------------------------------------
# _point_to_yolo_label (smoke checks on the existing math)
# ---------------------------------------------------------------------------


def test_point_to_yolo_label_centered_box_is_normalized() -> None:
    line = _point_to_yolo_label(100.0, 200.0, 50.0, 400, 400)
    cls, cx, cy, bw, bh = line.split()
    assert cls == "0"
    assert pytest.approx(float(cx), abs=1e-6) == 100.0 / 400.0
    assert pytest.approx(float(cy), abs=1e-6) == 200.0 / 400.0
    assert pytest.approx(float(bw), abs=1e-6) == 50.0 / 400.0
    assert pytest.approx(float(bh), abs=1e-6) == 50.0 / 400.0


# ---------------------------------------------------------------------------
# Dataset build: split has no train/val leakage and stays responsive
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_csv_and_video(tmp_path: Path) -> tuple[Path, Path, int]:
    """Generate a tiny synthetic video + matching getpixelvideo-style CSV."""
    cv2 = pytest.importorskip("cv2")
    pd = pytest.importorskip("pandas")
    import numpy as np

    n_frames = 20
    width, height = 64, 48
    video = tmp_path / "synth.mp4"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (width, height),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4v on this environment")
    for i in range(n_frames):
        frame = np.full((height, width, 3), i * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    csv = tmp_path / "points.csv"
    rows = [
        {"frame": i, "p1_x": 10.0 + i, "p1_y": 12.0 + i, "p2_x": 30.0, "p2_y": 20.0}
        for i in range(n_frames)
    ]
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv, video, n_frames


def _disjoint_image_basenames(out_dir: Path) -> dict[str, set[str]]:
    return {
        split: {p.name for p in (out_dir / split / "images").glob("*.jpg")}
        for split in ("train", "val", "test")
        if (out_dir / split / "images").exists()
    }


def test_build_dataset_no_train_val_leakage(
    tmp_path: Path, sample_csv_and_video: tuple[Path, Path, int]
) -> None:
    pytest.importorskip("cv2")
    csv, video, n_frames = sample_csv_and_video
    random.seed(0)
    app = yolotrain.YOLOTrainApp.__new__(yolotrain.YOLOTrainApp)
    out_root = tmp_path / "out"
    out_root.mkdir()

    out_dir, _yaml, _msg = app._build_tracking_dataset_from_pixel_csv(
        csv_path=str(csv),
        video_path=str(video),
        output_parent=str(out_root),
        class_name="athlete",
        box_size=20.0,
    )
    out = Path(out_dir)
    images = _disjoint_image_basenames(out)
    train = images.get("train", set())
    val = images.get("val", set())
    test = images.get("test", set())

    # No frame appears in more than one split.
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    # Train and val both have at least one image (no leakage fallback was used).
    assert len(train) >= 1
    assert len(val) >= 1
    # Total written equals total frames with visible points.
    assert len(train) + len(val) + len(test) == n_frames
    # data.yaml exists and references all three split image dirs.
    yaml_text = (out / "data.yaml").read_text(encoding="utf-8")
    assert "train:" in yaml_text and "val:" in yaml_text and "test:" in yaml_text


def test_build_dataset_refuses_single_frame(tmp_path: Path) -> None:
    """One-frame CSV must error out, not duplicate the single row into val."""
    cv2 = pytest.importorskip("cv2")
    pd = pytest.importorskip("pandas")
    import numpy as np

    width, height = 64, 48
    video = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4v on this environment")
    writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()

    csv = tmp_path / "one.csv"
    pd.DataFrame([{"frame": 0, "p1_x": 10.0, "p1_y": 10.0}]).to_csv(csv, index=False)

    app = yolotrain.YOLOTrainApp.__new__(yolotrain.YOLOTrainApp)
    out_root = tmp_path / "out"
    out_root.mkdir()
    with pytest.raises(ValueError, match="at least 2"):
        app._build_tracking_dataset_from_pixel_csv(
            csv_path=str(csv),
            video_path=str(video),
            output_parent=str(out_root),
            class_name="athlete",
            box_size=20.0,
        )
