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
_find_dataset_yaml_candidates = yolotrain._find_dataset_yaml_candidates
_format_trainer_metrics = yolotrain._format_trainer_metrics
_attach_yolo_progress_callbacks = yolotrain._attach_yolo_progress_callbacks
_print_class_name_hints = yolotrain._print_class_name_hints
_is_custom_weights_path = yolotrain._is_custom_weights_path
_model_path_for_training = yolotrain._model_path_for_training
_model_scale = yolotrain._model_scale
_raise_for_suspicious_pose_dataset = yolotrain._raise_for_suspicious_pose_dataset
_task_guide_text = yolotrain._task_guide_text
_training_help_uri = yolotrain._training_help_uri
_training_quick_guide_text = yolotrain._training_quick_guide_text
_format_training_cli_command = yolotrain._format_training_cli_command
_inspect_label_schema = yolotrain._inspect_label_schema


# ---------------------------------------------------------------------------
# _find_dataset_yaml_candidates (GUI folder auto-discovery)
# ---------------------------------------------------------------------------


def test_find_dataset_yaml_in_dataset_root(tmp_path: Path) -> None:
    dataset = tmp_path / "vaila_dataset_20260101_120000"
    dataset.mkdir()
    yaml_path = dataset / "data.yaml"
    yaml_path.write_text("path: .\ntrain: train/images\nval: val/images\n", encoding="utf-8")

    found = _find_dataset_yaml_candidates(dataset)
    assert found == [str(yaml_path.resolve())]


def test_find_dataset_yaml_from_parent_folder(tmp_path: Path) -> None:
    older = tmp_path / "vaila_dataset_old"
    newer = tmp_path / "vaila_dataset_new"
    older.mkdir()
    newer.mkdir()
    older_yaml = older / "data.yaml"
    newer_yaml = newer / "data.yaml"
    older_yaml.write_text("path: .\ntrain: train/images\nval: val/images\n", encoding="utf-8")
    newer_yaml.write_text("path: .\ntrain: train/images\nval: val/images\n", encoding="utf-8")
    newer_yaml.touch()

    found = _find_dataset_yaml_candidates(tmp_path)
    assert found[0] == str(newer_yaml.resolve())
    assert str(older_yaml.resolve()) in found


def test_find_dataset_yaml_skips_train_val_subfolders(tmp_path: Path) -> None:
    dataset = tmp_path / "my_dataset"
    (dataset / "train" / "images").mkdir(parents=True)
    root_yaml = dataset / "data.yaml"
    root_yaml.write_text("path: .\ntrain: train/images\nval: val/images\n", encoding="utf-8")
    # A stray yaml under train/ must not be picked up (depth-limited walk skips train/).
    (dataset / "train" / "data.yaml").write_text("path: .\n", encoding="utf-8")

    found = _find_dataset_yaml_candidates(dataset)
    assert found == [str(root_yaml.resolve())]


def test_find_dataset_yaml_accepts_yaml_file_path(tmp_path: Path) -> None:
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text("path: .\n", encoding="utf-8")
    found = _find_dataset_yaml_candidates(yaml_path)
    assert found == [str(yaml_path.resolve())]


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


def test_pose_looks_like_misexported_detection_flags_single_class_many_kpts() -> None:
    # SAM3-tracks-collapsed-to-keypoints mistake: nc=1 + many keypoints.
    suspect, nkp = yolotrain.pose_looks_like_misexported_detection(
        {"names": ["object"], "kpt_shape": [62, 3]}
    )
    assert suspect is True
    assert nkp == 62


def test_pose_looks_like_misexported_detection_ok_for_real_pose() -> None:
    # A genuine multi-class or few-keypoint pose set should NOT be flagged.
    suspect_field, _ = yolotrain.pose_looks_like_misexported_detection(
        {"names": ["field"], "kpt_shape": [4, 3]}
    )
    assert suspect_field is False
    # No kpt_shape at all -> detection dataset, never flagged.
    suspect_detect, nkp = yolotrain.pose_looks_like_misexported_detection({"names": ["person"]})
    assert suspect_detect is False
    assert nkp == 0


def test_pose_looks_like_misexported_detection_allows_named_body_keypoints() -> None:
    suspect, nkp = yolotrain.pose_looks_like_misexported_detection(
        {
            "names": ["person"],
            "kpt_shape": [17, 3],
            "kpt_names": [f"joint_{idx}" for idx in range(17)],
        }
    )
    assert suspect is False
    assert nkp == 17


def test_pose_looks_like_misexported_detection_flags_generic_keypoint_slots() -> None:
    suspect, nkp = yolotrain.pose_looks_like_misexported_detection(
        {
            "names": ["person"],
            "kpt_shape": [32, 3],
            "kpt_names": [f"kp_{idx}" for idx in range(1, 33)],
        }
    )
    assert suspect is True
    assert nkp == 32


def test_suspicious_pose_requires_explicit_override() -> None:
    yaml_data = {"names": ["person"], "kpt_shape": [62, 3]}
    with pytest.raises(ValueError, match="Training blocked.*individual player"):
        _raise_for_suspicious_pose_dataset(yaml_data)
    _raise_for_suspicious_pose_dataset(yaml_data, allow=True)


def test_task_guide_explains_tracking_task_distinction() -> None:
    assert "one box per player" in _task_guide_text("detect")
    assert "not a tracking algorithm" in _task_guide_text("auto")
    warning = _task_guide_text("pose", {"names": ["person"], "kpt_shape": [62, 3]})
    assert warning.startswith("STOP:")


def test_training_help_uri_points_to_local_html() -> None:
    uri = _training_help_uri("model-guide")
    assert uri.startswith("file:")
    assert uri.endswith("yolotrain.html#model-guide")


def test_model_scale_ignores_pose_and_segment_suffix_text() -> None:
    assert _model_scale("yolo26m-pose.pt") == "m"
    assert _model_scale("yolo26s-seg.pt") == "s"
    assert _model_scale("yolov9e.pt") == "e"


def test_model_characteristics_reports_pose_m_as_general_purpose(capsys) -> None:
    app = yolotrain.YOLOTrainApp.__new__(yolotrain.YOLOTrainApp)
    app._show_model_characteristics("yolo26m-pose.pt")
    output = capsys.readouterr().out
    assert "General purpose (recommended)" in output
    assert "Mobile applications" not in output


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


def test_training_config_rejects_non_positive_epochs(tmp_path: Path) -> None:
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "val" / "images").mkdir(parents=True)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        "path: .\ntrain: train/images\nval: val/images\nnames: [person]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Epochs must be at least 1"):
        _resolve_training_config(str(yaml_path), epochs=0, device="cpu")


def test_cli_help_contains_didactic_task_and_epochs_guide() -> None:
    help_text = yolotrain.build_arg_parser().format_help()
    assert "detect  One box per player" in help_text
    assert "Maximum training epochs" in help_text
    assert "--allow-suspicious-pose" in help_text
    assert "--print-guide" in help_text


def test_quick_guide_has_correct_sam_build_and_train_commands() -> None:
    guide = _training_quick_guide_text()
    assert "vaila.sam_to_yolo build" in guide
    assert "--class-name person" in guide
    assert "kpt_shape: [62, 3] still means POSE" in guide


def test_format_training_cli_command_matches_resolved_config(tmp_path: Path) -> None:
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "val" / "images").mkdir(parents=True)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        "path: .\ntrain: train/images\nval: val/images\nnames: [person]\n",
        encoding="utf-8",
    )
    config = _resolve_training_config(
        str(yaml_path), task="auto", model="auto", epochs=20, device="cpu"
    )
    command = _format_training_cli_command(config, dry_run=True)
    assert "vaila.yolotrain" in command
    assert "--task detect" in command
    assert "--model yolo26m.pt" in command
    assert "--epochs 20" in command
    assert command.endswith("--dry-run")


def test_inspect_label_schema_distinguishes_detect_and_pose(tmp_path: Path) -> None:
    images = tmp_path / "train" / "images"
    labels = tmp_path / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (labels / "frame_000001.txt").write_text(
        "0 0.5 0.5 0.2 0.3\n0 0.2 0.4 0.1 0.2\n", encoding="utf-8"
    )
    detect = _inspect_label_schema(str(images), "detect", {"names": ["person"]})
    assert detect["rows_max"] == 2
    assert detect["token_counts"] == {5: 2}
    assert detect["invalid_rows"] == 0

    pose = _inspect_label_schema(
        str(images), "pose", {"names": ["person"], "kpt_shape": [62, 3]}
    )
    assert pose["invalid_rows"] == 2


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


# ---------------------------------------------------------------------------
# Training progress helpers (v0.3.57)
# ---------------------------------------------------------------------------


class _FakeTrainer:
    def __init__(self) -> None:
        self.epoch = 4
        self.epochs = 100
        self.device = "cuda:0"
        self.tloss = 1.2345
        self.metrics = {
            "metrics/pose/mAP50": 0.8123,
            "metrics/pose/mAP50-95": 0.4567,
        }


def test_format_trainer_metrics_pose() -> None:
    text = _format_trainer_metrics(_FakeTrainer())
    assert "train_loss=1.2345" in text
    assert "mAP50=0.8123" in text


def test_attach_yolo_progress_callbacks_registers_events() -> None:
    class _FakeModel:
        def __init__(self) -> None:
            self.callbacks: dict[str, list] = {}

        def add_callback(self, event: str, func) -> None:
            self.callbacks.setdefault(event, []).append(func)

    model = _FakeModel()
    lines: list[str] = []
    _attach_yolo_progress_callbacks(model, emit=lines.append)
    assert "on_fit_epoch_end" in model.callbacks
    model.callbacks["on_fit_epoch_end"][0](_FakeTrainer())
    assert any("Epoch 5/100 complete" in line for line in lines)


def test_print_class_name_hints_detect_object(capsys) -> None:
    _print_class_name_hints("detect", ["object"])
    captured = capsys.readouterr().out
    assert "person" in captured


# ---------------------------------------------------------------------------
# Custom weights path resolution (v0.3.58)
# ---------------------------------------------------------------------------


def test_is_custom_weights_path_detects_absolute_and_relative(tmp_path: Path) -> None:
    weights = tmp_path / "runs" / "exp" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"fake")

    assert _is_custom_weights_path(str(weights))
    assert _is_custom_weights_path("runs/exp/weights/best.pt")
    assert _is_custom_weights_path("./best.pt")
    assert not _is_custom_weights_path("yolo26m.pt")
    assert not _is_custom_weights_path("auto")


def test_model_path_for_training_uses_custom_absolute_path(tmp_path: Path) -> None:
    weights = tmp_path / "my_best.pt"
    weights.write_bytes(b"fake")

    resolved = _model_path_for_training(str(weights))
    assert resolved == str(weights.resolve())


def test_model_path_for_training_uses_cwd_best_pt_before_vaila_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")
    monkeypatch.chdir(tmp_path)

    resolved = _model_path_for_training("best.pt")
    assert resolved == str(weights.resolve())


def test_model_path_for_training_relative_run_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rel = tmp_path / "runs" / "yolo_training" / "weights" / "last.pt"
    rel.parent.mkdir(parents=True)
    rel.write_bytes(b"fake")
    monkeypatch.chdir(tmp_path)

    resolved = _model_path_for_training("runs/yolo_training/weights/last.pt")
    assert resolved == str(rel.resolve())


def test_training_config_accepts_custom_weights_path(tmp_path: Path) -> None:
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "val" / "images").mkdir(parents=True)
    weights = tmp_path / "external" / "fine_tuned.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"fake")
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        "path: .\ntrain: train/images\nval: val/images\nnames: [object]\n",
        encoding="utf-8",
    )

    cfg = _resolve_training_config(str(yaml_path), task="detect", model=str(weights), device="cpu")
    assert cfg.model_name == str(weights.resolve())
