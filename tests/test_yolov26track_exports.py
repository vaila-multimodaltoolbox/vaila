from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def test_mask_to_polygons_smoke() -> None:
    from vaila.yolov26track import _mask_to_polygons

    m = np.zeros((64, 64), dtype=np.uint8)
    m[10:20, 10:30] = 255
    polys = _mask_to_polygons(m)
    assert isinstance(polys, list)
    assert len(polys) >= 1
    assert len(polys[0]) >= 3


def test_discover_tracking_csv_roots_nested(tmp_path: Path) -> None:
    from vaila.yolov26track import _discover_tracking_csv_roots

    leaf = tmp_path / "run" / "clipA"
    leaf.mkdir(parents=True)
    (leaf / "person_id_01.csv").write_text("frame,x1,y1,x2,y2\n")
    roots = _discover_tracking_csv_roots(tmp_path)
    assert roots == [leaf]


def test_discover_tracking_csv_roots_ignores_all_id(tmp_path: Path) -> None:
    from vaila.yolov26track import _discover_tracking_csv_roots

    (tmp_path / "all_id_merge.csv").write_text("x\n")
    assert _discover_tracking_csv_roots(tmp_path) == []


def test_discover_tracking_csv_roots_max_depth(tmp_path: Path) -> None:
    from vaila.yolov26track import _discover_tracking_csv_roots

    deep = tmp_path / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    (deep / "person_id_01.csv").write_text("x\n")
    assert _discover_tracking_csv_roots(tmp_path, max_depth=2) == []
    found = _discover_tracking_csv_roots(tmp_path, max_depth=8)
    assert deep in found


def test_yolo_contours_v1_schema_smoke() -> None:
    """SAM-style top-level keys expected from run_all contour export."""
    sample = {
        "schema": "vaila_yolo_contours_v1",
        "video": "clip.mp4",
        "width": 640,
        "height": 480,
        "fps": 30.0,
        "n_frames": 100,
        "object_ids": [1, 2],
        "frames": [
            {"frame": 0, "objects": [{"obj_id": 1, "polygons": [[[0, 0], [1, 0], [1, 1]]]}]}
        ],
    }
    for key in (
        "schema",
        "video",
        "width",
        "height",
        "fps",
        "n_frames",
        "object_ids",
        "frames",
    ):
        assert key in sample


def test_configure_ultralytics_dirs_creates_tree(tmp_path: Path) -> None:
    from vaila.yolov26track import _configure_ultralytics_dirs

    models_dir = tmp_path / "models"
    _configure_ultralytics_dirs(models_dir)
    assert (models_dir / "ultralytics").is_dir()


def test_is_custom_model_path_absolute_and_relative(tmp_path: Path) -> None:
    from vaila.yolov26track import _is_custom_model_path

    weights = tmp_path / "runs" / "exp" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"x")

    assert _is_custom_model_path(str(weights))
    assert _is_custom_model_path("runs/exp/weights/best.pt")
    assert not _is_custom_model_path("yolo26m.pt")


def test_progress_chunk_size_short_and_long_video() -> None:
    from vaila.yolov26track import PROGRESS_FRAME_CHUNK, _progress_chunk_size

    assert _progress_chunk_size(500) == PROGRESS_FRAME_CHUNK
    assert _progress_chunk_size(16_693) > PROGRESS_FRAME_CHUNK
    assert _progress_chunk_size(16_693) % PROGRESS_FRAME_CHUNK == 0


def test_emit_frame_progress_writes_to_stdout() -> None:
    import io

    from vaila.yolov26track import _emit_frame_progress

    buf = io.StringIO()
    old = sys.__stdout__
    sys.__stdout__ = buf
    try:
        _emit_frame_progress(99, 1000, phase="Track inference", chunk_frames=100)
        _emit_frame_progress(199, 1000, phase="Track inference", chunk_frames=100)
        out = buf.getvalue()
    finally:
        sys.__stdout__ = old

    assert ">> yolov26track: Track inference: 10% (100/1000)" in out
    assert ">> yolov26track: Track inference: 20% (200/1000)" in out
    assert out.count(">> yolov26track:") == 2


def test_make_frame_progress_logger_invokes_emit() -> None:
    import io

    from vaila.yolov26track import _make_frame_progress_logger

    buf = io.StringIO()
    old = sys.__stdout__
    sys.__stdout__ = buf
    try:
        cb = _make_frame_progress_logger(1000, "Track inference", chunk_frames=100)
        cb(99)
        cb(150)  # between chunks — no log
        out = buf.getvalue()
    finally:
        sys.__stdout__ = old

    assert "10% (100/1000)" in out
    assert out.count(">> yolov26track:") == 1


def test_auto_export_custom_path_uses_source_pt(tmp_path: Path) -> None:
    from vaila.hardware_manager import HardwareManager

    custom = tmp_path / "external" / "best.pt"
    custom.parent.mkdir(parents=True)
    custom.write_bytes(b"not-a-real-pt")

    hw = HardwareManager(models_dir=tmp_path / "models")
    hw.gpu_info["cuda_capable"] = False
    resolved = hw.auto_export(str(custom))
    assert resolved == str(custom.resolve())
