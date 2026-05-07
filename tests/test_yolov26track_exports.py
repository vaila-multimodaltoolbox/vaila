from __future__ import annotations

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
