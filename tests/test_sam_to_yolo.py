"""Unit tests for vaila.sam_to_yolo (SAM tracks -> YOLO detection dataset).

These tests are pure-Python (no GPU / no real video): they build a tiny
``sam_tracks.csv`` and a couple of fake frame images, then assert the produced
dataset is a valid YOLO *detection* set (multi-box labels, nc:1, no kpt_shape).

Run with::

    uv run pytest tests/test_sam_to_yolo.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the vaila package importable when tests are run from the repo root.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vaila.sam_to_yolo import (  # noqa: E402
    build_detection_dataset_from_sam_tracks,
    index_existing_frames,
    normalize_class_names,
    parse_sam_tracks,
    set_detection_class_names,
)

_HEADER = (
    "frame,obj_id,x_px,y_px,w_px,h_px,score,area_px,n_polygons,largest_polygon_pts,cx_px,cy_px\n"
)


def _write_tracks(path: Path, rows: list[tuple]) -> None:
    lines = [_HEADER]
    for frame, oid, x, y, w, h, score in rows:
        cx = x + w / 2.0
        cy = y + h / 2.0
        lines.append(f"{frame},{oid},{x},{y},{w},{h},{score},{int(w * h)},1,4,{cx},{cy}\n")
    path.write_text("".join(lines), encoding="utf-8")


def test_parse_sam_tracks_normalizes_and_drops_degenerate(tmp_path: Path) -> None:
    csv = tmp_path / "sam_tracks.csv"
    _write_tracks(
        csv,
        [
            (0, 0, 100, 200, 40, 80, 0.9),  # valid
            (0, 1, 0, 0, 0, 0, 0.9),  # degenerate -> dropped
            (1, 2, 960, 540, 100, 100, 0.4),  # valid (center of 1920x1080)
        ],
    )
    per_frame = parse_sam_tracks(csv, 1920, 1080)
    assert set(per_frame) == {0, 1}
    assert len(per_frame[0]) == 1  # degenerate dropped
    box = per_frame[0][0]
    assert abs(box.cx - (100 + 20) / 1920) < 1e-6
    assert abs(box.w - 40 / 1920) < 1e-6
    assert len(per_frame[1]) == 1


def test_min_score_filter(tmp_path: Path) -> None:
    csv = tmp_path / "sam_tracks.csv"
    _write_tracks(csv, [(0, 0, 10, 10, 20, 20, 0.3), (0, 1, 30, 30, 20, 20, 0.8)])
    per_frame = parse_sam_tracks(csv, 100, 100, min_score=0.5)
    assert len(per_frame[0]) == 1  # only the 0.8 box survives


def test_index_existing_frames(tmp_path: Path) -> None:
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "val" / "images").mkdir(parents=True)
    (tmp_path / "train" / "images" / "frame_000000.jpg").write_bytes(b"x")
    (tmp_path / "val" / "images" / "frame_000005.png").write_bytes(b"y")
    idx = index_existing_frames(tmp_path)
    assert set(idx) == {0, 5}
    assert idx[0].name == "frame_000000.jpg"


def test_build_detection_dataset_multibox_and_yaml(tmp_path: Path) -> None:
    # Two frames, frame 0 has 3 boxes (3 instances), frame 1 has 2.
    csv = tmp_path / "sam_tracks.csv"
    rows = [
        (0, 0, 10, 10, 20, 20, 0.9),
        (0, 1, 50, 50, 20, 20, 0.9),
        (0, 2, 80, 80, 10, 10, 0.9),
        (1, 0, 12, 12, 20, 20, 0.9),
        (1, 1, 55, 55, 20, 20, 0.9),
    ]
    _write_tracks(csv, rows)

    # Provide fake extracted frames so no video decode is needed.
    reuse = tmp_path / "frames"
    reuse.mkdir()
    for fr in (0, 1):
        (reuse / f"frame_{fr:06d}.jpg").write_bytes(b"img")

    out = tmp_path / "out"
    dataset_dir, msg, stats = build_detection_dataset_from_sam_tracks(
        csv,
        frame_width=200,
        frame_height=200,
        class_name="person",
        output_dir=out,
        reuse_images_dir=reuse,
        split_ratios=(0.5, 0.5, 0.0),
        seed=0,
    )

    assert Path(dataset_dir) == out
    assert stats.frames_with_boxes == 2
    assert stats.total_boxes == 5
    assert stats.images_missing == 0

    # data.yaml must be a DETECTION set: nc:1, no kpt_shape.
    yaml_text = (out / "data.yaml").read_text(encoding="utf-8")
    assert "nc: 1" in yaml_text
    assert "names: ['person']" in yaml_text
    # No actual kpt_shape *key* (mention inside a comment line is fine).
    assert not any(line.strip().startswith("kpt_shape:") for line in yaml_text.splitlines())

    # Every label file holds one line per instance, format `0 cx cy w h`.
    label_files = list((out / "train" / "labels").glob("*.txt")) + list(
        (out / "val" / "labels").glob("*.txt")
    )
    assert label_files
    total_lines = 0
    for lf in label_files:
        for line in lf.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            assert len(parts) == 5
            assert parts[0] == "0"
            total_lines += 1
    assert total_lines == 5

    assert "classes.txt" in {p.name for p in out.iterdir()}
    assert (out / "classes.txt").read_text(encoding="utf-8").strip() == "person"


def test_frame_stride_keeps_every_nth_frame(tmp_path: Path) -> None:
    # 20 frames (0..19), one box each. stride=5 -> keep frames 0,5,10,15.
    csv = tmp_path / "sam_tracks.csv"
    rows = [(fr, 0, 10, 10, 20, 20, 0.9) for fr in range(20)]
    _write_tracks(csv, rows)

    reuse = tmp_path / "frames"
    reuse.mkdir()
    for fr in range(20):
        (reuse / f"frame_{fr:06d}.jpg").write_bytes(b"img")

    out = tmp_path / "out_stride"
    _dir, _msg, stats = build_detection_dataset_from_sam_tracks(
        csv,
        frame_width=200,
        frame_height=200,
        class_name="person",
        output_dir=out,
        reuse_images_dir=reuse,
        split_mode="temporal",
        frame_stride=5,
    )

    kept = sorted(
        int(p.stem.split("_")[1])
        for split in ("train", "val", "test")
        for p in (out / split / "labels").glob("frame_*.txt")
    )
    assert kept == [0, 5, 10, 15]
    assert stats.frames_with_boxes == 4


def test_temporal_split_is_chronological(tmp_path: Path) -> None:
    # 10 frames, one box each. Temporal split must put early frames in train,
    # middle in val, late in test (no shuffle), so train < val < test by frame.
    csv = tmp_path / "sam_tracks.csv"
    rows = [(fr, 0, 10, 10, 20, 20, 0.9) for fr in range(10)]
    _write_tracks(csv, rows)

    reuse = tmp_path / "frames"
    reuse.mkdir()
    for fr in range(10):
        (reuse / f"frame_{fr:06d}.jpg").write_bytes(b"img")

    out = tmp_path / "out_temporal"
    build_detection_dataset_from_sam_tracks(
        csv,
        frame_width=200,
        frame_height=200,
        class_name="person",
        output_dir=out,
        reuse_images_dir=reuse,
        split_ratios=(0.7, 0.2, 0.1),
        split_mode="temporal",
    )

    def _frames(split: str) -> list[int]:
        files = (out / split / "labels").glob("frame_*.txt")
        return sorted(int(p.stem.split("_")[1]) for p in files)

    train, val, test = _frames("train"), _frames("val"), _frames("test")
    assert train == [0, 1, 2, 3, 4, 5, 6]
    assert val == [7, 8]
    assert test == [9]
    # No frame appears in more than one split.
    assert not (set(train) & set(val))
    assert max(train) < min(val) < min(test)


def test_normalize_class_names_single_list_and_csv() -> None:
    assert normalize_class_names("person") == ["person"]
    assert normalize_class_names("person, ball") == ["person", "ball"]
    assert normalize_class_names(["a", " b ", ""]) == ["a", "b"]


def test_set_detection_class_names_is_metadata_only(tmp_path: Path) -> None:
    # Build a tiny dataset with class 'object', then rename to 'person'.
    csv = tmp_path / "sam_tracks.csv"
    _write_tracks(csv, [(0, 0, 10, 10, 20, 20, 0.9), (0, 1, 50, 50, 20, 20, 0.9)])
    reuse = tmp_path / "frames"
    reuse.mkdir()
    (reuse / "frame_000000.jpg").write_bytes(b"img")
    out = tmp_path / "out"
    build_detection_dataset_from_sam_tracks(
        csv,
        frame_width=200,
        frame_height=200,
        class_name="object",
        output_dir=out,
        reuse_images_dir=reuse,
        split_ratios=(1.0, 0.0, 0.0),
    )
    label = out / "train" / "labels" / "frame_000000.txt"
    before = label.read_text(encoding="utf-8")

    msg = set_detection_class_names(out, "person")
    assert "person" in msg
    yaml_text = (out / "data.yaml").read_text(encoding="utf-8")
    assert "names: ['person']" in yaml_text
    assert "nc: 1" in yaml_text
    assert (out / "classes.txt").read_text(encoding="utf-8").strip() == "person"
    # Index-based labels are untouched by a rename.
    assert label.read_text(encoding="utf-8") == before


def test_set_detection_class_names_rejects_pose_dataset(tmp_path: Path) -> None:
    import pytest

    ds = tmp_path / "pose"
    ds.mkdir()
    (ds / "data.yaml").write_text(
        "nc: 1\nnames: ['object']\nkpt_shape: [62, 3]\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="kpt_shape"):
        set_detection_class_names(ds, "person")
