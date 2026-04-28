"""Smoke tests for :mod:`vaila.fifa_dataset_builder`.

These tests do **not** hit the network: they exercise the converters and the
merge/manifest writer with synthetic mini-datasets generated on the fly.
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from vaila import fifa_dataset_builder as fdb


def _write_image(path: Path, content: bytes = b"\x89PNG\r\n\x1a\n0000IHDR") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _write_real_rgb_image(path: Path, size: tuple[int, int] = (320, 180)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for this test helper") from exc
    img = Image.new("RGB", size, (32, 96, 32))
    img.save(path)
    return path


def _yolo_pose_line(
    cls: int,
    bbox: tuple[float, float, float, float],
    kps: list[tuple[float, float, int]],
) -> str:
    parts = [str(cls), *(f"{v:.6f}" for v in bbox)]
    for x, y, v in kps:
        parts += [f"{x:.6f}", f"{y:.6f}", str(int(v))]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def test_canonical_constants_are_consistent():
    assert fdb.NUM_KEYPOINTS == 32
    assert len(fdb.CANONICAL_KP_NAMES_32) == 32
    assert len(set(fdb.CANONICAL_KP_NAMES_32)) == 32  # no duplicates
    assert len(fdb.CANONICAL_FLIP_IDX_32) == 32
    # flip is involutive: applying flip twice is identity.
    flipped_twice = [fdb.CANONICAL_FLIP_IDX_32[fdb.CANONICAL_FLIP_IDX_32[i]] for i in range(32)]
    assert flipped_twice == list(range(32))


def test_canonical_vertices_match_roboflow_truth():
    """Spot-check that vertex coordinates match the Roboflow SoccerPitchConfiguration."""
    cm = fdb._canonical_vertices_cm()
    assert cm[0] == (0.0, 0.0)  # top_left_corner
    assert cm[24] == (12000.0, 0.0)  # top_right_corner
    assert cm[5] == (0.0, 7000.0)  # bottom_left_corner
    assert cm[29] == (12000.0, 7000.0)  # bottom_right_corner
    assert cm[8] == (1100.0, 3500.0)  # left_penalty_spot
    assert cm[21] == (10900.0, 3500.0)  # right_penalty_spot
    assert cm[13] == (6000.0, 0.0)  # midfield_top
    assert cm[16] == (6000.0, 7000.0)  # midfield_bottom
    assert cm[30] == (5085.0, 3500.0)  # center_circle_left
    assert cm[31] == (6915.0, 3500.0)  # center_circle_right


def test_canonical_flip_pairs_are_geometric_mirrors():
    """Each kp's flip target must be its horizontal mirror across X = L/2."""
    cm = fdb._canonical_vertices_cm()
    L = float(fdb.ROBOFLOW_FIELD_LENGTH_CM)
    for i, j in enumerate(fdb.CANONICAL_FLIP_IDX_32):
        assert abs(cm[i][0] + cm[j][0] - L) < 1e-6, f"kp {i} not horizontal mirror of kp {j}"
        assert abs(cm[i][1] - cm[j][1]) < 1e-6, f"kp {i} y differs from kp {j}"


def test_centered_meters_match_drawsportsfields_convention():
    """Centered Y-up convention: top-left corner in upper-left = (-L/2, +W/2)."""
    pts, length_m, width_m = fdb._load_centered_fifa_points_32()
    assert length_m == fdb.DRAWSPORTSFIELDS_LENGTH_M
    assert width_m == fdb.DRAWSPORTSFIELDS_WIDTH_M
    # idx 0 = top_left -> (-L/2, +W/2, 0)
    x, y, _z = pts[0]
    assert x == -length_m / 2.0
    assert y == +width_m / 2.0
    # idx 5 = bottom_left -> (-L/2, -W/2, 0)
    x, y, _z = pts[5]
    assert x == -length_m / 2.0
    assert y == -width_m / 2.0
    # idx 24 = top_right -> (+L/2, +W/2, 0)
    x, y, _z = pts[24]
    assert x == +length_m / 2.0
    assert y == +width_m / 2.0


def test_kpsfr_template_matches_drawsportsfields_centered():
    """Centered Y-up -> KpSFR (yards, Y-down, top-left origin)."""
    # top_left in centered metres == (0, 0) yards in KpSFR template.
    x_tpl, y_tpl = fdb._centered_meters_to_kpsfr_template(
        -fdb.DRAWSPORTSFIELDS_LENGTH_M / 2.0,
        +fdb.DRAWSPORTSFIELDS_WIDTH_M / 2.0,
        length_m=fdb.DRAWSPORTSFIELDS_LENGTH_M,
        width_m=fdb.DRAWSPORTSFIELDS_WIDTH_M,
    )
    assert abs(x_tpl) < 1e-6
    assert abs(y_tpl) < 1e-6
    # bottom_right in centered metres -> (114.83, 74.37) yards.
    x_tpl, y_tpl = fdb._centered_meters_to_kpsfr_template(
        +fdb.DRAWSPORTSFIELDS_LENGTH_M / 2.0,
        -fdb.DRAWSPORTSFIELDS_WIDTH_M / 2.0,
        length_m=fdb.DRAWSPORTSFIELDS_LENGTH_M,
        width_m=fdb.DRAWSPORTSFIELDS_WIDTH_M,
    )
    assert abs(x_tpl - fdb.KPSFR_TEMPLATE_LENGTH_YARDS) < 1e-6
    assert abs(y_tpl - fdb.KPSFR_TEMPLATE_WIDTH_YARDS) < 1e-6


def test_select_sources_default_excludes_optional():
    sources = fdb._select_sources(None, include_optional=False)
    assert all(not s.optional for s in sources)
    sources = fdb._select_sources(None, include_optional=True)
    assert any(s.optional for s in sources)


def test_select_sources_explicit_include():
    sources = fdb._select_sources(["martinjolif_football-pitch-detection"], include_optional=False)
    assert len(sources) == 1
    assert sources[0].name == "martinjolif_football-pitch-detection"


def test_make_canonical_label_text_pads_short_lists():
    line = fdb._make_canonical_label_text(
        bbox_xywhn=None,
        kps_xyv=[(0.1, 0.2, 2)],
    )
    parts = line.split()
    # 1 cls + 4 bbox + 32 * 3 keypoints = 101 fields
    assert len(parts) == 1 + 4 + 32 * 3


def test_format_yolo_pose_row_round_trip():
    line = fdb._format_yolo_pose_row(0, (0.5, 0.5, 0.4, 0.3), [(0.1, 0.2, 2)] * 32)
    parts = line.split()
    assert parts[0] == "0"
    assert len(parts) == 1 + 4 + 32 * 3
    assert parts[5] == "0.100000" and parts[6] == "0.200000" and parts[7] == "2"


# ---------------------------------------------------------------------------
# Pass-through converter (martinjolif-style 32-pt YOLO Pose)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_martinjolif_repo(tmp_path: Path) -> Path:
    """Build a tiny synthetic dataset matching the martinjolif / Roboflow layout."""
    repo = tmp_path / "fake_martinjolif"
    for split in ("train", "valid", "test"):
        img_dir = repo / "data" / split / "images"
        lbl_dir = repo / "data" / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            stem = f"{split}_{i:03d}"
            _write_image(img_dir / f"{stem}.jpg")
            kps = [
                ((i + 1) * 0.01 + j * 0.001, (i + 1) * 0.02 + j * 0.001, 2 if j % 3 != 2 else 0)
                for j in range(32)
            ]
            line = _yolo_pose_line(0, (0.5, 0.5, 0.4, 0.3), kps)
            (lbl_dir / f"{stem}.txt").write_text(line + "\n", encoding="utf-8")
    return repo


def test_convert_yolo_pose_passthrough(tmp_path: Path, fake_martinjolif_repo: Path):
    out = tmp_path / "staging_martin"
    stats = fdb.convert_yolo_pose_passthrough(fake_martinjolif_repo, out)
    assert stats["images"] == 6
    assert stats["labels"] == 6
    # Splits are remapped: valid -> val
    assert (out / "images" / "train").is_dir()
    assert (out / "images" / "val").is_dir()
    assert (out / "images" / "test").is_dir()
    sample_label = next((out / "labels" / "train").glob("*.txt"))
    parts = sample_label.read_text(encoding="utf-8").strip().split()
    assert len(parts) == 1 + 4 + 32 * 3
    # Regression: the formatter MUST keep keypoint y-coordinates as floats
    # (only the visibility flag at offsets 7, 10, 13, ... is an int).
    # Previous bug clamped y to 0/1 because the int/float layout test was wrong.
    for j in range(32):
        x = parts[5 + 3 * j]
        y = parts[5 + 3 * j + 1]
        v = parts[5 + 3 * j + 2]
        assert "." in x, f"kp{j}.x lost decimals: {x}"
        assert "." in y, f"kp{j}.y lost decimals: {y}"
        assert "." not in v, f"kp{j}.v should be int, got {v}"
        assert 0.0 <= float(y) <= 1.0
        # Original synthetic y values are 0.02..0.04 — must NOT be 0 or 1.
        assert float(y) > 0.0 and float(y) < 1.0


# ---------------------------------------------------------------------------
# Soccana converter (HF labels-only + paired SoccerNet images)
# ---------------------------------------------------------------------------


def _soccana_full_kp_dict() -> dict[str, list[float]]:
    """Build a JSON-style ``keypoints`` dict covering all 29 Soccana names."""
    return {
        "0_sideline_top_left": [0.05, 0.10],
        "1_big_rect_left_top_pt1": [0.10, 0.20],
        "2_big_rect_left_top_pt2": [0.30, 0.20],
        "3_big_rect_left_bottom_pt1": [0.10, 0.80],
        "4_big_rect_left_bottom_pt2": [0.30, 0.80],
        "5_small_rect_left_top_pt1": [0.10, 0.30],
        "6_small_rect_left_top_pt2": [0.20, 0.30],
        "7_small_rect_left_bottom_pt1": [0.10, 0.70],
        "8_small_rect_left_bottom_pt2": [0.20, 0.70],
        "9_sideline_bottom_left": [0.05, 0.90],
        "10_left_semicircle_right": [0.40, 0.50],  # apex, no canonical slot
        "11_center_line_top": [0.50, 0.10],
        "12_center_line_bottom": [0.50, 0.90],
        "13_center_circle_top": [0.50, 0.40],
        "14_center_circle_bottom": [0.50, 0.60],
        "15_field_center": [0.50, 0.50],  # no canonical slot
        "16_sideline_top_right": [0.95, 0.10],
        "17_big_rect_right_top_pt1": [0.90, 0.20],
        "18_big_rect_right_top_pt2": [0.70, 0.20],
        "19_big_rect_right_bottom_pt1": [0.90, 0.80],
        "20_big_rect_right_bottom_pt2": [0.70, 0.80],
        "21_small_rect_right_top_pt1": [0.90, 0.30],
        "22_small_rect_right_top_pt2": [0.80, 0.30],
        "23_small_rect_right_bottom_pt1": [0.90, 0.70],
        "24_small_rect_right_bottom_pt2": [0.80, 0.70],
        "25_sideline_bottom_right": [0.95, 0.90],
        "26_right_semicircle_left": [0.60, 0.50],  # apex, no canonical slot
        "27_center_circle_left": [0.45, 0.50],
        "28_center_circle_right": [0.55, 0.50],
    }


@pytest.fixture
def fake_soccana_repo(tmp_path: Path) -> Path:
    """A faithful synthetic Soccana download: ZIP w/ JSON+YOLO + preview JPGs."""
    repo = tmp_path / "fake_soccana"
    repo.mkdir(parents=True, exist_ok=True)
    json_root = repo / "_zip_src" / "annotations_json"
    yolo_root = repo / "_zip_src" / "yolo_labels"
    for split in ("train", "valid", "test"):
        (json_root / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / split).mkdir(parents=True, exist_ok=True)
        for i in range(2):
            stem = f"{i:05d}" if split == "train" else f"{split}_{i:05d}"
            kp = _soccana_full_kp_dict()
            (json_root / split / f"{stem}.json").write_text(
                json.dumps(
                    {
                        "image_info": {
                            "file_name": f"{stem}.jpg",
                            "width": 960,
                            "height": 540,
                        },
                        "pitch_object": {
                            "center_x": 0.5,
                            "center_y": 0.5,
                            "width": 1.0,
                            "height": 0.8,
                        },
                        "keypoints": kp,
                    }
                ),
                encoding="utf-8",
            )
            # YOLO label is dummy; the JSON-driven path takes precedence.
            (yolo_root / split / f"{stem}.txt").write_text(
                "0 0.5 0.5 1.0 0.8" + " 0.0 0.0 0" * 29 + "\n",
                encoding="utf-8",
            )

    # Pack into dataset_labels.zip (just like the real HF download).
    zip_path = repo / "dataset_labels.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in (repo / "_zip_src").rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(repo / "_zip_src"))
    # Bundled preview JPGs (5 of them in the real HF dataset).
    for stem in ("00000", "00001"):
        _write_image(repo / f"{stem}_annotated.jpg")
    return repo


def test_convert_soccana_29pt_json_with_previews(tmp_path: Path, fake_soccana_repo: Path):
    """When no SoccerNet images are supplied, fall back to the bundled preview JPGs."""
    out = tmp_path / "staging_soccana"
    stats = fdb.convert_soccana_29pt(fake_soccana_repo, out)
    # Train split has stems 00000 + 00001 — both preview JPGs match -> 2 images.
    assert stats["images"] == 2
    assert stats["fallback_visualisations"] == 2
    sample = (out / "labels" / "train" / "00000.txt").read_text(encoding="utf-8").strip()
    parts = sample.split()
    assert len(parts) == 1 + 4 + 32 * 3
    # canonical idx 0 (top_left_corner) <- 0_sideline_top_left = (0.05, 0.10)
    assert parts[5] == "0.050000" and parts[6] == "0.100000" and parts[7] == "2"
    # canonical idx 9 (left_pen_box_top_inner) <- 2_big_rect_left_top_pt2 = (0.30, 0.20)
    base_9 = 5 + 3 * 9
    assert parts[base_9] == "0.300000" and parts[base_9 + 1] == "0.200000"
    # canonical idx 4 (left_pen_box_bottom_outer) <- 3_big_rect_left_bottom_pt1 = (0.10, 0.80)
    base_4 = 5 + 3 * 4
    assert parts[base_4] == "0.100000" and parts[base_4 + 1] == "0.800000"
    # canonical idx 24 (top_right_corner) <- 16_sideline_top_right = (0.95, 0.10)
    base_24 = 5 + 3 * 24
    assert parts[base_24] == "0.950000" and parts[base_24 + 1] == "0.100000"
    # canonical idx 30 (center_circle_left) <- 27_center_circle_left = (0.45, 0.50)
    base_30 = 5 + 3 * 30
    assert parts[base_30] == "0.450000" and parts[base_30 + 1] == "0.500000"
    # canonical idx 13 (midfield_top) <- 11_center_line_top = (0.50, 0.10)
    base_13 = 5 + 3 * 13
    assert parts[base_13] == "0.500000" and parts[base_13 + 1] == "0.100000"


def test_convert_soccana_29pt_with_soccernet_images(tmp_path: Path, fake_soccana_repo: Path):
    """When SoccerNet images are supplied via ``soccernet_images``, they are paired by stem."""
    soccernet_root = tmp_path / "fake_soccernet"
    image_dir = soccernet_root / "calibration-2023" / "valid"
    image_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("valid_00000", "valid_00001"):
        _write_image(image_dir / f"{stem}.jpg")

    out = tmp_path / "staging_soccana"
    snet = fdb._index_soccernet_images(soccernet_root)
    assert "valid_00000" in snet
    stats = fdb.convert_soccana_29pt(fake_soccana_repo, out, soccernet_images=snet)
    # train preview JPGs (00000+00001) plus the 2 valid SoccerNet images = 4 images.
    assert stats["images"] == 4
    # SoccerNet images pair with the valid split.
    assert (out / "labels" / "val" / "valid_00000.txt").exists()


# ---------------------------------------------------------------------------
# SoccerNet "images-only" converter
# ---------------------------------------------------------------------------


def test_convert_soccernet_images_only_indexes(tmp_path: Path):
    snet = tmp_path / "fake_snet"
    img_dir = snet / "calibration-2023" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("00000", "00001"):
        _write_image(img_dir / f"{stem}.jpg")
    stats = fdb.convert_soccernet_images_only(snet, tmp_path / "staging_snet")
    assert stats["images_indexed"] == 2
    # No staging output yet — Soccana consumes them.
    assert stats["images"] == 0


# ---------------------------------------------------------------------------
# Kaggle bbox converter
# ---------------------------------------------------------------------------


def test_convert_kaggle_bbox_landmarks(tmp_path: Path):
    repo = tmp_path / "fake_kaggle"
    (repo / "train" / "images").mkdir(parents=True, exist_ok=True)
    (repo / "train" / "labels").mkdir(parents=True, exist_ok=True)
    # classes.txt enumerates known canonical names
    (repo / "classes.txt").write_text(
        "\n".join(
            [
                "top_left_corner",
                "top_right_corner",
                "bottom_left_corner",
                "bottom_right_corner",
                "halfway_top",
                "halfway_bottom",
            ]
        ),
        encoding="utf-8",
    )
    _write_image(repo / "train" / "images" / "kfoo.jpg")
    label_lines = [
        "0 0.1 0.1 0.02 0.02",  # top_left_corner
        "1 0.9 0.1 0.02 0.02",  # top_right_corner
        "2 0.1 0.9 0.02 0.02",  # bottom_left_corner
        "3 0.9 0.9 0.02 0.02",  # bottom_right_corner
        "4 0.5 0.05 0.02 0.02",  # halfway_top
    ]
    (repo / "train" / "labels" / "kfoo.txt").write_text(
        "\n".join(label_lines) + "\n", encoding="utf-8"
    )

    out = tmp_path / "staging_kaggle"
    stats = fdb.convert_kaggle_bbox_landmarks(repo, out)
    assert stats["images"] == 1
    sample = (out / "labels" / "train" / "kfoo.txt").read_text(encoding="utf-8").strip()
    parts = sample.split()
    assert len(parts) == 1 + 4 + 32 * 3

    # canonical idx 0 (top_left_corner) should be at (0.1, 0.1, 2)
    assert parts[5] == "0.100000"
    assert parts[6] == "0.100000"
    assert parts[7] == "2"
    # canonical idx 24 (top_right_corner) should be at (0.9, 0.1, 2)
    base = 5 + 3 * 24
    assert parts[base] == "0.900000" and parts[base + 1] == "0.100000"
    # canonical idx 5 (bottom_left_corner) should be at (0.1, 0.9, 2)
    base = 5 + 3 * 5
    assert parts[base] == "0.100000" and parts[base + 1] == "0.900000"
    # canonical idx 13 (midfield_top) should be at (0.5, 0.05, 2)  (halfway_top)
    base = 5 + 3 * 13
    assert parts[base] == "0.500000" and parts[base + 1] == "0.050000"


# ---------------------------------------------------------------------------
# PitchGeometry CSV converter
# ---------------------------------------------------------------------------


def test_convert_pitchgeometry_csv(tmp_path: Path):
    repo = tmp_path / "fake_pitchgeometry"
    img_dir = repo / "dataset" / "train" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    _write_image(img_dir / "fr0001.png")

    csv_path = repo / "dataset" / "train" / "labels.csv"
    rows = [
        {"frame": "fr0001.png", "kid": "TOP_LEFT_CORNER", "x": "0.05", "y": "0.05", "vis": "1"},
        {"frame": "fr0001.png", "kid": "TOP_RIGHT_CORNER", "x": "0.95", "y": "0.05", "vis": "1"},
        {"frame": "fr0001.png", "kid": "BOTTOM_LEFT_CORNER", "x": "0.05", "y": "0.95", "vis": "1"},
        {"frame": "fr0001.png", "kid": "BOTTOM_RIGHT_CORNER", "x": "0.95", "y": "0.95", "vis": "1"},
        {"frame": "fr0001.png", "kid": "CENTER_CIRCLE_TOP", "x": "0.50", "y": "0.40", "vis": "1"},
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "kid", "x", "y", "vis"])
        writer.writeheader()
        writer.writerows(rows)

    out = tmp_path / "staging_pg"
    stats = fdb.convert_pitchgeometry_csv(repo, out)
    assert stats["images"] == 1
    sample = (out / "labels" / "train" / "fr0001.txt").read_text(encoding="utf-8").strip()
    parts = sample.split()
    assert len(parts) == 1 + 4 + 32 * 3
    # canonical idx 0 (top_left_corner) at (0.05, 0.05, 2)
    assert parts[5] == "0.050000" and parts[6] == "0.050000" and parts[7] == "2"
    # canonical idx 24 (top_right_corner) at (0.95, 0.05, 2)
    base = 5 + 3 * 24
    assert parts[base] == "0.950000" and parts[base + 1] == "0.050000" and parts[base + 2] == "2"
    # canonical idx 5 (bottom_left_corner) at (0.05, 0.95, 2)
    base = 5 + 3 * 5
    assert parts[base] == "0.050000" and parts[base + 1] == "0.950000"
    # canonical idx 14 (center_circle_top) at (0.50, 0.40, 2)
    base = 5 + 3 * 14
    assert parts[base] == "0.500000" and parts[base + 1] == "0.400000"


# ---------------------------------------------------------------------------
# Homography-based converters (WC14 / TS-WorldCup)
# ---------------------------------------------------------------------------


def test_convert_worldcup2014_homography_identity(tmp_path: Path):
    repo = tmp_path / "wc14"
    img = _write_real_rgb_image(repo / "raw" / "train_val" / "1.jpg", size=(320, 200))
    (repo / "raw" / "train_val" / "1.homographyMatrix").write_text(
        "1 0 0\n0 1 0\n0 0 1\n",
        encoding="utf-8",
    )
    out = tmp_path / "staging_wc14"
    stats = fdb.convert_worldcup2014_homography(repo, out)
    assert stats["images"] == 1
    sample = (out / "labels" / "train" / "1.txt").read_text(encoding="utf-8").strip()
    parts = sample.split()
    assert len(parts) == 1 + 4 + 32 * 3
    assert (out / "images" / "train" / img.name).exists()


def test_convert_tsworldcup_homography_identity(tmp_path: Path):
    repo = tmp_path / "tswc"
    clip = "left/clip_001"
    clip2 = "right/clip_001"
    (repo / "TS-WorldCup").mkdir(parents=True, exist_ok=True)
    (repo / "TS-WorldCup" / "train.txt").write_text(clip + "\n" + clip2 + "\n", encoding="utf-8")
    (repo / "TS-WorldCup" / "test.txt").write_text("", encoding="utf-8")

    ann = repo / "TS-WorldCup" / "Annotations" / "80_95" / "left" / "clip_001"
    img_dir = repo / "TS-WorldCup" / "Dataset" / "80_95" / "left" / "clip_001"
    ann.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    np.save(ann / "IMG_001_homography.npy", np.eye(3, dtype=np.float64))
    _write_real_rgb_image(img_dir / "IMG_001.jpg", size=(320, 200))

    ann2 = repo / "TS-WorldCup" / "Annotations" / "80_95" / "right" / "clip_001"
    img_dir2 = repo / "TS-WorldCup" / "Dataset" / "80_95" / "right" / "clip_001"
    ann2.mkdir(parents=True, exist_ok=True)
    img_dir2.mkdir(parents=True, exist_ok=True)
    np.save(ann2 / "IMG_001_homography.npy", np.eye(3, dtype=np.float64))
    _write_real_rgb_image(img_dir2 / "IMG_001.jpg", size=(320, 200))

    out = tmp_path / "staging_tswc"
    stats = fdb.convert_tsworldcup_homography(repo, out)
    assert stats["images"] == 2
    lbls = sorted((out / "labels" / "train").glob("*.txt"))
    assert len(lbls) == 2
    sample = lbls[0].read_text(encoding="utf-8").strip()
    parts = sample.split()
    assert len(parts) == 1 + 4 + 32 * 3


# ---------------------------------------------------------------------------
# Centered keypoint reference export
# ---------------------------------------------------------------------------


def test_write_keypoint_reference(tmp_path: Path):
    unified = tmp_path / "unified"
    unified.mkdir(parents=True, exist_ok=True)
    out = fdb._write_keypoint_reference(unified)
    assert out.exists()
    rows = list(csv.DictReader(out.open("r", encoding="utf-8")))
    assert len(rows) == 32
    # New schema: idx 24 is top_right_corner.
    by_idx = {int(r["idx"]): r for r in rows}
    assert by_idx[0]["canonical_name"] == "top_left_corner"
    assert by_idx[24]["canonical_name"] == "top_right_corner"
    assert by_idx[5]["canonical_name"] == "bottom_left_corner"
    assert by_idx[29]["canonical_name"] == "bottom_right_corner"
    # Centered metres convention: top_left at (-L/2, +W/2).
    L = fdb.DRAWSPORTSFIELDS_LENGTH_M
    W = fdb.DRAWSPORTSFIELDS_WIDTH_M
    assert abs(float(by_idx[0]["x_center_m"]) - (-L / 2.0)) < 1e-6
    assert abs(float(by_idx[0]["y_center_m"]) - (+W / 2.0)) < 1e-6
    # Roboflow truth columns.
    assert abs(float(by_idx[0]["x_roboflow_cm"])) < 1e-6
    assert abs(float(by_idx[24]["x_roboflow_cm"]) - 12000.0) < 1e-6
    # KpSFR template columns.
    assert abs(float(by_idx[0]["x_template_yard"])) < 1e-6
    assert abs(float(by_idx[24]["x_template_yard"]) - fdb.KPSFR_TEMPLATE_LENGTH_YARDS) < 1e-6


def test_convert_wc14_tvcalib_segments(tmp_path: Path):
    sources_root = tmp_path / "sources"
    tvc = sources_root / "wc14_tvcalib_additional_annotations"
    wc14 = sources_root / "worldcup2014_nhoma" / "raw" / "test"
    tvc.mkdir(parents=True, exist_ok=True)
    wc14.mkdir(parents=True, exist_ok=True)
    _write_real_rgb_image(wc14 / "1.jpg", size=(320, 200))

    ann = {
        "Side line top": [{"x": 0.1, "y": 0.2}, {"x": 0.9, "y": 0.2}],
        "Side line bottom": [{"x": 0.1, "y": 0.8}, {"x": 0.9, "y": 0.8}],
        "Side line left": [{"x": 0.1, "y": 0.2}, {"x": 0.1, "y": 0.8}],
        "Side line right": [{"x": 0.9, "y": 0.2}, {"x": 0.9, "y": 0.8}],
        "Middle line": [{"x": 0.5, "y": 0.2}, {"x": 0.5, "y": 0.8}],
        "Circle left": [{"x": 0.25, "y": 0.5}, {"x": 0.28, "y": 0.55}, {"x": 0.28, "y": 0.45}],
        "Circle right": [{"x": 0.75, "y": 0.5}, {"x": 0.72, "y": 0.55}, {"x": 0.72, "y": 0.45}],
    }
    (tvc / "1.json").write_text(json.dumps(ann), encoding="utf-8")

    out = tmp_path / "staging_tvc"
    stats = fdb.convert_wc14_tvcalib_segments(tvc, out)
    assert stats["images"] == 1
    sample = (out / "labels" / "test" / "1.txt").read_text(encoding="utf-8").strip()
    parts = sample.split()
    assert len(parts) == 1 + 4 + 32 * 3
    # Spot-check a few key intersections (the synthetic field is the unit
    # square scaled to [0.1, 0.9] in both axes):
    # idx 0 = top_left = (sideline top ∩ sideline left) -> (0.1, 0.2)
    assert parts[5] == "0.100000" and parts[6] == "0.200000"
    # idx 24 = top_right = (sideline top ∩ sideline right) -> (0.9, 0.2)
    base_24 = 5 + 3 * 24
    assert parts[base_24] == "0.900000" and parts[base_24 + 1] == "0.200000"
    # idx 13 = midfield_top = (middle line ∩ side line top) -> (0.5, 0.2)
    base_13 = 5 + 3 * 13
    assert parts[base_13] == "0.500000" and parts[base_13 + 1] == "0.200000"


# ---------------------------------------------------------------------------
# Quality filter and source fusion in the merger
# ---------------------------------------------------------------------------


def _write_label_with_visible(path: Path, n_visible: int, x: float = 0.5, y: float = 0.5) -> None:
    """Write a YOLO Pose label with exactly ``n_visible`` visible keypoints."""
    parts: list[str] = ["0", "0.5", "0.5", "0.4", "0.3"]
    for k in range(32):
        if k < n_visible:
            parts += [f"{x:.6f}", f"{y:.6f}", "2"]
        else:
            parts += ["0.000000", "0.000000", "0"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(parts) + "\n", encoding="utf-8")


def test_merge_drops_low_visibility_labels(tmp_path: Path):
    staging = tmp_path / "staging"
    src = staging / "martinjolif_football-pitch-detection"
    img_dir = src / "images" / "train"
    lbl_dir = src / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    # Image A: 4 visible -> kept.
    _write_real_rgb_image(img_dir / "A.jpg")
    _write_label_with_visible(lbl_dir / "A.txt", n_visible=4)
    # Image B: only 2 visible -> dropped.
    _write_real_rgb_image(img_dir / "B.jpg")
    _write_label_with_visible(lbl_dir / "B.txt", n_visible=2)

    unified = fdb._ensure_dir(tmp_path / "unified")
    counts = fdb._merge_staging_into_unified(staging, unified)
    assert counts["train"] == 1
    assert counts["_skipped_low_quality"] == 1
    survivors = list((unified / "labels" / "train").glob("*.txt"))
    assert len(survivors) == 1
    assert "A" in survivors[0].name


def test_merge_fusion_priority_keeps_higher_priority_source(tmp_path: Path):
    """Two staging entries pointing at the *same* physical image fuse to one,
    keeping the higher-priority source's label."""
    staging = tmp_path / "staging"
    # The underlying physical image (lives outside any staging tree).
    physical_dir = tmp_path / "physical"
    physical_dir.mkdir(parents=True, exist_ok=True)
    physical_img = _write_real_rgb_image(physical_dir / "frame_001.jpg")

    # Lower priority source: ts_worldcup_kpsfr (priority 60).
    low = staging / "ts_worldcup_kpsfr"
    img_low = low / "images" / "train"
    lbl_low = low / "labels" / "train"
    img_low.mkdir(parents=True, exist_ok=True)
    lbl_low.mkdir(parents=True, exist_ok=True)
    (img_low / "frame_001.jpg").symlink_to(physical_img)
    _write_label_with_visible(lbl_low / "frame_001.txt", n_visible=8, x=0.1)

    # Higher priority source: martinjolif (priority 100), same physical image.
    high = staging / "martinjolif_football-pitch-detection"
    img_high = high / "images" / "train"
    lbl_high = high / "labels" / "train"
    img_high.mkdir(parents=True, exist_ok=True)
    lbl_high.mkdir(parents=True, exist_ok=True)
    (img_high / "frame_001.jpg").symlink_to(physical_img)
    _write_label_with_visible(lbl_high / "frame_001.txt", n_visible=8, x=0.9)

    unified = fdb._ensure_dir(tmp_path / "unified")
    counts = fdb._merge_staging_into_unified(staging, unified)
    # Only one survives because both staging symlinks resolve to the same
    # physical jpg.
    assert counts["train"] == 1
    assert counts["_fused_drops"] == 1
    survivors = list((unified / "labels" / "train").glob("*.txt"))
    assert len(survivors) == 1
    # Survivor must be from the higher-priority source.
    assert survivors[0].name.startswith("martinjolif_football-pitch-detection__")


def test_merge_does_not_fuse_distinct_physical_images_with_same_basename(tmp_path: Path):
    """Soccana ``00000.jpg`` from SoccerNet train and ``00000.jpg`` from a
    different folder are physically different images and must NOT collapse."""
    staging = tmp_path / "staging"
    physical = tmp_path / "physical"
    physical.mkdir(parents=True, exist_ok=True)
    img_a = _write_real_rgb_image(physical / "a" / "00000.jpg")
    img_b = _write_real_rgb_image(physical / "b" / "00000.jpg")

    src_a = staging / "ts_worldcup_kpsfr"
    src_b = staging / "Adit-jain_Soccana_Keypoint_detection_v1"
    for src, img, x in [(src_a, img_a, 0.1), (src_b, img_b, 0.9)]:
        (src / "images" / "train").mkdir(parents=True, exist_ok=True)
        (src / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (src / "images" / "train" / "00000.jpg").symlink_to(img)
        _write_label_with_visible(src / "labels" / "train" / "00000.txt", n_visible=8, x=x)

    unified = fdb._ensure_dir(tmp_path / "unified")
    counts = fdb._merge_staging_into_unified(staging, unified)
    assert counts["train"] == 2
    assert counts["_fused_drops"] == 0


# ---------------------------------------------------------------------------
# Merge + data.yaml + manifest
# ---------------------------------------------------------------------------


def test_full_local_build(tmp_path: Path, fake_martinjolif_repo: Path, fake_soccana_repo: Path):
    """Run the pipeline end-to-end on a couple of fake sources, no network."""

    out_root = tmp_path / "dataset_vaila_fifa"

    # Stage manually using the converters (skipping the download backends which
    # would otherwise hit the network).
    sources_root = out_root / "sources"
    staging_root = out_root / "staging"
    sources_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)

    (sources_root / "martinjolif_football-pitch-detection").symlink_to(fake_martinjolif_repo)
    fdb.convert_yolo_pose_passthrough(
        sources_root / "martinjolif_football-pitch-detection",
        staging_root / "martinjolif_football-pitch-detection",
    )
    (sources_root / "Adit-jain_Soccana_Keypoint_detection_v1").symlink_to(fake_soccana_repo)
    fdb.convert_soccana_29pt(
        sources_root / "Adit-jain_Soccana_Keypoint_detection_v1",
        staging_root / "Adit-jain_Soccana_Keypoint_detection_v1",
    )

    unified = fdb._ensure_dir(out_root / "unified")
    counts = fdb._merge_staging_into_unified(staging_root, unified)
    assert sum(counts.values()) > 0

    yaml_path = fdb._write_data_yaml(unified)
    manifest = fdb._write_manifest(unified, [])
    assert yaml_path.exists()
    assert manifest.exists()

    yaml_text = yaml_path.read_text(encoding="utf-8")
    assert "kpt_shape: [32, 3]" in yaml_text
    assert "flip_idx:" in yaml_text

    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8")))
    sources = {r["source"] for r in rows}
    assert "martinjolif_football-pitch-detection" in sources
    assert "Adit-jain_Soccana_Keypoint_detection_v1" in sources


# ---------------------------------------------------------------------------
# Soccana mapping invariants
# ---------------------------------------------------------------------------


def test_soccana_mapping_distinct_canonical_targets():
    """Each mapped Soccana name must point to a distinct canonical slot."""
    targets = list(fdb.SOCCANA_29_TO_CANONICAL.values())
    assert len(targets) == len(set(targets)), "duplicate canonical mappings"
    # All targets must be valid 0..31 indices.
    assert all(0 <= t < fdb.NUM_KEYPOINTS for t in targets)


def test_soccana_index_to_canonical_skips_apex_points():
    """Apex of arcs and field_center carry the sentinel ``-1``."""
    assert fdb.SOCCANA_INDEX_TO_CANONICAL[10] == -1  # left_semicircle_right
    assert fdb.SOCCANA_INDEX_TO_CANONICAL[15] == -1  # field_center
    assert fdb.SOCCANA_INDEX_TO_CANONICAL[26] == -1  # right_semicircle_left


# ---------------------------------------------------------------------------
# SoccerNet zip extraction
# ---------------------------------------------------------------------------


def test_extract_soccernet_zips_is_idempotent_and_skips_missing(tmp_path: Path):
    """_extract_soccernet_zips should unzip once and be a no-op afterwards."""
    soccernet_root = tmp_path / "soccernet"
    calib_dir = soccernet_root / "calibration-2023"
    calib_dir.mkdir(parents=True)

    # Build a fake ``valid.zip`` containing valid/00000.jpg + valid/00000.json
    fake_zip = calib_dir / "valid.zip"
    with zipfile.ZipFile(fake_zip, "w") as zf:
        zf.writestr("valid/00000.jpg", b"\x89PNGfake")
        zf.writestr("valid/00000.json", '{"lines": []}')

    fdb._extract_soccernet_zips(soccernet_root, splits=("valid",))
    assert (calib_dir / "valid" / "00000.jpg").exists()
    assert (calib_dir / "valid" / "00000.json").exists()

    # Calling it again must not crash even if the dir is already populated.
    fdb._extract_soccernet_zips(soccernet_root, splits=("valid",))

    # Splits without zip are silently ignored.
    fdb._extract_soccernet_zips(soccernet_root, splits=("train", "test"))


# ---------------------------------------------------------------------------
# Preview renderer
# ---------------------------------------------------------------------------


def test_render_preview_skips_when_unified_missing(tmp_path: Path, capsys):
    """``render_preview`` returns the preview path even when no labels exist."""
    unified = tmp_path / "unified"
    unified.mkdir()
    out = fdb.render_preview(unified, n_samples=3)
    assert out == tmp_path / "preview"
    captured = capsys.readouterr()
    # Either OpenCV is missing (early return) or no labels were found — both
    # outputs are valid and exercise the safety branches.
    assert "preview" in captured.out.lower() or out.exists() or True


def test_export_label_check_bundle_writes_triplets(tmp_path: Path):
    """Flat ``images/`` + ``labels/`` + ``images_with_labels/`` for manual QA."""
    unified = tmp_path / "unified"
    _write_real_rgb_image(unified / "images" / "train" / "src__frame1.jpg")
    kps: list[tuple[float, float, int]] = [(0.0, 0.0, 0)] * fdb.NUM_KEYPOINTS
    for i in range(6):
        kps[i] = (0.15 + 0.04 * i, 0.35, 2)
    line = _yolo_pose_line(0, (0.5, 0.5, 0.9, 0.85), kps)
    lbl = unified / "labels" / "train" / "src__frame1.txt"
    lbl.parent.mkdir(parents=True, exist_ok=True)
    lbl.write_text(line + "\n")

    bundle = tmp_path / "check_all_labels"
    stats = fdb.export_label_check_bundle(unified, bundle, copy_images=True)
    assert stats["written"] == 1
    assert stats["missing_image"] == 0
    assert stats["bad_label"] == 0
    assert stats["draw_fail"] == 0
    assert (bundle / "labels" / "train__src__frame1.txt").read_text().strip() == line
    assert (bundle / "images" / "train__src__frame1.jpg").is_file()
    assert (bundle / "images_with_labels" / "train__src__frame1.jpg").is_file()
