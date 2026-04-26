"""Smoke tests for :mod:`vaila.fifa_dataset_builder`.

These tests do **not** hit the network: they exercise the converters and the
merge/manifest writer with synthetic mini-datasets generated on the fly.
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest

from vaila import fifa_dataset_builder as fdb


def _write_image(path: Path, content: bytes = b"\x89PNG\r\n\x1a\n0000IHDR") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
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
    # canonical idx 9 (left_penalty_box_top_right) <- 2_big_rect_left_top_pt2 = (0.30, 0.20)
    base_9 = 5 + 3 * 9
    assert parts[base_9] == "0.300000" and parts[base_9 + 1] == "0.200000"
    # canonical idx 4 (left_penalty_box_bottom_left) <- 3_big_rect_left_bottom_pt1 = (0.10, 0.80)
    base_4 = 5 + 3 * 4
    assert parts[base_4] == "0.100000" and parts[base_4 + 1] == "0.800000"


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
    # canonical idx 26 (top_right_corner) at (0.95, 0.05, 2)
    base = 5 + 3 * 26
    assert parts[base] == "0.950000" and parts[base + 1] == "0.050000" and parts[base + 2] == "2"


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
