"""Tests for :mod:`vaila.sam_postprocess`."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from vaila.sam_postprocess import (
    discover_sam_run,
    extract_points_for_batch,
    extract_points_from_sam_run,
    frame_size,
    mask_centroid,
    read_sam_meta,
)

W, H = 800, 600


def _write_circle_mask(path: Path, cx: int, cy: int, radius: int, w: int = W, h: int = H) -> None:
    img = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(img, (cx, cy), radius, 255, thickness=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _write_overlay_mp4(path: Path, n_frames: int = 3, w: int = W, h: int = H) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (w, h))
    try:
        for _ in range(n_frames):
            writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    finally:
        writer.release()


def _build_synthetic_run(root: Path) -> Path:
    sam_dir = root / "BRA_KOR_test"
    sam_dir.mkdir(parents=True)

    _write_overlay_mp4(sam_dir / "BRA_KOR_test_sam_overlay.mp4", n_frames=3)

    cells_id0_f0 = (0.10, 0.10, 0.10, 0.10)
    cells_id0_f1 = (0.20, 0.20, 0.10, 0.10)
    cells_id1_f0 = (0.50, 0.50, 0.20, 0.20)

    def fmt(t):
        return ",".join(f"{v:.6f}" for v in t)

    header = "frame,box_x_0,box_y_0,box_w_0,box_h_0,prob_0,box_x_1,box_y_1,box_w_1,box_h_1,prob_1"
    rows = [
        f"0,{fmt(cells_id0_f0)},0.9,{fmt(cells_id1_f0)},0.8",
        f"1,{fmt(cells_id0_f1)},0.7,,,,,",
        "2,,,,,,,,,,",
    ]
    (sam_dir / "sam_frames_meta.csv").write_text(header + "\n" + "\n".join(rows) + "\n")

    masks_dir = sam_dir / "masks"
    cx0_f0 = int(round(0.10 * W + 0.10 * W * 0.5))
    cy0_f0 = int(round(0.10 * H + 0.10 * H * 0.5))
    _write_circle_mask(masks_dir / "frame_000000_obj_0.png", cx0_f0, cy0_f0, radius=15)

    cx1_f0 = int(round(0.50 * W + 0.20 * W * 0.5))
    cy1_f0 = int(round(0.50 * H + 0.20 * H * 0.5))
    _write_circle_mask(masks_dir / "frame_000000_obj_1.png", cx1_f0, cy1_f0, radius=25)

    cx0_f1 = int(round(0.20 * W + 0.10 * W * 0.5))
    cy0_f1 = int(round(0.20 * H + 0.10 * H * 0.5))
    _write_circle_mask(masks_dir / "frame_000001_obj_0.png", cx0_f1, cy0_f1, radius=15)

    return sam_dir


class TestDiscoverAndRead:
    def test_discover(self, tmp_path: Path) -> None:
        sam_dir = _build_synthetic_run(tmp_path)
        art = discover_sam_run(sam_dir)
        assert art.meta_csv.is_file()
        assert art.masks_dir.is_dir()
        assert art.overlay_mp4 is not None
        assert art.stem == "BRA_KOR_test"

    def test_read_meta(self, tmp_path: Path) -> None:
        sam_dir = _build_synthetic_run(tmp_path)
        df, oids = read_sam_meta(sam_dir)
        assert oids == [0, 1]
        assert len(df) == 3
        assert pd.isna(df.loc[1, "box_x_1"])
        assert pd.isna(df.loc[2, "box_x_0"])

    def test_frame_size_from_overlay(self, tmp_path: Path) -> None:
        sam_dir = _build_synthetic_run(tmp_path)
        art = discover_sam_run(sam_dir)
        assert frame_size(art) == (W, H)

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discover_sam_run(tmp_path / "nope")

    def test_missing_meta_raises(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            discover_sam_run(d)


class TestMaskCentroid:
    def test_circle_centroid(self, tmp_path: Path) -> None:
        png = tmp_path / "m.png"
        _write_circle_mask(png, cx=300, cy=200, radius=40)
        cx, cy = mask_centroid(png)
        assert cx == pytest.approx(300.0, abs=0.5)
        assert cy == pytest.approx(200.0, abs=0.5)

    def test_empty_mask(self, tmp_path: Path) -> None:
        png = tmp_path / "empty.png"
        cv2.imwrite(str(png), np.zeros((100, 100), dtype=np.uint8))
        assert mask_centroid(png) is None

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert mask_centroid(tmp_path / "missing.png") is None


class TestExtractPointsAll:
    def test_all_columns_and_values(self, tmp_path: Path) -> None:
        sam_dir = _build_synthetic_run(tmp_path)
        out = extract_points_from_sam_run(sam_dir, mode="all", canonical="foot")
        assert out == sam_dir / "sam_points.csv"
        df = pd.read_csv(out)

        expected = [
            "frame",
            "p1_x",
            "p1_y",
            "p1_cx",
            "p1_cy",
            "p1_mx",
            "p1_my",
            "p2_x",
            "p2_y",
            "p2_cx",
            "p2_cy",
            "p2_mx",
            "p2_my",
        ]
        assert list(df.columns) == expected

        row0 = df.loc[0]
        assert row0["p1_x"] == pytest.approx(0.10 * W + 0.10 * W * 0.5, abs=1e-3)
        assert row0["p1_y"] == pytest.approx(0.10 * H + 0.10 * H, abs=1e-3)
        assert row0["p1_cx"] == pytest.approx(0.10 * W + 0.10 * W * 0.5, abs=1e-3)
        assert row0["p1_cy"] == pytest.approx(0.10 * H + 0.10 * H * 0.5, abs=1e-3)
        cx0 = int(round(0.10 * W + 0.10 * W * 0.5))
        cy0 = int(round(0.10 * H + 0.10 * H * 0.5))
        assert row0["p1_mx"] == pytest.approx(cx0, abs=1.0)
        assert row0["p1_my"] == pytest.approx(cy0, abs=1.0)

        row1 = df.loc[1]
        assert pd.isna(row1["p2_x"])
        assert pd.isna(row1["p2_mx"])

        row2 = df.loc[2]
        for col in expected[1:]:
            assert pd.isna(row2[col])

    def test_id_map(self, tmp_path: Path) -> None:
        sam_dir = _build_synthetic_run(tmp_path)
        extract_points_from_sam_run(sam_dir, mode="all")
        id_map = pd.read_csv(sam_dir / "sam_id_map.csv")
        assert list(id_map.columns) == ["pN", "obj_id", "n_frames", "first_frame", "last_frame"]
        row = id_map.set_index("pN").loc[1]
        assert row["obj_id"] == 0
        assert row["n_frames"] == 2
        assert row["first_frame"] == 0
        assert row["last_frame"] == 1
        row2 = id_map.set_index("pN").loc[2]
        assert row2["n_frames"] == 1


class TestExtractPointsFootOnly:
    def test_foot_mode_excludes_extras(self, tmp_path: Path) -> None:
        sam_dir = _build_synthetic_run(tmp_path)
        out = extract_points_from_sam_run(sam_dir, mode="foot")
        df = pd.read_csv(out)
        assert list(df.columns) == ["frame", "p1_x", "p1_y", "p2_x", "p2_y"]


class TestExtractBatch:
    def test_batch_runs_all_subdirs(self, tmp_path: Path) -> None:
        batch = tmp_path / "processed_sam_TEST"
        batch.mkdir()
        run_a = _build_synthetic_run(batch)
        run_a.rename(batch / "video_a")
        run_b = _build_synthetic_run(batch)
        run_b.rename(batch / "video_b")
        outs = extract_points_for_batch(batch)
        assert len(outs) == 2
        for o in outs:
            assert o.name == "sam_points.csv"

    def test_batch_skips_non_sam_dirs(self, tmp_path: Path) -> None:
        batch = tmp_path / "processed_sam_TEST"
        batch.mkdir()
        run_a = _build_synthetic_run(batch)
        run_a.rename(batch / "video_a")
        (batch / "junk").mkdir()
        outs = extract_points_for_batch(batch)
        assert len(outs) == 1
