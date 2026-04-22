"""Tests for :mod:`vaila.sam_validate`."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from vaila.sam_validate import (
    associate_ids_hungarian,
    iou_xyxy,
    load_dlt2d_params,
    validate_sam_run,
)

W, H = 800, 600
BODY25 = 25
BODY25_MIDHIP = 8


def _identity_dlt2d_csv(path: Path) -> Path:
    df = pd.DataFrame(
        [[0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        columns=["frame", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"],
    )
    df.to_csv(path, index=False)
    return path


class TestIoU:
    def test_perfect_overlap(self) -> None:
        a = np.array([10.0, 10.0, 20.0, 20.0])
        assert iou_xyxy(a, a) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        a = np.array([0.0, 0.0, 5.0, 5.0])
        b = np.array([10.0, 10.0, 15.0, 15.0])
        assert iou_xyxy(a, b) == 0.0

    def test_half_overlap(self) -> None:
        a = np.array([0.0, 0.0, 10.0, 10.0])
        b = np.array([5.0, 0.0, 15.0, 10.0])
        assert iou_xyxy(a, b) == pytest.approx(50.0 / 150.0, abs=1e-9)


class TestHungarian:
    def test_matches_diagonal(self) -> None:
        iou_per_frame = {
            0: np.array([[1.0, 0.0], [0.0, 1.0]]),
            1: np.array([[1.0, 0.0], [0.0, 1.0]]),
        }
        sam_ids = {0: [0, 1], 1: [0, 1]}
        mapping, switches, ious = associate_ids_hungarian(iou_per_frame, sam_ids, fifa_n_players=2)
        assert mapping == {0: 0, 1: 1}
        assert switches == 0
        assert len(ious) == 4
        assert all(v == 1.0 for v in ious)

    def test_majority_vote_with_one_swap(self) -> None:
        iou_per_frame = {
            0: np.array([[1.0, 0.0], [0.0, 1.0]]),
            1: np.array([[1.0, 0.0], [0.0, 1.0]]),
            2: np.array([[0.0, 1.0], [1.0, 0.0]]),
        }
        sam_ids = {0: [10, 11], 1: [10, 11], 2: [10, 11]}
        mapping, switches, _ = associate_ids_hungarian(iou_per_frame, sam_ids, fifa_n_players=2)
        assert mapping == {10: 0, 11: 1}
        assert switches == 2


class TestLoadDLT2D:
    def test_parse_identity(self, tmp_path: Path) -> None:
        path = _identity_dlt2d_csv(tmp_path / "calib.dlt2d")
        params = load_dlt2d_params(path)
        np.testing.assert_allclose(params, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def _build_synth_run(
    tmp_path: Path,
    *,
    stem: str = "TEST_MATCH",
    n_frames: int = 4,
    p_fifa: int = 2,
) -> tuple[Path, Path]:
    sam_dir = tmp_path / "sam_run" / stem
    sam_dir.mkdir(parents=True)
    masks_dir = sam_dir / "masks"
    masks_dir.mkdir()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(sam_dir / f"{stem}_sam_overlay.mp4"), fourcc, 25.0, (W, H))
    try:
        for _ in range(n_frames):
            writer.write(np.zeros((H, W, 3), dtype=np.uint8))
    finally:
        writer.release()

    fifa_root = tmp_path / "fifa"
    (fifa_root / "boxes").mkdir(parents=True)
    (fifa_root / "skel_2d").mkdir(parents=True)
    (fifa_root / "skel_3d").mkdir(parents=True)
    (fifa_root / "cameras").mkdir(parents=True)

    fifa_boxes = np.zeros((n_frames, p_fifa, 4), dtype=float)
    fifa_skel2d = np.zeros((n_frames, p_fifa, BODY25, 2), dtype=float)
    fifa_skel3d = np.zeros((n_frames, p_fifa, BODY25, 3), dtype=float)

    box_specs = [
        (0.10, 0.10, 0.10, 0.10),
        (0.50, 0.50, 0.20, 0.20),
    ]

    header = "frame," + ",".join(
        f"box_x_{oid},box_y_{oid},box_w_{oid},box_h_{oid},prob_{oid}" for oid in (0, 1)
    )
    rows = [header]

    for f in range(n_frames):
        cells = [str(f)]
        for oid, (bx, by, bw, bh) in enumerate(box_specs):
            cells.append(f"{bx:.6f},{by:.6f},{bw:.6f},{bh:.6f},0.9")
            x1, y1 = bx * W, by * H
            x2, y2 = (bx + bw) * W, (by + bh) * H
            fifa_boxes[f, oid] = [x1, y1, x2, y2]
            mid_x_pixel = (x1 + x2) * 0.5
            mid_y_pixel = (y1 + y2) * 0.5
            fifa_skel2d[f, oid, BODY25_MIDHIP] = [mid_x_pixel, mid_y_pixel]
            fifa_skel3d[f, oid, BODY25_MIDHIP] = [mid_x_pixel, mid_y_pixel + bh * H * 0.5, 0.0]

            mask_img = np.zeros((H, W), dtype=np.uint8)
            cv2.rectangle(mask_img, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
            cv2.imwrite(str(masks_dir / f"frame_{f:06d}_obj_{oid}.png"), mask_img)

        rows.append(",".join(cells))

    (sam_dir / "sam_frames_meta.csv").write_text("\n".join(rows) + "\n")

    np.save(fifa_root / "boxes" / f"{stem}.npy", fifa_boxes)
    np.save(fifa_root / "skel_2d" / f"{stem}.npy", fifa_skel2d)
    np.save(fifa_root / "skel_3d" / f"{stem}.npy", fifa_skel3d)

    from vaila.sam_postprocess import extract_points_from_sam_run

    extract_points_from_sam_run(sam_dir, mode="all", canonical="foot")

    return sam_dir, fifa_root


class TestValidateSamRun:
    def test_identity_homography_zero_error(self, tmp_path: Path) -> None:
        sam_dir, fifa_root = _build_synth_run(tmp_path, stem="X")
        dlt = _identity_dlt2d_csv(tmp_path / "X.dlt2d")
        out = tmp_path / "out"
        res = validate_sam_run(
            sam_dir=sam_dir,
            fifa_data=fifa_root,
            match_stem="X",
            dlt2d=dlt,
            out_dir=out,
            iou_threshold=0.1,
        )
        assert res.overall_iou == pytest.approx(1.0, abs=1e-6)
        assert res.n_id_switches == 0
        assert res.summary_csv.is_file()
        assert res.report_html.is_file()
        df = pd.read_csv(res.summary_csv)
        overall_2d = df.query("metric == 'rmse_skel2d_m' and scope == 'all'")["value"].iloc[0]
        # FIFA skel_2d mid-hip lies at the BBOX CENTER while SAM foot lies at the
        # bbox bottom-center. With identity DLT2D the per-pair vertical error
        # equals half the bbox height (in metres in our identity world).
        h_small = 0.10 * H
        h_big = 0.20 * H
        expected_rmse_2d = float(np.sqrt(((h_small * 0.5) ** 2 + (h_big * 0.5) ** 2) / 2.0))
        assert overall_2d == pytest.approx(expected_rmse_2d, abs=1e-3)
        overall_3d = df.query("metric == 'rmse_skel3d_m' and scope == 'all'")["value"].iloc[0]
        # FIFA skel_3d mid-hip in our synth dataset is placed at the SAM foot
        # location, so identity DLT2D gives zero error.
        assert overall_3d == pytest.approx(0.0, abs=1e-6)

    def test_unmatched_box_drops_to_zero_iou(self, tmp_path: Path) -> None:
        sam_dir, fifa_root = _build_synth_run(tmp_path, stem="Y")
        boxes = np.load(fifa_root / "boxes" / "Y.npy")
        boxes[:, 0, :] = [0.0, 0.0, 1.0, 1.0]
        np.save(fifa_root / "boxes" / "Y.npy", boxes)
        dlt = _identity_dlt2d_csv(tmp_path / "Y.dlt2d")
        out = tmp_path / "out_y"
        res = validate_sam_run(
            sam_dir=sam_dir,
            fifa_data=fifa_root,
            match_stem="Y",
            dlt2d=dlt,
            out_dir=out,
            iou_threshold=0.1,
        )
        assert res.n_matched_frames < 8
