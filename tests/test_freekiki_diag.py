"""Tests for vaila/freekiki_diag.py (point identity, metrics, homography, temporal, gate).

Update Date: 28 September 2026
Version: 0.4.5
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from vaila import freekiki as fk
from vaila import freekiki_diag as diag

NKP = diag.NKP


# --------------------------------------------------------------------------- #
# Point order / identity
# --------------------------------------------------------------------------- #
def test_schema_order_and_names_agree_everywhere() -> None:
    names, flip, xyz = diag.load_field_points()
    fk_names, fk_flip = fk.load_schema()
    assert len(names) == NKP == fk.NKP == 49
    assert names == fk_names and flip.tolist() == list(fk_flip)
    assert names[0] == "top_left_corner" and names[5] == "bottom_left_corner"
    assert xyz.shape == (NKP, 3)
    # "top" is the far touchline (y > 0); elevated points have z > 0
    assert xyz[0, 1] > 0 > xyz[5, 1]
    for top, bottom in diag.ABOVE_PAIRS:
        assert xyz[top, 2] > 0 and xyz[bottom, 2] == 0
        assert np.allclose(xyz[top, :2], xyz[bottom, :2])


def test_flip_and_rot180_maps_are_involutions() -> None:
    _, flip, xyz = diag.load_field_points()
    mirror, rot = diag.symmetry_maps(xyz)
    ids = np.arange(NKP)
    assert np.all(flip[flip] == ids)
    assert np.all(flip == mirror), "flip_idx must be the x -> -x geometric mirror"
    assert np.all(rot >= 0) and np.all(rot[rot] == ids)
    assert rot[5] == 24 and rot[29] == 0  # bottom-left corner <-> top-right corner
    assert rot[39] == 46 and rot[47] == 38  # flags follow their corners


# --------------------------------------------------------------------------- #
# Matching and metric denominators
# --------------------------------------------------------------------------- #
def _one_image():
    gt = np.zeros((1, NKP, 2))
    vis = np.zeros((1, NKP))
    vis[0, :4] = 2  # p0..p3 labelled
    pred = gt.copy()
    pred[0, 1] = [3.0, 4.0]  # 5 px at 960 wide -> 10 px@1920
    pred[0, 2] = [30.0, 40.0]  # 50 px -> 100 px@1920 -> mislocalized
    conf = np.zeros((1, NKP))
    conf[0, [0, 1, 2, 7]] = 0.9  # p3 missed, p7 false positive
    return pred, conf, gt, vis


def test_match_keypoints_statuses_and_px_scale() -> None:
    pred, conf, gt, vis = _one_image()
    status, dist = diag.match_keypoints(pred, conf, gt, vis, [960], kp_conf=0.5, match_px=25)
    assert status[0, :4].tolist() == [diag.TP, diag.TP, diag.MIS, diag.FN]
    assert status[0, 7] == diag.FP and status[0, 8] == diag.TN
    assert dist[0, 1] == pytest.approx(10.0) and dist[0, 2] == pytest.approx(100.0)
    # a missing point is undefined, never zero error
    assert np.isnan(dist[0, 3]) and np.isnan(dist[0, 7]) and np.isnan(dist[0, 8])


def test_no_detection_means_all_labelled_points_are_misses() -> None:
    _, _, gt, vis = _one_image()
    conf = np.full((1, NKP), np.nan)  # image without an accepted box
    status, dist = diag.match_keypoints(gt, conf, gt, vis, [1920], 0.5, 25)
    assert (status[0, :4] == diag.FN).all() and np.isnan(dist).all()


def test_keypoint_table_denominators() -> None:
    pred, conf, gt, vis = _one_image()
    status, dist = diag.match_keypoints(pred, conf, gt, vis, [960], 0.5, 25)
    names, _, _ = diag.load_field_points()
    overall, rows = diag.keypoint_table(status, dist, names, pck=(5, 10, 25))
    assert (overall["n_ref"], overall["n_pred"], overall["n_tp"]) == (4, 4, 2)
    assert (overall["n_mislocalized"], overall["n_fn"], overall["n_fp"]) == (1, 1, 1)
    assert overall["recall"] == 0.5 and overall["precision"] == 0.5
    # PCK over matched pairs (TP + mislocalized = 3) vs over all references (4)
    assert overall["pck10_pred"] == pytest.approx(2 / 3, abs=1e-4)
    assert overall["pck10_all"] == 0.5
    assert overall["pck5_all"] == 0.25  # only p0 (0 px)
    assert overall["err_median"] == 5.0  # TP errors only: 0 and 10
    assert overall["err_any_median"] == 10.0
    assert overall["units"] == "px@1920"
    assert rows[3]["n_ref"] == 1 and rows[3]["recall"] == 0.0
    assert rows[3]["err_median"] is None and rows[3]["pck10_pred"] is None
    assert rows[3]["pck10_all"] == 0.0
    assert rows[10]["n_ref"] == 0 and rows[10]["recall"] is None
    assert rows[7]["n_fp"] == 1 and rows[7]["precision"] == 0.0


def test_identity_swaps_flags_rot180() -> None:
    names, _, xyz = diag.load_field_points()
    mirror, rot = diag.symmetry_maps(xyz)
    gt = np.zeros((1, NKP, 2))
    vis = np.zeros((1, NKP))
    vis[0, 5] = 2
    gt[0, 5] = [100.0, 900.0]
    pred = np.full((1, NKP, 2), 5000.0)
    pred[0, 24] = [102.0, 901.0]  # the network calls the bottom-left corner p24
    conf = np.zeros((1, NKP))
    conf[0, 24] = 0.9
    status, _ = diag.match_keypoints(pred, conf, gt, vis, [1920], 0.5, 25)
    codes, found = diag.identity_swaps(pred, conf >= 0.5, gt, status, [1920], 25, mirror, rot)
    assert status[0, 5] == diag.FN
    assert codes[0, 5] == diag.SWAP_ROT180 and found[0, 5] == 24
    _, rows = diag.keypoint_table(status, np.full(status.shape, np.nan), names, swaps=codes)
    assert rows[5]["swap_rot180"] == 1 and rows[5]["swap_mirror"] == 0


# --------------------------------------------------------------------------- #
# Homography criteria
# --------------------------------------------------------------------------- #
def _camera(world_xy: np.ndarray, sign_u: float = 1.0) -> np.ndarray:
    H = np.array([[15.0 * sign_u, 0.0, 960.0], [0.0, -12.0, 540.0], [0.0, 0.002, 1.0]])
    ph = np.c_[world_xy, np.ones(len(world_xy))] @ H.T
    return ph[:, :2] / ph[:, 2:3]


def test_fit_field_homography_ok_mirrored_degenerate_few() -> None:
    _, _, xyz = diag.load_field_points()
    planar = np.flatnonzero(xyz[:, 2] == 0)
    world = xyz[planar, :2]
    fit = diag.fit_field_homography(_camera(world), world)
    assert fit["status"] == "ok" and fit["n_inliers"] == len(planar)
    assert fit["rmse_px"] < 0.5 and fit["orientation"] == diag.EXPECTED_ORIENTATION

    mirrored = diag.fit_field_homography(_camera(world, sign_u=-1.0), world)
    assert mirrored["status"] == "mirrored"

    goal_line = [i for i in planar if xyz[i, 0] == -52.45]  # collinear
    assert len(goal_line) >= 4
    w = xyz[goal_line, :2]
    assert diag.fit_field_homography(_camera(w), w)["status"] == "degenerate"

    assert diag.fit_field_homography(_camera(world[:3]), world[:3])["status"] == "few_points"
    four = world[[0, 5, 24, 29]]  # 4 points always fit: redundancy required
    assert diag.fit_field_homography(_camera(four), four)["status"] == "few_inliers"


def test_frame_homography_uses_only_accepted_planar_points() -> None:
    _, _, xyz = diag.load_field_points()
    planar = xyz[:, 2] == 0
    xy = np.full((NKP, 2), np.nan)
    xy[planar] = _camera(xyz[planar, :2])
    xy[~planar] = [5000.0, 5000.0]  # elevated points must never enter the fit
    conf = np.where(planar, 0.9, 0.99)
    fit = diag.frame_homography(xy, conf, 0.5, planar, xyz[:, :2], 1920)
    assert fit["status"] == "ok" and fit["n_points"] == int(planar.sum())
    low = diag.frame_homography(xy, np.full(NKP, 0.1), 0.5, planar, xyz[:, :2], 1920)
    assert low["status"] == "few_points"


def test_px_at_1920_scaling_in_homography() -> None:
    _, _, xyz = diag.load_field_points()
    planar = np.flatnonzero(xyz[:, 2] == 0)
    world = xyz[planar, :2]
    img = _camera(world) / 2.0  # same view at 960 px width
    img[0] += [2.0, 0.0]  # 2 px at 960 = 4 px@1920 on one point
    fit = diag.fit_field_homography(img, world, width=960)
    assert fit["status"] == "ok"
    full = diag.fit_field_homography(img * 2.0, world, width=1920)
    assert fit["rmse_px"] == pytest.approx(full["rmse_px"], rel=1e-2)


# --------------------------------------------------------------------------- #
# Temporal behaviour
# --------------------------------------------------------------------------- #
def _pan(frames: int) -> np.ndarray:
    _, _, xyz = diag.load_field_points()
    base = np.full((NKP, 2), np.nan)
    base[:8] = _camera(xyz[:8, :2] * 0.3)
    return np.stack([base + [7.0 * t, 0.0] for t in range(frames)])


def test_temporal_residual_removes_camera_motion() -> None:
    _, _, xyz = diag.load_field_points()
    pts = _pan(5)
    cuts = np.zeros(5, dtype=bool)
    m = diag.temporal_metrics(pts, cuts, xyz[:, 2] == 0, 1920)
    assert m["displacement_median_px"] == pytest.approx(7.0, abs=0.01)
    assert m["residual_median_px"] == pytest.approx(0.0, abs=0.01)
    assert m["pairs_homography"] == 4 and m["pairs_across_cut"] == 0

    cuts[2] = True
    m = diag.temporal_metrics(pts, cuts, xyz[:, 2] == 0, 1920)
    assert m["pairs_across_cut"] == 1 and m["pairs_homography"] == 3

    empty = diag.temporal_metrics(np.full((3, NKP, 2), np.nan), np.zeros(3, bool), cuts, 1920)
    assert empty["residual_median_px"] is None and empty["pairs_skipped"] == 2


def test_fill_short_gaps_marks_and_respects_cuts_and_edges() -> None:
    pts = np.full((9, NKP, 2), np.nan)
    track = np.array([0, 1, np.nan, np.nan, 4, np.nan, np.nan, np.nan, 8], dtype=float)
    pts[:, 0, 0] = pts[:, 0, 1] = track
    pts[:, 1, 0] = pts[:, 1, 1] = [np.nan, 1, 2, 3, np.nan, np.nan, np.nan, np.nan, np.nan]
    cuts = np.zeros(9, dtype=bool)
    filled, interp = diag.fill_short_gaps(pts, cuts, max_gap=2)
    assert filled[:5, 0, 0].tolist() == [0, 1, 2, 3, 4]
    assert interp[:, 0].tolist() == [False, False, True, True, False, False, False, False, False]
    assert np.isnan(filled[5:8, 0]).all()  # gap of 3 > max_gap stays missing
    assert np.isnan(filled[0, 1]).all() and np.isnan(filled[4:, 1]).all()  # no extrapolation
    assert np.isnan(pts[2, 0]).all()  # input untouched

    cuts[3] = True  # the short gap now spans a cut
    filled, interp = diag.fill_short_gaps(pts, cuts, max_gap=2)
    assert np.isnan(filled[2:4, 0]).all() and not interp[:, 0].any()
    same, none = diag.fill_short_gaps(pts, cuts, max_gap=0)
    assert not none.any() and np.array_equal(same, pts, equal_nan=True)


def test_duplicate_points_detects_identity_clash() -> None:
    xy = np.zeros((NKP, 2))
    xy[:, 0] = np.arange(NKP) * 50.0
    xy[7] = xy[3] + [0.5, 0.5]  # two indices on the same pixel
    accepted = np.ones(NKP, dtype=bool)
    assert diag.duplicate_points(xy, accepted, 1920) == [(3, 7)]
    accepted[7] = False
    assert diag.duplicate_points(xy, accepted, 1920) == []


def test_calibration_table_valid_but_wrong_layout_is_not_correct() -> None:
    # A model that always outputs one plausible field layout gives a valid
    # homography on every image; only the label check shows it is wrong.
    _, _, xyz = diag.load_field_points()
    planar = xyz[:, 2] == 0
    gt = np.zeros((NKP, 2))
    gt[planar] = _camera(xyz[planar, :2])
    shifted = gt.copy()
    shifted[planar] += [300.0, 0.0]  # same shape, wrong place
    pred = {
        "width": [1920.0, 1920.0],
        "gt_xy": [gt, gt],
        "gt_vis": [np.where(planar, 2, 0)] * 2,
        "pred_xy": [gt, shifted],
    }
    kc = np.tile(np.where(planar, 0.9, 0.0), (2, 1))
    c = diag.calibration_table(pred, kc, 0.5)
    assert c["n_images_gt_calibratable"] == 2
    assert c["ok_rate"] == 1.0 and c["correct_rate"] == 0.5
    assert [row[1] for row in c["per_image"]] == ["ok", "ok"]
    assert c["per_image"][1][2] > diag.CALIB_CORRECT_PX


# --------------------------------------------------------------------------- #
# Promotion gate
# --------------------------------------------------------------------------- #
def _fake_eval(root: Path, *, pmap: float, pck: float, calib: float, kp: dict, split="val"):
    root.mkdir(parents=True)
    summary = {
        "model": str(root / "m.pt"),
        "split": split,
        "n_images": 10,
        "det_conf": 0.25,
        "kp_conf": 0.5,
        "match_px": 25,
        "images_sha1": diag.names_digest(["a.jpg", "b.jpg"]),
        "ultralytics": {"metrics/mAP50-95(P)": pmap},
        "keypoints": {"pck10_all": pck},
        "eval_schema": diag.EVAL_SCHEMA,
        "calib": {"correct_rate": calib},
    }
    (root / "eval_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (root / "per_keypoint.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["kp", "n_ref", "recall", "err_median"])
        w.writeheader()
        for i in range(NKP):
            n_ref, recall, err = kp.get(i, (100, 0.9, 4.0))
            w.writerow({"kp": f"p{i}", "n_ref": n_ref, "recall": recall, "err_median": err})
    return root


def test_compare_evals_blocks_critical_regression(tmp_path: Path) -> None:
    base = _fake_eval(tmp_path / "b", pmap=0.85, pck=0.86, calib=0.9, kp={16: (10, 0.5, 9.0)})
    better = _fake_eval(tmp_path / "c1", pmap=0.87, pck=0.88, calib=0.92, kp={16: (10, 0.1, 30.0)})
    ok = diag.compare_evals(base, better)
    assert ok["promote"], ok["reasons"]
    assert any(n.startswith("p16: support 10") for n in ok["notes"])  # not gated
    # a better mean hiding a collapsed critical point is rejected
    hidden = _fake_eval(tmp_path / "c2", pmap=0.88, pck=0.89, calib=0.93, kp={5: (100, 0.5, 4.0)})
    bad = diag.compare_evals(base, hidden)
    assert not bad["promote"] and any(r.startswith("p5_recall") for r in bad["reasons"])
    loose = diag.compare_evals(base, hidden, {"max_recall_drop": 0.5})
    assert loose["promote"]


def test_compare_evals_refuses_different_sets(tmp_path: Path) -> None:
    base = _fake_eval(tmp_path / "b", pmap=0.85, pck=0.86, calib=0.9, kp={})
    other = _fake_eval(tmp_path / "c", pmap=0.99, pck=0.99, calib=1.0, kp={}, split="test")
    d = diag.compare_evals(base, other)
    assert not d["promote"] and any("split differs" in r for r in d["reasons"])


def test_names_digest_is_order_invariant() -> None:
    assert diag.names_digest(["b", "a"]) == diag.names_digest(["a", "b"])
    assert diag.names_digest(["a"]) != diag.names_digest(["a", "b"])


def test_repeat_factors_boost_rare_keypoints_and_respect_cap() -> None:
    vis = np.zeros((40, NKP), dtype=bool)
    vis[:, 0] = True
    vis[0, 5] = True  # f_5 = 1/40 = 0.025
    r_img, r_kp, f_kp = diag.repeat_factors(vis, t=0.05, cap=4.0)
    assert f_kp[5] == pytest.approx(0.025)
    assert r_kp[5] == pytest.approx((0.05 / 0.025) ** 0.5)
    assert r_kp[0] == pytest.approx(1.0)
    assert np.isnan(r_kp[7])
    assert r_img[0] == pytest.approx(r_kp[5])
    assert r_img[1] == pytest.approx(1.0)
    capped, _, _ = diag.repeat_factors(vis, t=0.05, cap=1.0)
    assert capped[0] == pytest.approx(1.0)


def test_stochastic_copies_are_seeded_and_stay_near_the_factor() -> None:
    rates = np.array([1.0, 1.25, 3.0, 4.0])
    a = diag.stochastic_copies(rates, 0)
    b = diag.stochastic_copies(rates, 0)
    assert np.array_equal(a, b)
    assert (a[0], a[2], a[3]) == (1, 3, 4)
    assert a[1] in (1, 2)
    draws = diag.stochastic_copies(np.full(400, 1.5), 2)
    assert set(draws.tolist()) <= {1, 2}
    assert draws.mean() == pytest.approx(1.5, abs=0.08)


# --------------------------------------------------------------------------- #
# Oversampling manifest
# --------------------------------------------------------------------------- #
def test_repeat_factors_rare_point_cap_and_absent() -> None:
    vis = np.zeros((10, NKP), dtype=bool)
    vis[:, 0] = True  # everywhere: f=1, r=1
    vis[0, 5] = True  # one image in ten: f=0.1, r=sqrt(0.4/0.1)=2
    r_img, r_kp, f_kp = diag.repeat_factors(vis, t=0.4, cap=4.0)
    assert f_kp[5] == pytest.approx(0.1) and r_kp[5] == pytest.approx(2.0)
    assert r_kp[0] == 1.0 and np.isnan(r_kp[29])  # never labelled: undefined
    assert r_img[0] == pytest.approx(2.0) and np.all(r_img[1:] == 1.0)
    r_capped, _, _ = diag.repeat_factors(vis, t=0.4, cap=1.5)
    assert r_capped[0] == 1.5
    r_none, _, _ = diag.repeat_factors(np.zeros((3, NKP), dtype=bool), t=0.4, cap=4.0)
    assert np.all(r_none == 1.0)  # images without any visible point keep one copy


def test_stochastic_copies_seeded_and_unbiased() -> None:
    r = np.full(20000, 1.3)
    a = diag.stochastic_copies(r, seed=0)
    assert np.array_equal(a, diag.stochastic_copies(r, seed=0))
    assert set(np.unique(a)) == {1, 2} and a.mean() == pytest.approx(1.3, abs=0.02)
    assert np.all(diag.stochastic_copies(np.array([1.0, 2.0, 3.0]), seed=5) == [1, 2, 3])


def _rfs_workspace(root: Path, n: int = 20) -> Path:
    """Workspace whose dataset has ``n`` train images; p5 is visible only in img00."""
    ws = fk.init_workspace(root)
    ds = fk.dataset_dir(ws)
    (ds / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (ds / "data.yaml").write_text(
        "path: /elsewhere\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"kpt_shape: [{NKP}, 3]\n",
        encoding="utf-8",
    )
    rows = ["split,image,label,source,group"]
    for split, count in (("train", n), ("val", 2)):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
        for i in range(count):
            vis = ["0"] * NKP
            vis[0] = "2"
            if split == "train" and i == 0:
                vis[5] = "2"
            kps = " ".join(f"0.5 0.5 {v}" for v in vis)
            img, lab = f"images/{split}/img{i:02d}.jpg", f"labels/{split}/img{i:02d}.txt"
            (ds / img).write_bytes(b"jpg")
            (ds / lab).write_text(f"0 0.5 0.5 1 1 {kps}\n", encoding="utf-8")
            rows.append(f"{split},{img},{lab},{'rare' if i == 0 else 'main'},g{i % 3}")
    (ds / "manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return ws


def test_build_rfs_manifest_counts_and_exclude(tmp_path: Path) -> None:
    ws = _rfs_workspace(tmp_path / "ws")
    ds = fk.dataset_dir(ws)
    before = {p: p.read_bytes() for p in ds.rglob("*") if p.is_file()}
    info = diag.build_rfs_manifest(ds, tmp_path / "m1", t=0.2, cap=4.0, log=lambda *_: None)
    # f5 = 1/20 -> r = sqrt(0.2/0.05) = 2 exactly: img00 twice, everything else once
    lines = (tmp_path / "m1" / "train_list.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 21 and lines.count("images/train/img00.jpg") == 2
    c = info["counts"]
    assert (c["train_images"], c["entries_after"], c["images_repeated"]) == (20, 21, 1)
    assert info["counts"]["by_source_after"] == {"main": 19, "rare": 2}
    with (tmp_path / "m1" / "kp_counts.csv").open(encoding="utf-8") as f:
        kp = {r["kp"]: r for r in csv.DictReader(f)}
    assert (kp["p5"]["images_before"], kp["p5"]["instances_after"]) == ("1", "2")
    assert kp["p29"]["repeat_factor_kp"] == ""  # undefined stays blank, not 0/1
    with (tmp_path / "m1" / "repeats.csv").open(encoding="utf-8") as f:
        reps = {r["image"]: r for r in csv.DictReader(f)}
    assert reps["images/train/img00.jpg"]["reason"] == "p5 bottom_left_corner"
    # the dataset is read-only for the builder
    assert before == {p: p.read_bytes() for p in ds.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        diag.build_rfs_manifest(ds, tmp_path / "m1", log=lambda *_: None)
    info2 = diag.build_rfs_manifest(
        ds, tmp_path / "m2", t=0.2, exclude={"img00.jpg"}, log=lambda *_: None
    )
    assert info2["counts"]["excluded"] == 1 and info2["counts"]["entries_after"] == 19
    with (tmp_path / "m2" / "repeats.csv").open(encoding="utf-8") as f:
        assert any(r["reason"] == "excluded" and r["copies"] == "0" for r in csv.DictReader(f))


def test_make_manifest_rejects_bad_name_and_reads_csv_exclude(tmp_path: Path) -> None:
    ws = _rfs_workspace(tmp_path / "ws")
    with pytest.raises(ValueError):
        fk.make_manifest(ws, name="../escape")
    ex = tmp_path / "exclude.csv"  # audit CSV format (image column)
    ex.write_text("split,image,issue\ntrain,images/train/img03.jpg,mirrored\n", encoding="utf-8")
    v1 = fk.make_manifest(ws, t=0.2, exclude=ex)
    info = json.loads((v1 / "manifest.json").read_text(encoding="utf-8"))
    assert info["method"]["exclude"]["matched_train_images"] == 1
    assert info["counts"]["kept_images"] == 19
    assert not list((ws / fk.MANIFESTS_DIR).glob(".*partial"))


def test_materialize_manifest_writes_run_yaml_and_detects_changes(tmp_path: Path) -> None:
    ws = _rfs_workspace(tmp_path / "ws")
    fk.make_manifest(ws, t=0.2)
    run = ws / "runs" / "r1"
    yaml_path, sampling = fk.materialize_manifest(ws, "v001", run)
    ds = fk.dataset_dir(ws).resolve()
    text = yaml_path.read_text(encoding="utf-8")
    assert f'train: "{(run / "sampling" / "train.txt").resolve().as_posix()}"' in text
    assert f'path: "{ds.as_posix()}"' in text and "val: images/val" in text
    lines = (run / "sampling" / "train.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == sampling["entries"] == 21
    assert all(Path(p).is_absolute() and Path(p).is_file() for p in lines)
    # train labels edited after the build: refuse
    (ds / "labels" / "train" / "img04.txt").write_text("", encoding="utf-8")
    with pytest.raises(fk.RunStateError, match="labels changed"):
        fk.materialize_manifest(ws, "v001", run)
    with pytest.raises(FileNotFoundError):
        fk.materialize_manifest(ws, "v999", run)


def test_manifest_cli_defaults() -> None:
    a = fk.build_parser().parse_args(["manifest", "-w", "ws"])
    assert (a.rfs_t, a.cap, a.seed, a.exclude, a.name) == (0.05, 4.0, 0, None, None)


def test_train_and_resume_keep_the_manifest_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    import torch

    from vaila import yolotrain

    ws = _rfs_workspace(tmp_path / "ws")
    fk.make_manifest(ws, t=0.2)
    (ws / "models").mkdir(exist_ok=True)
    (ws / "models" / "base.pt").write_bytes(b"w")
    seen: dict = {}

    def fake_train(yaml_file, **kw):
        seen["yaml"] = yaml_file
        return {}

    monkeypatch.setattr(yolotrain, "train_yolo_dataset", fake_train)
    monkeypatch.setattr(fk, "register_run", lambda *a, **k: {})
    fk.train(ws, base=str(ws / "models" / "base.pt"), name="r1", manifest="v001")
    run = ws / "runs" / "r1"
    assert Path(seen["yaml"]) == run / "sampling" / "data.yaml"
    info = json.loads((run / "train_manifest.json").read_text(encoding="utf-8"))
    assert info["sampling"]["manifest"] == "v001" and info["sampling"]["entries"] == 21

    # interrupted after epoch 1: resume must train on the same list, not images/train
    (run / "weights").mkdir(exist_ok=True)
    (run / "args.yaml").write_text("model: base.pt\nepochs: 5\nimgsz: 640\n", encoding="utf-8")
    (run / "results.csv").write_text("epoch,metrics/mAP50-95(P)\n1,0.1\n", encoding="utf-8")
    torch.save({"epoch": 1, "optimizer": {"state": {}}}, run / "weights" / "last.pt")
    calls: list = []

    class FakeYOLO:
        def __init__(self, weights):
            pass

        def add_callback(self, *args):
            pass

        def train(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    fk.resume(ws, name="r1")
    assert Path(calls[0]["data"]) == run / "sampling" / "data.yaml"
