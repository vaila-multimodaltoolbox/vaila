"""FreeKiki field geometry: camera from keypoints, fill / fix, evaluate scoring.

Version: 0.4.7
Update Date: 07 October 2026
"""

import numpy as np
import pytest

from vaila import freekiki, freekiki_diag, freekiki_geom

W, H = 1920, 1080


def _camera(C=(10.0, -60.0, 25.0), target=(30.0, 0.0, 0.0), f=2000.0):
    """Broadcast-like pinhole camera (P[2,3] = 1) looking at ``target`` from ``C``."""
    C, target = np.asarray(C, float), np.asarray(target, float)
    z = (target - C) / np.linalg.norm(target - C)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    R = np.vstack([x, np.cross(z, x), z])
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    P = K @ np.c_[R, -R @ C]
    return P / P[2, 3]


def _view(P=None, noise=0.0, seed=0):
    xyz, planar = freekiki_geom._field()
    uv, front = freekiki_geom._project(_camera() if P is None else P, xyz)
    uv = uv + np.random.default_rng(seed).normal(0, noise, uv.shape)
    inside = front & (uv[:, 0] > 5) & (uv[:, 0] < W - 5) & (uv[:, 1] > 5) & (uv[:, 1] < H - 5)
    return uv, inside, planar


def test_homography_camera_recovers_focal_and_projects_points_off_the_ground():
    uv, inside, planar = _view()
    model = freekiki_geom.fit_field_camera(uv, inside & planar, W, H)
    assert model["status"] == "ok" and model["source"] == "homography"
    assert model["focal_px"] == pytest.approx(2000.0, rel=0.01)
    assert model["cam_height_m"] == pytest.approx(25.0, abs=0.5)
    proj, ok = freekiki_geom.project_field(model, W, H)
    off = inside & ~planar
    assert off.sum() >= 2 and ok[off].all()
    assert np.abs(proj[off] - uv[off]).max() < 0.5


def test_dlt3d_camera_when_points_off_the_ground_are_found():
    uv, inside, _ = _view(noise=0.3)
    model = freekiki_geom.fit_field_camera(uv, inside, W, H)
    assert model["status"] == "ok" and model["source"] == "dlt3d"
    proj, _ = freekiki_geom.project_field(model, W, H)
    assert np.abs(proj[inside] - uv[inside]).max() < 2.0


def _codes(ok):
    return ["D" if o else "Rk" for o in ok]


def test_fill_needs_the_network_to_half_see_the_point():
    uv, inside, planar = _view()
    target = int(np.flatnonzero(inside & planar)[-1])
    raw = uv.copy()
    raw[target] += (30.0, 0.0)  # network far from the truth, but conf 0.2
    kc = np.where(inside, 0.9, 0.0)
    kc[target] = 0.2
    codes = _codes(inside & (np.arange(49) != target))
    xy, kc2, g, model = freekiki_geom.refine_keypoints(
        raw, kc, codes, W, H, mode="fill", kp_conf=0.5
    )
    assert model["status"] == "ok" and g[target] == "G"
    assert np.allclose(xy[target], uv[target], atol=0.5) and kc2[target] == 0.5
    kc[target] = 0.01  # the network does not see it: never invented
    _, _, g, _ = freekiki_geom.refine_keypoints(raw, kc, codes, W, H, mode="fill", kp_conf=0.5)
    assert g[target] == "Rk"


def test_fix_replaces_a_wrong_accepted_point_and_fill_only_flags_it():
    uv, inside, planar = _view()
    bad = int(np.flatnonzero(inside & planar)[0])
    raw = uv.copy()
    raw[bad] += (60.0, 0.0)
    kc = np.where(inside, 0.9, 0.0)
    xy, _, g, _ = freekiki_geom.refine_keypoints(
        raw, kc, _codes(inside), W, H, mode="fix", kp_conf=0.5
    )
    assert g[bad] == "Gx" and np.allclose(xy[bad], uv[bad], atol=0.5)
    xy, _, g, _ = freekiki_geom.refine_keypoints(
        raw, kc, _codes(inside), W, H, mode="fill", kp_conf=0.5
    )
    assert g[bad] == "Dg" and np.allclose(xy[bad], raw[bad])


def test_nothing_changes_without_a_valid_camera_or_when_off():
    uv, inside, planar = _view()
    few = np.zeros(49, bool)
    few[np.flatnonzero(inside & planar)[:3]] = True
    kc = np.where(inside, 0.3, 0.0)
    kc[few] = 0.9
    xy, kc2, g, model = freekiki_geom.refine_keypoints(
        uv, kc, _codes(few), W, H, mode="fix", kp_conf=0.5
    )
    assert model["status"] == "few_points" and g == _codes(few)
    assert np.array_equal(kc2, kc)
    _, _, g, model = freekiki_geom.refine_keypoints(
        uv, kc, _codes(inside), W, H, mode="off", kp_conf=0.5
    )
    assert model is None and g == _codes(inside)
    with pytest.raises(ValueError):
        freekiki_geom.refine_keypoints(uv, kc, _codes(inside), W, H, mode="guess", kp_conf=0.5)


def test_camera_must_be_physical():
    uv, inside, planar = _view()
    model = freekiki_geom.fit_field_camera(uv, inside & planar, W, H)
    assert (
        freekiki_geom.camera_from_homography(model["H"], W, H, {"max_cam_height_m": 10.0}) is None
    )
    assert freekiki_geom.camera_from_homography(model["H"], W, H, {"max_focal_ratio": 0.5}) is None


def test_flag_label_far_from_the_camera_of_the_others_is_reported():
    uv, inside, _ = _view(noise=0.3)
    xyz, planar = freekiki_geom._field()
    flag = next(i for i in np.flatnonzero(inside & ~planar))
    assert freekiki_geom.nonplanar_label_outliers(uv, inside, W, H) == []
    wrong = uv.copy()
    wrong[flag] += (0.0, 40.0)
    found = dict(freekiki_geom.nonplanar_label_outliers(wrong, inside, W, H))
    assert flag in found and found[flag] > 30


def test_geometry_raises_recall_when_the_network_half_sees_missing_points():
    rows = []
    for seed in range(3):
        uv, inside, planar = _view(noise=0.3, seed=seed)
        kc = np.where(inside, 0.9, 0.0)
        hidden = np.flatnonzero(inside)[[1, 4]]
        kc[hidden] = 0.2  # rare points: below kp_conf but seen
        rows.append((uv, kc, inside))
    n = len(rows)
    pred = {
        "images": np.array([f"i{k}.jpg" for k in range(n)]),
        "groups": np.array(["g"] * n),
        "sources": np.array(["synthetic"] * n),
        "width": np.full(n, W),
        "height": np.full(n, H),
        "box_conf": np.full(n, 0.9),
        "pred_xy": np.stack([r[0] for r in rows]),
        "pred_kc": np.stack([r[1] for r in rows]),
        "gt_xy": np.stack([np.where(r[2][:, None], r[0], 0.0) for r in rows]),
        "gt_vis": np.stack([np.where(r[2], 2, 0) for r in rows]),
    }
    net = freekiki_diag.score_predictions(pred, det_conf=0.25, kp_conf=0.5, match_px=25)
    gpred, stats = freekiki_geom.refine_predictions(pred, det_conf=0.25, kp_conf=0.5, mode="fill")
    geo = freekiki_diag.score_predictions(gpred, det_conf=0.25, kp_conf=0.5, match_px=25)
    assert stats["camera_ok_rate"] == 1.0 and sum(stats["fills"].values()) == 2 * n
    assert geo["overall"]["recall"] > net["overall"]["recall"]
    assert geo["overall"]["precision"] == pytest.approx(net["overall"]["precision"])


def test_cli_geometry_options():
    parser = freekiki.build_parser()
    args = parser.parse_args(["detect", "-w", "WS", "--video", "v.mp4", "--geom", "fix"])
    assert args.geom == "fix"
    assert parser.parse_args(["detect", "-w", "WS", "--video", "v.mp4"]).geom is None
    args = parser.parse_args(
        ["evaluate", "-w", "WS", "--geom-min-conf", "0.1", "--geom-fix-px", "30"]
    )
    assert (args.geom_min_conf, args.geom_fix_px) == (0.1, 30.0)
    with pytest.raises(SystemExit):
        parser.parse_args(["detect", "-w", "WS", "--video", "v.mp4", "--geom", "maybe"])


def test_dlt_coefficients_match_rec2d_and_the_camera():
    from vaila.rec2d import rec2d

    P = _camera()
    uv, inside, planar = _view(P)
    model = freekiki_geom.fit_field_camera(uv, inside, W, H)
    d2, d3 = freekiki_geom.dlt_coefficients(model)
    assert d2 is not None and d3 is not None and len(d2) == 8 and len(d3) == 11
    xyz, _ = freekiki_geom._field()
    ground = inside & planar  # rec2d: pixel -> field metres on the ground
    np.testing.assert_allclose(rec2d(d2, uv[ground]), xyz[ground, :2], atol=0.02)
    L = np.append(d3, 1.0).reshape(3, 4)  # rec3d layout: L1..L11, P[2,3] = 1
    proj, _ = freekiki_geom._project(L, xyz[inside])
    np.testing.assert_allclose(proj, uv[inside], atol=0.5)
    assert freekiki_geom.dlt_coefficients(None) == (None, None)
    assert freekiki_geom.dlt_coefficients({"status": "few_points"}) == (None, None)


def test_detect_writes_per_frame_dlt_files_for_rec2d_rec3d(tmp_path, monkeypatch):
    import cv2
    import pandas as pd

    w, h = 640, 360
    P = np.diag([w / W, h / H, 1.0]) @ _camera()  # same camera, smaller image
    xyz, _ = freekiki_geom._field()
    uv, front = freekiki_geom._project(P / P[2, 3], xyz)
    seen = front & (uv[:, 0] > 2) & (uv[:, 0] < w - 2) & (uv[:, 1] > 2) & (uv[:, 1] < h - 2)
    video = tmp_path / "clip.mp4"
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))  # ty: ignore[unresolved-attribute]
    for _ in range(3):
        out.write(np.zeros((h, w, 3), np.uint8))
    out.release()
    ws = freekiki.init_workspace(tmp_path / "ws")
    model = tmp_path / "fake.pt"
    model.write_bytes(b"fake")

    class FakePredictor:
        backend, imgsz, calls = "fake", 640, 0

        def predict(self, frame):
            FakePredictor.calls += 1
            if FakePredictor.calls == 1:
                return float("nan"), None, None  # frame 0: no field
            return 0.9, uv.copy(), np.where(seen, 0.9, 0.0)

    monkeypatch.setattr(freekiki, "resolve_model", lambda *_a, **_k: str(model))
    monkeypatch.setattr(freekiki, "load_predictor", lambda *_a, **_k: FakePredictor())
    run = freekiki.detect_video(ws, video, output_dir=tmp_path, overlay=False, diag_frames=0)
    d2 = pd.read_csv(run / "clip.dlt2d")
    d3 = pd.read_csv(run / "clip.dlt3d")
    assert list(d2.columns) == ["frame"] + [f"p{j}" for j in range(1, 9)]
    assert list(d3.columns) == ["frame"] + [f"p{j}" for j in range(1, 12)]
    assert d3["frame"].tolist() == [0, 1, 2]
    assert d3.iloc[0, 1:].isna().all() and d2.iloc[0, 1:].isna().all()  # no camera
    L = np.append(d3.iloc[1, 1:].to_numpy(float), 1.0).reshape(3, 4)
    proj, _ = freekiki_geom._project(L, xyz[seen])
    np.testing.assert_allclose(proj, uv[seen], atol=0.5)
    quality = __import__("json").loads((run / "quality.json").read_text())
    assert quality["geom_dlt3d_rate"] == pytest.approx(2 / 3, abs=1e-3)


def test_homography_with_no_inliers_is_rejected_not_a_crash(monkeypatch):
    import cv2

    uv, inside, planar = _view()
    use = inside & planar
    xyz, _ = freekiki_geom._field()
    H_true = np.eye(3)

    def no_inlier(src, dst, method, thr):  # what OpenCV returned on a real video frame
        return H_true, np.zeros((len(src), 1), np.uint8)

    monkeypatch.setattr(cv2, "findHomography", no_inlier)
    fit = freekiki_diag.fit_field_homography(uv[use], xyz[use, :2], width=W)
    assert fit["status"] == "few_inliers" and fit["n_inliers"] == 0
    assert freekiki_diag._orientation(H_true, np.zeros((0, 2))) == 0


def test_origin_view_hides_projected_labels_without_counting_them_as_errors():
    from vaila import freekiki_diag as diag

    n = 3
    pred = {
        "images": np.array(["a.jpg", "b.jpg", "c.jpg"]),
        "groups": np.array(["a", "b", "c"]),
        "sources": np.array(["s"] * n),
        "width": np.full(n, 1920),
        "box_conf": np.full(n, 0.9),
        "pred_xy": np.zeros((n, 49, 2)),
        "pred_kc": np.full((n, 49), 0.9),
        "gt_xy": np.zeros((n, 49, 2)),
        "gt_vis": np.zeros((n, 49)),
    }
    pred["gt_xy"][:, 5] = (100.0, 100.0)
    pred["gt_vis"][:, 5] = 2
    pred["pred_xy"][:, 5] = (100.0, 100.0)
    pred["pred_kc"][2, 5] = 0.0  # missed where the label was only projected
    origins = ["1" * 49, "1" * 5 + "2" + "1" * 43, "human_reviewed"]
    origins[2] = "3" * 49
    mask = diag.origin_mask(origins, 49, "1")
    assert mask[:, 5].tolist() == [True, False, False]
    assert diag.origin_mask(["human_reviewed"], 49, "1").all()
    kw = {"det_conf": 0.25, "kp_conf": 0.5, "match_px": 25.0, "with_calib": False}
    full = diag.score_predictions(pred, **kw)["per_keypoint"][5]
    view = diag.score_predictions(diag.restrict_to_labelled(pred, mask), **kw)["per_keypoint"][5]
    assert full["n_ref"] == 3 and full["recall"] == round(2 / 3, 4)
    assert view["n_ref"] == 1 and view["recall"] == 1.0 and view["n_fp"] == 0
