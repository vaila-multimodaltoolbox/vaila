"""FreeKiki snap: keypoints moved onto the painted lines.

Version: 0.4.9
Update Date: 16 October 2026
"""

import numpy as np

from vaila import freekiki
from vaila import freekiki_snap as fs


def _box(offset=0.0):
    """Right penalty box corners of a synthetic top view (p17, p20, p25, p28)."""
    pts = np.full((49, 2), np.nan)
    pts[17] = (600 - offset, 200 - offset)
    pts[20] = (600 - offset, 800 + offset)
    pts[25] = (1500 + offset, 200 - offset)
    pts[28] = (1500 + offset, 800 + offset)
    pts[18] = (600 - offset, 420)
    pts[19] = (600 - offset, 580)
    return pts


def _paint(w=1920, h=1080):
    """Mask of the painted box: lines 3 px wide through the true corners."""
    m = np.zeros((h, w), dtype=bool)
    m[199:202, 600:1501] = True  # box top
    m[799:802, 600:1501] = True  # box bottom
    m[200:801, 599:602] = True  # box side
    m[200:801, 1499:1502] = True  # goal line
    return m


def test_a_box_drawn_too_large_snaps_back_to_the_paint():
    xy = _box(offset=25.0)
    acc = np.isfinite(xy).all(axis=1)
    new, how = fs.snap_keypoints(xy, acc, _paint())
    for k, true in ((17, (600, 200)), (20, (600, 800)), (25, (1500, 200)), (28, (1500, 800))):
        assert how[k] == "corner"
        assert np.hypot(*(new[k] - true)) < 2.0, (k, new[k])
    # arc crossings keep their place along the line but land on it
    assert how[18] == "line" and abs(new[18][0] - 600) < 2.0 and abs(new[18][1] - 420) < 1.0


def test_points_already_on_the_paint_barely_move():
    xy = _box(offset=0.0)
    acc = np.isfinite(xy).all(axis=1)
    new, _ = fs.snap_keypoints(xy, acc, _paint())
    assert np.nanmax(np.hypot(*(new[acc] - xy[acc]).T)) < 2.0


def test_no_paint_or_too_far_leaves_the_points_alone():
    xy = _box(offset=25.0)
    acc = np.isfinite(xy).all(axis=1)
    new, how = fs.snap_keypoints(xy, acc, np.zeros((1080, 1920), dtype=bool))
    assert how == {} and np.allclose(new[acc], xy[acc])
    far = _box(offset=80.0)  # the paint is outside the search band
    new, how = fs.snap_keypoints(far, np.isfinite(far).all(axis=1), _paint())
    assert all(np.hypot(*(new[k] - far[k])) <= fs.MAX_MOVE_PX for k in how)


def test_a_line_with_one_accepted_point_is_not_refitted():
    xy = _box(offset=25.0)
    acc = np.zeros(49, dtype=bool)
    acc[17] = True  # alone: no segment to search around
    _, how = fs.snap_keypoints(xy, acc, _paint())
    assert how == {}


def test_every_painted_line_member_is_a_kiki_point():
    for name, pts in fs.PAINTED_LINES.items():
        assert len(pts) >= 2 and all(0 <= p < 49 for p in pts), name
    assert sorted(fs.lines_of(17)) == ["box_right_side", "box_right_top"]


def test_snap_cli_parses():
    a = freekiki.build_parser().parse_args(["snap", "-w", "W", "--run", "R", "--max-move", "30"])
    assert a.run == "R" and a.max_move == 30.0 and not a.no_overlay


def _tracks(T=60):
    """A point panning right at 4 px/frame; network shaky, with a 30 px bias."""
    t = np.arange(T)[:, None]
    truth = np.c_[500 + 4.0 * t, np.full((T, 1), 300.0)]
    rng = np.random.default_rng(3)
    net = truth + [30.0, 0.0] + rng.normal(0, 3.0, (T, 2))
    out = np.full((T, 49, 2), np.nan)
    return truth, net, out


def test_stabilize_keeps_the_per_frame_snap_when_the_paint_is_steady():
    truth, net, base = _tracks()
    N, S = base.copy(), base.copy()
    N[:, 17], S[:, 17] = net, truth  # the snap sits on the paint every frame
    out = fs.stabilize(N, S, 1920)
    assert np.allclose(out[:, 17], truth)


def test_stabilize_smooths_a_snap_that_alternates_between_two_strokes():
    truth, net, base = _tracks()
    alt = truth.copy()
    alt[::2] += [9.0, 0.0]  # every other frame the snap catches the wrong stroke
    N, S = base.copy(), base.copy()
    N[:, 22], S[:, 22] = truth + [8.0, 0.0], alt  # smooth network, flickering snap
    out = fs.stabilize(N, S, 1920)
    d2 = out[2:, 22] - 2 * out[1:-1, 22] + out[:-2, 22]
    assert np.nanmax(np.hypot(*d2.T)) < 3.0  # the 9 px flicker is gone
    assert np.allclose(np.isnan(out[:, 30]), True)  # points never accepted stay empty


def test_points_already_on_their_lines_are_not_moved():
    """A camera the network knows well: points 1-2 px from the paint stay exactly where they are."""
    xy = _box(offset=0.0)
    xy[17] += (1.5, 1.0)
    xy[28] += (-1.0, 1.5)
    acc = np.isfinite(xy).all(axis=1)
    new, how = fs.snap_keypoints(xy, acc, _paint())
    assert 17 not in how and 28 not in how
    assert np.allclose(new[17], xy[17]) and np.allclose(new[28], xy[28])


def test_a_line_fitted_off_the_paint_is_refused():
    """Paint only near one end of a segment (e.g. a goal post): the refit is not trusted."""
    xy = _box(offset=25.0)
    acc = np.isfinite(xy).all(axis=1)
    m = np.zeros((1080, 1920), dtype=bool)
    m[199:202, 600:700] = True  # only 100 px of the 900 px box top are painted
    _, how = fs.snap_keypoints(xy, acc, m)
    assert 25 not in how


def test_near_parallel_lines_give_no_corner():
    d1 = np.array([1.0, 0.0])
    d2 = np.array([np.cos(np.radians(5)), np.sin(np.radians(5))])
    assert fs._intersect((np.zeros(2), d1), (np.array([0.0, 10.0]), d2)) is None
    d3 = np.array([0.0, 1.0])
    assert np.allclose(fs._intersect((np.zeros(2), d1), (np.array([5.0, 3.0]), d3)), (5.0, 0.0))


def test_stabilize_removes_a_single_frame_pop():
    """Snap kept the network point except in one frame: the pop must not survive."""
    truth, _, base = _tracks()
    N, S = base.copy(), base.copy()
    N[:, 22] = truth
    S[:, 22] = truth
    S[30, 22] += (10.0, 0.0)
    out = fs.stabilize(N, S, 1920)
    d1 = np.hypot(*np.diff(out[:, 22], axis=0).T)
    assert d1.max() < 6.0  # truth moves 4 px/frame; the 10 px pop is gone


def test_stabilize_keeps_the_network_when_its_correction_is_small():
    truth, net, base = _tracks()
    N, S = base.copy(), base.copy()
    N[:, 17] = truth
    S[:, 17] = truth + np.random.default_rng(1).normal(0, 1.0, truth.shape)  # tiny, noisy moves
    out = fs.stabilize(N, S, 1920)
    assert np.allclose(out[:, 17], truth)  # identical to the network


def test_fit_line_in_band_ignores_curved_penalty_arc():
    """A curved penalty arc inside the search band must not pull the straight box line."""
    m = _paint()
    # Add a thick curved penalty arc bowing 28 px left of x=600 between y=360 and y=640
    for y in range(360, 641):
        u = (y - 500.0) / 140.0
        x_arc = int(round(600.0 - 28.0 * (1.0 - u * u)))
        m[y, x_arc - 2 : x_arc + 3] = True
    ys, xs = np.nonzero(m)
    pts = np.c_[xs, ys].astype(float)
    line = fs.fit_line_in_band(pts, (580.0, 200.0), (580.0, 800.0), fs.BAND_PX, m)
    assert line is not None
    c, d = line
    assert abs(c[0] - 600.0) < 1.5 and abs(d[0]) < 0.01


def test_tilted_line_arc_crossing_follows_cross_line_direction():
    """On a tilted box line in perspective, p18 slides along the cross-line rather than orthogonally."""
    import cv2

    m = np.zeros((1080, 1920), dtype=np.uint8)
    # Tilted box_right_side from (900, 200) to (600, 800), horizontal top/bottom to x=1500
    cv2.line(m, (900, 200), (1500, 200), 1, 3)
    cv2.line(m, (600, 800), (1500, 800), 1, 3)
    cv2.line(m, (900, 200), (600, 800), 1, 3)
    cv2.line(m, (1500, 200), (1500, 800), 1, 3)
    paint = m.astype(bool)
    pts = np.full((49, 2), np.nan)
    pts[17] = (875.0, 200.0)  # shifted 25 px left
    pts[18] = (785.0, 380.0)  # true point on line at y=380 is x=810
    pts[19] = (685.0, 580.0)  # true point on line at y=580 is x=710
    pts[20] = (575.0, 800.0)
    pts[25] = (1500.0, 200.0)
    pts[28] = (1500.0, 800.0)
    acc = np.isfinite(pts).all(axis=1)
    new, how = fs.snap_keypoints(pts, acc, paint)
    assert how[18] == "line"
    assert abs(new[18, 0] - 810.0) < 2.0 and abs(new[18, 1] - 380.0) < 1.5


def test_stabilize_fills_dropout_frames_inside_snapped_stretch():
    """If the snap drops out for 2 frames inside a snapped clip, stabilize fills the gap."""
    truth, net, base = _tracks()
    N, S = base.copy(), base.copy()
    N[:, 17], S[:, 17] = net, truth
    S[20:22, 17] = net[20:22]  # 2 dropout frames where snap_keypoints kept the network point
    out = fs.stabilize(N, S, 1920)
    assert np.nanmax(np.hypot(*(out[20:22, 17] - truth[20:22]).T)) < 5.0


def test_detect_cli_snap_defaults_and_opt_out():
    p = freekiki.build_parser()
    args_def = p.parse_args(["detect", "-w", "W", "--video", "V.mp4"])
    assert args_def.snap is True
    args_nosnap = p.parse_args(["detect", "-w", "W", "--video", "V.mp4", "--no-snap"])
    assert args_nosnap.snap is False


def test_detect_video_applies_snap_by_default_and_respects_no_snap(tmp_path, monkeypatch):
    import json

    import cv2

    w, h = 640, 480
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
            pts = np.full((49, 2), 100.0)
            return 0.9, pts, np.full(49, 0.9)

    monkeypatch.setattr(freekiki, "resolve_model", lambda *_a, **_k: str(model))
    monkeypatch.setattr(freekiki, "load_predictor", lambda *_a, **_k: FakePredictor())

    # Default: snap=True
    run1 = freekiki.detect_video(
        ws, video, output_dir=tmp_path / "out1", overlay=False, diag_frames=0
    )
    assert (run1 / "field_kps_getpixelvideo.csv").is_file()
    assert (run1 / "field_kps_snap_getpixelvideo.csv").is_file()
    assert (run1 / "snap_moves.csv").is_file()
    q1 = json.loads((run1 / "quality.json").read_text())
    assert q1["snap"] is True

    # Opt-out: snap=False
    run2 = freekiki.detect_video(
        ws, video, output_dir=tmp_path / "out2", overlay=False, diag_frames=0, snap=False
    )
    assert (run2 / "field_kps_getpixelvideo.csv").is_file()
    assert not (run2 / "snap_moves.csv").is_file()
    q2 = json.loads((run2 / "quality.json").read_text())
    assert q2["snap"] is False


def test_barrel_distorted_box_side_snaps_top_corner_locally():
    """On a zoom/wide lens where box_right_side bends ~2.5 deg from p20 to p17, p17 must snap to its local corner."""
    import cv2

    m = np.zeros((1080, 1920), dtype=np.uint8)
    # True corner p17 at (400, 350), p25 at (1200, 310) -> shallow crossing
    cv2.line(m, (400, 350), (1200, 310), 1, 3)
    # Piecewise barrel-curved box_right_side: p17(400,350) -> p18(620,445) -> p19(950,600) -> p20(1350,805)
    cv2.line(m, (400, 350), (620, 445), 1, 3)
    cv2.line(m, (620, 445), (950, 600), 1, 3)
    cv2.line(m, (950, 600), (1350, 805), 1, 4)
    paint = m.astype(bool)

    pts = np.full((49, 2), np.nan)
    pts[17] = (402.0, 345.0)  # network slightly above the true corner (400, 350)
    pts[18] = (620.0, 445.0)
    pts[19] = (950.0, 600.0)
    pts[20] = (1350.0, 805.0)
    pts[25] = (1200.0, 310.0)
    acc = np.isfinite(pts).all(axis=1)

    new, _ = fs.snap_keypoints(pts, acc, paint)
    # Must land near the true corner (400, 350), not 25-35 px to the right along the top line
    assert np.hypot(*(new[17] - np.array([400.0, 350.0]))) < 4.0


def test_stabilize_does_not_copy_stale_offset_outside_snapped_window():
    """When the camera pans away and a point is no longer snapped for >window frames, do not copy old offsets."""
    truth, net, base = _tracks(T=80)
    N, S = base.copy(), base.copy()
    N[:, 9] = net
    S[:25, 9] = truth[:25]  # snapped during frames 0..24
    S[25:, 9] = net[25:]  # untouched by snap from frame 25..79 (e.g. crossing line off-screen)
    out = fs.stabilize(N, S, 1920)
    assert np.allclose(out[40:, 9], net[40:])


def test_corner_snap_rejects_perimeter_board_and_keeps_180_deg_colinearity():
    """When an ad board above touch_top and a flagpole above p24 are bright white, p24 must stay colinear (~180 deg) with p25-p26-p27."""
    import cv2

    m = np.zeros((1080, 1920), dtype=np.uint8)
    # True right goal line: p24(1041, 294) -> p25(1235, 341) -> p26(1391, 378) -> p27(1509, 407)
    cv2.line(m, (1041, 294), (1509, 407), 1, 3)
    # True top touchline: p13(350, 278) -> p24(1041, 294)
    cv2.line(m, (350, 278), (1041, 294), 1, 2)
    # Distractor: bright thick perimeter ad board 22 px above touch_top + vertical flagpole at p24
    cv2.line(m, (350, 255), (1050, 272), 1, 8)
    cv2.line(m, (1041, 294), (1044, 258), 1, 5)
    # Penalty box right side & top for planar context
    cv2.line(m, (841, 369), (1235, 341), 1, 3)
    cv2.line(m, (841, 369), (1380, 505), 1, 3)
    paint = m.astype(bool)

    pts = np.full((49, 2), np.nan)
    pts[13] = (350.0, 278.0)
    pts[17] = (841.0, 369.0)
    pts[18] = (1003.0, 410.0)
    pts[19] = (1205.0, 461.0)
    pts[20] = (1380.0, 505.0)
    pts[24] = (1041.0, 294.0)
    pts[25] = (1235.0, 341.0)
    pts[26] = (1391.0, 378.0)
    pts[27] = (1509.0, 407.0)
    pts[47] = (1044.0, 258.0)  # top_right_flag_top directly above p24
    acc = np.isfinite(pts).all(axis=1)

    new, _ = fs.snap_keypoints(pts, acc, paint)
    # p24 must not be pulled up onto the ad board / flagpole (y=272)
    assert np.hypot(*(new[24] - np.array([1041.0, 294.0]))) < 4.0
    # Verify 180-deg colinearity of p24-p25-p26
    v1 = new[24] - new[25]
    v2 = new[26] - new[25]
    cos_ang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    ang_deg = float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))
    assert ang_deg > 178.5


def test_projective_1d_cross_ratio_along_line_and_vanishing_points():
    """1D projective homography along a multi-point line predicts exact perspective spacing, and parallel lines yield their vanishing point."""
    xyz, _ = fs._field_xyz()
    # Build a synthetic perspective homography mapping pitch (X, Y) -> image (u, v)
    H = np.array(
        [
            [24.0, -8.5, 960.0],
            [3.2, -16.0, 540.0],
            [0.0015, -0.008, 1.0],
        ],
        dtype=float,
    )
    ph = np.c_[xyz[:, :2], np.ones(len(xyz))] @ H.T
    uv_gt = ph[:, :2] / ph[:, 2:3]

    pts = uv_gt.copy()
    pts[24] = np.nan
    members_gr = list(fs.PAINTED_LINES["goal_line_right"])
    d_gr = uv_gt[29] - uv_gt[25]
    d_gr = d_gr / np.linalg.norm(d_gr)
    line_gr = (uv_gt[25], d_gr)
    pred_24 = fs._projective_1d_along_line("goal_line_right", 24, members_gr, pts, line_gr, 1.0)
    assert pred_24 is not None
    np.testing.assert_allclose(pred_24, uv_gt[24], atol=1e-3)

    # Verify vanishing point of longitudinal lines matches H[:, 0] ideal point
    fitted_long = {}
    members_on = {}
    for name in fs.LONGITUDINAL_LINES:
        m = list(fs.PAINTED_LINES[name])
        d = uv_gt[m[-1]] - uv_gt[m[0]]
        fitted_long[name] = (uv_gt[m[0]], d / np.linalg.norm(d))
        members_on[name] = m
    vp_x = fs._vanishing_point(fitted_long, members_on, fs.LONGITUDINAL_LINES)
    assert vp_x is not None
    vp_gt = H[:2, 0] / H[2, 0]
    np.testing.assert_allclose(vp_x, vp_gt, rtol=1e-3)

