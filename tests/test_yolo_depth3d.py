"""Tests of the pure geometry of vaila/yolo_depth3d.py (synthetic depth maps, no network, no GPU)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vaila import yolo_depth3d as yd
from vaila.monocular_dlt_align import decompose_dlt3d, dlt_project


def _camera(f=1500.0, cx=540.0, cy=960.0, centre=(0.5, -4.0, 1.2)):
    """A DLT3D of a camera looking along world +Y with world Z up."""
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # world -> camera
    C = np.asarray(centre, float)
    P = K @ np.hstack([R, (-R @ C)[:, None]])
    P = P / P[2, 3]
    return np.array([*P[0], *P[1], *P[2, :3]]), K, R, C


def _write_dlt(path: Path, L: np.ndarray) -> None:
    columns = pd.Index(["frame"] + [f"L{i}" for i in range(1, 12)])
    pd.DataFrame([[0, *L]], columns=columns).to_csv(path, index=False)


def test_pinhole_backprojection_is_exact():
    K = yd.intrinsics_from_focal(1000.0, 640, 480)
    xyz = np.array([[0.3, -0.2, 2.0], [-0.5, 0.4, 5.0], [0.0, 0.0, 1.0]])
    uv = (xyz[:, :2] / xyz[:, 2:3]) * 1000.0 + [320.0, 240.0]
    got = yd.backproject_pinhole(uv, xyz[:, 2], K)
    assert np.allclose(got, xyz, atol=1e-9)
    assert yd.intrinsics_from_fov(60.0, 640, 480)[0, 0] == pytest.approx(
        320 / np.tan(np.radians(30))
    )
    assert yd.default_focal_px(640, 480) == pytest.approx(800.0)  # the image diagonal


def test_dlt_point_sits_on_the_pixel_ray_at_the_asked_camera_depth():
    L, _K, R, C = _camera()
    world = np.array([[0.2, 0.3, 1.0], [-0.4, 1.0, 0.1], [0.9, -0.5, 1.7]])
    uv = dlt_project(L, world)
    z_true = (R @ (world - C).T).T[:, 2]
    assert np.all(z_true > 0)
    got = yd.dlt_point_at_depth(L, uv, z_true)
    assert np.allclose(got, world, atol=1e-6)
    assert np.allclose(yd.camera_depth(decompose_dlt3d(L), world), z_true, atol=1e-6)
    # a wrong depth still leaves the point on the same ray: it reprojects on its pixel
    off = yd.dlt_point_at_depth(L, uv, z_true * 1.3)
    assert np.allclose(dlt_project(L, off), uv, atol=1e-6)
    assert not np.allclose(off, world, atol=0.05)


def test_the_window_ignores_the_background_behind_a_thin_limb():
    depth = np.full((200, 200), 6.0, np.float32)  # the far wall
    depth[:, 98:103] = 3.0  # a limb 5 px wide, 3 m away
    z, n = yd.sample_depth(depth, np.array([[100.0, 100.0]]), radius=8)
    assert z[0] == pytest.approx(3.0) and n[0] > 0
    # plain patch median would have answered the wall
    assert np.median(depth[92:109, 92:109]) == pytest.approx(6.0)
    # outside the image and not-a-number pixels give no depth, never a guess
    z, n = yd.sample_depth(depth, np.array([[-5.0, 10.0], [np.nan, 3.0], [500.0, 500.0]]), radius=4)
    assert np.all(np.isnan(z)) and np.all(n == 0)


def test_the_person_mask_restricts_the_window():
    depth = np.full((100, 100), 2.0, np.float32)  # something nearer in front of the patch
    depth[40:60, 40:60] = 4.0
    mask = np.zeros((100, 100), bool)
    mask[40:60, 40:60] = True
    z, _ = yd.sample_depth(depth, np.array([[41.0, 41.0]]), radius=6, mask=mask)
    assert z[0] == pytest.approx(4.0)


def test_control_scale_recovers_a_known_scale_and_a_known_log_affine_fit():
    rng = np.random.default_rng(3)
    z_true = rng.uniform(2.5, 6.0, 12)
    a, b = yd.fit_depth_scale(z_true / 1.25, z_true, mode="scale")
    assert a == 1.0 and np.exp(b) == pytest.approx(1.25)
    pred = np.exp((np.log(z_true) - 0.2) / 0.9)  # log z_true = 0.9 log pred + 0.2
    a, b = yd.fit_depth_scale(pred, z_true, mode="logaffine")
    assert (a, b) == pytest.approx((0.9, 0.2), abs=1e-6)
    assert np.allclose(yd.apply_depth_scale(pred, a, b), z_true)
    # too few points for two parameters: fall back to one scale instead of a wild fit
    a, _b = yd.fit_depth_scale(pred[:3], z_true[:3], mode="logaffine")
    assert a == 1.0


def test_floor_scale_puts_the_standing_foot_on_the_floor():
    L, _K, R, C = _camera()
    foot_h = 0.08
    feet = np.array([[0.1, 0.5, foot_h], [0.3, 0.6, 0.45]])  # one on the floor, one lifted
    uv = dlt_project(L, feet)
    z_true = (R @ (feet - C).T).T[:, 2]
    ratio = yd.floor_depth_ratio(L, uv, z_true / 1.2, foot_height_m=foot_h)
    assert ratio[0] == pytest.approx(1.2, rel=1e-6)  # the standing foot gives the scale
    assert ratio[1] > ratio[0]  # the lifted foot's ray meets the floor further away
    fixed = yd.dlt_point_at_depth(L, uv[:1], (z_true[:1] / 1.2) * np.nanmin(ratio))
    assert fixed[0, 2] == pytest.approx(foot_h, abs=1e-6)


def test_bone_fix_keeps_the_points_on_their_rays_and_steadies_the_segments():
    L, _K, R, C = _camera()
    rng = np.random.default_rng(7)
    n_frames = 40
    t = np.linspace(0, 1, n_frames)
    hip = np.stack([0.2 + 0.3 * t, 0.5 + 0.2 * np.sin(6 * t), np.full(n_frames, 0.9)], axis=1)
    knee = hip + [0.0, 0.05, -0.42]
    ankle = knee + np.stack([0.1 * np.sin(5 * t), 0 * t, -0.40 * np.ones_like(t)], axis=1)
    ankle = knee + (ankle - knee) / np.linalg.norm(ankle - knee, axis=1)[:, None] * 0.40
    world = np.stack([hip, knee, ankle], axis=1)  # (frames, 3 markers, 3)
    bones = [(0, 1), (1, 2)]
    uv = np.stack([dlt_project(L, w) for w in world])
    z = np.stack([(R @ (w - C).T).T[:, 2] for w in world])
    noisy = z + rng.normal(0.0, 0.08, z.shape)
    raw = np.stack([yd.dlt_point_at_depth(L, uv[i], noisy[i]) for i in range(n_frames)])
    fixed_z = yd.bone_fix_depths(L, uv, noisy, bones, weight=4.0)
    fixed = np.stack([yd.dlt_point_at_depth(L, uv[i], fixed_z[i]) for i in range(n_frames)])
    for i in range(n_frames):
        assert np.allclose(dlt_project(L, fixed[i]), uv[i], atol=1e-5)  # still on the rays
    cv_raw = yd.segment_variation(raw, bones)
    cv_fixed = yd.segment_variation(fixed, bones)
    assert max(cv_fixed.values()) < 0.5 * max(cv_raw.values())
    err_raw = np.linalg.norm(raw - world, axis=2).mean()
    err_fixed = np.linalg.norm(fixed - world, axis=2).mean()
    assert err_fixed < err_raw


def test_missing_points_stay_missing_and_the_csv_has_the_rec3d_layout(tmp_path: Path):
    xyz = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
    xyz[1, 2] = np.nan  # a marker with no depth: written empty, never filled
    out = tmp_path / "x.csv"
    yd.write_wide_xyz(out, np.array([10, 11]), xyz)
    df = pd.read_csv(out)
    assert list(df.columns)[:4] == ["frame", "p0_x", "p0_y", "p0_z"] and df.shape == (2, 10)
    assert df["frame"].tolist() == [10, 11] and np.isnan(df.loc[1, "p2_z"])
    frames, back = yd.read_wide_xyz(out)
    assert frames.tolist() == [10, 11] and np.allclose(back, xyz, equal_nan=True)


def test_reference_comparison_reports_error_and_bias_per_axis():
    rng = np.random.default_rng(1)
    ref = rng.uniform(-1, 1, (20, 5, 3))
    got = ref + [0.0, 0.10, 0.0]
    got[3, 2] = np.nan
    rep = yd.compare_with_reference(got, ref)
    assert rep["mpjpe_m"] == pytest.approx(0.10) and rep["n_points"] == 99
    assert rep["bias_m"] == pytest.approx([0.0, 0.10, 0.0], abs=1e-9)
    assert len(rep["per_marker_m"]) == 5
    # rigid alignment (no scale) removes a pure rotation + shift, so only the shape error is left
    ang = np.radians(30)
    Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    moved = ref @ Rz.T + [3.0, -1.0, 0.5]
    assert yd.compare_with_reference(moved, ref, rigid_align=True)["mpjpe_m"] < 1e-9


def test_layouts_and_the_cli_parser():
    assert yd.layout_for(17).name == "coco17" and yd.layout_for(33).name == "mediapipe33"
    assert yd.layout_for(5).bones == []  # unknown layout: no skeleton, the lift still works
    args = yd.build_parser().parse_args(["-i", "v.mp4", "--dlt3d", "c.dlt3d", "--scale", "floor"])
    assert (
        args.scale == "floor" and args.depth_model == yd.DEFAULT_DEPTH_MODEL and not args.bone_fix
    )
    assert args.joint_depth == yd.DEFAULT_JOINT_DEPTH
    with pytest.raises(SystemExit):
        yd.build_parser().parse_args(["-i", "v.mp4", "--scale", "bogus"])


def test_the_frame_scale_comes_from_the_standing_foot_at_each_markers_height():
    L, _K, R, C = _camera()
    lay = yd.MEDIAPIPE33
    world = np.full((33, 3), np.nan)
    world[27], world[29] = [0.1, 0.5, 0.12], [0.1, 0.45, 0.08]  # left foot on the floor
    world[28], world[30] = [0.3, 0.6, 0.50], [0.3, 0.55, 0.46]  # right foot lifted
    uv = np.full((33, 2), np.nan)
    z = np.full(33, np.nan)
    for m in (27, 28, 29, 30):
        uv[m] = dlt_project(L, world[[m]])[0]
        z[m] = (R @ (world[m] - C))[2] / 1.5
    assert yd.floor_scale_of_frame(L, uv, z, lay) == pytest.approx(1.5, rel=1e-6)
    # one height for every marker is a different (wrong here) assumption, so a different scale
    assert yd.floor_scale_of_frame(L, uv, z, lay, foot_height_m=0.0) != pytest.approx(1.5, rel=1e-3)
    z[:] = np.nan
    assert np.isnan(yd.floor_scale_of_frame(L, uv, z, lay))


def test_shrinking_keeps_the_persons_depth_and_only_part_of_the_joint_depth():
    z = np.array([[3.0, 3.2, 2.6, np.nan], [4.0, 4.0, 4.4, 4.2]])
    out = yd.shrink_to_person(z, trunk=(0, 1), keep=0.25)
    assert out[0, :3] == pytest.approx([3.1 - 0.025, 3.1 + 0.025, 3.1 - 0.125])
    assert np.isnan(out[0, 3])  # a missing depth stays missing
    assert yd.shrink_to_person(z, (0, 1), 1.0)[1] == pytest.approx(z[1])
    assert yd.shrink_to_person(z, (0, 1), 0.0)[1] == pytest.approx([4.0] * 4)
    assert yd.shrink_to_person(z, (), 0.0)[1] == pytest.approx([4.1] * 4)  # no trunk: all markers


def test_one_person_is_written_with_world_csv_c3d_and_quality(tmp_path: Path):
    """The whole post-processing of a person on synthetic depths (no model): files and summary."""
    L, _K, R, C = _camera()
    n_frames, lay = 30, yd.MEDIAPIPE33
    rng = np.random.default_rng(5)
    base = rng.uniform([-0.2, 0.3, 0.3], [0.4, 0.7, 1.6], (33, 3))
    base[27], base[29] = [0.0, 0.5, 0.12], [0.0, 0.46, 0.08]
    base[28], base[30] = [0.25, 0.5, 0.12], [0.25, 0.46, 0.08]
    rows = {}
    for f in range(n_frames):
        world = base + [0.002 * f, 0.0, 0.0]
        uv = dlt_project(L, world)
        z = (R @ (world - C).T).T[:, 2] / 1.6  # the network reads 1.6 times too near
        rows[f] = (uv, np.ones(33), z, np.full(33, 9))
    dlt = tmp_path / "cam.dlt3d"
    _write_dlt(dlt, L)
    args = yd.build_parser().parse_args(
        ["-i", "v.mp4", "--dlt3d", str(dlt), "--scale", "floor", "--joint-depth", "1"]
        + ["--fit", "off"]
    )
    frames, coeffs = yd.load_dlt3d_file(dlt)
    info = yd._finish_person(rows, "v", args, tmp_path, 30.0, 1080, 1920, frames, coeffs, None)
    assert info["scale_applied"]["scale"] == pytest.approx(1.6, rel=1e-6)
    assert info["layout"] == lay.name and "c3d_error" not in info
    assert (tmp_path / info["c3d"]).stat().st_size > 0
    _f, world = yd.read_wide_xyz(tmp_path / info["world_csv"])
    assert np.allclose(world[0], base, atol=1e-4)  # scale found, so the lift is exact here
    assert info["quality"]["lowest_foot_z_m"]["median"] == pytest.approx(0.08, abs=1e-4)
    assert info["scale_change_per_cm_of_foot_height"] != 0


# --- the whole-trial fit ------------------------------------------------------------


def _standing_person(n_frames=45):
    """A 33-marker person facing the camera of :func:`_camera`, arms swinging in depth."""
    t = np.linspace(0.0, 1.0, n_frames)
    xyz = np.zeros((n_frames, 33, 3))
    side = {"L": 1.0, "R": -1.0}  # the subject's left is world +X when it faces -Y
    idx = {
        "L": {
            "sh": 11,
            "el": 13,
            "wr": 15,
            "pk": 17,
            "ix": 19,
            "th": 21,
            "hip": 23,
            "kn": 25,
            "an": 27,
            "he": 29,
            "toe": 31,
        },
        "R": {
            "sh": 12,
            "el": 14,
            "wr": 16,
            "pk": 18,
            "ix": 20,
            "th": 22,
            "hip": 24,
            "kn": 26,
            "an": 28,
            "he": 30,
            "toe": 32,
        },
    }
    for name, sg in side.items():
        m = idx[name]
        xyz[:, m["an"]] = [0.10 * sg, 0.0, 0.12]
        xyz[:, m["he"]] = [0.10 * sg, 0.03, 0.08]
        xyz[:, m["toe"]] = [0.10 * sg, -0.15, 0.04]
        xyz[:, m["kn"]] = [0.10 * sg, -0.03, 0.50]
        xyz[:, m["hip"]] = [0.09 * sg, -0.04, 0.90]  # the pelvis is above the middle of the feet
        xyz[:, m["sh"]] = [0.17 * sg, -0.04, 1.35]
        phase = 0.0 if name == "L" else 0.4
        up = 0.2 + 1.0 * np.sin(np.pi * (t + phase)) ** 2  # upper arm swings forward
        fore = up + 0.5 + 0.4 * np.cos(2 * np.pi * t)  # the elbow stays flexed
        elbow = xyz[:, m["sh"]] + 0.28 * np.column_stack(
            [0.3 * sg + 0 * t, -np.sin(up), -np.cos(up)]
        )
        wrist = elbow + 0.25 * np.column_stack([0 * t, -np.sin(fore), -np.cos(fore)])
        xyz[:, m["el"]], xyz[:, m["wr"]] = elbow, wrist
        along = (wrist - elbow) / 0.25
        xyz[:, m["pk"]] = wrist + 0.08 * along + [0.02 * sg, 0, 0]
        xyz[:, m["ix"]] = wrist + 0.09 * along
        xyz[:, m["th"]] = wrist + 0.05 * along - [0.02 * sg, 0, 0]
    xyz[:, 0] = [0.0, -0.10, 1.52]
    for k, (x, y, z) in {
        1: (0.02, -0.08, 1.55),
        2: (0.03, -0.08, 1.55),
        3: (0.04, -0.07, 1.55),
        4: (-0.02, -0.08, 1.55),
        5: (-0.03, -0.08, 1.55),
        6: (-0.04, -0.07, 1.55),
        7: (0.07, 0.0, 1.52),
        8: (-0.07, 0.0, 1.52),
        9: (0.03, -0.08, 1.46),
        10: (-0.03, -0.08, 1.46),
    }.items():
        xyz[:, k] = [x, y, z]
    return xyz + [0.3, 0.2, 0.0]  # somewhere in the lab


def _rays_of(L, xyz):
    uv = np.stack([dlt_project(L, frame) for frame in xyz])
    origin = yd.dlt_rays(L, uv[0])[0]
    steps = np.stack([yd.dlt_rays(L, frame_uv)[1] for frame_uv in uv])
    z = np.stack([yd.camera_depth(L, frame) for frame in xyz])
    return uv, origin, steps, z


def test_the_jacobian_of_the_fit_matches_finite_differences():
    L, *_ = _camera()
    lay = yd.MEDIAPIPE33
    xyz = _standing_person(6)
    _uv, origin, steps, z_true = _rays_of(L, xyz)
    rng = np.random.default_rng(3)
    z = z_true + rng.normal(0, 0.04, z_true.shape)
    bones = np.asarray(lay.fit_bones, int)
    lengths = yd.estimate_segment_lengths(xyz, lay.fit_bones)
    cue = rng.normal(0, 0.08, (6, len(bones)))
    cue[0, 3] = np.nan
    order = rng.normal(0, 0.2, (6, len(bones)))
    weight = np.ones_like(z)
    weight[2, 15] = 0.0
    feet = (
        np.array(sorted(lay.foot_heights)),
        np.array([lay.foot_heights[m] for m in sorted(lay.foot_heights)]),
    )
    balance = (np.array(lay.balance[0]), np.array(lay.balance[1]))
    full = np.broadcast_to(origin, steps.shape)

    def terms(zz):
        return yd._fit_terms(
            zz, full, steps, weight, bones, lengths, 0.02, cue, order, feet, balance, 0.01
        )

    r0, rows, cols, vals = terms(z)
    jac = np.zeros((r0.size, z.size))
    np.add.at(jac, (rows, cols), vals)
    eps = 1e-6
    for k in rng.choice(z.size, 25, replace=False):
        dz = np.zeros(z.size)
        dz[k] = eps
        numeric = (terms(z + dz.reshape(z.shape))[0] - terms(z - dz.reshape(z.shape))[0]) / (
            2 * eps
        )
        assert np.allclose(jac[:, k], numeric, atol=1e-4, rtol=1e-4)


def test_the_fit_recovers_the_depth_between_joints_from_order_and_segment_lengths():
    """A noisy map with gross misreadings + the order of the ends + constant lengths."""
    L, *_ = _camera()
    lay = yd.MEDIAPIPE33
    xyz = _standing_person()
    _uv, origin, steps, z_true = _rays_of(L, xyz)
    rng = np.random.default_rng(11)
    person = np.median(z_true[:, list(lay.trunk)], axis=1, keepdims=True)
    z_map = z_true + rng.normal(0, 0.04, z_true.shape)  # 4 cm of noise at every joint ...
    z_map[rng.random(z_true.shape) < 0.05] += 1.5  # ... and one reading in 20 is the wall behind
    pose_z = 3.0 * (z_true - person)  # right order, three times too large

    def error(z):
        return float(np.linalg.norm(origin + steps * z[..., None] - xyz, axis=2).mean())

    fitted = yd.kinematic_fit(origin, steps, z_map, lay, fps=30.0, pose_z=pose_z)
    assert error(z_map) > 0.08
    assert error(fitted["z"]) < 0.03
    assert not fitted["lengths_given"].any()
    # every point is still on its own pixel ray
    back = np.stack([dlt_project(L, f) for f in origin + steps * fitted["z"][..., None]])
    assert np.allclose(back, np.stack([dlt_project(L, f) for f in xyz]), atol=1e-6)
    # with the person's true segment lengths the result is better still
    true_lengths = yd.estimate_segment_lengths(xyz, lay.fit_bones, lay.mirror)
    known = yd.kinematic_fit(
        origin, steps, z_map, lay, fps=30.0, pose_z=pose_z, lengths=true_lengths
    )
    assert known["lengths_given"].all()
    assert error(known["z"]) < 0.02
    variation = yd.segment_variation(origin + steps * known["z"][..., None], lay.bones)
    assert np.median(list(variation.values())) < 0.02


def test_a_marker_with_no_ray_takes_no_part_in_the_fit():
    L, *_ = _camera()
    lay = yd.MEDIAPIPE33
    xyz = _standing_person(20)
    _uv, origin, steps, z_true = _rays_of(L, xyz)
    steps[5:9, 16] = np.nan  # the right wrist is lost for four frames
    z_map = z_true.copy()
    z_map[3, 13] = np.nan  # and one depth was not read
    fitted = yd.kinematic_fit(origin, steps, z_map, lay, fps=30.0)
    assert np.isnan(fitted["z"][5:9, 16]).all()
    assert np.isfinite(np.delete(fitted["z"], 16, axis=1)).all()


def test_segment_lengths_are_shared_by_left_and_right_and_round_trip(tmp_path: Path):
    lay = yd.MEDIAPIPE33
    xyz = _standing_person(10)
    xyz[:, 13] += [0.0, 0.0, 0.02]  # the left upper arm 2 cm shorter than the right one
    lengths = yd.estimate_segment_lengths(xyz, lay.fit_bones, lay.mirror)
    left, right = lay.fit_bones.index((11, 13)), lay.fit_bones.index((12, 14))
    assert lengths[left] == pytest.approx(lengths[right])
    given = np.zeros(len(lengths), bool)
    yd.write_segment_lengths(tmp_path / "s.csv", lay.fit_bones, lengths, given)
    assert np.allclose(
        yd.read_segment_lengths(tmp_path / "s.csv", lay.fit_bones), lengths, atol=1e-5
    )
    # a 3D CSV of the same person works too
    yd.write_wide_xyz(tmp_path / "rec3d.csv", np.arange(10), xyz)
    from_3d = yd.read_segment_lengths(tmp_path / "rec3d.csv", lay.fit_bones, lay.mirror)
    assert np.allclose(from_3d, lengths, atol=1e-4)
    # a segment the file does not have is left to be estimated
    assert np.isnan(yd.read_segment_lengths(tmp_path / "s.csv", [(0, 32)])).all()


def test_crop_box_is_square_and_padded_outside_the_image():
    uv = np.array([[100.0, 50.0], [300.0, 850.0], [np.nan, np.nan]])
    box = yd.crop_box(uv, margin=0.25)
    assert box is not None
    x0, y0, side = box
    assert side == 1200 and (x0, y0) == (-400, -150)
    image = np.arange(1080 * 720, dtype=np.float32).reshape(1080, 720)
    crop = yd.crop_image(image, (x0, y0, side))
    assert crop.shape == (1200, 1200)
    assert crop[150 + 50, 400 + 100] == image[50, 100]  # the pixel keeps its place in the crop
    assert yd.crop_image(image, (x0, y0, side), fill="zero")[0, 0] == 0
    assert yd.crop_box(np.full((3, 2), np.nan)) is None


def test_the_pose_z_file_is_found_near_the_pixel_csv_and_read(tmp_path: Path):
    stem = "trial_c1"
    run = tmp_path / "mediapipe_x"
    (run / stem).mkdir(parents=True)
    cols = ["frame_index"] + [f"{n}_{a}" for n in ("nose", "left_eye") for a in "xyz"]
    pd.DataFrame(
        [[0, 0.5, 0.3, -0.2, 0.52, 0.29, -0.18], [1, 0.5, 0.3, -0.1, 0.52, 0.29, 0.05]],
        columns=pd.Index(cols),
    ).to_csv(run / stem / f"{stem}_mp_norm.csv", index=False)
    only_pixels = tmp_path / "inputs"
    only_pixels.mkdir()
    pixels = only_pixels / f"{stem}_pixel_vaila.csv"
    pixels.write_text("frame,p1_x,p1_y\n0,1,2\n")
    video = tmp_path / f"{stem}.mp4"
    found = yd.find_pose_z(pixels, video)
    assert found == run / stem / f"{stem}_mp_norm.csv"
    frames, z = yd.read_pose_z(found)
    assert frames.tolist() == [0, 1] and np.allclose(z, [[-0.2, -0.18], [-0.1, 0.05]])
    assert yd.find_pose_z(None, tmp_path / "other.mp4") is None


def test_a_person_is_fitted_and_the_segment_lengths_are_written(tmp_path: Path):
    L, _K, R, C = _camera()
    xyz = _standing_person(30)
    rows = {}
    for f, world in enumerate(xyz):
        z = (R @ (world - C).T).T[:, 2] / 1.6
        rows[f] = (dlt_project(L, world), np.ones(33), z, np.full(33, 9))
    dlt = tmp_path / "cam.dlt3d"
    _write_dlt(dlt, L)
    args = yd.build_parser().parse_args(["-i", "v.mp4", "--dlt3d", str(dlt)])
    frames, coeffs = yd.load_dlt3d_file(dlt)
    info = yd._finish_person(rows, "v", args, tmp_path, 30.0, 1080, 1920, frames, coeffs, None)
    assert info["scale_mode"] == "floor-frame" and info["fit"]["segment_lengths_given"] == 0
    table = pd.read_csv(tmp_path / info["fit"]["segment_lengths_csv"])
    assert len(table) == len(yd.MEDIAPIPE33.fit_bones) and (table["source"] == "estimated").all()
    _f, world = yd.read_wide_xyz(tmp_path / info["world_csv"])
    assert np.linalg.norm(world - xyz, axis=2).mean() < 0.03
    assert info["quality"]["segment_variation_median"] < 0.03


def test_the_printed_command_keeps_the_switches_that_are_on_by_default():
    args = yd.build_parser().parse_args(
        ["-i", "v.mp4", "--no-crop", "--no-balance", "--fit", "off"]
    )
    command = yd.cli_command(args)
    assert "--no-crop" in command and "--no-balance" in command and "--fit off" in command
    again = yd.build_parser().parse_args(__import__("shlex").split(command)[6:])
    assert again == args


def test_the_preview_hint_appears_only_when_the_preview_video_was_not_asked_for():
    off = yd.build_parser().parse_args(["-i", "v.mp4"])
    on = yd.build_parser().parse_args(["-i", "v.mp4", "--preview"])
    hint = yd.preview_hint(off)
    assert hint is not None and "--preview" in hint
    assert yd.preview_hint(on) is None
