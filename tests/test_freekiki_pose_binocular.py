"""Two-camera triangulation of the people's keypoints (vaila/freekiki_pose_binocular.py).

Two synthetic cameras look at a known person. Their 2D keypoints are generated from the true 3D
points, so the triangulation must recover them. The timing tests put the other camera on a
different clock, and a track test checks that a person is only read on its own frames.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from vaila import freekiki_pose_binocular as pb
from vaila.fifa_to_dlt import compute_dlt3d_from_KRt
from vaila.freekiki_stereo import CamView
from vaila.monocular_dlt_align import dlt_projection_matrix

W, H = 1920.0, 1080.0
K = np.array([[2400.0, 0.0, W / 2.0], [0.0, 2400.0, H / 2.0], [0.0, 0.0, 1.0]])
FPS = 59.94


def _camera(centre, target) -> np.ndarray:
    centre, target = np.asarray(centre, float), np.asarray(target, float)
    z = target - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return compute_dlt3d_from_KRt(K, R, -R @ centre)


L_A = _camera([-60.0, -45.0, 25.0], [-30.0, -10.0, 1.0])
L_B = _camera([-75.0, 20.0, 22.0], [-30.0, -10.0, 1.0])


def _pix(L: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    P = dlt_projection_matrix(L)
    h = P @ np.append(xyz, 1.0)
    return h[:2] / h[2]


def _skeleton(n_points: int = pb.KPT_COUNT) -> np.ndarray:
    """A standing person's keypoints around (-30, -10) on the pitch, 70 points."""
    rng = np.random.default_rng(7)
    base = np.array([-30.0, -10.0, 0.0])
    pts = base + rng.uniform([-0.2, -0.2, 0.0], [0.2, 0.2, 1.8], size=(n_points, 3))
    return pts


def _walk(sk: np.ndarray, n: int, velocity=(6.0, 0.0, 0.0)) -> np.ndarray:
    """``(n, K, 3)``: the skeleton moving at a constant velocity (metres per second)."""
    tau = np.arange(n) / FPS
    return sk[None, :, :] + tau[:, None, None] * np.asarray(velocity)[None, None, :]


def _kps(L: np.ndarray, world: np.ndarray, *, offset: int = 0, n_frames: int) -> pb.Keypoints:
    """2D keypoints of a camera, frame by frame, of the 3D points ``world`` (``(n, K, 3)``)."""
    arr = np.full((n_frames, pb.KPT_COUNT, 2), np.nan)
    for f in range(n_frames):
        src = f - offset
        if 0 <= src < len(world):
            for k in range(pb.KPT_COUNT):
                arr[f, k] = _pix(L, world[src, k])
    names = [f"kpt{k}" for k in range(pb.KPT_COUNT)]
    names[0], names[1] = "left-heel", "right-heel"
    return pb.Keypoints(names=names, by_person={1: arr})


def _view(name, L, n, *, sync, pts=None) -> CamView:
    dlt = dict.fromkeys(range(n), L)
    return CamView(
        name=name,
        dlt=dlt,
        markers={},
        cut_start=0,
        sync_cut=sync,
        pts_cut=np.arange(n) / FPS if pts is None else pts,
    )


def test_triangulation_recovers_the_true_keypoints() -> None:
    n = 6
    world = _walk(_skeleton(), n, velocity=(0.0, 0.0, 0.0))
    ref = _view("ref", L_A, n, sync=0)
    other = _view("other", L_B, n, sync=0)
    rows = pb.triangulate_people(
        ref,
        other,
        ref_tracks=[pb.TrackSpan(1, 5, 0, n - 1)],
        other_tracks=[pb.TrackSpan(1, 9, 0, n - 1)],
        ref_kps=pb.Keypoints(names=[f"k{k}" for k in range(pb.KPT_COUNT)], by_person={5: _kps(L_A, world, n_frames=n).by_person[1]}),
        other_kps=pb.Keypoints(names=[f"k{k}" for k in range(pb.KPT_COUNT)], by_person={9: _kps(L_B, world, n_frames=n).by_person[1]}),
    )  # fmt: skip
    valid = [r for r in rows if r["valid"]]
    assert len(valid) == n * pb.KPT_COUNT
    for r in valid[:: pb.KPT_COUNT]:
        k = r["kpt_idx"]
        xyz = np.array([r["x_m"], r["y_m"], r["z_m"]])
        assert np.linalg.norm(xyz - world[r["frame"], k]) < 1e-6
        assert r["resid_ref_px"] < 1e-3 and r["resid_other_px"] < 1e-3


def test_nearly_parallel_rays_are_not_valid() -> None:
    n = 2
    world = _walk(_skeleton(), n, velocity=(0.0, 0.0, 0.0))
    near = _camera([-60.0, -45.0, 25.0], [-30.0, -10.0, 1.0])
    near_other = _camera([-60.01, -45.0, 25.0], [-30.0, -10.0, 1.0])
    kps_a = _kps(near, world, n_frames=n)
    kps_b = _kps(near_other, world, n_frames=n)
    rows = pb.triangulate_people(
        _view("ref", near, n, sync=0),
        _view("other", near_other, n, sync=0),
        ref_tracks=[pb.TrackSpan(1, 5, 0, n - 1)],
        other_tracks=[pb.TrackSpan(1, 9, 0, n - 1)],
        ref_kps=pb.Keypoints(names=kps_a.names, by_person={5: kps_a.by_person[1]}),
        other_kps=pb.Keypoints(names=kps_b.names, by_person={9: kps_b.by_person[1]}),
    )
    assert rows and not any(r["valid"] for r in rows)
    assert all("apart" in r["reason"] for r in rows if r["kpt_idx"] >= 0)


def test_a_residual_above_the_limit_is_not_valid() -> None:
    """The linear triangulation spreads a disagreement over both views (about half in each)."""
    n = 1
    world = _walk(_skeleton(), n, velocity=(0.0, 0.0, 0.0))
    kps_a = _kps(L_A, world, n_frames=n)
    kps_b = _kps(L_B, world, n_frames=n)
    kps_b.by_person[1][0, :, 0] += 80.0  # the two rays disagree by about 2 * 16 px
    rows = pb.triangulate_people(
        _view("ref", L_A, n, sync=0),
        _view("other", L_B, n, sync=0),
        ref_tracks=[pb.TrackSpan(1, 5, 0, n - 1)],
        other_tracks=[pb.TrackSpan(1, 9, 0, n - 1)],
        ref_kps=pb.Keypoints(names=kps_a.names, by_person={5: kps_a.by_person[1]}),
        other_kps=pb.Keypoints(names=kps_b.names, by_person={9: kps_b.by_person[1]}),
    )
    assert not any(r["valid"] for r in rows)
    assert any("residual above" in r["reason"] for r in rows)


def test_the_other_camera_is_read_at_the_same_instant() -> None:
    """Reference 1-60 fps, other camera started 3 frames later: the moving person must still match."""
    n = 40
    world = _walk(_skeleton(), n, velocity=(6.0, 0.0, 0.0))
    # the other camera's frame f shows the instant of its own frame f; its sync is frame 3
    other_offset = 3
    kps_b = _kps(L_B, world, offset=other_offset, n_frames=n + other_offset)
    ref_kps = _kps(L_A, world, n_frames=n)
    ref = _view("ref", L_A, n, sync=0)
    other = _view("other", L_B, n + other_offset, sync=other_offset)
    rows = pb.triangulate_people(
        ref,
        other,
        ref_tracks=[pb.TrackSpan(1, 5, 0, n - 1)],
        other_tracks=[pb.TrackSpan(1, 9, 0, n + other_offset - 1)],
        ref_kps=pb.Keypoints(names=ref_kps.names, by_person={5: ref_kps.by_person[1]}),
        other_kps=pb.Keypoints(names=kps_b.names, by_person={9: kps_b.by_person[1]}),
    )
    valid = [r for r in rows if r["valid"] and r["frame"] > 0 and r["kpt_idx"] == 3]
    assert valid
    for r in valid:
        assert (
            np.linalg.norm(np.array([r["x_m"], r["y_m"], r["z_m"]]) - world[r["frame"], 3]) < 1e-5
        )


def test_a_person_is_read_only_on_its_own_track_frames() -> None:
    n = 10
    world = _walk(_skeleton(), n, velocity=(0.0, 0.0, 0.0))
    kps_a = _kps(L_A, world, n_frames=n)
    kps_b = _kps(L_B, world, n_frames=n)
    rows = pb.triangulate_people(
        _view("ref", L_A, n, sync=0),
        _view("other", L_B, n, sync=0),
        ref_tracks=[pb.TrackSpan(1, 5, 2, 4)],
        other_tracks=[pb.TrackSpan(1, 9, 0, n - 1)],
        ref_kps=pb.Keypoints(names=kps_a.names, by_person={5: kps_a.by_person[1]}),
        other_kps=pb.Keypoints(names=kps_b.names, by_person={9: kps_b.by_person[1]}),
    )
    frames = {r["frame"] for r in rows}
    assert frames == {2, 3, 4}


def test_the_vitruvian_dlt_is_recorded_per_row() -> None:
    n = 1
    world = _walk(_skeleton(), n, velocity=(0.0, 0.0, 0.0))
    kps_a = _kps(L_A, world, n_frames=n)
    kps_b = _kps(L_B, world, n_frames=n)
    rows = pb.triangulate_people(
        _view("ref", L_A, n, sync=0),
        _view("other", L_B, n, sync=0),
        ref_tracks=[pb.TrackSpan(1, 5, 0, 0)],
        other_tracks=[pb.TrackSpan(1, 9, 0, 0)],
        ref_kps=pb.Keypoints(names=kps_a.names, by_person={5: kps_a.by_person[1]}),
        other_kps=pb.Keypoints(names=kps_b.names, by_person={9: kps_b.by_person[1]}),
        dlt_source_ref={0: "vitruvian"},
    )
    assert {r["dlt_ref"] for r in rows} == {"vitruvian"}


def test_write_poses_leaves_missing_numbers_empty(tmp_path: Path) -> None:
    rows = [
        {
            "frame": 0,
            "time_s": 0.0,
            "player_id": 1,
            "kpt_idx": -1,
            "kpt_name": "",
            "x_m": np.nan,
            "y_m": np.nan,
            "z_m": np.nan,
            "resid_ref_px": np.nan,
            "resid_other_px": np.nan,
            "angle_deg": np.nan,
            "dlt_ref": "run",
            "dlt_other": "",
            "valid": False,
            "reason": "not paired on this instant",
        },  # fmt: skip
        {
            "frame": 1,
            "time_s": 0.0167,
            "player_id": 1,
            "kpt_idx": 3,
            "kpt_name": "nose",
            "x_m": 1.23456789,
            "y_m": 2.0,
            "z_m": 0.5,
            "resid_ref_px": 0.1,
            "resid_other_px": 0.2,
            "angle_deg": 30.0,
            "dlt_ref": "run",
            "dlt_other": "run",
            "valid": True,
            "reason": "",
        },  # fmt: skip
    ]
    path = pb.write_poses(rows, tmp_path / "poses.csv")
    with path.open(newline="", encoding="utf-8") as fh:
        out = list(csv.DictReader(fh))
    assert out[0]["x_m"] == "" and out[0]["valid"] == "False"
    assert out[1]["x_m"] == "1.234568" and out[1]["valid"] == "True"
    summary = pb.summarize(rows)
    assert summary["1"]["valid"] == 1 and summary["1"]["points"] == 1
