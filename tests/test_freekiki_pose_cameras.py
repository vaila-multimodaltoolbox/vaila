"""
================================================================================
Tests of the per-frame cameras of the people (``pose_cameras``) in freekiki_detect3d
================================================================================

Version: 0.4.19
Update Date: 24 November 2026

The poses, ground tracks and the pitch gate of the 3D stage used only the FreeKiki DLT3D frames, so
a broadcast that pans through frames FreeKiki did not calibrate left the kicker with no pose at the
kick. ``pose_cameras`` (frame -> 11 DLT coefficients or 3x4, ``approximate`` frames left out) place
the people with the per-frame cameras of every frame; None keeps today's behaviour. Synthetic
cameras and people only: no GPU, no video.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from tests.test_freekiki_detect3d import _look_at_camera
from vaila import freekiki_ball as ball
from vaila import freekiki_detect3d as d3
from vaila import freekiki_field3d as f3
from vaila.fifa_to_dlt import compute_dlt3d_from_KRt

NAMES = ["left-ankle", "right-ankle", "nose"]
# the person stands here, in every frame
WORLD = np.array([[5.0, 2.0, 0.0], [5.2, 2.0, 0.0], [5.1, 2.0, 1.8]])
FOOT_XY = WORLD[:2, :2].mean(axis=0)


def _camera(focal: float, centre) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(L, R, C)`` of a pinhole camera at ``centre`` looking at the centre spot."""
    K, R, t = _look_at_camera(focal, np.asarray(centre, dtype=float), np.zeros(3))
    return compute_dlt3d_from_KRt(K, R, t), R, -R.T @ t


def _write_person(tmp_path: Path, cams: dict[int, tuple]) -> tuple[Path, Path]:
    """The person's camera csv (box) and keypoints csv in every frame of ``cams``.

    ``cams`` is frame -> ``(L, R, C)``: the SAM 3D Body camera-frame points and the box of each
    frame are those of that frame's own camera, like a real run.
    """
    camera_csv = tmp_path / "cam.csv"
    keypoints_csv = tmp_path / "kp.csv"
    with (
        camera_csv.open("w", encoding="utf-8") as cam_fh,
        keypoints_csv.open("w", encoding="utf-8") as kp_fh,
    ):
        cam_fh.write("frame,person_id,focal_length_px,cam_t_x_m,cam_t_y_m,cam_t_z_m,")
        cam_fh.write("bbox_x1_px,bbox_y1_px,bbox_x2_px,bbox_y2_px\n")
        kp_fh.write("frame,person_id,kpt_idx,kpt_name,x_m,y_m,z_m,xcam_m,ycam_m,zcam_m\n")
        for frame, (L, R, C) in sorted(cams.items()):
            feet_px = f3.project_points(L, WORLD[:2]).mean(axis=0)
            box = (feet_px[0] - 15, feet_px[1] - 90, feet_px[0] + 15, feet_px[1])
            cam_fh.write(f"{frame},0,2400,0,0,0,{box[0]},{box[1]},{box[2]},{box[3]}\n")
            xcam = (R @ (WORLD - C).T).T
            for i, name in enumerate(NAMES):
                x, y, z = xcam[i]
                kp_fh.write(f"{frame},0,{i},{name},0,0,0,{x},{y},{z}\n")
    return camera_csv, keypoints_csv


def _write(tmp_path: Path, name: str, **kwargs) -> Path:
    out = tmp_path / name
    out.mkdir()
    n = kwargs.pop("n_frames")
    d3.write_field_3d(
        tmp_path,
        out,
        n_frames=n,
        rate_hz=25.0,
        kept={0},
        ball_rows=[ball.ball_row(i, None) for i in range(n)],
        **kwargs,
    )
    return out


def _frames(path: Path) -> set[int]:
    with path.open(encoding="utf-8") as fh:
        return {int(r["frame"]) for r in csv.DictReader(fh)}


@pytest.fixture
def scene(tmp_path: Path) -> dict:
    """Frame 0: a FreeKiki DLT3D. Frames 1-2: other cameras (a pan), a per-frame solve."""
    cams = {
        0: _camera(2400.0, (0.0, -60.0, 25.0)),
        1: _camera(2600.0, (8.0, -66.0, 22.0)),
        2: _camera(2900.0, (14.0, -70.0, 20.0)),
    }
    camera_csv, keypoints_csv = _write_person(tmp_path, cams)
    return {
        "tmp": tmp_path,
        "cams": cams,
        "camera_csv": camera_csv,
        "keypoints_csv": keypoints_csv,
        "freekiki": {0: cams[0][0]},
        "per_frame": {f: c[0] for f, c in cams.items()},
    }


def test_the_poses_use_the_per_frame_cameras_where_freekiki_has_none(scene: dict) -> None:
    tmp = scene["tmp"]
    common = {
        "cameras": scene["freekiki"],
        "n_frames": 3,
        "camera_csv": scene["camera_csv"],
        "keypoints_csv": scene["keypoints_csv"],
    }
    today = _write(tmp, "today", **common)
    assert _frames(today / "poses_3d_on_field.csv") == {0}  # one calibrated frame only
    after = _write(tmp, "after", pose_cameras=scene["per_frame"], **common)
    assert _frames(after / "poses_3d_on_field.csv") == {0, 1, 2}
    assert _frames(after / "ground_tracks.csv") == {0, 1, 2}
    with (after / "ground_tracks.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):  # the box bottom through each frame's own camera
            assert (float(row["raw_x_m"]), float(row["raw_y_m"])) == pytest.approx(
                tuple(FOOT_XY), abs=0.02
            )
    with (after / "poses_3d_on_field.csv").open(encoding="utf-8") as fh:
        ground = {
            int(r["frame"]): (float(r["x_m"]), float(r["y_m"]))
            for r in csv.DictReader(fh)
            if r["kpt_name"] == "ground"
        }
    for frame in (1, 2):  # every frame stands where the person is, with no jump between cameras
        assert ground[frame] == pytest.approx(tuple(FOOT_XY), abs=0.1)


def test_the_default_keeps_todays_poses_and_the_field_keeps_its_cameras(scene: dict) -> None:
    tmp = scene["tmp"]
    common = {
        "cameras": scene["freekiki"],
        "n_frames": 3,
        "camera_csv": scene["camera_csv"],
        "keypoints_csv": scene["keypoints_csv"],
    }
    default = _write(tmp, "default", **common)
    same = _write(tmp, "same", pose_cameras=scene["freekiki"], **common)
    per_frame = _write(tmp, "per_frame", pose_cameras=scene["per_frame"], **common)
    for name in ("poses_3d_on_field.csv", "ground_tracks.csv", "field_kps_3d.csv"):
        assert (default / name).read_bytes() == (same / name).read_bytes(), name
    # the field points are the FreeKiki DLT3D's, whatever the people use
    assert (default / "field_kps_3d.csv").read_bytes() == (
        per_frame / "field_kps_3d.csv"
    ).read_bytes()
    assert (default / "ball_3d.csv").read_bytes() == (per_frame / "ball_3d.csv").read_bytes()


def test_pose_cameras_accept_3x4_projections_and_leave_a_frame_out(scene: dict) -> None:
    projections = {f: f3.dlt_projection_matrix(L) for f, L in scene["per_frame"].items()}
    del projections[1]  # an `approximate` frame is not given
    out = _write(
        scene["tmp"],
        "p34",
        cameras=scene["freekiki"],
        n_frames=3,
        camera_csv=scene["camera_csv"],
        keypoints_csv=scene["keypoints_csv"],
        pose_cameras=projections,
    )
    assert _frames(out / "poses_3d_on_field.csv") == {0, 2}


def test_a_planar_camera_gives_no_pose_in_its_frame_and_does_not_stop_the_run(
    scene: dict,
) -> None:
    planar = dict(scene["per_frame"])
    broken = np.array(planar[1], dtype=float)
    broken[[2, 6, 10]] = 0.0  # L3 = L7 = L11 = 0: a ground homography, no camera centre
    planar[1] = broken
    out = _write(
        scene["tmp"],
        "planar",
        cameras=scene["freekiki"],
        n_frames=3,
        camera_csv=scene["camera_csv"],
        keypoints_csv=scene["keypoints_csv"],
        pose_cameras=planar,
    )
    assert _frames(out / "poses_3d_on_field.csv") == {0, 2}


def test_the_summary_says_how_many_pose_cameras_were_used(scene: dict) -> None:
    out = scene["tmp"] / "summary"
    out.mkdir()
    summary = d3.write_field_3d(
        scene["tmp"],
        out,
        cameras=scene["freekiki"],
        n_frames=3,
        rate_hz=25.0,
        camera_csv=scene["camera_csv"],
        keypoints_csv=scene["keypoints_csv"],
        kept={0},
        ball_rows=[ball.ball_row(i, None) for i in range(3)],
        pose_cameras=scene["per_frame"],
    )
    assert summary["cameras"] == 1 and summary["pose_cameras"] == 3


def test_ground_homographies_of_cameras_match_the_projection() -> None:
    L, _, _ = _camera(2400.0, (0.0, -60.0, 25.0))
    homographies = d3.homographies_from_cameras({3: L, 4: np.full(11, np.nan)})
    assert sorted(homographies) == [3]  # a camera that is not finite is left out
    H = homographies[3]
    assert H[2, 2] == pytest.approx(1.0)
    xy = np.array([[5.0, 2.0], [-20.0, 10.0]])
    via_h = np.column_stack((xy, np.ones(2))) @ H.T
    via_h = via_h[:, :2] / via_h[:, 2:3]
    direct = f3.project_points(L, np.column_stack((xy, np.zeros(2))))
    assert via_h == pytest.approx(direct, abs=1e-6)


# --- the pitch gate of detect_3d_run -------------------------------------------------------------


def _gate_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pose_cameras) -> dict:
    """detect_3d_run with the SAM output of a 2D stage, two people, and a one-row ``.dlt2d``.

    Person 0 is seen in frame 0 (the only row of the ``.dlt2d``); person 1 only in frames 1-2, which
    only the per-frame cameras know. Everything after the gate is stubbed.
    """
    per_frame = _gate_scene_cameras()  # the true cameras of the clip (the boxes are made with them)
    L0 = per_frame[0]
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "m.dlt3d").write_text(
        "frame," + ",".join(f"L{i}" for i in range(1, 12)) + "\n0," + ",".join(map(str, L0)) + "\n",
        encoding="utf-8",
    )
    H0 = d3.homographies_from_cameras({0: L0})[0].reshape(9)[:8]
    (run_dir / "m.dlt2d").write_text(
        "frame," + ",".join(f"L{i}" for i in range(1, 9)) + "\n0," + ",".join(map(str, H0)) + "\n",
        encoding="utf-8",
    )
    work = tmp_path / "work"
    sam = work / "sam3dinov3" / "processed_x" / "m"
    sam.mkdir(parents=True)
    rows = ["frame,person_id,bbox_x1_px,bbox_y1_px,bbox_x2_px,bbox_y2_px"]
    for pid, frames in ((0, (0,)), (1, (1, 2))):
        for frame in frames:
            u, v = f3.project_points(per_frame[frame], WORLD[:1])[0]
            rows.append(f"{frame},{pid},{u - 15},{v - 90},{u + 15},{v}")
    (sam / "m_sam3dinov3_camera.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    seen: dict = {}
    monkeypatch.setattr(d3, "_video_info", lambda _v: (3, 1920, 1080))
    monkeypatch.setattr(d3, "_video_fps", lambda _v: 30.0)
    monkeypatch.setattr(d3, "focal_sequence", lambda *_a, **_k: np.full(3, np.nan))
    monkeypatch.setattr(d3, "vitruvian_refit", lambda *_a, **_k: {"status": "stub"})
    monkeypatch.setattr(d3, "write_ball_dependent_outputs", lambda *_a, **kw: seen.update(kw) or {})
    d3.detect_3d_run(
        run_dir,
        tmp_path / "m.mp4",
        tmp_path / "out",
        work_dir=work,
        skip_sam=True,
        players_on_field=True,
        pose_cameras=pose_cameras,
        ball_csv=None,
    )
    seen["verdicts"] = {
        int(r["person_id"]): r
        for r in csv.DictReader((tmp_path / "out" / "players_on_field.csv").open(encoding="utf-8"))
    }
    return seen


def _gate_scene_cameras() -> dict[int, np.ndarray]:
    return {
        0: _camera(2400.0, (0.0, -60.0, 25.0))[0],
        1: _camera(2600.0, (8.0, -66.0, 22.0))[0],
        2: _camera(2900.0, (14.0, -70.0, 20.0))[0],
    }


def test_the_pitch_gate_judges_a_person_seen_only_in_the_per_frame_cameras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    today = _gate_run(tmp_path / "a", monkeypatch, None)
    assert today["verdicts"][0]["keep"] == "True"
    # no .dlt2d row for frames 1-2: not a single frame is checked, so the person is not kept
    assert today["verdicts"][1]["keep"] == "False" and today["verdicts"][1]["checked"] == "0"
    assert today["pose_cameras"] is None
    after = _gate_run(tmp_path / "b", monkeypatch, _gate_scene_cameras())
    assert after["verdicts"][0]["keep"] == "True" and after["verdicts"][0]["checked"] == "1"
    assert after["verdicts"][1]["keep"] == "True" and after["verdicts"][1]["checked"] == "2"
    assert sorted(after["pose_cameras"]) == [0, 1, 2]  # and they reach the writer of the poses


def test_a_dlt2d_row_wins_over_the_camera_of_the_same_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frame 0 has a ``.dlt2d`` row: person 0 is judged by it even when ``pose_cameras`` differ."""
    cameras = _gate_scene_cameras()
    # a wrong camera for frame 0 would put the person off the pitch if it replaced the row
    cameras[0] = _camera(2400.0, (0.0, 60.0, 25.0))[0]
    seen = _gate_run(tmp_path / "c", monkeypatch, cameras)
    assert seen["verdicts"][0]["keep"] == "True" and seen["verdicts"][0]["checked"] == "1"


def test_rebuild_ball_in_place_hands_the_pose_cameras_to_the_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rebuild rewrites the poses, so it needs the cameras of the 3D stage (else they regress)."""
    work = tmp_path / "work"
    run_dir = work / "freekiki_predict_m_20261124"
    run_dir.mkdir(parents=True)
    (run_dir / "m.dlt3d").write_text("frame\n", encoding="utf-8")
    sam = work / "sam3dinov3" / "processed_x" / "m"
    sam.mkdir(parents=True)
    (sam / "m_sam3dinov3_camera.csv").write_text("frame,person_id\n", encoding="utf-8")
    (sam / "m_sam3dinov3_keypoints3d.csv").write_text("frame,person_id\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    markers = tmp_path / "m_markers.csv"
    markers.write_text("frame,p0_x,p0_y\n0,10,10\n", encoding="utf-8")
    seen: dict = {}
    monkeypatch.setattr(d3, "_video_info", lambda _v: (2, 1920, 1080))
    monkeypatch.setattr(d3, "_video_fps", lambda _v: 30.0)
    monkeypatch.setattr(d3, "write_ball_dependent_outputs", lambda *_a, **kw: seen.update(kw) or {})
    cams = _gate_scene_cameras()
    d3.rebuild_ball_in_place(out, tmp_path / "m.mp4", work, markers, pose_cameras=cams)
    assert seen["pose_cameras"] is cams
    seen.clear()
    d3.rebuild_ball_in_place(out, tmp_path / "m.mp4", work, markers)
    assert seen["pose_cameras"] is None  # today's behaviour for the other callers
