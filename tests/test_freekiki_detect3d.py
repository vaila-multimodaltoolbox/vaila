"""Pure-logic tests for the FreeKiki 3D chain (no GPU, no weights, no video model).

Covers the per-frame focal from the field DLT3D, the keypoint propagation gate,
the pitch / feet-on-field test, the ball row selection and the CSV helpers of
``freekiki_detect3d``. The SAM 3 and SAM 3D Body calls are not exercised here.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest

from vaila import freekiki_ball as ball
from vaila import freekiki_camera3d as cam3d
from vaila import freekiki_detect3d as d3
from vaila import freekiki_players as players
from vaila import freekiki_tracking as trk
from vaila.fifa_to_dlt import compute_dlt3d_from_KRt

W, H_IMG = 1920, 1080


def _look_at_camera(f: float, centre: np.ndarray, target: np.ndarray):
    """Pinhole K, R (world -> camera, OpenCV axes) and t for a camera looking at ``target``."""
    up = np.array([0.0, 0.0, 1.0])
    z = target - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    t = -R @ centre
    K = np.array([[f, 0.0, W / 2.0], [0.0, f, H_IMG / 2.0], [0.0, 0.0, 1.0]])
    return K, R, t


def _dlt3d_row(f: float) -> np.ndarray:
    K, R, t = _look_at_camera(f, np.array([0.0, -60.0, 25.0]), np.zeros(3))
    return compute_dlt3d_from_KRt(K, R, t)


def test_focal_is_recovered_from_a_synthetic_dlt3d() -> None:
    got = cam3d.focal_from_dlt3d(_dlt3d_row(2400.0), W)
    assert got is not None
    assert got == pytest.approx(2400.0, rel=1e-3)


def test_focal_rejects_planar_and_out_of_range_dlt3d() -> None:
    # A planar DLT3D (all control points on Z=0) cannot be decomposed.
    planar = np.zeros(11)
    planar[[0, 1, 2, 3, 4, 5, 6, 7]] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    planar[8] = 0.0
    planar[9] = 0.0
    planar[10] = 0.0
    assert cam3d.focal_from_dlt3d(planar, W) is None
    # A focal far outside the image width is a failed fit, not a lens.
    assert cam3d.focal_from_dlt3d(_dlt3d_row(2400.0), 10.0) is None


def test_focal_sequence_reads_frames_and_leaves_gaps_nan(tmp_path: Path) -> None:
    path = tmp_path / "clip.dlt3d"
    with path.open("w", newline="") as fh:
        fh.write("frame," + ",".join(f"L{i}" for i in range(1, 12)) + "\n")
        for frame, f in ((0, 2000.0), (2, 2200.0)):
            row = _dlt3d_row(f)
            fh.write(f"{frame}," + ",".join(f"{v:.12g}" for v in row) + "\n")
    focals = cam3d.focal_sequence(path, n_frames=4, width=float(W))
    assert focals[0] == pytest.approx(2000.0, rel=1e-3)
    assert np.isnan(focals[1])
    assert focals[2] == pytest.approx(2200.0, rel=1e-3)
    assert np.isnan(focals[3])


def test_cam_int_sequence_centres_principal_point_and_keeps_nan_rows() -> None:
    cam = cam3d.cam_int_sequence(np.array([1500.0, np.nan]), 1920, 1080)
    assert cam.shape == (2, 3, 3)
    assert cam[0, 0, 0] == 1500.0 and cam[0, 1, 1] == 1500.0
    assert cam[0, 0, 2] == 960.0 and cam[0, 1, 2] == 540.0
    assert cam[0, 0, 1] == 0.0
    assert np.isnan(cam[1]).all()


def test_propagation_detected_near_prior_replaces_the_prior() -> None:
    prev = np.array([[100.0, 100.0]])
    out_xy, out_ok, src = trk.propagate_keypoints(
        prev, np.array([True]), np.array([[104.0, 98.0]]), np.array([0.9]),
        kp_conf=0.5, band_px=10.0,
    )  # fmt: skip
    assert src == [trk.SOURCE_DETECTED]
    assert out_ok[0]
    assert out_xy[0] == pytest.approx([104.0, 98.0])


def test_propagation_far_detection_keeps_the_prior() -> None:
    prev = np.array([[100.0, 100.0]])
    out_xy, out_ok, src = trk.propagate_keypoints(
        prev, np.array([True]), np.array([[400.0, 400.0]]), np.array([0.9]),
        kp_conf=0.5, band_px=10.0,
    )  # fmt: skip
    assert src == [trk.SOURCE_PROPAGATED]
    assert out_xy[0] == pytest.approx([100.0, 100.0])
    assert out_ok[0]


def test_propagation_low_confidence_and_missing_prior() -> None:
    xy = np.array([[50.0, 50.0], [60.0, 60.0], [70.0, 70.0]])
    kc = np.array([0.1, 0.9, 0.2])
    prev = np.array([[50.0, 50.0], [0.0, 0.0], [0.0, 0.0]])
    prev_ok = np.array([True, False, False])
    out_xy, out_ok, src = trk.propagate_keypoints(prev, prev_ok, xy, kc, kp_conf=0.5, band_px=10.0)
    assert src == [trk.SOURCE_PROPAGATED, trk.SOURCE_DETECTED, trk.SOURCE_NONE]
    assert list(out_ok) == [True, True, False]


def test_track_sequence_carries_a_point_through_a_gap() -> None:
    frames_xy = [np.array([[100.0, 100.0]]), np.array([[100.0, 100.0]]), np.array([[103.0, 101.0]])]
    frames_kc = [np.array([0.9]), np.array([0.1]), np.array([0.9])]
    results = trk.track_sequence(frames_xy, frames_kc, kp_conf=0.5, band_px=10.0)
    assert [r[2][0] for r in results] == [
        trk.SOURCE_DETECTED,
        trk.SOURCE_PROPAGATED,
        trk.SOURCE_DETECTED,
    ]


def test_pitch_outline_projects_the_corners_of_a_known_homography() -> None:
    # World (X, Y) -> pixel: x = 10*X + 1000, y = -10*Y + 500.
    coeffs = np.array([10.0, 0.0, 1000.0, 0.0, -10.0, 500.0, 0.0, 0.0])
    outline = players.pitch_outline_px(players.homography_from_dlt2d(coeffs))
    expected = np.array(
        [
            [1000.0 - 10 * players.PITCH_HALF_LENGTH_M, 500.0 + 10 * players.PITCH_HALF_WIDTH_M],
            [1000.0 + 10 * players.PITCH_HALF_LENGTH_M, 500.0 + 10 * players.PITCH_HALF_WIDTH_M],
            [1000.0 + 10 * players.PITCH_HALF_LENGTH_M, 500.0 - 10 * players.PITCH_HALF_WIDTH_M],
            [1000.0 - 10 * players.PITCH_HALF_LENGTH_M, 500.0 - 10 * players.PITCH_HALF_WIDTH_M],
        ]
    )
    assert outline == pytest.approx(expected)


def test_feet_point_is_bottom_centre_of_the_box() -> None:
    assert players.feet_point((100.0, 200.0, 140.0, 400.0)) == (120.0, 400.0)


def test_point_on_pitch_inside_and_outside() -> None:
    coeffs = np.array([10.0, 0.0, 1000.0, 0.0, -10.0, 500.0, 0.0, 0.0])
    outline = players.pitch_outline_px(players.homography_from_dlt2d(coeffs))
    assert players.point_on_pitch(outline, (1000.0, 500.0))
    assert not players.point_on_pitch(outline, (10.0, 10.0))
    assert not players.point_on_pitch(np.full((4, 2), np.nan), (1000.0, 500.0))


def test_players_on_field_keeps_tracks_whose_feet_stay_on_the_pitch() -> None:
    coeffs = np.array([10.0, 0.0, 1000.0, 0.0, -10.0, 500.0, 0.0, 0.0])
    homographies = {f: players.homography_from_dlt2d(coeffs) for f in range(4)}
    on = (990.0, 100.0, 1010.0, 500.0)  # feet at (1000, 500), centre of the pitch
    off = (10.0, 100.0, 30.0, 120.0)  # feet far outside
    tracks = {
        1: [(0, np.array(on)), (1, np.array(on)), (2, np.array(on)), (3, np.array(off))],
        2: [(0, np.array(off)), (1, np.array(off)), (2, np.array(on)), (3, np.array(off))],
    }
    verdicts = players.players_on_field(tracks, homographies, min_frac=0.5)
    assert verdicts[1]["keep"] is True and verdicts[1]["on"] == 3
    assert verdicts[2]["keep"] is False and verdicts[2]["fraction"] == pytest.approx(0.25)


def test_players_on_field_ignores_frames_without_a_homography() -> None:
    coeffs = np.array([10.0, 0.0, 1000.0, 0.0, -10.0, 500.0, 0.0, 0.0])
    homographies = {0: players.homography_from_dlt2d(coeffs)}
    box = np.array([990.0, 100.0, 1010.0, 500.0])
    tracks = {7: [(0, box), (5, np.array([0.0, 0.0, 1.0, 1.0]))]}
    verdict = players.players_on_field(tracks, homographies, min_frac=1.0)[7]
    assert verdict["checked"] == 1 and verdict["keep"] is True
    with pytest.raises(ValueError):
        players.players_on_field(tracks, homographies, min_frac=1.5)


def test_best_ball_picks_the_most_confident_finite_box() -> None:
    detections = [
        (0.3, np.array([0.0, 0.0, 10.0, 10.0])),
        (0.8, np.array([20.0, 20.0, 30.0, 30.0])),
        (0.95, np.array([np.nan, 0.0, 1.0, 1.0])),
    ]
    score, box = ball.best_ball(detections)
    assert score == pytest.approx(0.8)
    assert box.tolist() == [20.0, 20.0, 30.0, 30.0]
    assert ball.best_ball([]) is None


def test_ball_row_centre_and_empty_frame(tmp_path: Path) -> None:
    row = ball.ball_row(3, (0.7, np.array([10.0, 20.0, 30.0, 60.0])))
    assert row["cx"] == 20.0 and row["cy"] == 40.0 and row["score"] == 0.7
    empty = ball.ball_row(4, None)
    assert empty["cx"] == "" and empty["score"] == ""
    out = ball.write_ball_csv(tmp_path / "ball.csv", [row, empty])
    with out.open(newline="") as fh:
        read = list(csv.DictReader(fh))
    assert list(read[0].keys()) == ball.BALL_COLUMNS
    assert read[0]["frame"] == "3" and read[1]["cx"] == ""


def test_run_folder_helpers_read_the_detect_outputs(tmp_path: Path) -> None:
    dlt2d = tmp_path / "clip.dlt2d"
    dlt2d.write_text(
        "frame," + ",".join(f"L{i}" for i in range(1, 9)) + "\n0,10,0,1000,0,-10,500,0,0\n",
        encoding="utf-8",
    )
    homographies = d3._read_homographies(dlt2d)
    assert set(homographies) == {0}
    assert homographies[0][0, 0] == pytest.approx(10.0)

    camera = tmp_path / "clip_sam3dinov3_camera.csv"
    camera.write_text(
        "frame,person_id,focal_length_px,cam_t_x_m,cam_t_y_m,cam_t_z_m,"
        "bbox_x1_px,bbox_y1_px,bbox_x2_px,bbox_y2_px\n"
        "0,3,1500,0,0,0,10,20,30,40\n"
        "1,3,1500,0,0,0,10,20,30,40\n"
        "1,9,1500,0,0,0,50,60,70,80\n",
        encoding="utf-8",
    )
    tracks = d3._read_person_tracks(camera)
    assert sorted(tracks) == [3, 9]
    assert [f for f, _ in tracks[3]] == [0, 1]

    kept = tmp_path / "kept.csv"
    d3._filter_person_csv(camera, kept, {9})
    lines = kept.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("frame,person_id")
    assert len(lines) == 2 and lines[1].split(",")[1] == "9"


def test_detect_soccer_3d_chains_detect_then_3d(monkeypatch, tmp_path: Path) -> None:
    # Orchestration only: detect_videos and detect_3d are stubbed, no model runs.
    from vaila import freekiki

    calls: dict[str, object] = {}

    def fake_detect_videos(ws, source, **kwargs):
        calls["detect"] = {"ws": ws, "source": source, **kwargs}
        return tmp_path / "run"

    def fake_detect_3d(run, **kwargs):
        calls["3d"] = {"run": run, **kwargs}
        # the real detect_3d returns one summary per video, with the folder it wrote ("out")
        return [{"video": "clip.mp4", "out": str(kwargs["out_root"])}]

    monkeypatch.setattr(freekiki, "detect_videos", fake_detect_videos)
    monkeypatch.setattr(freekiki, "detect_3d", fake_detect_3d)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")  # the folder name comes from the file, so it must exist
    out = freekiki.detect_soccer_3d(
        tmp_path / "ws",
        video,
        device="0",
        sam_device="cuda",
        ball=True,
        ball_weights="sam3.pt",
        players_on_field=True,
        snap=False,
    )

    assert [result["video"] for result in out] == ["clip.mp4"]
    # The detection device and the SAM 3D Body device are separate options.
    assert calls["detect"]["device"] == "0"
    assert calls["detect"]["snap"] is False
    # One flat folder "<clip>_3d_<timestamp>": the detect run goes under its work/ folder.
    final = calls["3d"]["out_root"]
    assert final.parent == tmp_path and final.name.startswith("clip_3d_")
    assert calls["detect"]["output_dir"] == final / "work"
    assert calls["3d"]["run"] == tmp_path / "run"
    assert calls["3d"]["device"] == "cuda"
    assert calls["3d"]["ball"] is True
    assert calls["3d"]["players_on_field"] is True


def test_vitruvian_refit_recovers_a_planar_frame(tmp_path: Path) -> None:
    # A weak frame (homography camera) with noise-free field points and four people:
    # the boxes must supply the vertical scale, and the refit must reproject the
    # planar schema points within a pixel.
    from vaila import freekiki_diag
    from vaila.soccerfield_vitruvian_dlt3d import project_dlt3d

    K, R, t = _look_at_camera(2000.0, np.array([0.0, -60.0, 25.0]), np.zeros(3))
    L = compute_dlt3d_from_KRt(K, R, t)
    _, _, xyz = freekiki_diag.load_field_points()
    field_px = project_dlt3d(L, xyz)

    run = tmp_path / "run"
    run.mkdir()
    names, _, _ = freekiki_diag.load_field_points()
    header = ["frame"] + [f"p{i}_{axis}" for i in range(len(names)) for axis in "xy"]
    with (run / "field_kps_getpixelvideo.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerow([0] + [f"{v:.4f}" for v in field_px.ravel()])
    with (run / "field_kps_geom_status.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "camera"])
        writer.writerow([0, "homography"])

    people = [(-20.0, -10.0), (5.0, 5.0), (15.0, -12.0), (-30.0, 8.0)]
    camera = tmp_path / "camera.csv"
    with camera.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["frame", "person_id", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px"]
        )
        for pid, (x, y) in enumerate(people):
            foot = project_dlt3d(L, np.array([[x, y, 0.0]]))[0]
            head = project_dlt3d(L, np.array([[x, y, 1.83]]))[0]  # the height the fit assumes
            writer.writerow([0, pid, foot[0] - 15, head[1], foot[0] + 15, foot[1]])

    result = d3.vitruvian_refit(run, camera, tmp_path / "out")
    assert result["status"] == "ok"
    assert result["refit"] == 1 and result["failed"] == 0

    with (tmp_path / "out" / "vitruvian.dlt3d").open(encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    L_fit = np.array([float(row[f"L{i}"]) for i in range(1, 12)])
    planar = xyz[:, 2] == 0
    reproj = project_dlt3d(L_fit, xyz[planar])
    assert np.max(np.linalg.norm(reproj - field_px[planar], axis=1)) < 1.0


def test_ball_mode_accepts_bools_and_the_three_names() -> None:
    assert d3._ball_mode(True) == "sam3"
    assert d3._ball_mode(False) == "off"
    assert d3._ball_mode("net") == "net"
    assert d3._ball_mode("off") == "off"
    with pytest.raises(ValueError):
        d3._ball_mode("yolo")


def test_write_field_3d_writes_pitch_tables_and_c3d(tmp_path: Path) -> None:
    pytest.importorskip("ezc3d")
    import ezc3d

    from vaila import freekiki_field3d as f3

    K, R, t = _look_at_camera(2400.0, np.array([0.0, -60.0, 25.0]), np.zeros(3))
    C = -R.T @ t
    L = compute_dlt3d_from_KRt(K, R, t)
    # One person standing at (5, 2) on the pitch, 1.8 m tall, in two frames.
    names = ["left-ankle", "right-ankle", "nose"]
    world = np.array([[5.0, 2.0, 0.0], [5.2, 2.0, 0.0], [5.1, 2.0, 1.8]])
    xcam = (R @ (world - C).T).T
    feet_px = f3.project_points(L, world[:2]).mean(axis=0)
    box = (feet_px[0] - 15, feet_px[1] - 90, feet_px[0] + 15, feet_px[1])
    camera_csv = tmp_path / "cam.csv"
    with camera_csv.open("w", encoding="utf-8") as fh:
        fh.write("frame,person_id,focal_length_px,cam_t_x_m,cam_t_y_m,cam_t_z_m,")
        fh.write("bbox_x1_px,bbox_y1_px,bbox_x2_px,bbox_y2_px\n")
        for frame in (0, 1):
            fh.write(f"{frame},0,2400,0,0,0,{box[0]},{box[1]},{box[2]},{box[3]}\n")
    keypoints_csv = tmp_path / "kp.csv"
    with keypoints_csv.open("w", encoding="utf-8") as fh:
        fh.write("frame,person_id,kpt_idx,kpt_name,x_m,y_m,z_m,xcam_m,ycam_m,zcam_m\n")
        for frame in (0, 1):
            for i, name in enumerate(names):
                x, y, z = xcam[i]
                fh.write(f"{frame},0,{i},{name},0,0,0,{x},{y},{z}\n")
    ball_px = f3.project_points(L, np.array([[20.0, 10.0, f3.BALL_RADIUS_M]]))[0]
    ball_rows = [
        ball.ball_row(
            0, (0.9, np.array([ball_px[0] - 4, ball_px[1] - 4, ball_px[0] + 4, ball_px[1] + 4]))
        ),
        ball.ball_row(1, None),
    ]
    out = tmp_path / "out"
    out.mkdir()
    summary = d3.write_field_3d(
        tmp_path,
        out,
        cameras={0: L, 1: L},
        n_frames=2,
        rate_hz=25.0,
        camera_csv=camera_csv,
        keypoints_csv=keypoints_csv,
        kept={0},
        ball_rows=ball_rows,
    )

    field = (out / "field_kps_3d.csv").read_text(encoding="utf-8").splitlines()
    assert len(field) == 1 + 2 * 49
    poses = (out / "poses_3d_on_field.csv").read_text(encoding="utf-8").splitlines()
    # Each frame: the pose keypoints plus the "ground" marker of the person.
    assert len(poses) == 1 + 2 * (len(names) + 1)
    assert (out / "ground_tracks.csv").is_file()
    ball_on = (out / "ball_3d_on_field.csv").read_text(encoding="utf-8").splitlines()
    # Frame 1 has no detected ball, but a gap of one frame is bridged by the physical track.
    assert len(ball_on) == 1 + 2
    # 49 field points, 3 pose keypoints of person 0, and the ball.
    # One C3D per group, flat in the output folder: field points and ball (49 + 1),
    # then one per person (3 here).
    back = ezc3d.c3d(str(out / "field_ball.c3d"))
    labels = back["parameters"]["POINT"]["LABELS"]["value"]
    assert len(labels) == 49 + 1 and labels[-1] == "BALL"
    person = ezc3d.c3d(str(out / "person_000.c3d"))
    assert person["parameters"]["POINT"]["LABELS"]["value"] == [
        f"P0_{n}" for n in [*names, "ground"]
    ]
    assert summary["c3d_files"] == 3  # field_ball, ball alone, and one person
    assert summary["markers"] == 49 + 1 + 1 + len(names) + 1
    ball_only = ezc3d.c3d(str(out / "ball.c3d"))
    assert ball_only["parameters"]["POINT"]["LABELS"]["value"] == ["BALL"]
    # The pose stands on the pitch: the feet mean is on the ground under the box.
    row = [r for r in csv.DictReader(poses) if r["kpt_name"] == "nose"][0]
    # the nose is scaled to the stature prior: 1.83 m minus 0.10 m to the top of the head
    assert float(row["z_m"]) == pytest.approx(1.83 - 0.10, abs=1e-3)


@pytest.mark.parametrize("as_script", [False, True], ids=["package", "script"])
def test_overlay_draws_field_points_kept_poses_and_ball(
    tmp_path: Path, monkeypatch, as_script: bool
) -> None:
    # as_script loads the module the way freekiki.py runs it: a top-level module, no package,
    # so the relative imports fall back to the plain module names.
    cv2 = pytest.importorskip("cv2")
    import importlib
    import sys

    from vaila.sam3dinov3 import MHR70_NAMES

    module = d3
    if as_script:
        monkeypatch.syspath_prepend(str(Path(d3.__file__).resolve().parent))
        monkeypatch.delitem(sys.modules, "freekiki_detect3d", raising=False)
        module = importlib.import_module("freekiki_detect3d")
        assert module.__package__ in (None, "")

    run, out, work = tmp_path / "run", tmp_path / "out", tmp_path / "work" / "sam3dinov3"
    for folder in (run, out, work):
        folder.mkdir(parents=True)
    video = tmp_path / "clip.mp4"
    size = (160, 120)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    for _ in range(2):
        writer.write(np.zeros((size[1], size[0], 3), np.uint8))
    writer.release()

    def csv_rows(path: Path, header: str, rows: list[list[str]]) -> None:
        path.write_text(
            header + "\n" + "\n".join(",".join(r) for r in rows) + "\n", encoding="utf-8"
        )

    # Field: point 0 accepted in both frames at (20, 20); the other 48 blank.
    field_header = "frame," + ",".join(f"p{i}_{c}" for i in range(49) for c in "xy")
    field_rows = []
    for f in (0, 1):
        vals = [""] * 98
        vals[0], vals[1] = "20", "20"
        field_rows.append([str(f), *vals])
    csv_rows(run / "field_kps_getpixelvideo.csv", field_header, field_rows)

    # Poses: person 3 (kept) has neck at (100, 80); person 4 (not kept) has a nose at (30, 90).
    index = {name: i for i, name in enumerate(MHR70_NAMES)}
    pose_header = "frame," + ",".join(f"p{i}_{c}" for i in range(70) for c in "xy")

    def pose_rows(name_index: int, xy: tuple[str, str]) -> list[list[str]]:
        rows = []
        for f in (0, 1):
            vals = [""] * 140
            vals[2 * name_index], vals[2 * name_index + 1] = xy
            rows.append([str(f), *vals])
        return rows

    csv_rows(work / "clip_id_03_markers.csv", pose_header, pose_rows(index["neck"], ("100", "80")))
    csv_rows(work / "clip_id_04_markers.csv", pose_header, pose_rows(index["nose"], ("30", "90")))

    # Ball: found in frame 0 with a high score, blank in frame 1 (the video ends at 2 frames).
    csv_rows(
        out / "ball.csv",
        "frame,cx,cy,x1,y1,x2,y2,score",
        [["0", "140", "100", "135", "95", "145", "105", "0.9"], ["1", "", "", "", "", "", "", ""]],
    )

    path = module.write_3d_overlay(video, run, out, work.parent, kept={3})
    assert path == out / "clip_3d_overlay.mp4"
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) == 2

    first = frames[0].astype(int)
    green_ring = first[90:111, 130:151]
    assert ((green_ring[..., 1] > 120) & (green_ring[..., 0] < 100)).any()  # ball, score >= 0.5
    assert first[76:85, 96:105].sum() > 0  # kept person's neck
    assert first[16:25, 16:25].sum() > 0  # accepted field point
    assert first[86:95, 26:35].sum() == 0  # person 4 is not kept, so not drawn

    # the id written beside a person is the player id when labels are given, else the track id
    texts: list[str] = []
    real_put_text = cv2.putText

    def spy(img, text, *args, **kwargs):
        texts.append(text)
        return real_put_text(img, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", spy)
    module.write_3d_overlay(video, run, out, work.parent, kept={3}, labels={3: 1})
    assert "ID 1" in texts and "ID 3" not in texts
    texts.clear()
    module.write_3d_overlay(video, run, out, work.parent, kept={3})
    assert "ID 3" in texts and "ID 1" not in texts


def test_track_ball_keeps_one_consistent_track_and_skips_gaps() -> None:
    from vaila import freekiki_ball as fb

    frames = [
        [(0.8, 100.0, 100.0)],
        [(0.95, 900.0, 900.0), (0.5, 110.0, 105.0)],  # a strong box far away, then the ball
        [(0.7, 120.0, 110.0)],
        [],  # the ball is not seen
        [(0.3, 130.0, 115.0)],  # weak, but it continues the track
    ]
    assert fb.track_ball(frames) == [0, 1, 0, None, 0]


def test_track_ball_returns_none_without_candidates() -> None:
    from vaila import freekiki_ball as fb

    assert fb.track_ball([[], []]) == [None, None]
    assert fb.track_ball([[(0.05, 10.0, 10.0)]]) == [None]  # below min_score: not worth a pick


def _pitch_camera():
    from vaila import freekiki_field3d as f3
    from vaila.fifa_to_dlt import compute_dlt3d_from_KRt

    K, R, t = _look_at_camera(2400.0, np.array([0.0, -60.0, 25.0]), np.zeros(3))
    return f3, compute_dlt3d_from_KRt(K, R, t)


def _box_at(f3, L, xyz) -> np.ndarray:
    """A 16 px box centred on the pixel of the ball at world ``xyz`` (radius height)."""
    u, v = f3.project_points(L, np.array([xyz], dtype=float))[0]
    return np.array([u - 8, v - 8, u + 8, v + 8])


def test_pitch_gate_drops_boxes_off_the_pitch() -> None:
    f3, L = _pitch_camera()
    on = _box_at(f3, L, [20.0, 10.0, f3.BALL_RADIUS_M])
    off = _box_at(f3, L, [f3.PITCH_HALF_LENGTH_M + 15.0, 0.0, f3.BALL_RADIUS_M])  # in the stands
    gated = f3.pitch_ball_candidates([[(0.9, off), (0.4, on)], []], {0: L})
    assert [round(s, 2) for s, *_ in gated[0]] == [0.4]  # the strong off-pitch box is gone
    assert gated[1] == []  # no camera for frame 1, so no candidate


def test_select_ball_rows_writes_the_track_and_the_diagnostics(tmp_path: Path) -> None:
    f3, L = _pitch_camera()
    on0 = _box_at(f3, L, [20.0, 10.0, f3.BALL_RADIUS_M])
    off = _box_at(f3, L, [f3.PITCH_HALF_LENGTH_M + 15.0, 0.0, f3.BALL_RADIUS_M])
    candidates = [[(0.9, off), (0.6, on0)], [(0.95, off)]]  # frame 1: only a stand box
    diag = tmp_path / "ball_candidates.csv"
    rows = d3.select_ball_rows(candidates, {0: L, 1: L}, diag)
    assert (
        rows[0]["cx"] != "" and abs(float(rows[0]["cx"]) - (on0[0] + on0[2]) / 2) < 0.01
    )  # ball_row rounds to 0.01 px
    assert rows[1]["cx"] == ""  # the only box is off the pitch: no ball in frame 1
    lines = diag.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",")[-2:] == ["on_pitch", "chosen"]
    assert len(lines) == 1 + 3  # three candidates in total
    chosen = [line for line in lines[1:] if line.endswith("True")]
    assert len(chosen) == 1 and chosen[0].startswith("0,1,")  # the on-pitch box of frame 0


def test_topview_writes_video_and_last_frame_with_the_paths(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    import cv2

    from vaila import freekiki_topview as tv

    out = tmp_path / "out"
    out.mkdir()
    (out / "ground_tracks.csv").write_text(
        "frame,person_id,raw_x_m,raw_y_m,x_m,y_m\n"
        + "\n".join(f"{f},7,0,0,{f},5" for f in range(10))
        + "\n",
        encoding="utf-8",
    )
    (out / "ball_3d_on_field.csv").write_text(
        "frame,cx,cy,score,x_m,y_m,z_m,on_field\n"
        + "\n".join(f"{f},1,1,0.9,{f},-10,0.11,True" for f in range(10))
        + "\n",
        encoding="utf-8",
    )
    written = tv.write_topview(out, "clip", fps=10.0, n_frames=10)
    assert written == {"video": "clip_topview.mp4", "png": "clip_topview.png"}
    png = cv2.imread(str(out / "clip_topview.png"))
    assert png.shape[:2] == (tv.canvas_size()[1], tv.canvas_size()[0])
    # The path of person 7 ends at (x=9, y=5) and the ball at (x=9, y=-10): both are drawn.
    u, v = tv.pitch_to_px(9.0, 5.0)
    assert (png[v - 3 : v + 4, u - 3 : u + 4] != tv.GRASS_BGR).any()
    u, v = tv.pitch_to_px(9.0, -10.0)
    assert (png[v - 3 : v + 4, u - 3 : u + 4] != tv.GRASS_BGR).any()
    cap = cv2.VideoCapture(str(out / "clip_topview.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    cap.release()


def test_ball_correction_csv_round_trips_through_read_corrected_ball(tmp_path: Path) -> None:
    folder = tmp_path / "clip_3d_1"
    folder.mkdir()
    (folder / "ball.csv").write_text(
        "frame,cx,cy,x1,y1,x2,y2,score\n"
        "0,100.5,200.25,92,192,109,208,0.8\n"
        "1,,,,,,,\n"
        "2,300,400,292,392,308,408,0.7\n",
        encoding="utf-8",
    )
    path = d3.write_ball_correction_csv(folder)
    assert path.name == "ball_for_correction.csv"
    assert path.read_text(encoding="utf-8").splitlines() == [
        "frame,p0_x,p0_y",
        "0,100.5,200.25",
        "1,,",
        "2,300,400",
    ]
    rows = d3.read_corrected_ball(path, 3)
    assert [r["cx"] for r in rows] == [100.5, "", 300.0]
    assert rows[0]["score"] == 1.0  # a corrected point is accepted by hand


def _write_3d_folder(root: Path, video: Path) -> Path:
    """A minimal finished 3D folder: work/ with a detect run, a DLT3D and a camera csv."""
    prev = root / "clip_3d_20260101_000000"
    run = prev / "work" / "freekiki_predict_clip_1"
    run.mkdir(parents=True)
    header = "frame," + ",".join(f"L{i}" for i in range(1, 12))
    row = ",".join(f"{v:.12g}" for v in _dlt3d_row(2400.0))
    (run / "clip.dlt3d").write_text(f"{header}\n0,{row}\n", encoding="utf-8")
    sam = prev / "work" / "sam3dinov3" / "clip"
    sam.mkdir(parents=True)
    (sam / "clip_sam3dinov3_camera.csv").write_text("frame,person_id\n", encoding="utf-8")
    (prev / "ball.csv").write_text("frame,cx\n0,1.0\n", encoding="utf-8")
    (prev / "README.txt").write_text(f"FreeKiki 3D output\nvideo: {video}\n", encoding="utf-8")
    return prev


def test_rebuild_ball_writes_a_new_folder_and_leaves_the_previous_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    prev = _write_3d_folder(tmp_path, video)
    ball_csv = tmp_path / "clip_markers.csv"
    ball_csv.write_text("frame,p0_x,p0_y\n0,100,200\n2,300,400\n", encoding="utf-8")

    monkeypatch.setattr(d3, "_video_info", lambda _v: (4, 1920, 1080))
    monkeypatch.setattr(d3, "_video_fps", lambda _v: 30.0)
    monkeypatch.setattr(d3, "write_field_3d", lambda *_a, **_k: {"stub": True})
    monkeypatch.setattr(d3, "write_topview", lambda *_a, **_k: {})
    monkeypatch.setattr(
        d3, "write_3d_overlay", lambda _v, _r, out_dir, *_a, **_k: Path(out_dir) / "o.mp4"
    )

    before = (prev / "ball.csv").read_bytes()
    result = d3.rebuild_ball_3d(prev, ball_csv, timestamp="20260102_000000")

    out = tmp_path / "clip_3d_20260102_000000"
    assert result["out"] == str(out)
    with (out / "ball.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert rows[0]["cx"] == "100.0" and rows[0]["score"] == "1.0"
    assert rows[1]["cx"] == ""  # frame 1 had no point: no ball
    assert rows[2]["cx"] == "300.0"
    assert result["ball"]["found"] == 2
    assert (prev / "ball.csv").read_bytes() == before  # the previous folder is untouched


def test_rebuild_ball_rejects_a_csv_without_the_point_columns(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    prev = _write_3d_folder(tmp_path, video)
    bad = tmp_path / "bad.csv"
    bad.write_text("frame,x,y\n0,1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frame,p0_x,p0_y"):
        d3.rebuild_ball_3d(prev, bad)


def test_rebuild_ball_needs_the_work_folder(tmp_path: Path) -> None:
    bare = tmp_path / "clip_3d_20260101_000000"
    bare.mkdir()
    csv_path = tmp_path / "c.csv"
    csv_path.write_text("frame,p0_x,p0_y\n", encoding="utf-8")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="work/"):
        d3.rebuild_ball_3d(bare, csv_path, video=video)


def _stub_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(d3, "_video_info", lambda _v: (4, 1920, 1080))
    monkeypatch.setattr(d3, "_video_fps", lambda _v: 30.0)
    monkeypatch.setattr(d3, "write_field_3d", lambda *_a, **_k: {"stub": True})
    monkeypatch.setattr(d3, "write_topview", lambda *_a, **_k: {})
    monkeypatch.setattr(
        d3, "write_3d_overlay", lambda _v, _r, out_dir, *_a, **_k: Path(out_dir) / "o.mp4"
    )


def test_rebuild_survives_deleting_the_previous_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a rebuild, the new folder owns work/; deleting the old one is safe."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    origin = _write_3d_folder(tmp_path, video)
    csv_path = tmp_path / "clip_markers.csv"
    csv_path.write_text("frame,p0_x,p0_y\n0,100,200\n", encoding="utf-8")
    _stub_outputs(monkeypatch)

    first = d3.rebuild_ball_3d(origin, csv_path, timestamp="20260102_000000")
    rebuilt1 = Path(first["out"])
    shutil.rmtree(origin)
    second = d3.rebuild_ball_3d(rebuilt1, csv_path, timestamp="20260103_000000")
    rebuilt2 = Path(second["out"])
    assert (rebuilt2 / "work" / "sam3dinov3").is_dir()
    assert second["work_from"] == str(rebuilt2)


def test_rebuild_twice_in_a_chain_reads_the_original_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    origin = _write_3d_folder(tmp_path, video)
    (origin / "players_on_field.csv").write_text(
        "person_id,fraction,on,checked,keep\n1,1.0,32,32,True\n", encoding="utf-8"
    )
    csv_path = tmp_path / "clip_markers.csv"
    csv_path.write_text("frame,p0_x,p0_y\n0,100,200\n", encoding="utf-8")
    _stub_outputs(monkeypatch)

    origin_before = (origin / "README.txt").read_bytes()
    first = d3.rebuild_ball_3d(origin, csv_path, timestamp="20260102_000000")
    rebuilt1 = Path(first["out"])
    # Flat owner: work/ moves into the new folder; origin keeps a symlink.
    assert (rebuilt1 / "work").is_dir()
    assert (rebuilt1 / "work" / "sam3dinov3").is_dir()
    assert first["work_from"] == str(rebuilt1)
    assert "work from: " + str(rebuilt1) in (rebuilt1 / "README.txt").read_text(encoding="utf-8")
    assert (origin / "work").is_dir()  # symlink back to the moved tree
    assert (rebuilt1 / "players_on_field.csv").read_bytes() == (
        origin / "players_on_field.csv"
    ).read_bytes()

    second = d3.rebuild_ball_3d(rebuilt1, csv_path, timestamp="20260103_000000")
    rebuilt2 = Path(second["out"])
    assert second["work_from"] == str(rebuilt2)
    assert (rebuilt2 / "work").is_dir()
    assert "work from: " + str(rebuilt2) in (rebuilt2 / "README.txt").read_text(encoding="utf-8")
    assert (rebuilt2 / "players_on_field.csv").is_file()
    assert (origin / "README.txt").read_bytes() == origin_before  # the owner README is untouched


def test_work_owner_follows_the_work_from_line_then_a_sibling(tmp_path: Path) -> None:
    origin = tmp_path / "clip_3d_20260101_000000"
    (origin / "work").mkdir(parents=True)
    rebuilt = tmp_path / "clip_3d_20260102_000000"
    rebuilt.mkdir()
    assert d3.work_owner(rebuilt) is None  # no work/, no README, no video
    (rebuilt / "README.txt").write_text(f"work from: {origin}\n", encoding="utf-8")
    assert d3.work_owner(rebuilt) == origin
    (rebuilt / "README.txt").unlink()
    assert d3.work_owner(rebuilt, "clip.mp4") == origin  # sibling with the same stem
    assert d3.work_owner(rebuilt, "other.mp4") is None


def _write_batch_3d_layout(tmp_path: Path, video: Path) -> Path:
    """Detect + 3D on a folder: outputs under <batch>_3d_<ts>/<stem>/, work under work/<stem>/."""
    stem = video.stem
    batch = tmp_path / "games_3d_20260101_000000"
    out = batch / stem
    out.mkdir(parents=True)
    work = batch / "work" / stem
    sam = work / "sam3dinov3" / stem
    sam.mkdir(parents=True)
    (sam / f"{stem}_sam3dinov3_camera.csv").write_text("frame,person_id\n", encoding="utf-8")
    run = batch / "work" / "processed_freekiki_batch_1" / f"freekiki_predict_{stem}_1"
    run.mkdir(parents=True)
    header = "frame," + ",".join(f"L{i}" for i in range(1, 12))
    row = ",".join(f"{v:.12g}" for v in _dlt3d_row(2400.0))
    (run / f"{stem}.dlt3d").write_text(f"{header}\n0,{row}\n", encoding="utf-8")
    (out / "ball.csv").write_text("frame,cx,cy,score\n0,1.0,2.0,0.9\n", encoding="utf-8")
    (out / "players_on_field.csv").write_text(
        "person_id,fraction,on,checked,keep\n1,1.0,32,32,True\n", encoding="utf-8"
    )
    (out / "README.txt").write_text(f"FreeKiki 3D output\nvideo: {video}\n", encoding="utf-8")
    return out


def test_work_owner_and_locate_find_batch_nested_work(tmp_path: Path) -> None:
    video = tmp_path / "clip_a.mp4"
    video.write_bytes(b"")
    out = _write_batch_3d_layout(tmp_path, video)
    batch = out.parent
    assert d3.work_owner(out, video) == batch
    bundle = d3.locate_3d_work(out, video)
    assert bundle is not None
    assert bundle.owner == batch
    assert bundle.work == batch / "work" / video.stem
    assert bundle.run_dir.name.startswith("freekiki_predict_clip_a_")
    assert bundle.players_dir == out


def test_rebuild_ball_uses_batch_nested_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip_a.mp4"
    video.write_bytes(b"")
    prev = _write_batch_3d_layout(tmp_path, video)
    ball_csv = tmp_path / "clip_a_markers.csv"
    ball_csv.write_text("frame,p0_x,p0_y\n0,100,200\n", encoding="utf-8")
    _stub_outputs(monkeypatch)

    result = d3.rebuild_ball_3d(prev, ball_csv, video=video, timestamp="20260102_000000")
    out = Path(result["out"])
    # Batch nested work stays under the batch parent; rebuild gets a local symlink.
    assert (out / "work").is_dir()
    assert (prev.parent / "work" / video.stem).is_dir()
    assert result["work_from"] == str(out)
    assert out.is_dir()
    assert (out / "ball.csv").is_file()
    assert (out / "players_on_field.csv").is_file()
    assert "work from: " + str(out) in (out / "README.txt").read_text(encoding="utf-8")


def test_a_hand_marked_ball_outside_the_pitch_keeps_its_frames() -> None:
    def row(f: int, x: float, y: float, on: bool) -> dict:
        return {"frame": f, "cx": "", "cy": "", "score": 1.0, "x_m": x, "y_m": y, "on_field": on}

    # frames 0..4 on the pitch; frame 5 is behind the goal line (x = -54 m, out of play)
    ball_all = [row(f, -20.0 + f, 2.0, True) for f in range(5)] + [row(5, -54.0, 3.5, False)]
    trusted = d3.physical_ball_rows(ball_all, 6, 30.0, trusted=True)
    assert {r["frame"] for r in trusted} == {0, 1, 2, 3, 4, 5}


def test_ball_anchor_spec_reads_json_and_rejects_unknown_keys() -> None:
    spec = d3.ball_anchor_spec_from_json('{"kicker_id": 30, "barrier_id": 15, "keeper_id": 21}')
    assert (spec.kicker_id, spec.barrier_id, spec.keeper_id) == (30, 15, 21)
    assert spec.barrier_frame == 9 and spec.barrier_height_m == pytest.approx(1.8)
    assert spec.keeper_parts == ("left-wrist", "right-wrist")
    with pytest.raises(ValueError, match="unknown"):
        d3.ball_anchor_spec_from_json(
            '{"kicker_id": 30, "barrier_id": 15, "keeper_id": 21, "x": 1}'
        )
    with pytest.raises(ValueError, match="JSON object"):
        d3.ball_anchor_spec_from_json("[1, 2]")


def test_anchors_are_read_from_the_pose_table() -> None:
    spec = d3.BallAnchorSpec(kicker_id=30, barrier_id=15, keeper_id=21)

    def row(pid: int, frame: int, name: str, x: float, y: float, z: float) -> dict:
        return {"person_id": pid, "frame": frame, "kpt_name": name, "x_m": x, "y_m": y, "z_m": z}

    poses = [
        row(30, 0, "left-ankle", -30.8, -20.2, 0.04),
        row(15, 9, "left-shoulder", -37.7, -15.0, 1.2),
        row(21, 4, "left-wrist", -52.0, -1.7, 0.9),
        row(21, 4, "right-wrist", -52.2, -1.5, 0.95),
    ]
    anchors = d3.anchors_from_poses(spec, poses, n_frames=5)
    assert anchors["rest"] == pytest.approx((-30.8, -20.2))
    assert anchors["barrier"]["xy"] == pytest.approx((-37.7, -15.0))
    assert anchors["keeper"]["frame"] == 4
    assert anchors["keeper"]["xyz"] == pytest.approx((-52.1, -1.6, 0.925))
    with pytest.raises(ValueError, match="kept people"):
        d3.anchors_from_poses(spec, poses[:1], n_frames=5)


def test_flight_check_video_draws_markers_fit_and_anchors(tmp_path: Path) -> None:
    import cv2

    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 240))
    for _ in range(3):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()
    P = np.array([[100.0, 0.0, 0.0, 160.0], [0.0, 100.0, 0.0, 120.0], [0.0, 0.0, 0.0, 1.0]])
    anchors = {
        "rest": (0.0, 0.0),
        "rest_z": 0.04,
        "barrier_shoulder_z": 1.2,
        "barrier": {"frame": 1, "xy": (0.5, 0.0), "z": 1.8, "offset": 0.3},
        "keeper": {"frame": 2, "xyz": (0.0, 0.5, 0.9)},
    }
    points = d3.flight_check_points(anchors)
    assert [p["frame"] for p in points] == [0, 1, 1, 2]
    out = d3.write_flight_check_video(
        video,
        tmp_path / "check.mp4",
        {0: P, 1: P, 2: P},
        {0: (0.0, 0.0, 0.11), 1: (0.5, 0.0, 1.8)},
        {0: (160.0, 120.0), 1: (210.0, 120.0)},
        points,
    )
    cap = cv2.VideoCapture(str(out))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) == 3
    assert frames[0].any()  # something was drawn on the frame


def test_point_anchors_need_three_points_and_round_trip_through_json() -> None:
    text = d3.anchor_points_json(
        [
            {"frame": 0, "xyz": [-31.0, -19.4, 0.11], "sigma": 0.2, "label": "início"},
            {"frame": 9, "xyz": [-38.0, -14.0, 1.0], "sigma": 0.8, "label": "meio"},
            {"frame": 30, "xyz": [-52.45, 1.2, 0.11], "sigma": 0.2, "label": "gol"},
        ]
    )
    spec = d3.ball_anchor_spec_from_json(text)
    assert isinstance(spec, d3.BallAnchorPoints) and len(spec.points) == 3
    assert spec.points[1]["label"] == "meio" and spec.points[1]["frame"] == 9
    with pytest.raises(ValueError, match="at least 3"):
        d3.ball_anchor_spec_from_json('{"points": [{"frame": 0, "xyz": [0, 0, 0.1]}]}')
    with pytest.raises(ValueError, match="sigma"):
        d3.ball_anchor_spec_from_json(
            '{"points": [{"frame": 0, "xyz": [0, 0, 0.1], "sigma": 0}, '
            '{"frame": 1, "xyz": [0, 0, 0.1]}, {"frame": 2, "xyz": [0, 0, 0.1]}]}'
        )


def test_anchor_sources_and_points_come_from_the_run_files(tmp_path: Path) -> None:
    (tmp_path / "ball_3d.csv").write_text(
        "frame,cx,cy,score,x_m,y_m,z_m,on_field\n0,1,1,1,-31.0,-19.4,0.11,True\n5,2,2,1,-40.0,-10.0,0.11,True\n",
        encoding="utf-8",
    )
    (tmp_path / "poses_3d_on_field.csv").write_text(
        "frame,person_id,kpt_idx,kpt_name,x_m,y_m,z_m\n"
        "0,30,0,left-ankle,-30.8,-20.2,0.04\n5,30,0,left-ankle,-35.0,-16.0,0.05\n",
        encoding="utf-8",
    )
    sources = d3.anchor_sources(tmp_path)
    assert sources["ball"][5] == (-40.0, -10.0)
    assert d3.source_point(sources, "bola", 0, z=0.11)[2] == pytest.approx(0.11)
    assert d3.source_point(sources, "bola", 3) is None  # no ball at that frame
    foot = d3.source_point(sources, "jogador", 0, person=30, kpt="left-ankle")
    assert foot == pytest.approx([-30.8, -20.2, 0.04])
    goal = d3.source_point(sources, "gol", 5, option="centro do gol, no chão", side=-1)
    assert goal == pytest.approx([-52.45, 0.0, 0.11])
    with pytest.raises(ValueError, match="unknown anchor source"):
        d3.source_point(sources, "vento", 0)


def test_check_points_and_record_for_point_anchors() -> None:
    spec = d3.BallAnchorPoints(
        (
            {"frame": 0, "xyz": [-31.0, -19.4, 0.11], "sigma": 0.2, "label": "início"},
            {"frame": 9, "xyz": [-38.0, -14.0, 1.0], "sigma": 0.8, "label": "meio"},
            {"frame": 30, "xyz": [-52.45, 1.2, 0.11], "sigma": 0.2, "label": "gol"},
        )
    )
    anchors = d3.anchors_from_points(spec)
    points = d3.flight_check_points(anchors)
    assert [p["label"] for p in points] == ["início", "meio", "gol"]
    assert points[1]["frame"] == 9 and len(points[0]["color"]) == 3
    record = d3._anchor_points_record(anchors)
    assert record["p1"] == {"label": "meio", "frame": 9, "xyz": [-38.0, -14.0, 1.0]}


def test_pose_rows_skip_a_frame_without_its_own_dlt(tmp_path: Path, monkeypatch) -> None:
    """A frame with no DLT3D (dropped by the detect run) must not crash the pose stage.

    The smoothed track can fill that frame; without a camera the pose cannot be placed on the
    pitch, so the person is simply absent there (found in the one-click run on the veiga pair).
    """
    from vaila import freekiki_detect3d as d3

    rows = ["frame,person_id,kpt_idx,kpt_name,xcam_m,ycam_m,zcam_m"]
    for f in range(3):
        rows.append(f"{f},0,0,nose,0.0,0.0,10.0")
    csv_path = tmp_path / "kp3d.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    cameras = {0: np.zeros(11), 1: np.zeros(11)}  # frame 2 has no DLT3D
    boxes = {(f, 0): np.array([0.0, 0.0, 10.0, 10.0]) for f in range(3)}
    monkeypatch.setattr(d3, "box_ground", lambda L, box: (0.5, 0.5))
    monkeypatch.setattr(
        d3, "smooth_track", lambda pts, n, fps, model: dict.fromkeys(range(n), (1.0, 2.0))
    )
    monkeypatch.setattr(
        d3, "pose_to_pitch", lambda names, xcam, L, box, ground_xy=None: np.zeros((len(names), 3))
    )
    poses, ground = d3._pose_rows(csv_path, cameras, boxes, None, 60.0, 3)
    assert {r["frame"] for r in poses} == {0, 1}
    assert {r["frame"] for r in ground} == {0, 1}
