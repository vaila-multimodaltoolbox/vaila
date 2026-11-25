"""One analysis (vaila/freekiki_team_analysis.py) on a synthetic camera: who is on the pitch, the folders.

A small video paints four boxes with shirt colours and a SAM predictions file gives their boxes. The
test checks that only the people on the pitch are labelled and written, that the goalkeeper is opposite
the kicker, that each team folder holds its players' C3D files and an overlay, and that a height is not
reported when the camera's vertical scale is not checked.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from vaila import freekiki_team_analysis as ta
from vaila.sam3dinov3 import MHR70_NAMES


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


FRAMES = 3
SIZE = (320, 240)
BOXES = {  # SAM id -> bbox (x1, y1, x2, y2)
    1: (10, 40, 70, 200),  # the keeper, yellow
    2: (100, 40, 160, 200),  # the kicker, green
    3: (190, 40, 250, 200),  # a field player, blue (on the pitch)
    4: (260, 40, 310, 200),  # a blue person off the pitch
}
COLOURS = {1: (0, 230, 255), 2: (0, 200, 0), 3: (230, 140, 70), 4: (230, 140, 70)}  # BGR
PLAYER = {1: "0", 2: "1", 3: "2"}  # SAM id -> player id (4 has no player id: off the pitch)
ROLE = {"0": "keeper", "1": "kicker", "2": "other"}


def _make_video(path: Path, n_frames: int = FRAMES) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 30.0, SIZE)
    for _ in range(n_frames):
        frame = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
        for sam, (x1, y1, x2, y2) in BOXES.items():
            frame[y1:y2, x1:x2] = COLOURS[sam]
        writer.write(frame)
    writer.release()


def _instances(frame: int = 0, skeletons: dict | None = None) -> list[dict]:
    """The SAM instances of a frame; ``skeletons[sam](frame)`` -> ``{name: (u, v)}`` replaces keypoints."""
    out = []
    for sam, (x1, y1, x2, y2) in BOXES.items():
        xs = np.linspace(x1 + 5, x2 - 5, len(MHR70_NAMES))
        ys = np.linspace(y1 + 5, y2 - 5, len(MHR70_NAMES))
        kp = np.column_stack([xs, ys]).tolist()
        if skeletons and sam in skeletons:
            for name, uv in skeletons[sam](frame).items():
                kp[MHR70_NAMES.index(name)] = [float(uv[0]), float(uv[1])]
        out.append({
            "person_id": sam, "sam_obj_id": sam, "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "keypoints_2d_px": kp, "cam_t_m": [0.0, 0.0, 10.0], "keypoints_3d_m": [[0.0, 0.0, 0.0]] * 70,
        })  # fmt: skip
    return out


def _predictions(work: Path, n_frames: int = FRAMES, skeletons: dict | None = None) -> None:
    folder = work / "sam3dinov3" / "processed" / "clip"
    folder.mkdir(parents=True)
    data = {
        "schema": "test", "keypoint_names": list(MHR70_NAMES), "fps": 30.0,
        "frames": [
            {"frame": f, "instances": _instances(f, skeletons)} for f in range(n_frames)
        ],
    }  # fmt: skip
    with gzip.open(folder / "clip_sam3dinov3_predictions.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(data, fh)


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _camera(
    folder: Path, work: Path, video: Path, n_frames: int = FRAMES, skeletons: dict | None = None
) -> None:
    folder.mkdir(parents=True)
    _write(folder / "players_on_field.csv",
           "person_id,fraction,on,checked,keep\n1,1.0,3,3,True\n2,1.0,3,3,True\n"
           "3,1.0,3,3,True\n4,0.0,0,3,False\n")  # fmt: skip
    rows = [
        "player_id,role,camera,sam_id,first_frame,last_frame,frames,partner,median_match_m,group"
    ]
    for sam, pid in PLAYER.items():
        rows.append(f"{pid},{ROLE[pid]},cam,{sam},0,2,3,,0.1,{pid}")
    _write(folder / "player_map.csv", "\n".join(rows) + "\n")
    _write(folder / "ball.csv", "frame,cx,cy,x1,y1,x2,y2,score\n0,160.0,220.0,0,0,0,0,1.0\n")
    poses = ["frame,person_id,kpt_idx,kpt_name,x_m,y_m,z_m,player_id"]
    for f in range(n_frames):
        for sam, pid in PLAYER.items():
            poses.append(f"{f},{sam},0,nose,{10.0 + int(pid)},0.0,1.7,{pid}")
    _write(folder / "poses_3d_on_field_players.csv", "\n".join(poses) + "\n")
    _predictions(work, n_frames, skeletons)
    _make_video(video, n_frames)


def test_only_the_people_on_the_pitch_are_labelled_and_written(tmp_path: Path) -> None:
    folder = tmp_path / "cam_3d"
    work = tmp_path / "work"
    video = tmp_path / "clip.avi"
    _camera(folder, work, video)
    source = ta.Source("ref", folder, video, work, dlt=None)
    summary = ta.run_team_analysis(
        [source], folder, poses_csv=folder / "poses_3d_on_field_players.csv",
        ball_csv=folder / "ball.csv", kind="mono-ref", rate_hz=30.0,
    )  # fmt: skip

    team_map = {r["player_id"]: r for r in _rows(Path(folder / "data" / "team_map.csv"))}
    assert set(team_map) == {"0", "1", "2"}  # the person off the pitch is not a player
    assert team_map["1"]["team"] == "TEAM_A"  # the kicker wears green
    assert team_map["2"]["team"] == "TEAM_B"  # the other colour most field players wear
    assert team_map["0"]["team"] == "TEAM_B"  # the keeper: opposite of the kicker
    assert summary["counts"] == {"TEAM_A": 1, "TEAM_B": 2, "REF": 0}

    off = list(_rows(Path(folder / "data" / "not_on_pitch.csv")))
    assert [(r["sam_id"], r["reason"]) for r in off] == [("4", "not on the pitch")]
    assert summary["kind"] == "mono-ref"

    assert (folder / "team_A" / "P01_kicker.c3d").is_file()
    assert (folder / "team_B" / "P00_keeper.c3d").is_file()
    assert (folder / "team_B" / "P02_other.c3d").is_file()
    assert (folder / "team_A" / "overlay_ref.mp4").is_file()
    assert (folder / "team_B" / "overlay_ref.mp4").is_file()
    assert not (folder / "referees" / "overlay_ref.mp4").exists()  # no referee: no overlay written


def test_the_height_is_not_reported_when_the_vertical_scale_is_not_checked(tmp_path: Path) -> None:
    folder = tmp_path / "cam_3d"
    work = tmp_path / "work"
    video = tmp_path / "clip.avi"
    _camera(folder, work, video)
    source = ta.Source("ref", folder, video, work, dlt=None)
    ta.run_team_analysis([source], folder, poses_csv=folder / "poses_3d_on_field_players.csv",
                         ball_csv=folder / "ball.csv", kind="mono-ref", rate_hz=30.0)  # fmt: skip
    rows = list(_rows(Path(folder / "data" / "team_map.csv")))
    assert all(r["height_m"] == "" for r in rows)
    assert all("vertical scale" in r["reason"] for r in rows)
    analysis = json.loads((folder / "data" / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["heights_validated"] is False


def test_on_pitch_reads_the_keep_flag(tmp_path: Path) -> None:
    _write(
        tmp_path / "players_on_field.csv",
        "person_id,fraction,on,checked,keep\n5,1.0,2,2,True\n6,0,0,2,False\n",
    )
    assert ta.on_pitch(tmp_path) == {5}


def test_read_person_points_accepts_poses_without_a_valid_column(tmp_path: Path) -> None:
    path = tmp_path / "poses.csv"
    _write(path, "frame,person_id,kpt_idx,kpt_name,x_m,y_m,z_m,player_id\n"
           "0,7,0,nose,1.0,2.0,1.5,4\n1,7,0,nose,1.1,2.0,1.5,4\n")  # fmt: skip
    people, names = ta.read_person_points(path)
    assert set(people) == {4} and people[4].shape[0] == 2
    assert np.allclose(people[4][1, 0], (1.1, 2.0, 1.5))
    assert names[0] == "nose"


def test_the_vertical_check_is_nan_without_field_points(tmp_path: Path) -> None:
    assert np.isnan(ta.vertical_residual_px(tmp_path, tmp_path))


def _shifted_camera() -> np.ndarray:
    """A broadcast-like camera (f = 5000 px, 90 m away, 16 m high) whose picture of the person at
    the pitch origin lands in the box of SAM 3 of the small test video."""
    from tests import test_freekiki_stature_chain as sc

    P0 = sc._look_at([2.0, -90.0, 16.0], [0.0, 0.0, 0.0])[0]
    foot = sc._project(P0, np.zeros(3))
    shift = np.array([[1, 0, 220.0 - foot[0]], [0, 1, 185.0 - foot[1]], [0, 0, 1]])
    return shift @ P0


def _dlt_of(P: np.ndarray) -> np.ndarray:
    P = P / P[2, 3]
    return np.array([*P[0], *P[1], *P[2, :3]])


def _skeletons(P: np.ndarray, stature: float) -> dict:
    from tests import test_freekiki_stature_chain as sc

    def of_frame(frame: int) -> dict:
        joints = sc.skeleton(stature, (0.0, 0.0, 0.0), 0.05 * frame, "run", phase=0.5 * frame)
        return {name: sc._project(P, joints[name]) for name in sc.NAMES}

    return {3: of_frame}


def test_the_height_comes_from_the_chain_with_the_cameras_of_all_frames(tmp_path: Path) -> None:
    """No goal posts to check, yet the person gets his height: the ground-plane scale of each frame."""
    P = _shifted_camera()
    folder, work, video = tmp_path / "cam_3d", tmp_path / "work", tmp_path / "clip.avi"
    _camera(folder, work, video, n_frames=8, skeletons=_skeletons(P, 1.80))
    source = ta.Source("ref", folder, video, work, dlt=dict.fromkeys(range(8), _dlt_of(P)))
    summary = ta.run_team_analysis(
        [source], folder, poses_csv=folder / "poses_3d_on_field_players.csv",
        ball_csv=folder / "ball.csv", kind="mono-ref", rate_hz=30.0,
    )  # fmt: skip
    team_map = {r["player_id"]: r for r in _rows(folder / "data" / "team_map.csv")}
    assert abs(float(team_map["2"]["height_m"]) - 1.80) / 1.80 < 0.03, team_map["2"]
    assert int(team_map["2"]["height_frames"]) == 8 and team_map["2"]["height_iqr_m"] != ""
    assert summary["heights_validated"] is False  # the flag stays, the height is still reported
    assert summary["height_method"] == "chain-p80" and summary["height_cameras"] == "freekiki dlt3d"
    labels = json.loads((folder / "data" / "labels_ref.json").read_text(encoding="utf-8"))
    assert "1.8" in labels["3"] or "1.7" in labels["3"]  # the label carries the height


def test_frame_cameras_prefers_the_per_frame_file_and_drops_the_approximate_frames(
    tmp_path: Path, monkeypatch
) -> None:
    from vaila import kick_measure as km

    P = _shifted_camera()
    track = km.CameraTrack.from_matrices(dict.fromkeys(range(6), P), n_frames=10)  # 6-9 approximate
    monkeypatch.setattr(km, "find_frame_calibration", lambda work, stem: Path(f"{stem}.dlt3d"), raising=False)  # fmt: skip
    monkeypatch.setattr(km, "read_frame_calibration", lambda path: track, raising=False)
    source = ta.Source("ref", tmp_path, tmp_path / "v.avi", tmp_path / "work" / "clip", dlt={0: _dlt_of(P)})  # fmt: skip
    cameras, origin = ta.frame_cameras(source)
    assert cameras is not None and sorted(cameras) == list(range(6))
    assert origin == "clip.dlt3d"


def test_frame_cameras_falls_back_to_the_freekiki_dlt_and_to_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    from vaila import kick_measure as km

    monkeypatch.setattr(km, "find_frame_calibration", lambda work, stem: None, raising=False)
    monkeypatch.setattr(km, "read_frame_calibration", lambda path: None, raising=False)
    dlt = {3: np.arange(11.0)}
    source = ta.Source("ref", tmp_path, tmp_path / "v.avi", tmp_path / "work" / "clip", dlt=dlt)
    assert ta.frame_cameras(source) == (dlt, "freekiki dlt3d")
    source = ta.Source("ref", tmp_path, tmp_path / "v.avi", tmp_path / "work" / "clip", dlt=None)
    assert ta.frame_cameras(source) == (None, "")
    monkeypatch.delattr(km, "find_frame_calibration", raising=False)  # kick_measure without the API
    monkeypatch.delattr(km, "read_frame_calibration", raising=False)
    assert ta.frame_cameras(source) == (None, "")
