"""The one-click kick analysis (vaila/freekiki_kick_pipeline.py) with the heavy stages replaced.

The GPU stages (detection, SAM 3D Body) and the analyses (teams, kicker, reports) are stubs that
write the files each stage must leave. The tests check the order of the stages, the input checks
(one or two cameras), the reuse of finished stages, a failing monocular analysis, the failure
message and the parser. Nothing here needs a video, a model or a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vaila import freekiki as fk
from vaila import freekiki_detect3d as d3
from vaila import freekiki_kick_pipeline as kp
from vaila import freekiki_physics as fk_physics
from vaila import freekiki_player_ids as pids
from vaila import freekiki_report as rep
from vaila import freekiki_stereo as st

CAM1 = "c1_main_sync_15_frames_2_to_79"
CAM2 = "c2_16mm_sync_14_frames_1_to_78"


def _cameras(folder: Path, *, markers: bool = True, both: bool = True) -> list[Path]:
    videos = []
    for stem in (CAM1, CAM2) if both else (CAM1,):
        video = folder / f"{stem}.mp4"
        video.write_bytes(b"")
        if markers:
            (folder / f"{stem}_markers.csv").write_text(
                "frame,p0_x,p0_y\n0,1,1\n", encoding="utf-8"
            )
        videos.append(video)
    return videos


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """Replace the heavy stages and the analyses; ``calls`` counts how often each one ran."""
    calls: dict[str, int] = {}

    def counted(name, fn):
        def wrapper(*args, **kwargs):
            calls[name] = calls.get(name, 0) + 1
            return fn(*args, **kwargs)

        return wrapper

    def detect_video(ws, video, *, model, output_dir, device=None, **_):
        predict = Path(output_dir) / f"freekiki_predict_{Path(video).stem}_20260101_000000"
        predict.mkdir(parents=True, exist_ok=True)
        (predict / "quality.json").write_text("{}", encoding="utf-8")
        (predict / "x.dlt3d").write_text("frame,L1\n", encoding="utf-8")
        return predict

    def detect_3d_run(run_dir, video, out_dir, *, work_dir, device=None, ball_csv=None, **_):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "poses_3d_on_field.csv").write_text("frame\n", encoding="utf-8")
        (out_dir / "ball.csv").write_text("frame,cx,cy\n", encoding="utf-8")
        (out_dir / "README.txt").write_text(f"video: {video}\n", encoding="utf-8")
        return {"out": str(out_dir)}

    def run_player_ids(folders):
        for folder in folders:
            (Path(folder) / "player_map.csv").write_text(
                "player_id,role,camera,sam_id,first_frame,last_frame,frames,partner,median_match_m,group\n"
                "0,keeper,c,4,0,77,78,c:18,0.4,0\n1,kicker,c,11,0,77,78,c:16,0.3,1\n",
                encoding="utf-8",
            )
        return {"pairs": 1}

    def run_binocular(run_a, run_b, *, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ball_flight.json").write_text(
            json.dumps({"kick_time_s": 0.22, "peak_speed_mps": 30.0, "spin_rps": 9.5}),
            encoding="utf-8",
        )
        (out_dir / "ball_3d_on_field.csv").write_text("frame,x_m,y_m,z_m\n", encoding="utf-8")
        return {"fit": {"peak_speed_mps": 30.0, "kick_time_s": 0.22, "spin_rps": 9.5}}

    def triangulate(ref3d, other3d, work_ref, work_other, out_csv, ref_video, other_video):
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        Path(out_csv).write_text("frame\n", encoding="utf-8")
        return {"rows": 0, "valid_rows": 0, "height_check_m": {},
                "vitruvian_frames": {"ref": 0, "other": 0}}  # fmt: skip

    def c3d_binocular(ref3d, other3d, ref_work, other_work, fit, data, ref_video):
        data = Path(data)
        data.mkdir(parents=True, exist_ok=True)
        (data / "field_ball.c3d").write_bytes(b"c3d")
        return {"out": str(data / "field_ball.c3d"), "summary": {"field_kept": 0}}

    def teams_binocular(ref3d, other3d, ref_video, other_video, ref_work, other_work, bino_dir,
                        poses_csv, ball_csv, *, team_a, fps):  # fmt: skip
        data = Path(bino_dir) / "data"
        data.mkdir(parents=True, exist_ok=True)
        (data / "team_map.csv").write_text("player_id\n", encoding="utf-8")
        return {"counts": {"TEAM_A": 0, "TEAM_B": 0, "REF": 0}}

    def teams_mono(cam3d, video, work, *, is_ref, key, team_a, fps, sync=None):
        data = Path(cam3d) / "data"
        data.mkdir(parents=True, exist_ok=True)
        (data / "team_map.csv").write_text("player_id\n", encoding="utf-8")
        return {"counts": {"TEAM_A": 0, "TEAM_B": 0, "REF": 0}}

    def kicker(ref3d, binocular, poses_csv, out, ref_video, *, kick_foot, **_):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "kicker_metrics.csv").write_text("metric\n", encoding="utf-8")
        (out / "kicker_curves_pt.png").write_bytes(b"png")
        return {"values": {"contact_frame": 40}, "warnings": [], "fps": 59.94, "files": {}}

    def build_report(run_dir, *, lang="both", **_):
        base = Path(run_dir) / "report_tv"
        for sub, name in (("pt", "relatorio_bola.html"), ("en", "ball_report.html")):
            (base / sub).mkdir(parents=True, exist_ok=True)
            (base / sub / name).write_text("<html></html>", encoding="utf-8")
        return {"dir": str(base), "tier": "alta"}

    monkeypatch.setattr(fk, "detect_video", counted("detect", detect_video))
    monkeypatch.setattr(d3, "detect_3d_run", counted("3d", detect_3d_run))
    monkeypatch.setattr(pids, "run_player_ids", counted("players", run_player_ids))
    monkeypatch.setattr(st, "run_binocular", counted("binocular", run_binocular))
    monkeypatch.setattr(kp, "_triangulate", counted("triangulate", triangulate))
    monkeypatch.setattr(kp, "_c3d_binocular", counted("c3d", c3d_binocular))
    monkeypatch.setattr(kp, "_teams_binocular", counted("teams", teams_binocular))
    monkeypatch.setattr(kp, "_teams_mono", counted("teams_mono", teams_mono))
    monkeypatch.setattr(kp, "_kicker", counted("kicker", kicker))
    monkeypatch.setattr(rep, "build_report", counted("report", build_report))
    monkeypatch.setattr(
        kp, "_screen_or_raise", lambda ws, video, model, min_camera_ok: {"camera_ok": 1.0}
    )
    monkeypatch.setattr(kp, "nominal_clock", lambda path: np.arange(80) / 59.94)
    monkeypatch.setattr(d3, "_video_fps", lambda path: 59.94)
    return calls


def test_pick_videos_takes_the_two_cut_videos_of_a_folder(tmp_path: Path) -> None:
    _cameras(tmp_path)
    (tmp_path / "notes.mp4").write_bytes(b"")  # not a cut video: no sync in the name
    picked = kp.pick_videos(tmp_path)
    assert [p.stem for p in picked] == sorted([CAM1, CAM2])


def test_pick_videos_takes_one_video_for_the_monocular_analysis(tmp_path: Path) -> None:
    _cameras(tmp_path, both=False)
    assert [p.stem for p in kp.pick_videos(tmp_path)] == [CAM1]


def test_pick_videos_takes_a_folder_video_with_no_sync_name_alone(tmp_path: Path) -> None:
    clip = tmp_path / "c1_main_camera_platform_frame_63_to_81.mp4"  # the cut tool's name, no sync
    clip.write_bytes(b"")
    assert kp.pick_videos(tmp_path) == [clip]


def test_pick_videos_still_refuses_two_videos_with_no_sync_names(tmp_path: Path) -> None:
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "b.mp4").write_bytes(b"")
    with pytest.raises(kp.KickInputError, match="one or two"):
        kp.pick_videos(tmp_path)


def test_pick_videos_refuses_a_folder_with_three_cut_videos(tmp_path: Path) -> None:
    _cameras(tmp_path)
    (tmp_path / "c3_sync_1_frames_0_to_5.mp4").write_bytes(b"")
    with pytest.raises(kp.KickInputError, match="one or two"):
        kp.pick_videos(tmp_path)


def test_validate_inputs_names_each_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kp, "nominal_clock", lambda path: np.arange(80) / 59.94)
    monkeypatch.setattr(d3, "_video_fps", lambda path: 59.94)
    videos = _cameras(tmp_path)
    cams = kp.validate_inputs(videos, reference=2)
    assert cams[0].stem == CAM2 and cams[1].stem == CAM1 and cams[0].sync_frame == 14

    no_sync = tmp_path / "plain.mp4"
    no_sync.write_bytes(b"")
    (tmp_path / "plain_markers.csv").write_text("frame,p0_x,p0_y\n", encoding="utf-8")
    with pytest.raises(kp.KickInputError, match="no sync"):
        kp.validate_inputs([no_sync, videos[1]], reference=1)

    bare = tmp_path / "nomarkers"
    bare.mkdir()
    bare_videos = _cameras(bare, markers=False)
    with pytest.raises(kp.KickInputError, match="ball markers not found"):
        kp.validate_inputs(bare_videos, reference=1)

    with pytest.raises(kp.KickInputError, match="video not found"):
        kp.validate_inputs([videos[0], tmp_path / "missing_sync_1_frames_0_to_5.mp4"], reference=1)

    with pytest.raises(kp.KickInputError, match="same file"):
        kp.validate_inputs([videos[0], videos[0]], reference=1)

    monkeypatch.setattr(d3, "_video_fps", lambda path: 30.0 if path.stem == CAM2 else 59.94)
    with pytest.raises(kp.KickInputError, match="frame rates"):
        kp.validate_inputs(videos, reference=1)


def test_one_video_needs_no_sync_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kp, "nominal_clock", lambda path: np.arange(80) / 59.94)
    monkeypatch.setattr(d3, "_video_fps", lambda path: 59.94)
    video = tmp_path / "lance.mp4"
    video.write_bytes(b"")
    (tmp_path / "lance_markers.csv").write_text("frame,p0_x,p0_y\n0,1,1\n", encoding="utf-8")
    cams = kp.validate_inputs([video], reference=1)
    assert len(cams) == 1 and cams[0].stem == "lance"


def _equalized(folder: Path, *, markers: bool = True) -> list[Path]:
    """The two videos of the event equalizer: no sync in the names, the manifest has it."""
    videos = []
    for stem in ("reference_c2_16mm_left_equalized", "source_c1_main_equalized"):
        video = folder / f"{stem}.mp4"
        video.write_bytes(b"")
        if markers:
            (folder / f"{stem}_markers.csv").write_text(
                "frame,p0_x,p0_y\n0,1,1\n", encoding="utf-8"
            )
        videos.append(video)
    manifest = {
        "vaila_event_equalizer": 1,
        "frame_base": 1,
        "output_event_frames": [1, 33, 85],
        "cameras": {
            "reference": {"video": videos[0].name},
            "source": {"video": videos[1].name},
        },
    }
    (folder / "event_equalizer.json").write_text(json.dumps(manifest), encoding="utf-8")
    return videos


def test_pick_videos_takes_the_two_videos_of_the_event_equalizer(tmp_path: Path) -> None:
    reference, source = _equalized(tmp_path, markers=False)
    assert kp.pick_videos(tmp_path) == [reference, source]  # the folder the equalizer wrote


def test_the_two_equalized_videos_share_the_contact_as_their_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kp, "nominal_clock", lambda path: np.arange(85) / 120.0)
    monkeypatch.setattr(d3, "_video_fps", lambda path: 120.0)
    cams = kp.validate_inputs(kp.pick_videos(_equalized(tmp_path)[0].parent), reference=1)
    assert [(c.sync_frame, c.cut_start) for c in cams] == [(32, 0), (32, 0)]
    assert cams[0].stem == "reference_c2_16mm_left_equalized"
    swapped = kp.validate_inputs(kp.pick_videos(tmp_path), reference=2)
    assert swapped[0].stem == "source_c1_main_equalized"


def test_missing_ball_markers_are_listed_for_every_camera_with_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kp, "nominal_clock", lambda path: np.arange(85) / 120.0)
    monkeypatch.setattr(d3, "_video_fps", lambda path: 120.0)
    videos = _equalized(tmp_path, markers=False)
    with pytest.raises(kp.KickInputError) as raised:
        kp.validate_inputs(videos, reference=1)
    text = str(raised.value)
    assert "ball markers not found" in text
    for video in videos:
        assert f"{video.stem}_markers.csv" in text
        assert f"getpixelvideo.py -f {video}" in text  # the exact command that writes the file


def test_one_equalized_video_uses_its_manifest_sync_unless_one_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kp, "nominal_clock", lambda path: np.arange(85) / 120.0)
    monkeypatch.setattr(d3, "_video_fps", lambda path: 120.0)
    reference = _equalized(tmp_path)[0]
    alone = kp.validate_inputs([reference], reference=1)
    assert (alone[0].sync_frame, alone[0].cut_start) == (32, 0)
    given = kp.validate_inputs([reference], reference=1, sync=(40, 5))
    assert (given[0].sync_frame, given[0].cut_start) == (40, 5)


def test_the_equalized_pair_runs_the_binocular_and_both_monocular_analyses(
    tmp_path: Path, stubs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kp, "nominal_clock", lambda path: np.arange(85) / 120.0)
    monkeypatch.setattr(d3, "_video_fps", lambda path: 120.0)
    _equalized(tmp_path)
    result = kp.run_kick_analysis("ws", tmp_path, reference=1, out_dir=tmp_path / "kick")
    status = json.loads(Path(result["status_file"]).read_text(encoding="utf-8"))
    assert all(rec["status"] == "ok" for rec in status["stages"].values())
    assert {"binocular", "triangulate", "kicker:binocular", "kicker:mono-other"} <= set(
        status["stages"]
    )
    assert result["summary"]["chosen"] == "binocular"


def test_the_player_ids_align_two_equalized_cameras_with_no_offset(tmp_path: Path) -> None:
    folders = []
    for video in _equalized(tmp_path):
        folder = tmp_path / f"{video.stem}_3d_20261029_000000"
        folder.mkdir()
        (folder / "README.txt").write_text(f"video: {video}\n", encoding="utf-8")
        folders.append(folder)
    assert [pids._cut_sync(f) for f in folders] == [(32, 0), (32, 0)]


def test_a_single_camera_with_no_sync_anywhere_still_gets_its_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "plain_clip.mp4"
    video.write_bytes(b"")
    folder = tmp_path / "plain_clip_3d_20261029_000000"
    folder.mkdir()
    (folder / "README.txt").write_text(f"video: {video}\n", encoding="utf-8")
    seen: dict = {}

    def load_view(run_dir, *, markers, timing, sync_frame, cut_start, name):
        seen.update(sync_frame=sync_frame, cut_start=cut_start)
        return "view", video

    monkeypatch.setattr(st, "load_view", load_view)
    assert kp._camera_view(folder) == "view"  # a monocular free kick needs only the DLT
    assert seen == {"sync_frame": 0, "cut_start": 0}
    kp._camera_view(folder, (71, 63))
    assert seen == {"sync_frame": 71, "cut_start": 63}


def test_a_finished_stage_is_reused_until_its_inputs_change(tmp_path: Path) -> None:
    state = kp._State(tmp_path, "t")
    out_file = tmp_path / "result.txt"
    runs: list[int] = []

    def fn() -> dict:
        runs.append(1)
        out_file.write_text("x", encoding="utf-8")
        return {"path": str(out_file)}

    first = state.stage("demo", {"v": 1}, [out_file], fn, reuse=True)
    again = state.stage("demo", {"v": 1}, [out_file], fn, reuse=True)
    assert first == again and len(runs) == 1
    state.stage("demo", {"v": 1}, [out_file], fn, reuse=False)
    assert len(runs) == 2
    state.stage("demo", {"v": 2}, [out_file], fn, reuse=True)
    assert len(runs) == 3


def test_a_failed_stage_is_recorded_and_named(tmp_path: Path) -> None:
    state = kp._State(tmp_path, "t")

    def boom() -> dict:
        raise RuntimeError("the GPU is busy")

    with pytest.raises(kp.KickStageError, match="demo failed: RuntimeError: the GPU is busy"):
        state.stage("demo", {}, [], boom, reuse=True)
    saved = json.loads((tmp_path / kp.STATUS_NAME).read_text(encoding="utf-8"))
    assert saved["stages"]["demo"]["status"] == "failed"
    assert "GPU is busy" in saved["stages"]["demo"]["error"]


def test_the_whole_run_writes_every_stage_in_order(tmp_path: Path, stubs: dict) -> None:
    videos = _cameras(tmp_path)
    result = kp.run_kick_analysis("ws", videos, reference=1, out_dir=tmp_path / "kick")
    out = Path(result["out"])
    status = json.loads(Path(result["status_file"]).read_text(encoding="utf-8"))
    assert list(status["stages"]) == [
        f"detect:{CAM1}", f"3d:{CAM1}", f"detect:{CAM2}", f"3d:{CAM2}",
        "players", "setpiece", "binocular", "triangulate",
        "teams:mono-ref", "kicker:mono-ref", "report:mono-ref",
        "teams:mono-other", "kicker:mono-other", "report:mono-other",
        "c3d:binocular", "teams:binocular", "kicker:binocular", "report:binocular",
    ]  # fmt: skip
    assert all(rec["status"] == "ok" for rec in status["stages"].values())
    assert (out / "README.txt").is_file()
    readme = (out / "README.txt").read_text(encoding="utf-8")
    assert "binocular  <- chosen" in readme
    assert stubs["report"] == 3 and stubs["teams"] == 1 and stubs["teams_mono"] == 2


def test_a_second_run_in_the_same_folder_reuses_every_stage(tmp_path: Path, stubs: dict) -> None:
    videos = _cameras(tmp_path)
    out = tmp_path / "kick"
    kp.run_kick_analysis("ws", videos, reference=1, out_dir=out)
    before = dict(stubs)
    kp.run_kick_analysis("ws", videos, reference=1, out_dir=out)
    assert stubs == before


def test_a_run_from_before_the_flight_version_redoes_what_is_built_from_it(
    tmp_path: Path, stubs: dict
) -> None:
    """The stored flight of an older fit must not be reused, nor the stages that read it."""
    videos = _cameras(tmp_path)
    out = tmp_path / "kick"
    kp.run_kick_analysis("ws", videos, reference=1, out_dir=out)
    status_path = out / kp.STATUS_NAME
    status = json.loads(status_path.read_text(encoding="utf-8"))
    threed = status["stages"][f"3d:{CAM1}"]["inputs"]
    assert threed["flight_model"] == fk_physics.FLIGHT_MODEL
    del threed["flight_model"]  # what a folder made before the tag looks like
    status_path.write_text(json.dumps(status), encoding="utf-8")
    before = dict(stubs)
    kp.run_kick_analysis("ws", videos, reference=1, out_dir=out)
    assert stubs["3d"] == before["3d"] + 1  # the stage of that camera, not of the other one
    assert stubs["detect"] == before["detect"]  # the detection does not depend on the fit
    for built_from_it in ("players", "binocular", "triangulate", "kicker", "report"):
        assert stubs[built_from_it] > before[built_from_it], built_from_it
    again = dict(stubs)
    kp.run_kick_analysis("ws", videos, reference=1, out_dir=out)  # and now everything is current
    assert stubs == again


def test_a_new_detection_redoes_the_3d_stage_of_that_camera(tmp_path: Path, stubs: dict) -> None:
    videos = _cameras(tmp_path)
    out = tmp_path / "kick"
    kp.run_kick_analysis("ws", videos, reference=1, out_dir=out)
    status_path = out / kp.STATUS_NAME
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["stages"][f"detect:{CAM1}"]["inputs"]["model"] = "an older model"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    before = dict(stubs)
    kp.run_kick_analysis("ws", videos, reference=1, out_dir=out)
    assert stubs["detect"] == before["detect"] + 1 and stubs["3d"] == before["3d"] + 1


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (None, None),  # no ball_flight.json: a smoothed ball, nothing to be stale
        ({"flight_model": "CURRENT"}, None),
        ({"rms_px": 28.1}, "first flight fit"),  # a folder from before the tag
        ({"flight_model": "flight-v0"}, "flight-v0 flight fit"),
    ],
)
def test_stale_flight_names_the_fit_and_the_way_out(
    tmp_path: Path, record: dict | None, expected: str | None
) -> None:
    if record is not None:
        record = {k: (fk_physics.FLIGHT_MODEL if v == "CURRENT" else v) for k, v in record.items()}
        (tmp_path / "ball_flight.json").write_text(json.dumps(record), encoding="utf-8")
    reason = kp.stale_flight(tmp_path)
    if expected is None:
        assert reason is None
    else:
        assert reason is not None and expected in reason
        assert "ball-rebuild" in reason and str(tmp_path) in reason


def test_stale_flight_reports_a_file_it_cannot_read(tmp_path: Path) -> None:
    (tmp_path / "ball_flight.json").write_text("{not json", encoding="utf-8")
    assert "cannot be read" in (kp.stale_flight(tmp_path) or "")


def test_a_camera_the_screen_rejects_stops_the_run(
    tmp_path: Path, stubs: dict, monkeypatch
) -> None:
    videos = _cameras(tmp_path)

    def reject(ws, video, model, min_camera_ok):
        raise kp.KickInputError(f"{video.name} is not a camera the network was trained for")

    monkeypatch.setattr(kp, "_screen_or_raise", reject)
    with pytest.raises(kp.KickStageError, match=f"detect:{CAM1} failed.*not a camera"):
        kp.run_kick_analysis("ws", videos, reference=1, out_dir=tmp_path / "kick")
    status = json.loads((tmp_path / "kick" / kp.STATUS_NAME).read_text(encoding="utf-8"))
    assert status["stages"][f"detect:{CAM1}"]["status"] == "failed"


def test_a_camera_with_its_kiki49_file_is_detected_by_hand_and_not_screened(
    tmp_path: Path, stubs: dict, monkeypatch
) -> None:
    video = _cameras(tmp_path, both=False)[0]
    keypoints = tmp_path / f"{CAM1}_kiki49.csv"
    keypoints.write_text("frame\n", encoding="utf-8")
    screened: list[str] = []
    marked: list[Path | None] = []
    network_detect = fk.detect_video  # the stub of the stubs fixture

    def screen(ws, video, model, min_camera_ok):
        screened.append(video.name)
        return {}

    def detect_video(ws, video, **kwargs):
        marked.append(kwargs.get("keypoints_csv"))
        return network_detect(
            ws, video, model="active", output_dir=kwargs["output_dir"], device=kwargs.get("device")
        )

    monkeypatch.setattr(kp, "_screen_or_raise", screen)
    monkeypatch.setattr(fk, "detect_video", detect_video)
    kp.run_kick_analysis("ws", [video], out_dir=tmp_path / "kick")
    assert screened == []
    assert marked == [keypoints]


def test_a_keypoints_file_named_as_the_ball_is_refused_with_the_fix(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(kp, "nominal_clock", lambda path: np.arange(80) / 59.94)
    video = _cameras(tmp_path, both=False)[0]
    (tmp_path / f"{CAM1}_markers.csv").write_text(
        "frame,p0_x,p0_y,p1_x,p1_y\n0,1,1,2,2\n", encoding="utf-8"
    )
    with pytest.raises(kp.KickInputError, match="holds 49 keypoints"):
        kp.validate_inputs([video], 1)


def test_one_video_gives_only_the_monocular_analysis(tmp_path: Path, stubs: dict) -> None:
    video = _cameras(tmp_path, both=False)[0]
    result = kp.run_kick_analysis("ws", [video], out_dir=tmp_path / "kick")
    status = json.loads(Path(result["status_file"]).read_text(encoding="utf-8"))
    assert list(status["stages"]) == [
        f"detect:{CAM1}", f"3d:{CAM1}", "players", "setpiece",
        "teams:mono-ref", "kicker:mono-ref", "report:mono-ref",
    ]  # fmt: skip
    assert "binocular" not in " ".join(status["stages"])
    assert result["summary"]["chosen"] == "mono-ref"


def test_a_video_file_is_picked_and_its_sync_comes_from_the_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "c1_main_cam_frame_63_to_81.mp4_events.mp4"  # no sync in the name
    video.write_bytes(b"")
    (tmp_path / "c1_main_cam_frame_63_to_81.mp4_events_markers.csv").write_text(
        "frame,p0_x,p0_y\n0,1,1\n", encoding="utf-8"
    )
    monkeypatch.setattr(kp, "nominal_clock", lambda path: np.arange(19) / 29.97)
    monkeypatch.setattr(d3, "_video_fps", lambda path: 29.97)
    assert kp.pick_videos(str(video)) == [video]
    with_sync = kp.validate_inputs([video], 1, sync=(71, 63))
    assert (with_sync[0].sync_frame, with_sync[0].cut_start) == (71, 63)
    without = kp.validate_inputs([video], 1)
    assert (without[0].sync_frame, without[0].cut_start) == (0, 0)


def test_the_sync_argument_is_refused_for_two_videos(tmp_path: Path) -> None:
    videos = _cameras(tmp_path)
    with pytest.raises(kp.KickInputError, match="one video"):
        kp.validate_inputs(videos, 1, sync=(71, 63))


def test_a_sync_given_for_one_clip_reaches_the_monocular_teams_and_the_set_piece_rule(
    tmp_path: Path, stubs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip_events.mp4"
    video.write_bytes(b"")
    (tmp_path / "clip_events_markers.csv").write_text("frame,p0_x,p0_y\n0,1,1\n", encoding="utf-8")
    monkeypatch.setattr(kp, "nominal_clock", lambda path: np.arange(80) / 59.94)
    monkeypatch.setattr(d3, "_video_fps", lambda path: 29.97)
    seen: list = []

    def teams(cam3d, video_, work, *, is_ref, key, team_a, fps, sync=None):
        seen.append((key, sync))
        return {"counts": {"TEAM_A": 0, "TEAM_B": 0, "REF": 0}}

    monkeypatch.setattr(kp, "_teams_mono", teams)
    result = kp.run_kick_analysis("ws", [video], out_dir=tmp_path / "kick", sync=(71, 63))
    assert seen == [("mono-ref", (71, 63))]
    status = json.loads(Path(result["status_file"]).read_text(encoding="utf-8"))
    assert status["stages"]["teams:mono-ref"]["inputs"]["sync"] == [71, 63]
    assert status["stages"]["setpiece"]["inputs"]["penalty_tolerance_m"] == 1.5


def test_a_penalty_gets_the_pynalty_metrics_and_the_freekiki_report(
    tmp_path: Path, stubs: dict, monkeypatch
) -> None:
    video = _cameras(tmp_path, both=False)[0]
    monkeypatch.setattr(
        kp,
        "_setpiece",
        lambda ref3d: {
            "kind": "penalty",
            "reason": "stub",
            "goal_x_m": 52.45,
            "wall_count": 0,
            "wall_ids": [],
        },  # fmt: skip
    )
    calls: list[tuple[str, float]] = []

    def fake_penalty(ref3d, ref_video, ref_work, folder, *, goal_x, track_csv=None):
        calls.append((folder.name, goal_x))
        return {
            "available": True,
            "reason": "",
            "metrics": {"coord_x": 3.0},
            "files": {},
            "report": {},
        }

    monkeypatch.setattr(kp, "_penalty", fake_penalty)
    result = kp.run_kick_analysis("ws", [video], out_dir=tmp_path / "kick")
    status = json.loads(Path(result["status_file"]).read_text(encoding="utf-8"))
    assert "penalty:mono-ref" in status["stages"]
    assert (
        "report:mono-ref" in status["stages"]
    )  # the freekiki report (report_tv) is made for a penalty too
    assert "teams:mono-ref" not in status["stages"]  # no team split for a penalty
    assert len(calls) == 1 and calls[0][0].startswith(CAM1) and calls[0][1] == 52.45
    readme = (Path(result["out"]) / "README.txt").read_text(encoding="utf-8")
    assert "set piece: penalty (stub)" in readme
    assert "coord_x: 3.0" in readme


def test_the_readme_says_whether_the_penalty_ball_path_is_straight_or_bowed() -> None:
    path = {
        "available": True,
        "source": "binocular",
        "shape": "curved",
        "bow_side": "right",
        "max_dev_m": 0.33,
        "apex_z_m": 1.2,
        "spot_fit_offset_m": 0.14,
        "last_frame": 48,
        "goal_frame": 48,
        "arrival_in_goal_mouth": True,
    }
    lines = kp._penalty_lines({"available": True, "metrics": {"coord_x": 3.0}, "path": path})
    assert any(
        "Ball path (data/penalty_path.csv, binocular ball): curved bowing right, max 0.33 m" in line
        and "crosses the goal line at frame 48 (inside the goal mouth)" in line
        for line in lines
    )
    straight = dict(path, shape="straight", max_dev_m=0.01)
    lines = kp._penalty_lines({"available": True, "metrics": {}, "path": straight})
    assert any(
        line.startswith("Ball path (data/penalty_path.csv, binocular ball): straight, max 0.01 m")
        for line in lines
    )


def test_a_failing_monocular_analysis_does_not_stop_the_others(
    tmp_path: Path, stubs: dict, monkeypatch
) -> None:
    videos = _cameras(tmp_path)

    def broken(cam3d, video, work, *, is_ref, key, team_a, fps, sync=None):
        if key == "mono-other":
            raise RuntimeError("the other camera has no colours")
        return {"counts": {"TEAM_A": 0, "TEAM_B": 0, "REF": 0}}

    monkeypatch.setattr(kp, "_teams_mono", broken)
    result = kp.run_kick_analysis("ws", videos, reference=1, out_dir=tmp_path / "kick")
    analyses = result["summary"]["analyses"]
    assert analyses["mono-other"]["status"] == "failed"
    assert "colours" in analyses["mono-other"]["error"]
    assert analyses["binocular"]["status"] == "ok"


def test_the_cli_knows_kick_analysis() -> None:
    args = fk.build_parser().parse_args(
        ["kick-analysis", "-w", "ws", "--videos", "a.mp4", "b.mp4", "--kick-foot", "left"]
    )
    assert args.command == "kick-analysis"
    assert args.videos == ["a.mp4", "b.mp4"] and args.kick_foot == "left" and args.reference == "1"


def test_the_cli_knows_kick_figures() -> None:
    args = fk.build_parser().parse_args(["kick-figures", "kick_analysis_20261028_110255"])
    assert args.command == "kick-figures"
    assert args.kick_dir == "kick_analysis_20261028_110255"


def test_the_one_click_line_names_the_set_piece(tmp_path: Path, monkeypatch, capsys) -> None:
    video = _cameras(tmp_path, both=False)[0]
    summary = {"setpiece": {"kind": "penalty", "reason": "the ball is 11.00 m from the goal line"}}

    def fake_run(ws, videos, **kwargs):
        return {
            "out": str(tmp_path / "kick"),
            "summary": summary,
            "status_file": str(tmp_path / "kick" / kp.STATUS_NAME),
        }

    monkeypatch.setattr(kp, "run_kick_analysis", fake_run)
    assert fk.main(["kick-analysis", "-w", str(tmp_path), "--videos", str(video)]) == 0
    assert "set piece: penalty (the ball is 11.00 m from the goal line)" in capsys.readouterr().out


def test_a_forced_free_kick_is_not_routed_to_the_penalty(
    tmp_path: Path, stubs: dict, monkeypatch
) -> None:
    video = _cameras(tmp_path, both=False)[0]
    monkeypatch.setattr(
        kp,
        "_setpiece",
        lambda ref3d: {
            "kind": "penalty",
            "reason": "stub",
            "goal_x_m": 52.45,
            "wall_count": 0,
            "wall_ids": [],
        },  # fmt: skip
    )
    result = kp.run_kick_analysis("ws", [video], out_dir=tmp_path / "kick", set_piece="free_kick")
    status = json.loads(Path(result["status_file"]).read_text(encoding="utf-8"))
    assert "penalty:mono-ref" not in status["stages"]
    assert "report:mono-ref" in status["stages"]  # the freekiki report, as for any free kick
    readme = (Path(result["out"]) / "README.txt").read_text(encoding="utf-8")
    assert (
        "set piece: free_kick (chosen by the user (free_kick); the ball rule said penalty" in readme
    )


def test_a_penalty_without_a_ball_at_rest_is_reported_as_unavailable(
    tmp_path: Path, stubs: dict
) -> None:
    video = _cameras(tmp_path, both=False)[0]
    result = kp.run_kick_analysis("ws", [video], out_dir=tmp_path / "kick", set_piece="penalty")
    readme = (Path(result["out"]) / "README.txt").read_text(encoding="utf-8")
    assert "set piece: penalty (chosen by the user (penalty)" in readme
    assert "no ball at rest on the field" in readme
    status = json.loads(Path(result["status_file"]).read_text(encoding="utf-8"))
    assert "penalty:mono-ref" not in status["stages"]


def test_the_cli_passes_the_set_piece_choice_and_refuses_others(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    video = _cameras(tmp_path, both=False)[0]
    seen: dict = {}

    def fake_run(ws, videos, **kwargs):
        seen.update(kwargs)
        return {
            "out": str(tmp_path / "kick"),
            "summary": {},
            "status_file": str(tmp_path / "kick" / kp.STATUS_NAME),
        }

    monkeypatch.setattr(kp, "run_kick_analysis", fake_run)
    argv = ["kick-analysis", "-w", str(tmp_path), "--videos", str(video)]
    assert fk.main([*argv, "--set-piece", "penalty"]) == 0
    assert seen["set_piece"] == "penalty"
    assert fk.main([*argv]) == 0
    assert seen["set_piece"] == "auto"
    with pytest.raises(SystemExit):
        fk.main([*argv, "--set-piece", "corner"])


def test_a_figures_rewrite_keeps_the_sync_of_a_clip_without_sync_in_its_name(
    tmp_path: Path, stubs: dict, monkeypatch
) -> None:
    video = tmp_path / "clip.mp4"  # no sync numbers in the name: the sync comes from the argument
    video.write_bytes(b"")
    (tmp_path / "clip_markers.csv").write_text("frame,p0_x,p0_y\n0,1,1\n", encoding="utf-8")
    result = kp.run_kick_analysis("ws", [video], out_dir=tmp_path / "kick", sync=(71, 63))
    status = json.loads(Path(result["status_file"]).read_text(encoding="utf-8"))
    assert status["sync"] == [71, 63]

    seen: list = []
    original = kp._analyses

    def record(*args, **kwargs):
        seen.append(kwargs.get("ref_sync"))
        return original(*args, **kwargs)

    monkeypatch.setattr(kp, "_analyses", record)
    kp.regenerate_figures(result["out"])
    assert seen == [(71, 63)]


def test_a_penalty_writes_the_other_people_to_other_ids_and_no_one_else(tmp_path: Path) -> None:
    player_map = tmp_path / "player_map.csv"
    player_map.write_text(
        "player_id,role,camera,sam_id\n0,keeper,c1,12\n1,kicker,c1,7\n2,other,c1,1\n3,other,c1,18\n",
        encoding="utf-8",
    )
    poses = tmp_path / "poses_3d_on_field.csv"
    poses.write_text(
        "frame,person_id,kpt_idx,kpt_name,x_m,y_m,z_m\n"
        "0,12,0,nose,50.0,0.0,1.7\n0,7,0,nose,41.0,0.0,1.8\n0,1,0,nose,30.0,5.0,1.7\n0,18,0,nose,20.0,-6.0,1.7\n",
        encoding="utf-8",
    )
    out = kp._penalty_others(player_map, poses, "person_id", tmp_path / "data")
    assert out == tmp_path / "data" / "other_IDs" / "poses_other_persons.csv"
    kept = [line.split(",")[1] for line in out.read_text(encoding="utf-8").splitlines()[1:]]
    assert kept == ["1", "18"]  # the keeper (12) and the kicker (7) are not in other_IDs


def test_the_fitted_contact_replaces_the_marker_only_when_the_fit_is_reliable_and_far() -> None:
    fps = 30.0
    flight = {"kick_time_s": 8 / fps, "rms_px": 1.0}
    assert kp._fit_contact(flight, 5, fps) == 8  # the marker drifted 3 frames early
    assert kp._fit_contact(flight, 7, fps) is None  # within one frame: the marker stays
    assert kp._fit_contact({**flight, "rms_px": 50.0}, 5, fps) is None  # a rejected fit
    assert kp._fit_contact({"rms_px": 1.0}, 5, fps) is None  # no kick time in the fit
    assert kp._fit_contact(flight, None, fps) is None  # no marker to compare with


def test_the_penalty_overlay_is_redrawn_with_the_player_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cam3d, work = tmp_path / "cam3d", tmp_path / "work"
    cam3d.mkdir()
    work.mkdir()
    (cam3d / "player_map.csv").write_text(
        "player_id,role,camera,sam_id\n0,keeper,c1,12\n1,kicker,c1,7\n2,other,c1,1\n",
        encoding="utf-8",
    )
    (cam3d / "poses_3d_on_field.csv").write_text(
        "frame,person_id,kpt_idx,kpt_name,x_m,y_m,z_m\n"
        "0,12,0,nose,50.0,0.0,1.7\n0,7,0,nose,41.0,0.0,1.8\n",
        encoding="utf-8",
    )
    detect_run = tmp_path / "detect"
    seen: dict = {}
    monkeypatch.setattr(d3, "_detect_run_dir", lambda _work, _stem: detect_run)

    def fake_overlay(video, run_dir, out_dir, work_dir, *, kept=None, labels=None):
        seen.update(video=video, run_dir=run_dir, out_dir=out_dir, kept=kept, labels=labels)
        return Path(out_dir) / "o.mp4"

    monkeypatch.setattr(d3, "write_3d_overlay", fake_overlay)
    kp._penalty_overlay(cam3d, tmp_path / "clip.mp4", work)
    assert seen["labels"] == {12: 0, 7: 1, 1: 2}  # SAM id -> player id: ID0 keeper, ID1 kicker
    assert seen["kept"] == {12, 7}  # the people that are on the pitch, not every track
    assert seen["run_dir"] == detect_run and seen["out_dir"] == cam3d

    # no detect run or no poses: nothing is redrawn
    seen.clear()
    monkeypatch.setattr(d3, "_detect_run_dir", lambda _work, _stem: None)
    kp._penalty_overlay(cam3d, tmp_path / "clip.mp4", work)
    assert not seen


def test_a_failing_overlay_redraw_does_not_stop_the_penalty_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cam3d, work = tmp_path / "cam3d", tmp_path / "work"
    cam3d.mkdir()
    work.mkdir()
    (cam3d / "player_map.csv").write_text(
        "player_id,role,camera,sam_id\n0,keeper,c1,12\n1,kicker,c1,7\n", encoding="utf-8"
    )
    (cam3d / "poses_3d_on_field.csv").write_text(
        "frame,person_id,kpt_idx,kpt_name,x_m,y_m,z_m\n0,12,0,nose,50.0,0.0,1.7\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(d3, "_detect_run_dir", lambda _work, _stem: tmp_path)

    def broken(*_args, **_kwargs):
        raise FileNotFoundError("field_kps_getpixelvideo.csv")

    monkeypatch.setattr(d3, "write_3d_overlay", broken)
    kp._penalty_overlay(cam3d, tmp_path / "clip.mp4", work)  # does not raise
    assert "overlay not redrawn" in capsys.readouterr().out
