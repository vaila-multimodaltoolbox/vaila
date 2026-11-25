"""The per-frame cameras in the kick pipeline: stage ``cameras:<stem>``, the ball wiring, the CLI.

Update Date: 24 November 2026
Version: 0.4.19

No GPU, no model: the heavy stages are the stubs of ``test_freekiki_kick_pipeline`` (``stubs``,
which also replaces ``write_frame_cameras`` by a writer of the four files). What is checked here:

* the stage runs right after ``sam:`` of each camera, lists the four files as outputs, is reused,
  and a new detection quality / raw field points / method version redoes the cameras and the ball,
  not SAM nor the 3D stage; ``stop_after="sam"`` writes the cameras too;
* the ball and flight options (per-frame cameras, contact from the feet, the roster's kicker) reach
  ``detect_3d_run`` and ``rebuild_ball_in_place`` unless a contact was given by hand;
* a failing build does not stop the analysis (said in the log and the README, tried again next time);
* the real ``write_frame_cameras`` on a hand-made kick folder (files, replaced, restored on error);
* the set piece takes the attacked goal from the flight's ``side``; the ``feet`` contact labels;
* ``kick-cameras``: argv, parser, mirror lines that the real parser accepts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests import test_freekiki_kick_correct as kc_tests
from tests import test_freekiki_kick_pipeline as kp_tests
from tests import test_freekiki_roster_pipeline as rp_tests
from tests import test_kick_measure as km_tests
from vaila import freekiki as fk
from vaila import freekiki_detect3d as d3
from vaila import freekiki_kick_pipeline as kp
from vaila import freekiki_kicker as kicker
from vaila import freekiki_report as rep
from vaila import freekiki_report_i18n as i18n
from vaila import freekiki_setpiece as sp
from vaila import kick_measure as km

stubs = kp_tests.stubs
kick_folder = km_tests.kick_folder
kick_run = kc_tests.kick_run
CAM1, CAM2 = kp_tests.CAM1, kp_tests.CAM2
_cameras = kp_tests._cameras
_counts = kp_tests._counts


def _status(out: Path) -> dict:
    return json.loads((out / kp.STATUS_NAME).read_text(encoding="utf-8"))


# --- the stage --------------------------------------------------------------------------------


def test_each_camera_gets_its_cameras_stage_right_after_sam(tmp_path: Path, stubs: dict) -> None:
    out = tmp_path / "kick"
    kp.run_kick_analysis("ws", _cameras(tmp_path), reference=1, out_dir=out)
    stages = _status(out)["stages"]
    keys = list(stages)
    for stem in (CAM1, CAM2):
        assert keys.index(f"cameras:{stem}") == keys.index(f"sam:{stem}") + 1
    record = stages[f"cameras:{CAM1}"]
    assert [Path(p).name for p in record["outputs"]] == [
        f"{CAM1}_frames.dlt3d", f"{CAM1}_frames.dlt2d",
        f"{CAM1}_frames_cameras.csv", f"{CAM1}_frames_cameras.json",
    ]  # fmt: skip
    assert all(Path(p).parent.name == CAM1 for p in record["outputs"])  # the SAM 3 stem folder
    assert {"video", "quality", "model"} <= set(record["inputs"])
    assert record["inputs"]["model"] == kp.CAMERAS_MODEL
    # reference first: the stub saw ("reference", "other") as the camera names
    assert stubs["cameras_args"] == [(CAM1, "reference"), (CAM2, "other")]


def test_the_cameras_stage_is_reused_and_redone_by_its_inputs_only(
    tmp_path: Path, stubs: dict
) -> None:
    videos = _cameras(tmp_path)
    out = tmp_path / "kick"
    kp.run_kick_analysis("ws", videos, reference=1, out_dir=out)
    before = _counts(stubs)
    kp.run_kick_analysis("ws", videos, reference=1, out_dir=out)
    assert _counts(stubs) == before  # a second run reuses it, and everything else
    # new raw field points (or a new quality.json) or a new method version: the cameras again
    status = _status(out)
    status["stages"][f"cameras:{CAM1}"]["inputs"]["model"] = "an older tracking"
    (out / kp.STATUS_NAME).write_text(json.dumps(status), encoding="utf-8")
    kp.run_kick_analysis("ws", videos, reference=1, out_dir=out)
    after = _counts(stubs)
    assert after["cameras"] == before["cameras"] + 1
    assert after["detect"] == before["detect"] and after["sam"] == before["sam"]
    # a new camera file redoes the 3D stage of that camera (its poses use them); the stage reads
    # the SAM 3D Body output of the 2D stage, which is not run again
    assert after["3d"] == before["3d"] + 1 and after["sam"] == before["sam"]
    for built_from_it in ("players", "kicker", "report"):  # a camera stage that ran: later ones too
        assert after[built_from_it] > before[built_from_it], built_from_it


def test_a_new_camera_file_redoes_the_3d_stage_and_the_ball_of_that_camera(
    tmp_path: Path, stubs: dict
) -> None:
    videos = _cameras(tmp_path, both=False)
    out = tmp_path / "kick"
    kp.run_kick_analysis("ws", videos, out_dir=out)
    before = _counts(stubs)
    dlt3d = next((out / "work").rglob(f"{CAM1}_frames.dlt3d"))
    dlt3d.write_text("frame,L1\n0,1\n", encoding="utf-8")  # a refreshed file: new size and mtime
    kp.run_kick_analysis("ws", videos, out_dir=out)
    # the 3D run writes the ball itself with the same cameras: no separate ball rebuild after it
    assert stubs["3d"] == before["3d"] + 1 and stubs.get("ball", 0) == before.get("ball", 0)
    assert _status(out)["stages"][f"ball:{CAM1}"]["result"] == {"from": "3d"}
    assert stubs["sam"] == before["sam"] and stubs["detect"] == before["detect"]
    assert stubs["cameras"] == before["cameras"]  # the stage itself had nothing new


def test_an_older_run_without_a_cameras_record_writes_them_and_redoes_the_3d_and_the_ball(
    tmp_path: Path, stubs: dict
) -> None:
    videos = _cameras(tmp_path, both=False)
    out = tmp_path / "kick"
    kp.run_kick_analysis("ws", videos, out_dir=out)
    status = _status(out)
    del status["stages"][f"cameras:{CAM1}"]  # a run made before the stage existed
    for key in ("cameras", "feet_contact", "kicker_sam"):
        status["stages"][f"ball:{CAM1}"]["inputs"].pop(key, None)
    (out / kp.STATUS_NAME).write_text(json.dumps(status), encoding="utf-8")
    before = _counts(stubs)
    kp.run_kick_analysis("ws", videos, out_dir=out)
    assert stubs["cameras"] == before["cameras"] + 1
    assert stubs["detect"] == before["detect"] and stubs["sam"] == before["sam"]
    assert stubs["3d"] == before["3d"] + 1  # the poses again, with the cameras (SAM is not run)
    assert stubs.get("ball", 0) == before.get("ball", 0)  # written by that 3D run, same cameras
    assert _status(out)["stages"][f"ball:{CAM1}"]["result"] == {"from": "3d"}
    assert f"cameras:{CAM1}" in _status(out)["stages"]


def test_stop_after_sam_writes_the_cameras_too(
    tmp_path: Path, stubs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vaila import freekiki_roster as ro

    monkeypatch.setattr(
        ro, "write_roster_sheet",
        lambda predictions, video, folder, **_: {"sheet": str(folder / "s.png"),
                                                 "csv": str(folder / "roster.csv")},
    )  # fmt: skip
    out = tmp_path / "kick"
    found = kp.run_kick_analysis(
        "ws", _cameras(tmp_path, both=False), out_dir=out, stop_after="sam"
    )
    assert found["stopped_after"] == "sam"
    assert list(found["cameras"]) == [CAM1]
    assert Path(found["cameras"][CAM1]["files"]["dlt3d"]).is_file()
    assert f"cameras:{CAM1}" in _status(out)["stages"]
    assert f"3d:{CAM1}" not in _status(out)["stages"]  # the user measures before the 3D stage


def test_the_readme_lists_the_per_frame_cameras_and_the_approximate_frames(
    tmp_path: Path, stubs: dict
) -> None:
    out = tmp_path / "kick"
    kp.run_kick_analysis("ws", _cameras(tmp_path, both=False), out_dir=out)
    text = (out / "README.txt").read_text(encoding="utf-8")
    assert f"per-frame cameras of {CAM1}: 4 frames (1 approximate, 2 dlt3d, 1 solved)" in text
    assert f"{CAM1}_frames.dlt3d" in text and "1 frames are only a copy" in text


def test_a_failing_camera_build_does_not_stop_the_analysis(
    tmp_path: Path, stubs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    videos = _cameras(tmp_path, both=False)
    out = tmp_path / "kick"

    def broken(kick_dir, camera="reference"):
        stubs["broken"] = stubs.get("broken", 0) + 1
        raise ValueError("no calibrated frame")

    monkeypatch.setattr(kp, "write_frame_cameras", broken)
    kp.run_kick_analysis("ws", videos, out_dir=out)
    record = _status(out)["stages"][f"cameras:{CAM1}"]
    assert record["status"] == "ok" and "no calibrated frame" in record["result"]["error"]
    assert "ball_cameras" not in stubs["3d_kwargs"][0]  # the FreeKiki DLT3D frames serve
    readme = (out / "README.txt").read_text(encoding="utf-8")
    assert f"per-frame cameras of {CAM1}: NOT written (ValueError: no calibrated frame)" in readme
    kp.run_kick_analysis("ws", videos, out_dir=out)
    assert stubs["broken"] == 2  # a file that was not written is tried again on the next call


# --- the ball options -------------------------------------------------------------------------


def test_the_ball_gets_the_per_frame_cameras_and_the_feet_contact(
    tmp_path: Path, stubs: dict
) -> None:
    out = tmp_path / "kick"
    kp.run_kick_analysis("ws", _cameras(tmp_path), reference=1, out_dir=out)
    ref, other = stubs["3d_kwargs"][:2]
    for kwargs in (ref, other):
        assert sorted(kwargs["ball_cameras"]) == [0, 1] and kwargs["feet_contact"] is True
    assert ref["kicker_sam"] is None  # no roster: everybody's feet are looked at
    inputs = _status(out)["stages"][f"ball:{CAM1}"]["inputs"]
    assert inputs["cameras"]["path"].endswith(f"{CAM1}_frames.dlt3d")
    assert inputs["feet_contact"] is True and inputs["kicker_sam"] is None


def test_the_roster_kicker_is_the_one_whose_feet_are_read(tmp_path: Path, stubs: dict) -> None:
    out = tmp_path / "kick"
    roster = rp_tests._write_roster(
        tmp_path / "roster.csv",
        (22, "B", "kicker", "V"),
        (25, "A", "keeper", "R"),
        (1, "A", "", ""),
    )
    videos = _cameras(tmp_path)
    kp.run_kick_analysis("ws", videos, reference=1, out_dir=out, roster=roster)
    ref, other = stubs["3d_kwargs"][:2]
    assert ref["kicker_sam"] == 22 and other["kicker_sam"] is None  # only the reference is labelled
    assert _status(out)["stages"][f"ball:{CAM1}"]["inputs"]["kicker_sam"] == 22
    before = _counts(stubs)
    changed = rp_tests._write_roster(
        tmp_path / "roster.csv",
        (23, "B", "kicker", "V"),
        (25, "A", "keeper", "R"),
        (1, "A", "", ""),
    )
    kp.run_kick_analysis("ws", videos, reference=1, out_dir=out, roster=changed)
    assert stubs["ball"] == before.get("ball", 0) + 1  # a new kicker: the ball (the contact) again
    assert stubs["3d"] == before["3d"] and stubs["cameras"] == before["cameras"]
    assert stubs["ball_kwargs"][-1]["kicker_sam"] == 23


def test_a_contact_given_by_hand_wins_over_the_feet(tmp_path: Path, stubs: dict) -> None:
    out = tmp_path / "kick"
    kp.run_kick_analysis("ws", _cameras(tmp_path), reference=1, out_dir=out, contact_frame=7)
    ref, other = stubs["3d_kwargs"][:2]
    assert ref["contact_frame"] == 7
    assert "feet_contact" not in ref and "kicker_sam" not in ref  # the hand-given frame stands
    assert sorted(ref["ball_cameras"]) == [0, 1]  # the per-frame cameras still serve the fit
    assert other["feet_contact"] is True  # the other camera has no hand-given frame
    inputs = _status(out)["stages"][f"ball:{CAM1}"]["inputs"]
    assert "feet_contact" not in inputs and "kicker_sam" not in inputs


def test_the_3d_stage_places_the_people_with_the_per_frame_cameras(
    tmp_path: Path, stubs: dict
) -> None:
    videos = _cameras(tmp_path, both=False)
    out = tmp_path / "kick"
    kp.run_kick_analysis("ws", videos, out_dir=out)
    kwargs = stubs["3d_kwargs"][-1]
    # the cameras of the ball and of the poses are the same per-frame set
    assert sorted(kwargs["pose_cameras"]) == [0, 1] == sorted(kwargs["ball_cameras"])
    record = _status(out)["stages"][f"3d:{CAM1}"]["inputs"]
    assert record["cameras"]["path"].endswith(f"{CAM1}_frames.dlt3d")  # a new file reruns the 3D


def test_a_run_without_the_cameras_file_places_the_people_as_before(
    tmp_path: Path, stubs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_cameras(kick_dir, camera="reference"):
        raise ValueError("no SAM 3 folder")

    monkeypatch.setattr(kp, "write_frame_cameras", no_cameras)
    out = tmp_path / "kick"
    kp.run_kick_analysis("ws", _cameras(tmp_path, both=False), out_dir=out)
    kwargs = stubs["3d_kwargs"][-1]
    assert "pose_cameras" not in kwargs and "ball_cameras" not in kwargs
    assert "cameras" not in _status(out)["stages"][f"3d:{CAM1}"]["inputs"]


def test_the_ball_rebuild_of_the_ball_stage_carries_the_same_options(
    tmp_path: Path, stubs: dict
) -> None:
    videos = _cameras(tmp_path, both=False)
    out = tmp_path / "kick"
    kp.run_kick_analysis("ws", videos, out_dir=out)
    markers = tmp_path / f"{CAM1}_markers.csv"
    markers.write_text(markers.read_text(encoding="utf-8") + "1,2,2\n", encoding="utf-8")
    kp.run_kick_analysis("ws", videos, out_dir=out)
    kwargs = stubs["ball_kwargs"][-1]
    assert sorted(kwargs["ball_cameras"]) == [0, 1] and kwargs["feet_contact"] is True
    # a rebuild rewrites the poses: it needs the cameras of the 3D stage, or the poses fall back
    assert sorted(kwargs["pose_cameras"]) == [0, 1]


def test_the_refine_stage_rebuilds_the_ball_with_the_same_options(
    tmp_path: Path, stubs: dict
) -> None:
    video = _cameras(tmp_path, both=False)[0]
    kp.run_kick_analysis("ws", [video], out_dir=tmp_path / "kick", statures="keeper=2.04")
    assert stubs["ball_kwargs"], "the refine rebuild must have run"
    last = stubs["ball_kwargs"][-1]
    assert sorted(last["ball_cameras"]) == [0, 1] and last["feet_contact"] is True
    assert sorted(last["pose_cameras"]) == [0, 1]


# --- the real writer (hand-made kick folder) --------------------------------------------------


def _sam_folder(run: Path) -> Path:
    folder = run / "work" / km_tests.STEM / "sam3dinov3" / "processed_sam3dinov3_x" / km_tests.STEM
    folder.mkdir(parents=True)
    return folder


def test_write_frame_cameras_writes_the_four_files_next_to_the_sam_outputs(
    kick_folder: dict,
) -> None:
    run = kick_folder["run"]
    sam = _sam_folder(run)
    found = kp.write_frame_cameras(run)
    stem = km_tests.STEM
    assert Path(found["dir"]) == sam and found["stem"] == stem
    assert {k: Path(v).name for k, v in found["files"].items()} == {
        "dlt3d": f"{stem}_frames.dlt3d", "dlt2d": f"{stem}_frames.dlt2d",
        "cameras": f"{stem}_frames_cameras.csv", "summary": f"{stem}_frames_cameras.json",
    }  # fmt: skip
    table = pd.read_csv(sam / f"{stem}_frames.dlt3d")
    assert len(table) == found["n_frames"] and np.isfinite(table.to_numpy()).all()
    assert sum(found["counts"].values()) == found["n_frames"]
    track = km.read_frame_calibration(sam / f"{stem}_frames.dlt3d")
    assert track.n_frames == found["n_frames"]
    ball = kp.frame_cameras_for_ball(sam / f"{stem}_frames.dlt3d")
    assert ball and all(np.asarray(v).shape == (11,) for v in ball.values())
    assert len(ball) == found["n_frames"] - found["counts"].get("approximate", 0)


def test_write_frame_cameras_replaces_the_files_and_restores_them_on_an_error(
    kick_folder: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = kick_folder["run"]
    sam = _sam_folder(run)
    first = kp.write_frame_cameras(run)
    dlt3d = Path(first["files"]["dlt3d"])
    good = dlt3d.read_bytes()
    kp.write_frame_cameras(run)  # again: replaced, no ``.previous`` left behind
    assert dlt3d.read_bytes() == good and not list(sam.glob("*.previous"))

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(km, "write_frame_calibration", broken)
    with pytest.raises(OSError, match="disk full"):
        kp.write_frame_cameras(run)
    assert dlt3d.read_bytes() == good and not list(sam.glob("*.previous"))  # the files came back


def test_write_frame_cameras_needs_the_sam_folder(kick_folder: dict) -> None:
    with pytest.raises(ValueError, match="no SAM 3 output folder"):
        kp.write_frame_cameras(kick_folder["run"])


# --- set piece: the goal of the flight --------------------------------------------------------


def test_the_goal_is_the_one_the_flight_goes_to_not_the_one_nearer_to_the_ball() -> None:
    near = sp.classify((-11.07, -3.16), {}, "rolling")
    assert near["goal_x_m"] == -sp.GOAL_LINE_X_M  # the old rule: the sign of the kick x
    far = sp.classify((-11.07, -3.16), {}, "rolling", side=1)
    assert far["goal_x_m"] == sp.GOAL_LINE_X_M
    assert far["distance_to_goal_m"] == pytest.approx(63.52, abs=0.01)
    assert sp.classify((30.0, 0.0), {}, "rolling", side=-1)["goal_x_m"] == -sp.GOAL_LINE_X_M
    assert sp.classify((30.0, 0.0), {}, "rolling", side=None)["goal_x_m"] == sp.GOAL_LINE_X_M


def test_the_wall_ring_is_toward_the_goal_of_the_flight() -> None:
    feet = {5: (-2.0, -3.0), 6: (-2.0, -2.0)}  # 9.15 m toward +x of a ball at x = -11.1
    free = sp.classify((-11.1, -3.0), feet, None, side=1)
    assert free["wall_ids"] == [5, 6] and free["kind"] == "free_kick_wall"
    assert sp.classify((-11.1, -3.0), feet, None, side=-1)["wall_ids"] == []  # behind the ball


def test_classify_run_uses_the_side_of_the_fitted_flight(tmp_path: Path) -> None:
    folder = tmp_path / "threed"
    folder.mkdir()
    flight = {"contact_frame": 78, "contact_source": "feet", "pre_contact": "rolling",
              "side": 1.0, "anchor_points": {"p0": {"xyz": [-11.07, -3.16, 0.11]}}}  # fmt: skip
    (folder / "ball_flight.json").write_text(json.dumps(flight), encoding="utf-8")
    assert sp.classify_run(folder)["goal_x_m"] == sp.GOAL_LINE_X_M
    flight.pop("side")  # an older flight: the sign of the kick x, as before
    (folder / "ball_flight.json").write_text(json.dumps(flight), encoding="utf-8")
    assert sp.classify_run(folder)["goal_x_m"] == -sp.GOAL_LINE_X_M


# --- the feet contact in the readers ----------------------------------------------------------


def test_the_feet_contact_has_a_label_in_every_reader() -> None:
    assert "kicker's foot" in kp._contact_lines({"frame": 78, "source": "feet"})[0]
    for lang in ("pt", "en"):
        assert i18n.tr(lang, "contact_limit_feet") != i18n.tr(lang, "contact_limit_rest")
    lead, text, limit = rep._incoming_texts(
        {"incoming": {}, "launch_kmh": 120.0, "contact_source": "feet"}, "en"
    )
    assert limit == i18n.tr("en", "contact_limit_feet")
    assert "feet" in limit


def test_the_kicker_stage_takes_the_feet_contact_as_decided_by_the_flight() -> None:
    n = 100
    rows = [{"frame": f, "player_id": 1, "kpt_idx": 0, "kpt_name": "left-ankle", "valid": True,
             "x_m": float(f), "y_m": 0.0, "z_m": 0.0} for f in range(n)]  # fmt: skip
    ball = np.array([[float(f), 0.1, 0.1] for f in range(n)])
    result = kicker.analyze_kicker(
        rows, ball, {}, kicker_id=1, keeper_id=None, fps=30.0, kick_foot="auto",
        contact_override=78, override_source="feet",
    )  # fmt: skip
    metric = next(m for m in result["metrics"] if m["metric"] == "contact_frame")
    assert result["events"]["contact_source"] == "feet" and metric["value"] == 78
    assert metric["source"] == "inferido" and "kicker's foot" in metric["reason"]


# --- kick-cameras -----------------------------------------------------------------------------


def test_kick_cameras_argv_and_parser(kick_run: dict) -> None:
    run = str(kick_run["run"])
    assert fk.kick_cameras_argv(run) == ["kick-cameras", run, "--camera", "reference"]
    assert fk.kick_cameras_argv(kick_run["threed"])[1] == run  # the 3D folder resolves to the run
    with pytest.raises(ValueError, match="one camera"):
        fk.kick_cameras_argv(run, "other")
    parser = fk.build_parser()
    args = parser.parse_args(fk.kick_cameras_argv(run))
    assert args.command == "kick-cameras" and args.kick_dir == run and args.camera == "reference"
    assert args.workspace is None  # not needed
    assert (
        parser.parse_args(["kick-cameras", run, "--camera", "other", "-w", "ws"]).camera == "other"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["kick-cameras", run, "--camera", "third"])


def test_kick_cameras_writes_and_prints_commands_the_real_parser_accepts(
    kick_run: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = kick_run["run"]
    seen: list = []
    mirrored: list[list[str]] = []

    def fake_write(kick_dir, camera="reference"):
        seen.append((Path(kick_dir), camera))
        return {"stem": kc_tests.STEM, "dir": "/x/sam", "n_frames": 5,
                "files": {"dlt3d": "/x/sam/a.dlt3d", "dlt2d": "/x/sam/a.dlt2d"},
                "counts": {"dlt3d": 3, "approximate": 2}}  # fmt: skip

    monkeypatch.setattr(kp, "write_frame_cameras", fake_write)
    monkeypatch.setattr(fk, "print_gui_cli_mirror", lambda label, cmd, **k: mirrored.append(cmd))
    code = fk.main(["kick-cameras", str(run), "-w", "ws"])
    assert code == 0 and seen == [(run.resolve(), "reference")]
    out = capsys.readouterr().out
    assert "5 frames (2 approximate, 3 dlt3d)" in out and "copy of the nearest camera" in out
    assert len(mirrored) == 3  # the command, then measure, then the run again
    parser = fk.build_parser()
    freekiki = [c[4:] for c in mirrored if c[3] == "vaila/freekiki.py"]
    assert len(freekiki) == 2
    for argv in freekiki:  # every printed freekiki command parses with the real parser
        parser.parse_args(argv)
    assert freekiki[1][:3] == ["kick-analysis", "-w", "ws"]  # kick-analysis needs the workspace
    assert freekiki[1][3:] == ["--resume", str(run.resolve())]
    measure = next(c for c in mirrored if c[3] == "vaila/getpixelvideo.py")
    assert measure[4:] == fk.kick_measure_argv(run)


def test_kick_cameras_reports_an_error_with_exit_code_one(
    kick_run: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(kick_dir, camera="reference"):
        raise ValueError("no SAM 3 output folder for x")

    monkeypatch.setattr(kp, "write_frame_cameras", broken)
    monkeypatch.setattr(fk, "print_gui_cli_mirror", lambda *a, **k: None)
    assert fk.main(["kick-cameras", str(kick_run["run"])]) == 1
    assert "kick-cameras: no SAM 3 output folder" in capsys.readouterr().out


def test_the_open_run_window_has_the_calibration_button() -> None:
    source = Path(fk.__file__).read_text(encoding="utf-8")
    assert 'text="Calibration per frame"' in source
    assert 'run_cli(["kick-cameras", folder, "--camera", cam_var.get()])' in source


def test_the_stage_inputs_name_the_files_that_change_the_cameras(tmp_path: Path) -> None:
    predict = tmp_path / "predict"
    predict.mkdir()
    (predict / "quality.json").write_text("{}", encoding="utf-8")
    (predict / "field_kps_raw.csv").write_text("frame\n", encoding="utf-8")
    (predict / "x.dlt3d").write_text("frame\n", encoding="utf-8")
    inputs = kp.frame_cameras_inputs(predict, {"path": "v.mp4"})
    assert {"video", "model", "quality", "field_kps_raw", "dlt3d"} <= set(inputs)
    (predict / "field_kps_raw.csv").write_text("frame\n1\n", encoding="utf-8")
    assert kp.frame_cameras_inputs(predict, {"path": "v.mp4"}) != inputs
    assert d3 is not None  # the stubs patch freekiki_detect3d; keep the import explicit
