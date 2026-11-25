"""The roster in the pipeline: the 2D stage apart from the 3D one, and the user's labels used.

Update Date: 24 November 2026
Version: 0.4.19

No GPU, no model: the heavy stages are the stubs of ``test_freekiki_kick_pipeline`` (``stubs``).
What is checked here:

* ``freekiki_detect3d``: ``sam_2d_run`` never starts sam3dinov3 for a finished run, ``detect_3d_run``
  with ``skip_sam`` reads that output and ``drop_ids`` (roster OUT) leave the kept people;
* ``freekiki_player_ids``: the keeper and the kicker named by the roster take ids 0 and 1;
* ``freekiki_team_analysis``: the teams come from the labels, not from the shirt colour;
* ``freekiki_kick_pipeline``: ``stop_after="sam"``, the roster in the stage inputs, an old run
  without a ``sam:`` record, the other camera;
* ``freekiki_teams.label_text`` and ``kick_measure.load_people_info`` with a roster.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests import test_freekiki_kick_pipeline as kp_tests
from tests import test_freekiki_player_ids as ids_tests
from tests import test_freekiki_team_analysis as ta_tests
from tests import test_kick_measure as km_tests
from vaila import freekiki_detect3d as d3
from vaila import freekiki_kick_pipeline as kp
from vaila import freekiki_player_ids as pids
from vaila import freekiki_roster as ro
from vaila import freekiki_team_analysis as ta
from vaila import freekiki_teams as teams

stubs = kp_tests.stubs  # the heavy stages replaced (a fixture of the pipeline tests)
kick_folder = km_tests.kick_folder  # the hand-made kick folder and its opened run (kick_measure)
scene_run = km_tests.scene_run
CAM1, CAM2 = kp_tests.CAM1, kp_tests.CAM2


def _roster(*rows: tuple[int, str, str, str], path: Path | None = None) -> ro.Roster:
    entries = {s: ro.RosterEntry(s, t, r, n) for s, t, r, n in rows}
    ro.validate_entries(entries.values())
    return ro.Roster(path=path, entries=entries)


def _write_roster(path: Path, *rows: tuple[int, str, str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(ro.ROSTER_FIELDS)
        for sam, team, role, name in rows:
            writer.writerow([sam, team, role, name, "", "", "", ""])
    return path


# --- freekiki_detect3d --------------------------------------------------------------------------


def _sam_run(work: Path, stem: str, *, n_frames: int = 5, status: str = "complete") -> Path:
    """A sam3dinov3 output of ``stem`` as the real one lays it out."""
    folder = work / "sam3dinov3" / "processed_sam3dinov3_20261101_000000" / stem
    folder.mkdir(parents=True)
    (folder / f"{stem}_sam3dinov3_camera.csv").write_text(
        "frame,person_id,bbox_x1_px,bbox_y1_px,bbox_x2_px,bbox_y2_px\n"
        "0,1,0,0,10,10\n0,2,0,0,10,10\n0,3,0,0,10,10\n",
        encoding="utf-8",
    )
    (folder / f"{stem}_sam3dinov3_predictions.json.gz").write_bytes(b"x")
    (folder / "sam3dinov3_summary.json").write_text(
        json.dumps({"status": status, "n_frames": n_frames}), encoding="utf-8"
    )
    return folder


@pytest.fixture
def no_subprocess(monkeypatch: pytest.MonkeyPatch) -> list:
    """Any sam3dinov3 subprocess is recorded and refused."""
    started: list = []

    def run(cmd, **_):
        started.append(cmd)
        raise AssertionError(f"sam3dinov3 must not start: {cmd}")

    monkeypatch.setattr(d3.subprocess, "run", run)
    monkeypatch.setattr(d3, "_video_info", lambda _v: (5, 64, 48))
    monkeypatch.setattr(d3, "_video_fps", lambda _v: 30.0)
    return started


def test_sam_2d_run_does_not_start_sam_for_a_finished_run(
    tmp_path: Path, no_subprocess: list
) -> None:
    work = tmp_path / "work"
    folder = _sam_run(work, "clip")
    run_dir = tmp_path / "predict"
    run_dir.mkdir()
    result = d3.sam_2d_run(run_dir, tmp_path / "clip.mp4", work)
    assert no_subprocess == []
    assert result["reused"] is True
    assert Path(result["camera_csv"]).parent == folder
    assert result["predictions"].endswith("_sam3dinov3_predictions.json.gz")
    assert (work / "cam_int.npy").is_file()  # the focal array is still written


def test_sam_2d_run_starts_sam_when_the_run_is_new_incomplete_other_length_or_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(d3, "_video_info", lambda _v: (5, 64, 48))
    started: list = []

    def run(cmd, **_):
        started.append(cmd)
        sam_out = Path(cmd[cmd.index("-o") + 1])  # the subprocess leaves its camera csv here
        batches = sorted(sam_out.glob("processed_sam3dinov3_*"))  # sam3dinov3 resumes a match
        folder = (batches[0] if batches else sam_out / "processed_sam3dinov3_new") / "clip"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "clip_sam3dinov3_camera.csv").write_text("frame,person_id\n", encoding="utf-8")

        class Done:
            returncode = 0
            stdout = stderr = ""

        return Done()

    monkeypatch.setattr(d3.subprocess, "run", run)
    run_dir = tmp_path / "predict"
    run_dir.mkdir()
    video = tmp_path / "clip.mp4"
    out = d3.sam_2d_run(run_dir, video, tmp_path / "new")  # nothing there yet: the subprocess runs
    assert len(started) == 1 and out["reused"] is False
    assert "--cam-int-npy" in started[0] and "--fresh" not in started[0]

    running = tmp_path / "running"
    _sam_run(running, "clip", status="running")  # unfinished: left to sam3dinov3 (it resumes)
    d3.sam_2d_run(run_dir, video, running)
    assert len(started) == 2

    other = tmp_path / "other"
    _sam_run(other, "clip", n_frames=99)  # another video length: not the same run
    d3.sam_2d_run(run_dir, video, other)
    assert len(started) == 3

    done = tmp_path / "done"
    _sam_run(done, "clip")
    d3.sam_2d_run(run_dir, video, done, fresh_sam=True)  # a new detection: always a new run
    assert len(started) == 4 and "--fresh" in started[3]


def _gate_stubs(monkeypatch: pytest.MonkeyPatch, seen: dict) -> None:
    monkeypatch.setattr(d3, "_video_info", lambda _v: (5, 64, 48))
    monkeypatch.setattr(d3, "_video_fps", lambda _v: 30.0)
    monkeypatch.setattr(d3, "vitruvian_refit", lambda *a, **k: {})
    monkeypatch.setattr(d3, "_read_homographies", lambda _p: {})
    monkeypatch.setattr(
        d3,
        "pitch_verdicts",
        lambda tracks, homographies, min_frac=0.5: {
            pid: {"fraction": 1.0, "on": 3, "checked": 3, "keep": pid != 3} for pid in tracks
        },
    )

    def outputs(*args, **kwargs):
        seen["kept"] = kwargs["kept"]

    monkeypatch.setattr(d3, "write_ball_dependent_outputs", outputs)


def test_detect_3d_run_skips_sam_and_drops_the_roster_out_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_subprocess: list
) -> None:
    seen: dict = {}
    _gate_stubs(monkeypatch, seen)
    run_dir = tmp_path / "predict"
    run_dir.mkdir()
    (run_dir / "x.dlt2d").write_text("frame,L1\n", encoding="utf-8")
    work = tmp_path / "work"
    _sam_run(work, "clip")
    out = tmp_path / "out"
    summary = d3.detect_3d_run(
        run_dir,
        tmp_path / "clip.mp4",
        out,
        work_dir=work,
        players_on_field=True,
        skip_sam=True,
        drop_ids={2},
    )
    assert no_subprocess == []
    assert seen["kept"] == {1}  # 2 dropped by the roster, 3 is off the pitch
    assert summary["players"] == {"tracks": 3, "kept": 1, "dropped_by_roster": [2]}
    with (out / "players_on_field.csv").open(newline="", encoding="utf-8") as fh:
        rows = {int(r["person_id"]): r for r in csv.DictReader(fh)}
    assert (rows[1]["keep"], rows[2]["keep"], rows[3]["keep"]) == ("True", "False", "False")
    assert rows[2]["reason"] == "roster: OUT" and rows[1]["reason"] == ""
    assert ta.on_pitch(out) == {1}  # the readers keep working on the new column


def test_detect_3d_run_with_skip_sam_needs_the_2d_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_subprocess: list
) -> None:
    run_dir = tmp_path / "predict"
    run_dir.mkdir()
    with pytest.raises(RuntimeError, match="skip_sam: no sam3dinov3 output"):
        d3.detect_3d_run(
            run_dir, tmp_path / "clip.mp4", tmp_path / "out", work_dir=tmp_path / "w", skip_sam=True
        )
    assert no_subprocess == []


# --- freekiki_player_ids ------------------------------------------------------------------------


def test_the_roster_keeper_and_kicker_take_the_ids_0_and_1() -> None:
    tracks, feet = ids_tests._scene()
    # the heuristics say keeper 10 / kicker 11; the user says keeper 12 (A) and kicker 13 (B)
    groups = pids.link_tracks(tracks, {"c1": 0, "c2": 2})
    _, metres = ids_tests._ball()
    ball_xy = {f: (r["x_m"], r["y_m"]) for f, r in metres.items()}

    def assign(forced):
        ids, roles, info = pids.assign_player_ids(
            groups, offsets={"c1": 0, "c2": 2}, feet=feet, rest=ids_tests.BALL_REST,
            contact=ids_tests.CONTACT, side=1.0, ball_xy=ball_xy, fps=ids_tests.FPS,
            last_frame=ids_tests.N - 1, forced=forced,
        )  # fmt: skip

        def of(cam, sam):
            gi = next(
                i
                for i, g in enumerate(groups)
                for t in g.tracks
                if (t.camera, t.sam_id) == (cam, sam)
            )
            return ids[gi], roles[gi]

        return of, info

    of, info = assign(None)
    assert of("c1", 10) == (0, "keeper") and of("c1", 11) == (1, "kicker")
    assert (info["kicker_source"], info["keeper_source"]) == ("auto", "auto")

    of, info = assign({"keeper": 12, "kicker": 13})
    assert of("c1", 12) == (0, "keeper") and of("c1", 13) == (1, "kicker")
    assert of("c2", 22) == (0, "keeper")  # the same person in the other camera follows the group
    assert (info["kicker_source"], info["keeper_source"]) == ("roster", "roster")
    ranks = {of("c1", s)[0] for s in (10, 11, 14, 15)}
    assert ranks == {2, 3, 4, 5}  # the ones the heuristics chose are ordinary players now

    of, info = assign({"kicker": 13})  # only the kicker is named: the keeper is still found
    assert of("c1", 13) == (1, "kicker") and of("c1", 10) == (0, "keeper")
    assert (info["kicker_source"], info["keeper_source"]) == ("roster", "auto")


def test_a_roster_id_without_a_track_falls_back_to_the_heuristic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tracks, feet = ids_tests._scene()
    groups = pids.link_tracks(tracks, {"c1": 0, "c2": 2})
    _, metres = ids_tests._ball()
    ids, roles, info = pids.assign_player_ids(
        groups, offsets={"c1": 0, "c2": 2}, feet=feet, rest=ids_tests.BALL_REST,
        contact=ids_tests.CONTACT, side=1.0, ball_xy={f: (r["x_m"], r["y_m"]) for f, r in metres.items()},
        fps=ids_tests.FPS, last_frame=ids_tests.N - 1, forced={"kicker": 99, "keeper": 10},
    )  # fmt: skip
    assert "no track of the reference camera" in capsys.readouterr().out
    assert (info["kicker_source"], info["keeper_source"]) == ("auto", "roster")
    assert sorted(roles.values()).count("kicker") == 1  # the heuristic's kicker is still there


# --- freekiki_team_analysis ---------------------------------------------------------------------


def _scene_with_a_fourth_person(tmp_path: Path, *, keep_off_pitch: bool = True) -> dict:
    """The synthetic camera of the team tests plus SAM 4 as a player on the pitch (player id 3)."""
    folder, work, video = tmp_path / "cam_3d", tmp_path / "work", tmp_path / "clip.avi"
    ta_tests._camera(folder, work, video)
    if keep_off_pitch:
        text = (folder / "players_on_field.csv").read_text(encoding="utf-8")
        (folder / "players_on_field.csv").write_text(
            text.replace("4,0.0,0,3,False", "4,1.0,3,3,True"), encoding="utf-8"
        )
        with (folder / "player_map.csv").open("a", encoding="utf-8") as fh:
            fh.write("3,other,cam,4,0,2,3,,0.1,3\n")
        with (folder / "poses_3d_on_field_players.csv").open("a", encoding="utf-8") as fh:
            for f in range(ta_tests.FRAMES):
                fh.write(f"{f},4,0,nose,{10.0 + 3},0.0,1.7,3\n")
    return {"folder": folder, "source": ta.Source("ref", folder, video, work, dlt=None)}


def _analyse(scene: dict, roster, kind: str = "mono-ref") -> dict:
    folder = scene["folder"]
    return ta.run_team_analysis(
        [scene["source"]], folder, poses_csv=folder / "poses_3d_on_field_players.csv",
        ball_csv=folder / "ball.csv", kind=kind, rate_hz=30.0, roster=roster,
    )  # fmt: skip


def _team_map(folder: Path) -> dict[str, dict]:
    return {r["player_id"]: r for r in ta_tests._rows(folder / "data" / "team_map.csv")}


def test_the_teams_come_from_the_roster_not_from_the_shirt_colour(tmp_path: Path) -> None:
    scene = _scene_with_a_fourth_person(tmp_path)
    # the shirts say: keeper yellow, kicker green, 3 and 4 blue. The user says the opposite split.
    roster = _roster(
        (1, "A", "keeper", "Rossi"),
        (2, "B", "kicker", "Vargas"),
        (3, "A", "", ""),
        (4, "REF", "", ""),
    )
    summary = _analyse(scene, roster)
    folder = scene["folder"]
    team_map = _team_map(folder)
    assert (team_map["0"]["team"], team_map["0"]["role"]) == ("TEAM_A", "keeper")
    assert (team_map["1"]["team"], team_map["1"]["role"]) == ("TEAM_B", "kicker")
    assert team_map["2"]["team"] == "TEAM_A"
    assert (team_map["3"]["team"], team_map["3"]["role"]) == ("REF", "referee_main")
    assert team_map["1"]["name"] == "Vargas" and team_map["2"]["name"] == ""
    assert (
        team_map["3"]["colour"] == "blue" and team_map["1"]["colour"] == "green"
    )  # still measured
    assert summary["counts"] == {"TEAM_A": 2, "TEAM_B": 1, "REF": 1}
    assert summary["teams_source"] == "roster"
    analysis = json.loads((folder / "data" / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["teams_source"] == "roster"
    colours = json.loads((folder / "data" / "team_colours.json").read_text(encoding="utf-8"))
    assert colours["source"] == "roster" and colours["TEAM_B"] == "green"
    labels = json.loads((folder / "data" / "labels_ref.json").read_text(encoding="utf-8"))
    assert labels["2"] == "ID 1 KICKER TEAM_B Vargas"
    assert (folder / "team_A" / "P00_keeper.c3d").is_file()
    assert (folder / "team_B" / "P01_kicker.c3d").is_file()
    assert (folder / "referees" / "P03_referee_main.c3d").is_file()
    assert (folder / "referees" / "overlay_ref.mp4").is_file()


def test_a_rerun_with_other_teams_moves_the_earlier_files_aside(tmp_path: Path) -> None:
    scene = _scene_with_a_fourth_person(tmp_path)
    folder = scene["folder"]
    _analyse(scene, None)  # the colour split: SAM 2 (kicker) in TEAM_A, the blue people in TEAM_B
    assert (folder / "team_A" / "P01_kicker.c3d").is_file()
    assert (folder / "team_B" / "P03_other.c3d").is_file()
    roster = _roster((2, "B", "kicker", ""), (4, "REF", "referee_main", ""), (3, "A", "", ""))
    _analyse(scene, roster)
    assert not (folder / "team_A" / "P01_kicker.c3d").exists()  # it is in team_B now
    assert (folder / "team_B" / "P01_kicker.c3d").is_file()
    assert not (folder / "team_B" / "P03_other.c3d").exists()  # the referee is not a player
    assert (folder / "referees" / "P03_referee_main.c3d").is_file()
    moved = {p.name for p in (folder / "superseded").glob("*/*/*")}
    assert {"P01_kicker.c3d", "P03_other.c3d"} <= moved  # nothing was deleted


def test_without_a_roster_the_colour_split_is_unchanged(tmp_path: Path) -> None:
    scene = _scene_with_a_fourth_person(tmp_path, keep_off_pitch=False)
    summary = _analyse(scene, None)
    assert summary["teams_source"] == "colour" and "source" not in summary["team_colours"]
    unfilled = _roster((1, "", "", ""), (2, "", "", ""))  # nobody has a team: not filled
    assert not unfilled.filled
    assert _analyse(scene, unfilled)["teams_source"] == "colour"
    team_map = _team_map(scene["folder"])
    assert team_map["1"]["team"] == "TEAM_A" and team_map["1"]["name"] == ""


def test_people_the_user_left_blank_or_marked_out_have_no_team(tmp_path: Path) -> None:
    scene = _scene_with_a_fourth_person(tmp_path)
    roster = _roster((2, "A", "kicker", ""), (3, "OUT", "", "bench"))  # SAM 1 and 4 left blank
    _analyse(scene, roster)
    team_map = _team_map(scene["folder"])
    for pid in ("0", "3"):  # SAM 1 and SAM 4: left blank
        assert (
            team_map[pid]["team"] == "" and "not labelled in the roster" in team_map[pid]["reason"]
        )
    assert team_map["2"]["team"] == "" and "roster: OUT (not used)" in team_map["2"]["reason"]
    assert team_map["1"]["team"] == "TEAM_A"  # the kicker, SAM 2, is pid 1


def test_a_person_labelled_but_not_on_the_pitch_is_listed_with_the_reason(tmp_path: Path) -> None:
    scene = _scene_with_a_fourth_person(tmp_path, keep_off_pitch=False)  # SAM 4 is off the pitch
    roster = _roster((2, "A", "kicker", ""), (4, "B", "", ""), (3, "OUT", "", ""))
    _analyse(scene, roster)
    off = {
        r["sam_id"]: r["reason"]
        for r in ta_tests._rows(scene["folder"] / "data" / "not_on_pitch.csv")
    }
    assert off["4"] == "labelled B in the roster, but not on the pitch"


def test_referees_are_ranked_by_the_roster_then_by_the_distance_to_the_ball() -> None:
    pids_ = ["5", "6", "7"]
    roles = dict.fromkeys(pids_, "other")
    sam_of = {"5": 15, "6": 16, "7": 17}
    near = {"5": 30.0, "6": 10.0, "7": 20.0}  # pixel distance to the ball

    def rows_for(*people):
        roster = _roster(*people)
        rows, _ = ta._roster_rows(pids_, roles, {}, sam_of, roster, lambda pid: near[pid])
        return {pid: (r["team"], r["role"]) for pid, r in rows.items()}

    # no one is the main referee: the nearest of the people with no role is, then the aux by distance
    got = rows_for((15, "REF", "", ""), (16, "REF", "", ""), (17, "REF", "", ""))
    assert got == {"6": ("REF", "referee_main"), "7": ("REF", "referee_aux1"),
                   "5": ("REF", "referee_aux2")}  # fmt: skip
    # the user names the main one: the distance only orders the aux
    got = rows_for(
        (15, "REF", "referee_main", ""), (16, "REF", "", ""), (17, "REF", "referee_aux", "")
    )
    assert got["5"] == ("REF", "referee_main")
    assert got["6"] == ("REF", "referee_aux1") and got["7"] == ("REF", "referee_aux2")
    # only aux labelled: nobody is promoted to main
    got = rows_for((16, "REF", "referee_aux", ""))
    assert got["6"] == ("REF", "referee_aux1")


# --- freekiki_teams.label_text ------------------------------------------------------------------


def test_label_text_appends_the_roster_name_last() -> None:
    assert teams.label_text("0", "keeper", "TEAM_B", None, name="Rossi") == "ID 0 GK TEAM_B Rossi"
    assert (
        teams.label_text("1", "kicker", "TEAM_A", 1.54, "Vargas")
        == "ID 1 KICKER TEAM_A 1.54 m Vargas"
    )
    assert teams.label_text("8", "other", "", None) == "ID 8"  # the old calls are unchanged


# --- freekiki_kick_pipeline ---------------------------------------------------------------------


def test_stop_after_sam_writes_the_sheets_and_stops_before_the_3d_stage(
    tmp_path: Path, stubs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    videos = kp_tests._cameras(tmp_path)
    out = tmp_path / "kick"
    written: list = []

    def sheet(predictions, video, out_dir, **_):
        written.append((Path(predictions).name, Path(video).stem, Path(out_dir)))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        csv_path = Path(out_dir) / "roster.csv"
        csv_path.write_text(",".join(ro.ROSTER_FIELDS) + "\n", encoding="utf-8")
        return {"sheet": Path(out_dir) / "roster_ids.png", "csv": csv_path, "context": []}

    monkeypatch.setattr(ro, "write_roster_sheet", sheet)
    result = kp.run_kick_analysis("ws", videos, reference=1, out_dir=out, stop_after="sam")
    assert result["stopped_after"] == "sam" and set(result["roster"]) == {CAM1, CAM2}
    assert Path(result["roster"][CAM1]["csv"]) == out / "roster" / CAM1 / "roster.csv"
    assert [w[2] for w in written] == [out / "roster" / CAM1, out / "roster" / CAM2]
    status = json.loads(Path(result["status_file"]).read_text(encoding="utf-8"))
    assert list(status["stages"]) == [
        f"detect:{CAM1}",
        f"sam:{CAM1}",
        f"cameras:{CAM1}",
        f"detect:{CAM2}",
        f"sam:{CAM2}",
        f"cameras:{CAM2}",
    ]
    assert status["stopped_after"] == "sam"
    assert stubs.get("3d") is None and stubs["sam"] == 2

    # the second call (the labels done) continues: the 2D stages are reused, the 3D one runs
    roster_csv = _write_roster(
        out / "roster" / CAM1 / "roster.csv", (4, "A", "keeper", ""), (11, "B", "kicker", "")
    )
    before = dict(stubs)
    kp.run_kick_analysis("ws", videos, reference=1, out_dir=out, roster=roster_csv)
    assert stubs["detect"] == before["detect"] and stubs["sam"] == before["sam"]
    assert stubs["3d"] == 2
    assert all(kw["skip_sam"] is True for kw in stubs["3d_kwargs"])  # SAM is never started twice
    status = json.loads((out / kp.STATUS_NAME).read_text(encoding="utf-8"))
    assert "stopped_after" not in status


def test_stop_after_must_be_sam(tmp_path: Path, stubs: dict) -> None:
    videos = kp_tests._cameras(tmp_path, both=False)
    with pytest.raises(kp.KickInputError, match="--stop-after must be one of sam"):
        kp.run_kick_analysis("ws", videos, out_dir=tmp_path / "kick", stop_after="3d")
    assert not stubs


class _Recorder:
    """The stages that read the roster, recorded (the pipeline stubs do not take it)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, stubs: dict) -> None:
        self.roles: list = []
        self.teams: list = []
        original = pids.run_player_ids

        def run_player_ids(folders, roles=None):
            self.roles.append(roles)
            return original(folders)

        def teams_mono(cam3d, video, work, *, is_ref, key, team_a, fps, sync=None, roster=None):
            self.teams.append((key, roster))
            data = Path(cam3d) / "data"
            data.mkdir(parents=True, exist_ok=True)
            (data / "team_map.csv").write_text("player_id\n", encoding="utf-8")
            return {"counts": {"TEAM_A": 0, "TEAM_B": 0, "REF": 0}}

        monkeypatch.setattr(pids, "run_player_ids", run_player_ids)
        monkeypatch.setattr(kp, "_teams_mono", teams_mono)


def test_the_roster_reaches_the_3d_stage_the_players_and_the_teams(
    tmp_path: Path, stubs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    videos = kp_tests._cameras(tmp_path)
    out = tmp_path / "kick"
    seen = _Recorder(monkeypatch, stubs)
    csv_path = _write_roster(
        out / "roster" / CAM1 / "roster.csv",
        (4, "A", "keeper", "Rossi"), (11, "B", "kicker", "Vargas"), (6, "A", "", ""), (9, "OUT", "", ""),
    )  # fmt: skip
    kp.run_kick_analysis("ws", videos, reference=1, out_dir=out)  # found in roster/<ref stem>/
    first, second = stubs["3d_kwargs"][:2]
    assert (
        first["drop_ids"] == {9} and second["drop_ids"] == set()
    )  # only the reference is labelled
    assert seen.roles == [
        {"kicker": 11, "keeper": 4}
    ]  # the keeper of the other team than the kicker
    assert {k for k, _ in seen.teams} == {"mono-ref", "mono-other"}
    by_key = dict(seen.teams)
    assert by_key["mono-ref"] is not None and by_key["mono-ref"].kicker() == 11
    assert by_key["mono-other"] is None  # the other camera follows the pairing
    options = kp.resume_options(out)
    assert options["roster"] == str(csv_path.resolve())
    readme = (out / "README.txt").read_text(encoding="utf-8")
    assert "roster:" in readme and "only the reference camera is labelled" in readme

    # editing a label reruns the players and the analyses; the detection, SAM and the 3D (out ids
    # unchanged) are not touched
    before = dict(stubs)
    _write_roster(csv_path, (4, "A", "keeper", "Rossi"), (11, "B", "kicker", "V. Vargas"),
                  (6, "A", "", ""), (9, "OUT", "", ""))  # fmt: skip
    kp.run_kick_analysis("ws", videos, reference=1, out_dir=out)
    assert stubs["detect"] == before["detect"] and stubs["sam"] == before["sam"]
    assert stubs["3d"] == before["3d"]
    assert len(seen.roles) == 2 and len(seen.teams) == 4  # players once more, both teams again
    assert stubs["kicker"] > before["kicker"]

    # a new OUT person reruns the 3D stage of the reference camera only
    before = dict(stubs)
    _write_roster(csv_path, (4, "A", "keeper", ""), (11, "B", "kicker", ""), (9, "OUT", "", ""),
                  (7, "OUT", "", ""))  # fmt: skip
    kp.run_kick_analysis("ws", videos, reference=1, out_dir=out)
    assert stubs["3d"] == before["3d"] + 1
    assert stubs["3d_kwargs"][-1]["drop_ids"] == {7, 9}
    assert stubs["sam"] == before["sam"]


def test_the_roster_decides_the_keeper_of_the_team_opposite_to_the_kicker() -> None:
    both = _roster((4, "A", "keeper", ""), (5, "B", "keeper", ""), (11, "B", "kicker", ""))
    assert kp._roster_roles(both) == {"kicker": 11, "keeper": 4}
    own_team = _roster((5, "B", "keeper", ""), (11, "B", "kicker", ""))  # the kicker's own keeper
    assert kp._roster_roles(own_team) == {"kicker": 11}
    assert kp._roster_roles(_roster((4, "A", "keeper", ""), (6, "A", "", ""))) == {"keeper": 4}
    assert kp._roster_roles(_roster((4, "A", "keeper", ""), (5, "B", "keeper", ""))) == {}
    assert kp._roster_roles(_roster((6, "A", "", ""))) == {}


def test_an_unfilled_or_broken_roster(
    tmp_path: Path, stubs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    videos = kp_tests._cameras(tmp_path, both=False)
    out = tmp_path / "kick"
    seen = _Recorder(monkeypatch, stubs)
    _write_roster(out / "roster" / CAM1 / "roster.csv", (4, "", "", ""), (11, "", "", ""))
    kp.run_kick_analysis("ws", videos, out_dir=out)
    assert seen.roles == [None] and seen.teams[0][1] is None  # ignored: the colour split stays
    with pytest.raises(kp.KickInputError, match="does not exist"):
        kp.run_kick_analysis("ws", videos, out_dir=out, roster=tmp_path / "nope.csv")
    bad = _write_roster(tmp_path / "bad.csv", (4, "A", "kicker", ""), (5, "B", "kicker", ""))
    with pytest.raises(kp.KickInputError, match="more than one kicker"):
        kp.run_kick_analysis("ws", videos, out_dir=out, roster=bad)


def test_an_older_run_without_a_sam_record_does_not_redo_anything(
    tmp_path: Path, stubs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    videos = kp_tests._cameras(tmp_path, both=False)
    out = tmp_path / "kick"
    kp.run_kick_analysis("ws", videos, out_dir=out)
    status_path = out / kp.STATUS_NAME
    status = json.loads(status_path.read_text(encoding="utf-8"))
    del status["stages"][f"sam:{CAM1}"]  # a run made before the 2D stage had its own record
    status["stages"][f"3d:{CAM1}"]["inputs"]["sam_device"] = None  # and its 3D inputs held this
    status_path.write_text(json.dumps(status), encoding="utf-8")

    def finished(run_dir, video, work_dir, **_):  # the sam3dinov3 run is on disk and complete
        stubs["sam_finished"] = stubs.get("sam_finished", 0) + 1
        folder = Path(work_dir) / "sam3dinov3" / "processed_sam3dinov3_x" / Path(video).stem
        camera = folder / "c_sam3dinov3_camera.csv"
        (Path(work_dir) / "cam_int.npy").write_bytes(b"npy")
        return {"focal": {}, "camera_csv": str(camera), "predictions": "", "reused": True}

    monkeypatch.setattr(d3, "sam_2d_run", finished)
    (out / "work" / CAM1 / "sam3dinov3" / "processed_sam3dinov3_x" / CAM1).mkdir(
        parents=True, exist_ok=True
    )
    (
        out
        / "work"
        / CAM1
        / "sam3dinov3"
        / "processed_sam3dinov3_x"
        / CAM1
        / "c_sam3dinov3_camera.csv"
    ).write_text("frame\n", encoding="utf-8")
    before = dict(stubs)
    kp.run_kick_analysis("ws", videos, out_dir=out)
    assert stubs["sam_finished"] == 1  # recorded
    assert stubs["3d"] == before["3d"] and stubs["players"] == before["players"]
    assert stubs["teams_mono"] == before["teams_mono"]
    assert (
        json.loads(status_path.read_text(encoding="utf-8"))["stages"][f"sam:{CAM1}"]["status"]
        == "ok"
    )


def test_the_roster_dir_is_the_one_of_the_design(tmp_path: Path) -> None:
    assert ro.roster_dir(tmp_path, CAM1) == tmp_path / "roster" / CAM1


# --- kick_measure -------------------------------------------------------------------------------


def test_load_people_info_prefers_the_roster(scene_run) -> None:
    from vaila import kick_measure as km

    base = km.load_people_info(scene_run)
    assert (base[3].role, base[3].team) == ("kicker", "A")
    _write_roster(
        scene_run.run_dir / "roster" / scene_run.stem / "roster.csv",
        (3, "B", "kicker", "Vargas"), (4, "A", "player", ""), (5, "REF", "referee_main", ""),
        (6, "OUT", "", ""),
    )  # fmt: skip
    info = km.load_people_info(scene_run)
    assert (info[3].role, info[3].team) == ("kicker", "B")  # the user's team, not team_map's
    assert (info[4].role, info[4].team) == ("other", "A")
    assert (info[5].role, info[5].team) == ("referee", "")  # was the keeper of player_map
    assert (info[6].role, info[6].team) == ("other", "")  # was the referee of the c3d file
    assert info[3].n_frames == base[3].n_frames  # the frames still come from the run

    # a roster with no team is not a decision: the files answer as before
    _write_roster(
        scene_run.run_dir / "roster" / scene_run.stem / "roster.csv",
        (3, "", "", ""),
        (4, "", "", ""),
    )
    assert km.load_people_info(scene_run) == base
    # a roster that breaks a rule is the pipeline's to report: here it is not an error
    (scene_run.run_dir / "roster" / scene_run.stem / "roster.csv").write_text(
        "sam_id,team,role\n3,A,kicker\n4,A,kicker\n", encoding="utf-8"
    )
    assert km.load_people_info(scene_run) == base
    # people the roster leaves blank keep the answer of player_map / team_map
    _write_roster(
        scene_run.run_dir / "roster" / scene_run.stem / "roster.csv", (3, "B", "kicker", "")
    )
    info = km.load_people_info(scene_run)
    assert (info[3].role, info[3].team) == ("kicker", "B")
    assert (info[4].role, info[4].team) == (base[4].role, base[4].team)
    assert (info[6].role, info[6].team) == (base[6].role, base[6].team)
