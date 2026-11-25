"""Tests of vaila/freekiki_roster.py (no GPU, tiny synthetic video and predictions)."""

import csv
import gzip
import json
from pathlib import Path

import pytest

from vaila import freekiki_roster as fr

N_FRAMES = 30


def _inst(sam_id, box):
    return {"sam_obj_id": sam_id, "bbox_xyxy": box, "keypoints_2d_px": []}


def _build_frames() -> list[dict]:
    """SAM 1: whole clip, the box grows with time; SAM 2: frames 0-14, crossed by SAM 3 at 5-9;
    SAM 3: frames 5-29 (starts overlapping SAM 2); SAM 4: only frame 20."""
    frames = []
    for f in range(N_FRAMES):
        insts = [_inst(1, [10, 20, 40 + f, 100])]
        if f < 15:
            insts.append(_inst(2, [100, 20, 130, 100]))
        if f >= 5:
            insts.append(_inst(3, [105, 25, 135, 105] if f < 10 else [200, 20, 230, 100]))
        if f == 20:
            insts.append(_inst(4, [60, 30, 80, 60]))
        frames.append({"frame": f, "instances": insts})
    return frames


@pytest.fixture
def fake_run(tmp_path: Path) -> tuple[Path, Path]:
    import cv2
    import numpy as np

    video = tmp_path / "cam.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (260, 140))
    assert writer.isOpened()
    for f in range(N_FRAMES):
        img = np.full((140, 260, 3), (40, 120, 40), np.uint8)
        cv2.rectangle(img, (f, 10), (f + 40, 120), (200, 200, 200), -1)
        writer.write(img)
    writer.release()
    pred = tmp_path / "pred.json.gz"
    with gzip.open(pred, "wt", encoding="utf-8") as fh:
        json.dump({"keypoint_names": [], "frames": _build_frames()}, fh)
    return pred, video


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _frames_dict() -> dict[int, list[dict]]:
    return {f["frame"]: f["instances"] for f in _build_frames()}


# ------------------------------------------------------------------ track_summary


def test_track_summary_three_picks_per_id():
    summary = fr.track_summary(_frames_dict())
    assert sorted(summary) == [1, 2, 3, 4]
    s1 = summary[1]
    assert (s1["first"], s1["last"], s1["frames"]) == (0, 29, 30)
    assert len(s1["picks"]) == 3
    # the box of SAM 1 grows, so the largest of each third is its last frame
    assert [f for f, _ in s1["picks"]] == [9, 19, 29]
    # short tracks give fewer picks, never more than 3
    assert len(summary[4]["picks"]) == 1
    assert all(len(s["picks"]) <= 3 for s in summary.values())


def test_track_summary_overlap_rule_prefers_clean_frames():
    summary = fr.track_summary(_frames_dict())
    # SAM 2 (frames 0-14): SAM 3 sits on it in frames 5-9, which is its whole middle third, so that
    # pick falls back to the covered frames; the first and last thirds are clean
    picked = [f for f, _ in summary[2]["picks"]]
    assert len(picked) == 3
    assert 0 <= picked[0] <= 4 and 5 <= picked[1] <= 9 and 10 <= picked[2] <= 14
    # SAM 3, first third (frames 5-12): frames 5-9 overlap SAM 2 and have the biggest box (30x80
    # both, so ties go to the first): the clean frames 10-12 must win
    first = summary[3]["picks"][0][0]
    assert first >= 10


def test_track_summary_falls_back_when_every_frame_overlaps():
    frames = {
        0: [_inst(1, [0, 0, 10, 10]), _inst(2, [0, 0, 10, 10])],
        1: [_inst(1, [0, 0, 12, 12]), _inst(2, [0, 0, 12, 12])],
    }
    summary = fr.track_summary(frames)
    assert [f for f, _ in summary[1]["picks"]] == [0, 1]  # n=2: two picks, from covered frames


# ------------------------------------------------------------------ the sheet


def test_write_roster_sheet_outputs(fake_run, tmp_path):
    pred, video = fake_run
    out = tmp_path / "roster" / "cam"
    paths = fr.write_roster_sheet(pred, video, out)
    assert paths["sheet"] == out / "roster_ids.png" and paths["sheet"].stat().st_size > 0
    assert paths["csv"] == out / "roster.csv"
    assert 1 <= len(paths["context"]) <= 5
    assert all(p.name.startswith("roster_context_") and p.exists() for p in paths["context"])
    for sam_id in (1, 2, 3, 4):
        assert (out / "roster_crops" / f"id_{sam_id:03d}.png").exists()
    import cv2

    sheet = cv2.imread(str(paths["sheet"]))
    tile_w, tile_h = 3 * fr.CELL_W + 4, fr.CELL_H + fr.LABEL_H + 4
    assert sheet.shape[:2] == (tile_h, 4 * tile_w)  # 4 ids, 4 columns: one row
    rows = _csv_rows(paths["csv"])
    assert list(rows[0]) == fr.ROSTER_FIELDS
    assert [r["sam_id"] for r in rows] == ["1", "2", "3", "4"]
    assert rows[0]["first_frame"] == "0" and rows[0]["last_frame"] == "29"
    assert rows[0]["frames"] == "30"
    assert all(r["team"] == "" and r["role"] == "" for r in rows)


def test_write_roster_sheet_keeps_labels_on_second_call(fake_run, tmp_path):
    pred, video = fake_run
    out = tmp_path / "sheet"
    fr.write_roster_sheet(pred, video, out)
    # the user labels two ids by hand (odd case, a note, an extra id the sheet does not know)
    path = out / "roster.csv"
    path.write_text(
        "sam_id,team,role,name,first_frame,last_frame,frames,note\n"
        "1,a,Kicker,Vargas,0,0,0,he is the one\n"
        "3,ref,,,,,,\n"
        "99,OUT,,,,,,gone\n",
        encoding="utf-8",
    )
    # a video with one more id appears in the predictions
    frames = _build_frames()
    frames[3]["instances"].append(_inst(7, [150, 20, 180, 100]))
    with gzip.open(pred, "wt", encoding="utf-8") as fh:
        json.dump({"keypoint_names": [], "frames": frames}, fh)
    fr.write_roster_sheet(pred, video, out)
    rows = {int(r["sam_id"]): r for r in _csv_rows(path)}
    assert rows[1]["team"] == "A" and rows[1]["role"] == "kicker"
    assert rows[1]["name"] == "Vargas" and rows[1]["note"] == "he is the one"
    assert rows[1]["frames"] == "30"  # refreshed
    assert rows[3]["team"] == "REF"
    assert rows[99]["team"] == "OUT" and rows[99]["note"] == "gone"  # unknown id is not dropped
    assert rows[7]["team"] == "" and rows[7]["frames"] == "1"  # new id, blank
    assert rows[2]["team"] == ""


def test_write_roster_sheet_custom_context_and_columns(fake_run, tmp_path):
    pred, video = fake_run
    paths = fr.write_roster_sheet(pred, video, tmp_path / "o", columns=2, context_frames=[0, 12])
    assert [p.name for p in paths["context"]] == [
        "roster_context_000.png",
        "roster_context_012.png",
    ]
    import cv2

    sheet = cv2.imread(str(paths["sheet"]))
    assert sheet.shape[1] == 2 * (3 * fr.CELL_W + 4)
    assert sheet.shape[0] == 2 * (fr.CELL_H + fr.LABEL_H + 4)


def test_default_context_frames_pick_the_busiest_frame_of_each_part():
    ctx = fr.default_context_frames(_frames_dict())
    assert len(ctx) == 5 and ctx == sorted(ctx)
    assert fr.default_context_frames({}) == []


# ------------------------------------------------------------------ read_roster rules


def _write(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "roster.csv"
    path.write_text("sam_id,team,role,name\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_read_roster_ok_case_insensitive(tmp_path):
    roster = fr.read_roster(
        _write(
            tmp_path,
            ["1, a ,Keeper,Rossi", "2,B,KICKER,Vargas", "3,ref,Referee_Main,", "4,out,,", "5,,,"],
        )
    )
    assert roster.entries[1] == fr.RosterEntry(1, "A", "keeper", "Rossi")
    assert roster.entries[2].team == "B" and roster.entries[2].role == "kicker"
    assert roster.entries[3].role == "referee_main"
    assert roster.team_of(4) == "OUT" and roster.team_of(5) == "" and roster.team_of(99) == ""


@pytest.mark.parametrize(
    ("lines", "needles"),
    [
        (["7,X,,"], ["SAM 7", "unknown team"]),
        (["8,A,goalie,"], ["SAM 8", "unknown role"]),
        (["9,A,referee_main,"], ["SAM 9", "not allowed", "team A"]),
        (["10,REF,keeper,"], ["SAM 10", "not allowed", "team REF"]),
        (["11,OUT,kicker,"], ["SAM 11", "not allowed", "OUT"]),
        (["12,,player,"], ["SAM 12", "not allowed", "no team"]),
        (["1,A,kicker,", "2,B,kicker,"], ["more than one kicker", "SAM 1", "SAM 2"]),
        (["1,A,keeper,", "2,A,keeper,"], ["more than one keeper", "team A", "SAM 1", "SAM 2"]),
        (["3,REF,referee_main,", "4,REF,referee_main,"], ["more than one referee_main", "SAM 3"]),
        ([f"{i},REF,," for i in range(1, 5)], ["4 people labelled REF", "SAM 4"]),
        ([f"{i},B,," for i in range(1, 13)], ["team B has 12", "SAM 12"]),
        (["5,A,,", "5,B,,"], ["duplicate sam_id 5"]),
        (["abc,A,,"], ["sam_id 'abc'", "not an integer"]),
    ],
)
def test_read_roster_rule_errors_name_the_sam_id(tmp_path, lines, needles):
    with pytest.raises(fr.RosterError) as exc:
        fr.read_roster(_write(tmp_path, lines))
    for needle in needles:
        assert needle in str(exc.value)


def test_read_roster_allows_one_keeper_per_team_and_eleven_per_team(tmp_path):
    lines = ["100,A,keeper,", "101,B,keeper,"]
    lines += [f"{i},A,," for i in range(1, 11)] + [f"{i},B,," for i in range(20, 30)]
    roster = fr.read_roster(_write(tmp_path, lines))  # A and B: 11 each, keeper included
    assert roster.keeper("A") == 100 and roster.keeper("b") == 101


def test_read_roster_without_sam_id_column(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("id,team\n1,A\n", encoding="utf-8")
    with pytest.raises(fr.RosterError, match="sam_id"):
        fr.read_roster(path)


# ------------------------------------------------------------------ Roster helpers


def _roster(tmp_path: Path) -> fr.Roster:
    return fr.read_roster(
        _write(
            tmp_path,
            [
                "22,A,kicker,Vargas",
                "30,B,keeper,Rossi",
                "31,A,keeper,",
                "40,REF,referee_main,",
                "41,REF,,",
                "50,OUT,,",
                "60,,,",
            ],
        )
    )


def test_roster_helpers(tmp_path):
    roster = _roster(tmp_path)
    assert roster.filled
    assert roster.kicker() == 22
    assert roster.keeper("B") == 30 and roster.keeper("A") == 31
    assert roster.keeper() == 30  # smallest id with the role, any team
    assert roster.keeper("REF") is None
    assert roster.out_ids() == {50}
    assert roster.labelled_ids() == {22, 30, 31, 40, 41}
    assert roster.team_of(60) == ""
    assert not fr.Roster(path=None, entries={}).filled
    assert not fr.read_roster(_write(tmp_path, ["1,,,", "2,,,"])).filled
    assert fr.Roster(path=None).kicker() is None


def test_roster_key_is_stable_sorted_and_ignores_unlabelled(tmp_path):
    roster = _roster(tmp_path)
    key = roster.key()
    assert key["path"] == str(roster.path)
    assert [row[0] for row in key["labels"]] == [22, 30, 31, 40, 41, 50]  # 60 not labelled
    assert key["labels"][0] == [22, "A", "kicker", "Vargas"]
    json.dumps(key)  # json-able
    # same content, rows in another order and another case -> same digest
    shuffled = tmp_path / "again" / "roster.csv"
    shuffled.parent.mkdir()
    shuffled.write_text(
        "sam_id,team,role,name\n"
        "60,,,\n50,out,,\n41,ref,,\n40,REF,REFEREE_MAIN,\n31,a,keeper,\n30,b,KEEPER,Rossi\n"
        "22,A,kicker,Vargas\n",
        encoding="utf-8",
    )
    other = fr.read_roster(shuffled).key()
    assert other["labels"] == key["labels"]
    # a label change changes the digest; a blank id does not
    shuffled.write_text(shuffled.read_text().replace("50,out", "50,a"), encoding="utf-8")
    assert fr.read_roster(shuffled).key()["labels"] != key["labels"]


def test_write_roster_roundtrip_normalises_and_keeps_extra_columns(tmp_path):
    path = tmp_path / "roster.csv"
    path.write_text(
        "sam_id,team,role,name,first_frame,last_frame,frames,note\n5,,,,3,9,7,keep me\n",
        encoding="utf-8",
    )
    roster = fr.Roster(
        path=path,
        entries={
            5: fr.RosterEntry(5, "b", "KEEPER", "Rossi"),
            6: fr.RosterEntry(6, "ref", "", ""),
        },
    )
    fr.write_roster(path, roster, summary={6: {"first": 1, "last": 2, "frames": 2, "picks": []}})
    rows = {int(r["sam_id"]): r for r in _csv_rows(path)}
    assert rows[5]["team"] == "B" and rows[5]["role"] == "keeper"
    assert rows[5]["note"] == "keep me" and rows[5]["frames"] == "7"
    assert rows[6]["team"] == "REF" and rows[6]["frames"] == "2"
    assert fr.read_roster(path).keeper("B") == 5


# ------------------------------------------------------------------ paths, helpers, CLI


def test_roster_paths(tmp_path):
    d = fr.roster_dir(tmp_path, "cam1")
    assert d == tmp_path / "roster" / "cam1"
    assert fr.find_roster(tmp_path, "cam1") is None
    d.mkdir(parents=True)
    (d / "roster.csv").write_text("sam_id,team\n", encoding="utf-8")
    assert fr.find_roster(tmp_path, "cam1") == d / "roster.csv"


def test_dialog_helpers():
    assert fr.roles_for_team("a") == ("", "player", "keeper", "kicker")
    assert fr.roles_for_team("REF") == ("", "referee_main", "referee_aux")
    assert fr.roles_for_team("OUT") == ("",) and fr.roles_for_team("zzz") == ("",)
    assert fr.normalize_team(" ref ") == "REF" and fr.normalize_role(" Keeper ") == "keeper"


def test_team_colours_distinguish_every_team():
    codes = ["A", "B", "REF", "OUT", ""]
    colours = [fr.team_colour(c) for c in codes]
    assert len(set(colours)) == 5
    assert fr.team_colour("a") == fr.team_colour("A")  # case-insensitive
    assert fr.team_colour("zzz") not in colours  # unknown text is flagged in red


def test_dialog_without_ids_is_an_error(tmp_path):
    path = tmp_path / "roster.csv"
    path.write_text("sam_id,team,role,name\n", encoding="utf-8")
    with pytest.raises(fr.RosterError, match="no ids"):
        fr.roster_dialog(path, tmp_path)


def test_cli_check_output(tmp_path, capsys):
    path = _roster(tmp_path).path
    assert fr.main(["check", str(path)]) == 0
    out = capsys.readouterr().out
    assert "filled: yes" in out and "ids: 7" in out
    assert "team A: 2   keeper: SAM 31" in out
    assert "team B: 1   keeper: SAM 30" in out
    assert "REF: 2   referee_main: SAM 40" in out
    assert "OUT: 1   not labelled: 1" in out
    assert "kicker: SAM 22" in out


def test_cli_check_reports_invalid(tmp_path, capsys):
    path = _write(tmp_path, ["1,A,kicker,", "2,A,kicker,"])
    assert fr.main(["check", str(path)]) == 2
    assert "INVALID" in capsys.readouterr().err


def test_cli_sheet(fake_run, tmp_path, capsys):
    pred, video = fake_run
    out = tmp_path / "cli"
    assert (
        fr.main(["sheet", "--predictions", str(pred), "--video", str(video), "--out", str(out)])
        == 0
    )
    text = capsys.readouterr().out
    assert "sheet:" in text and "next: python -m vaila.freekiki_roster gui" in text
    assert (out / "roster_ids.png").exists() and (out / "roster.csv").exists()
