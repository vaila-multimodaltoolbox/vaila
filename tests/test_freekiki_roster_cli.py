"""CLI of the roster: ``kick-analysis --stop-after sam`` / ``--roster`` and ``kick-roster``.

Update Date: 23 November 2026
Version: 0.4.19

No GPU, no model, no window: the kick folder is the small hand-made one of
``test_freekiki_kick_correct.py`` plus fake SAM predictions (a gzipped JSON with the boxes of three
people). The labelling window (``roster_dialog``) and the pipeline are replaced by recorders.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest

from tests import test_freekiki_kick_correct as _kick_correct
from vaila import freekiki as fk

kick_run = _kick_correct.kick_run  # the hand-made kick folder, as a fixture of this file
STEM = _kick_correct.STEM


def _add_predictions(run: Path) -> Path:
    """Fake SAM 3 predictions of three people under work/<stem>/sam3dinov3/x/<stem>/."""
    folder = run / "work" / STEM / "sam3dinov3" / "x" / STEM
    folder.mkdir(parents=True)
    boxes = {7: (2, 4, 14, 40), 12: (20, 6, 32, 42), 30: (40, 8, 52, 44)}
    frames = []
    for frame in range(_kick_correct.N_FRAMES):
        instances = [
            {
                "person_id": sam,
                "sam_obj_id": sam,
                "bbox_xyxy": [x0 + frame, y0, x1 + frame, y1],
                "keypoints_2d_px": [[x0 + frame, y0]],
            }
            for sam, (x0, y0, x1, y1) in boxes.items()
        ]
        frames.append({"frame": frame, "instances": instances})
    path = folder / f"{STEM}_sam3dinov3_predictions.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(
            {"schema": "vaila_sam3dinov3_v1", "keypoint_names": ["nose"], "frames": frames}, fh
        )
    return path


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# --- parser and kwargs ------------------------------------------------------------------------


def test_kick_analysis_parses_stop_after_and_roster() -> None:
    parser = fk.build_parser()
    args = parser.parse_args(
        [
            "kick-analysis",
            "-w",
            "ws",
            "--videos",
            "clip",
            "--stop-after",
            "sam",
            "--roster",
            "r.csv",
        ]
    )
    assert args.stop_after == "sam" and args.roster == "r.csv"
    args = parser.parse_args(["kick-analysis", "-w", "ws", "--videos", "clip"])
    assert args.stop_after is None and args.roster is None
    with pytest.raises(SystemExit):
        parser.parse_args(["kick-analysis", "-w", "ws", "--videos", "clip", "--stop-after", "3d"])


def test_the_kwargs_carry_stop_after_and_roster() -> None:
    args = fk.build_parser().parse_args(
        [
            "kick-analysis",
            "-w",
            "ws",
            "--videos",
            "clip",
            "--stop-after",
            "sam",
            "--roster",
            "r.csv",
        ]
    )
    kwargs = fk._kick_analysis_kwargs(args)
    assert kwargs["stop_after"] == "sam" and kwargs["roster"] == "r.csv"
    plain = fk._kick_analysis_kwargs(
        fk.build_parser().parse_args(["kick-analysis", "-w", "ws", "--videos", "clip"])
    )
    assert plain["stop_after"] is None and plain["roster"] is None


def test_resume_repeats_the_stored_roster_but_not_the_stop(kick_run: dict) -> None:
    status_path = kick_run["run"] / fk.KICK_STATUS
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["options"]["roster"] = "/stored/roster.csv"
    status["options"]["stop_after"] = "sam"  # a stop is never inherited
    status_path.write_text(json.dumps(status), encoding="utf-8")
    parser = fk.build_parser()
    resumed = fk._kick_analysis_kwargs(
        parser.parse_args(["kick-analysis", "-w", "ws", "--resume", str(kick_run["run"])])
    )
    assert resumed["roster"] == "/stored/roster.csv"
    assert resumed["stop_after"] is None
    given = fk._kick_analysis_kwargs(
        parser.parse_args(
            ["kick-analysis", "-w", "ws", "--resume", str(kick_run["run"]), "--roster", "new.csv"]
        )
    )
    assert given["roster"] == "new.csv"  # the option given replaces the stored one


def test_a_stop_prints_the_roster_paths_and_the_two_next_steps(
    kick_run: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from vaila import freekiki_kick_pipeline as pipeline

    seen: dict = {}
    run = kick_run["run"]

    def fake_run(workspace, **kwargs):
        seen.update(kwargs)
        return {
            "out": str(run),
            "stopped_after": "sam",
            "roster": {STEM: {"sheet": f"{run}/roster/sheet.png", "csv": f"{run}/roster/r.csv"}},
            "status_file": str(run / fk.KICK_STATUS),
        }

    mirrored: list[tuple] = []
    monkeypatch.setattr(pipeline, "run_kick_analysis", fake_run)
    monkeypatch.setattr(fk, "print_gui_cli_mirror", lambda *a, **k: mirrored.append((a, k)))
    code = fk.main(
        ["kick-analysis", "-w", "ws", "--videos", str(run.parent), "--stop-after", "sam"]
    )
    assert code == 0 and seen["stop_after"] == "sam"
    out = capsys.readouterr().out
    assert "stopped after the sam stage" in out and "r.csv" in out
    commands = [a[1] for a, _ in mirrored]
    assert commands[0][-3:] == ["kick-roster", str(run), "--gui"]
    # kick-analysis needs the workspace: the printed next step carries it
    assert commands[1][-5:] == ["kick-analysis", "-w", "ws", "--resume", str(run)]
    assert all(c[:4] == ["uv", "run", "--no-sync", "vaila/freekiki.py"] for c in commands)


# --- kick-roster ------------------------------------------------------------------------------


def test_kick_roster_argv(kick_run: dict) -> None:
    run = str(kick_run["run"])
    assert fk.kick_roster_argv(run) == ["kick-roster", run, "--camera", "reference"]
    assert fk.kick_roster_argv(run, gui=True) == [
        "kick-roster",
        run,
        "--camera",
        "reference",
        "--gui",
    ]
    # the 3D folder resolves to the same run
    assert fk.kick_roster_argv(kick_run["threed"])[1] == run
    with pytest.raises(ValueError, match="one camera"):
        fk.kick_roster_argv(run, "other")
    with pytest.raises(ValueError, match="not a kick analysis folder"):
        fk.kick_roster_argv(kick_run["video"].parent)


def test_the_subcommand_parses() -> None:
    parser = fk.build_parser()
    args = parser.parse_args(["kick-roster", "some_run"])
    assert (args.command, args.kick_dir, args.camera, args.gui, args.force_sheet) == (
        "kick-roster",
        "some_run",
        "reference",
        False,
        False,
    )
    args = parser.parse_args(
        ["kick-roster", "some_run", "--camera", "other", "--gui", "--force-sheet", "-w", "ws"]
    )
    assert args.camera == "other" and args.gui and args.force_sheet and args.workspace == "ws"
    with pytest.raises(SystemExit):
        parser.parse_args(["kick-roster", "some_run", "--camera", "third"])


def test_without_predictions_it_says_what_to_run_first(kick_run: dict) -> None:
    with pytest.raises(ValueError, match="stop-after sam"):
        fk.kick_roster(kick_run["run"])
    assert fk.main(["kick-roster", str(kick_run["run"])]) == 1


def test_kick_roster_writes_the_sheet_and_keeps_the_labels(
    kick_run: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vaila import freekiki_roster as roster_mod

    _add_predictions(kick_run["run"])
    opened: list[tuple] = []
    mirrored: list[list[str]] = []
    monkeypatch.setattr(roster_mod, "roster_dialog", lambda *a, **k: opened.append((a, k)))
    monkeypatch.setattr(fk, "print_gui_cli_mirror", lambda m, cli, **k: mirrored.append(cli))
    monkeypatch.setattr(fk, "_last_workspace", lambda: "")  # no GUI workspace remembered

    first = fk.kick_roster(kick_run["run"])
    # no -w and no remembered workspace: a placeholder the user must replace, never a wrong path
    assert mirrored[-1][-5:] == [
        "kick-analysis",
        "-w",
        "WORKSPACE",
        "--resume",
        str(kick_run["run"]),
    ]
    csv_path = kick_run["run"] / "roster" / STEM / "roster.csv"
    assert first["csv"] == str(csv_path) and csv_path.is_file()
    assert Path(first["sheet"]).is_file() and first["labelled"] == {}
    rows = _read_rows(csv_path)
    assert sorted(int(r["sam_id"]) for r in rows) == [7, 12, 30]
    assert opened == []  # no --gui: no window

    for row in rows:  # the user labels two people and the keeper
        if row["sam_id"] == "7":
            row["team"], row["role"], row["name"] = "A", "keeper", "Rossi"
        elif row["sam_id"] == "12":
            row["team"] = "B"
    _write_rows(csv_path, rows)

    second = fk.kick_roster(kick_run["run"], force_sheet=True, gui=True, workspace="ws")
    assert second["labelled"] == {"A": 1, "B": 1}
    kept = {r["sam_id"]: r for r in _read_rows(csv_path)}
    assert (kept["7"]["team"], kept["7"]["role"], kept["7"]["name"]) == ("A", "keeper", "Rossi")
    assert kept["12"]["team"] == "B" and kept["30"]["team"] == ""
    assert len(opened) == 1 and Path(opened[0][0][0]) == csv_path  # the window got the csv
    # the mirror shows the roster command, then the two next steps
    assert any("kick-roster" in c for c in mirrored)
    assert mirrored[-1][-5:] == ["kick-analysis", "-w", "ws", "--resume", str(kick_run["run"])]

    third = fk.kick_roster(kick_run["run"])  # no --force-sheet: the sheet is reused
    assert third["labelled"] == {"A": 1, "B": 1}
    assert _read_rows(csv_path) == list(kept.values())


def test_main_runs_kick_roster(kick_run: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    _add_predictions(kick_run["run"])
    monkeypatch.setattr(fk, "print_gui_cli_mirror", lambda *a, **k: None)
    assert fk.main(["kick-roster", str(kick_run["run"])]) == 0
    assert (kick_run["run"] / "roster" / STEM / "roster.csv").is_file()


def test_the_gui_rerun_finds_the_roster_of_the_run(kick_run: dict) -> None:
    assert fk._kick_roster_csv(kick_run["run"]) is None
    folder = kick_run["run"] / "roster" / STEM
    folder.mkdir(parents=True)
    (folder / "roster.csv").write_text("sam_id,team\n", encoding="utf-8")
    assert fk._kick_roster_csv(kick_run["run"]) == folder / "roster.csv"
