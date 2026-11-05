"""Smoke test for the open-play dribble / pass TV report."""

from __future__ import annotations

import json
from pathlib import Path

from vaila import freekiki_openplay_report as op


def _dribble_folder(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    ball = ["frame,x_m,y_m,z_m,on_field\n"]
    ground = ["frame,person_id,x_m,y_m\n"]
    for f in range(15):
        t = f / 10.0
        ball.append(f"{f},{t},0.1,0.11,True\n")
        ground.append(f"{f},35,{t},0.0\n")
        ground.append(f"{f},23,{t + 1.5},0.8\n")
    (folder / "ball_3d_on_field.csv").write_text("".join(ball), encoding="utf-8")
    (folder / "ground_tracks.csv").write_text("".join(ground), encoding="utf-8")
    (folder / "summary.json").write_text(json.dumps({"fps": 10.0}), encoding="utf-8")
    return folder


def test_build_dribble_report_html(tmp_path: Path) -> None:
    folder = _dribble_folder(tmp_path / "drib_3d")
    result = op.build_openplay_report(
        folder,
        kind="dribble",
        dribbler_id=35,
        marker_ids=[23],
        lang="pt",
        fps=10.0,
    )
    assert result["kind"] == "dribble"
    html = Path(result["languages"]["pt"]["html"])
    assert html.is_file()
    text = html.read_text(encoding="utf-8")
    assert "drible" in text.lower() or "driblador" in text.lower()
    assert (folder / "data" / "dribble_analysis.json").is_file()
    assert (folder / "data" / "setpiece.json").is_file()
    setpiece = json.loads((folder / "data" / "setpiece.json").read_text(encoding="utf-8"))
    assert setpiece["kind"] == "dribble"
    assert setpiece.get("auto_kind") == "kick"


def _pass_folder(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    ball = ["frame,x_m,y_m,z_m,on_field\n"]
    ground = ["frame,person_id,x_m,y_m\n"]
    for f in range(25):
        t = f / 25.0
        ball.append(f"{f},{0.5 + 14.0 * t},0.0,0.11,True\n")  # already moving in frame 0
        ground.append(f"{f},21,0.0,-0.5\n")
        ground.append(f"{f},51,{12.0 + 2.0 * t},{4.0 - 4.0 * t}\n")  # runs from frame 0
    (folder / "ball_3d_on_field.csv").write_text("".join(ball), encoding="utf-8")
    (folder / "ground_tracks.csv").write_text("".join(ground), encoding="utf-8")
    (folder / "ball_flight.json").write_text(
        json.dumps({"kick_time_s": -0.03, "fps": 25.0}), encoding="utf-8"
    )
    shot = folder.parent / "shot_3d"
    (shot / "data").mkdir(parents=True, exist_ok=True)
    (shot / "data" / "setpiece.json").write_text(
        json.dumps({"ball_xy_m": [15.5, 0.5], "distance_to_goal_m": 10.0}), encoding="utf-8"
    )
    return folder


def test_build_pass_report_with_timing_and_shot(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(op, "print_pdf", lambda html, pdf: None)
    folder = _pass_folder(tmp_path / "pass_3d")
    result = op.build_openplay_report(
        folder,
        kind="pass",
        passer_id=21,
        receiver_id=51,
        shot_folder=tmp_path / "shot_3d",
        arrival_frame=24,
        lang="both",
    )
    pt = Path(result["languages"]["pt"]["html"]).read_text(encoding="utf-8")
    en = Path(result["languages"]["en"]["html"]).read_text(encoding="utf-8")
    assert "do passador até o ponto de chegada" in pt
    assert "o recebedor começou a correr antes do passe" in pt
    assert "Depois do passe: o chute" in pt
    assert "Posição, velocidade e tempo" in pt
    assert "1º quadro (logo após a saída)" in pt
    assert "from the passer to the arrival point" in en
    assert "After the pass: the shot" in en
    assert "{{" not in pt and "{{" not in en
    data = json.loads((folder / "data" / "pass_analysis.json").read_text(encoding="utf-8"))
    assert data["timing"]["arrival_source"] == "user"
    assert data["timing"]["release_source"] == "flight_fit"
    assert (folder / "data" / "pass_timeseries.csv").is_file()
