"""Dribble 1vN analysis from synthetic FreeKiki tracks."""

from __future__ import annotations

import json
from pathlib import Path

from vaila import freekiki_dribble as dr
from vaila import freekiki_setpiece as sp


def _write_run(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    # ball: moves along x
    ball = ["frame,x_m,y_m,z_m,on_field\n"]
    ground = ["frame,person_id,x_m,y_m\n"]
    for f in range(20):
        t = f / 10.0
        ball.append(f"{f},{t},0.2,0.11,True\n")
        ground.append(f"{f},35,{t},0.0\n")  # dribbler
        ground.append(f"{f},23,{t + 2.0},1.0\n")  # marker 1 (always ~2.2 m away)
        ground.append(f"{f},10,{5.0 - t},-1.0\n")  # marker 2 (approaches)
        ground.append(f"{f},5,{8.0},{-2.0}\n")  # marker 3 (static far)
    (folder / "ball_3d_on_field.csv").write_text("".join(ball), encoding="utf-8")
    (folder / "ground_tracks.csv").write_text("".join(ground), encoding="utf-8")
    (folder / "summary.json").write_text(json.dumps({"fps": 10.0}), encoding="utf-8")


def test_analyze_dribble_writes_json_and_min_distance(tmp_path: Path) -> None:
    folder = tmp_path / "clip_3d"
    _write_run(folder)
    result = dr.run_dribble(folder, 35, [23, 10, 5], fps=10.0)
    assert result["kind"] == "dribble"
    assert result["dribbler_id"] == 35
    assert result["marker_ids"] == [23, 10, 5]
    assert result["ball"]["path_length_m"] > 1.5
    assert result["dribbler"]["path_length_m"] > 1.5
    path = folder / "data" / dr.ANALYSIS_NAME
    assert path.is_file()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["markers"][0]["person_id"] == 23
    # marker 10 approaches the dribbler → smaller min than static marker 5
    mins = {m["person_id"]: m["vs_dribbler"]["min_distance_m"] for m in loaded["markers"]}
    assert mins[10] < mins[5]


def test_dribble_requires_a_marker() -> None:
    import pytest

    with pytest.raises(ValueError, match="marker"):
        dr.analyze_dribble("/tmp", 35, [])


def test_dribble_and_pass_are_set_piece_choices_without_wall() -> None:
    assert "dribble" in sp.SET_PIECE_CHOICES and "pass" in sp.SET_PIECE_CHOICES
    chosen = sp.apply_choice({"kind": "kick", "reason": "rule"}, "dribble")
    assert chosen["kind"] == "dribble" and chosen["auto_kind"] == "kick"
    assert not sp.has_wall("dribble") and not sp.has_wall("pass")
    assert sp.is_openplay_analysis("dribble") and not sp.is_openplay_analysis("open_play")
    assert sp.is_goal_kick("kick") and not sp.is_goal_kick("dribble")
