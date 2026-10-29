"""The kind of set piece (vaila/freekiki_setpiece.py): penalty, free kick with a wall, free kick.

The cases are placed on the FIFA distances (penalty mark 11 m, wall 9.15 m) with the margins written in
the module. The run of the 28 October clip (the free kick with a wall) is checked when it is on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaila import freekiki as fk
from vaila import freekiki_setpiece as sp

GOAL = sp.GOAL_LINE_X_M


def _wall(ball_x: float, distance: float, n: int) -> dict[int, tuple[float, float]]:
    """``n`` people standing ``distance`` m from the ball, between it and the goal, at distinct y."""
    return {100 + i: (ball_x + distance, -4.0 + 2.0 * i) for i in range(n)}


def test_a_ball_on_the_penalty_mark_with_nobody_in_the_ring_is_a_penalty() -> None:
    ball = (GOAL - 11.0, 0.5)
    result = sp.classify(ball, {1: (ball[0] - 1.0, 0.5)})  # the kicker, 1 m away: not in the ring
    assert result["kind"] == "penalty"
    assert result["wall_count"] == 0
    assert abs(result["distance_to_goal_m"] - 11.0) < 1e-6


def test_a_ball_far_out_with_a_wall_is_a_free_kick_with_a_wall() -> None:
    ball = (28.05, -0.72)
    feet = {1: (27.6, -1.9), **_wall(28.05, 9.15, 3)}
    result = sp.classify(ball, feet)
    assert result["kind"] == "free_kick_wall"
    assert result["wall_count"] == 3
    assert set(result["wall_ids"]) == {100, 101, 102}


def test_a_ball_far_out_without_a_wall_is_a_free_kick() -> None:
    ball = (28.05, -0.72)
    result = sp.classify(ball, {1: (27.6, -1.9), 2: (23.3, 1.2)})
    assert result["kind"] == "free_kick"
    assert result["wall_count"] == 0


def test_a_person_just_outside_the_wall_ring_does_not_make_a_wall() -> None:
    ball = (28.05, -0.72)
    # 10.9 m from the ball, between it and the goal: outside the ring, so no wall
    feet = {200 + i: (ball[0] + 10.9, -4.0 + 2.0 * i) for i in range(3)}
    assert sp.classify(ball, feet)["wall_count"] == 0


def test_a_ball_at_the_penalty_mark_but_at_the_side_is_not_a_penalty() -> None:
    result = sp.classify((GOAL - 11.0, 6.0), {})
    assert result["kind"] == "free_kick"
    assert "from the middle of the goal" in result["reason"]


def test_a_ball_at_the_other_goal_uses_the_other_goal_line() -> None:
    result = sp.classify((-(GOAL - 11.0), 0.0), {})
    assert result["goal_x_m"] == -GOAL
    assert result["kind"] == "penalty"


def test_the_wall_is_counted_only_between_the_ball_and_the_goal() -> None:
    ball = (28.05, -0.72)
    behind = {300 + i: (ball[0] - 9.15, -4.0 + 2.0 * i) for i in range(3)}  # on the other side
    assert sp.classify(ball, behind)["wall_count"] == 0


def test_a_folder_without_the_ball_file_is_unknown_not_an_error(tmp_path: Path) -> None:
    result = sp.classify_run(tmp_path)
    assert result["kind"] == "unknown"
    assert result["wall_count"] == 0


def test_a_ball_1_25_m_past_the_penalty_mark_is_a_penalty_with_the_1_5_m_tolerance() -> None:
    # the penalty clip of 28 October: 12.25 m from the goal line, 1.25 m outside the old 1.0 m
    result = sp.classify((GOAL - 12.25, -0.23), {1: (GOAL - 12.95, -0.2)})
    assert result["kind"] == "penalty"
    assert abs(result["distance_to_goal_m"] - 12.25) < 1e-6


def test_a_ball_beyond_the_penalty_tolerance_is_a_free_kick() -> None:
    assert sp.classify((GOAL - 12.6, 0.0), {})["kind"] == "free_kick"


def test_wall_ids_become_player_ids_and_keep_the_sam_ids() -> None:
    result = {"kind": "free_kick_wall", "wall_ids": [5, 10, 24], "wall_count": 3}
    out = sp.with_player_ids(result, {5: 3, 10: 4})  # SAM 24 is not in the player map
    assert out["wall_ids"] == [3, 4]
    assert out["wall_sam_ids"] == [5, 10, 24]
    assert out["wall_count"] == 2
    assert out["kind"] == "free_kick_wall"


VEIGA_RUN = Path("/home/preto/data/veiga/kickanalysis/kick_analysis_20261028_201548")


@pytest.mark.skipif(
    not VEIGA_RUN.is_dir(),
    reason="the 28 October clip is not on this machine",
)
def test_the_28_october_clip_is_a_free_kick_with_a_wall() -> None:
    folder = VEIGA_RUN / "c1_main_sync_15_frames_2_to_79_3d_20261028_201548"
    if not folder.is_dir():
        pytest.skip("the run folder was cleaned: run the analysis again first")
    result = sp.classify_run(folder)
    assert result["kind"] == "free_kick_wall"
    assert result["wall_count"] >= 2
    assert abs(result["distance_to_goal_m"] - 24.4) < 0.5


def _run_folder(folder: Path, ball_x: float | None, ball_y: float = 0.5) -> Path:
    """A 3D folder with the ball on the field (or none) and one kicker's feet 1 m from the ball."""
    folder.mkdir(parents=True, exist_ok=True)
    if ball_x is None:
        ball_rows = ["frame,x_m,y_m,on_field\n0,,,\n"]
    else:
        ball_rows = [f"frame,x_m,y_m,on_field\n0,{ball_x},{ball_y},True\n"]
    (folder / "ball_3d_on_field.csv").write_text("".join(ball_rows), encoding="utf-8")
    feet = "frame,person_id,kpt_name,x_m,y_m\n"
    if ball_x is not None:
        feet += f"0,1,left-ankle,{ball_x - 1.0},{ball_y}\n"
    (folder / "poses_3d_on_field.csv").write_text(feet, encoding="utf-8")
    return folder


def test_write_provisional_grades_the_run_ball_into_its_own_file(tmp_path: Path) -> None:
    folder = _run_folder(tmp_path / "clip_3d", ball_x=GOAL - 11.0)
    result = sp.write_provisional(folder, "sam3")
    assert result["kind"] == "penalty"
    assert result["provisional"] is True
    assert result["ball_mode"] == "sam3"
    written = folder / "data" / sp.PROVISIONAL_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["kind"] == "penalty"
    assert not (folder / "data" / sp.SETPIECE_NAME).exists()  # the one-click's file is not written


def test_write_provisional_without_a_ball_is_unknown_and_does_not_fail(tmp_path: Path) -> None:
    folder = _run_folder(tmp_path / "clip_3d", ball_x=None)
    result = sp.write_provisional(folder, "off")
    assert result["kind"] == "unknown"
    assert (folder / "data" / sp.PROVISIONAL_NAME).is_file()


def test_detect_soccer_3d_prints_the_provisional_set_piece(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    folder = _run_folder(tmp_path / "clip_3d_x", ball_x=GOAL - 11.0)
    monkeypatch.setattr(fk, "detect_videos", lambda ws, source, **kwargs: tmp_path / "batch")
    monkeypatch.setattr(fk, "detect_3d", lambda run, **kwargs: [{"out": str(folder)}])
    fk.detect_soccer_3d("ws", "clip.mp4", output_dir=tmp_path, ball="sam3")
    out = capsys.readouterr().out
    assert "set piece (provisional, ball of this run = sam3): penalty" in out
    assert (folder / "data" / sp.PROVISIONAL_NAME).is_file()


def test_apply_choice_forces_the_kind_and_keeps_the_rules_answer() -> None:
    rule = sp.classify((GOAL - 11.0, 0.0), {})  # a penalty by the rule
    forced = sp.apply_choice(rule, "free_kick")
    assert forced["kind"] == "free_kick"
    assert forced["auto_kind"] == "penalty"
    assert forced["choice"] == "free_kick"
    assert "chosen by the user (free_kick); the ball rule said penalty" in forced["reason"]
    assert forced["distance_to_goal_m"] == rule["distance_to_goal_m"]  # the measures stay


def test_apply_choice_auto_keeps_the_rule_and_a_bad_choice_is_refused() -> None:
    rule = sp.classify((GOAL - 20.0, 0.0), {})
    kept = sp.apply_choice(rule, "auto")
    assert kept["kind"] == rule["kind"] == "free_kick"
    assert kept["choice"] == "auto"
    assert kept["auto_kind"] == "free_kick"
    with pytest.raises(ValueError, match="set piece must be one of"):
        sp.apply_choice(rule, "corner")
