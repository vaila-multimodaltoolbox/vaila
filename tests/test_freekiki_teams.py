"""Roles, teams and stature (vaila/freekiki_teams.py) on synthetic colours and a synthetic camera.

A shirt is classified by its HSV colour inside the torso; the goalkeeper is in the team opposite the
kicker; the caps (11 per team, 3 referees) hold; a height is recovered exactly by the calibrated
camera when the top and the foot project from a known vertical.
"""

from __future__ import annotations

import numpy as np

from vaila import freekiki_teams as T  # noqa: N812 (the short name keeps the tests readable)
from vaila.fifa_to_dlt import compute_dlt3d_from_KRt
from vaila.freekiki_stereo import project
from vaila.monocular_dlt_align import dlt_projection_matrix

W, H = 1920.0, 1080.0
K = np.array([[2400.0, 0.0, W / 2.0], [0.0, 2400.0, H / 2.0], [0.0, 0.0, 1.0]])


def _solid(bgr) -> np.ndarray:
    return np.full((12, 12, 3), bgr, dtype=np.uint8)


def test_each_shirt_colour_is_its_own_class() -> None:
    for bgr, expected in (
        ((0, 200, 0), T.GREEN),
        ((230, 140, 70), T.BLUE),
        ((0, 120, 255), T.ORANGE),
        ((0, 230, 255), T.YELLOW),
    ):
        counts, n = T.colour_counts(_solid(bgr))
        assert n == 144
        assert max(counts, key=counts.get) == expected, (bgr, counts)


def test_grass_outside_the_torso_does_not_count() -> None:
    image = np.full((100, 100, 3), (60, 150, 60), dtype=np.uint8)  # grass everywhere
    image[10:50, 25:75] = (230, 140, 70)  # a blue shirt in the box of the person
    box = (25, 10, 75, 50)  # the person's box: the grass around it is not in it
    crop = T.torso_crop(image, box)
    counts, _ = T.colour_counts(crop)
    assert max(counts, key=counts.get) == T.BLUE


def test_a_colour_needs_its_share_of_the_torso() -> None:
    entry = {"counts": {T.GREEN: 5, T.BLUE: 5}, "pixels": 100, "frames": 3}
    assert T.dominant(entry) == (None, 0.05)  # 5 % < 15 %, nothing wins
    entry = {"counts": {T.GREEN: 40, T.BLUE: 5}, "pixels": 100, "frames": 3}
    colour, purity = T.dominant(entry)
    assert colour == T.GREEN and purity == 0.4


def test_the_goalkeeper_is_in_the_team_opposite_the_kicker() -> None:
    green = {"counts": {T.GREEN: 60}, "pixels": 100, "frames": 10}
    yellow = {"counts": {T.YELLOW: 60}, "pixels": 100, "frames": 10}
    rows = T.assign(
        {"1": green, "0": yellow},
        {"0": "keeper", "1": "kicker"},
        referee_order=[],
    )
    assert rows["1"]["team"] == "TEAM_A"  # the kicker wears green
    assert rows["0"]["team"] == "TEAM_B"  # the goalkeeper is opposite
    assert rows["0"]["reason"].startswith("goalkeeper")


def test_no_more_than_eleven_players_per_team_and_three_referees() -> None:
    blue = {"counts": {T.BLUE: 60}, "pixels": 100, "frames": 10}
    orange = {"counts": {T.ORANGE: 60}, "pixels": 100, "frames": 10}
    colours = {str(i): blue for i in range(2, 14)}  # twelve blue players
    colours.update({str(100 + i): orange for i in range(4)})  # four orange people
    roles = {str(i): "other" for i in range(2, 14)}
    roles.update({str(100 + i): "other" for i in range(4)})
    rows = T.assign(colours, roles, referee_order=[str(100 + i) for i in range(4)])
    assert sum(1 for r in rows.values() if r["team"] == "TEAM_B") == 11
    assert sum(1 for r in rows.values() if r["team"] == T.REF_TEAM) == 3
    over = [r for r in rows.values() if "not assigned" in r["reason"]]
    assert len(over) == 1
    assert "not a referee" in rows["103"]["reason"]


def test_referees_take_their_roles_in_the_order_given() -> None:
    orange = {"counts": {T.ORANGE: 60}, "pixels": 100, "frames": 10}
    rows = T.assign({"2": orange, "21": orange}, {"2": "other", "21": "other"}, ["2", "21"])
    assert rows["2"]["role"] == "referee_main" and rows["21"]["role"] == "referee_aux1"


def test_the_label_names_the_role_the_team_and_the_height() -> None:
    assert T.label_text("0", "keeper", "TEAM_B", None) == "ID 0 GK TEAM_B"
    assert T.label_text("1", "kicker", "TEAM_A", 1.54) == "ID 1 KICKER TEAM_A 1.54 m"
    assert T.label_text("2", "referee_main", T.REF_TEAM, 1.6) == "ID 2 REF_MAIN 1.60 m"
    assert T.label_text("8", "other", "", None) == "ID 8"


def _camera(centre, target) -> np.ndarray:
    centre, target = np.asarray(centre, float), np.asarray(target, float)
    z = target - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return compute_dlt3d_from_KRt(K, R, -R @ centre)


def test_the_height_is_recovered_exactly_from_the_foot_and_the_top() -> None:
    L = _camera([-60.0, -45.0, 25.0], [-30.0, -10.0, 1.0])
    P = dlt_projection_matrix(L)
    foot = np.array([-31.0, -9.0, 0.0])
    top = np.array([-31.0, -9.0, 1.8])
    uv_foot = project(P, foot)
    uv_top = project(P, top)
    assert abs(T.height_from_points(L, uv_foot, uv_top) - 1.8) < 1e-6


def test_a_stature_outside_the_plausible_range_is_not_reported() -> None:
    frames = {f: [] for f in range(6)}  # no keypoints: no upright frame, so no stature
    result = T.stature_from_keypoints(
        {f: np.zeros(11) for f in frames}, frames, 1, ["nose", "left-hip"]
    )
    assert result["height_m"] is None and result["reason"]


def _player(colour: str, frames: int = 50) -> dict:
    return {"counts": {colour: 60}, "pixels": 100, "frames": frames}


def test_team_b_is_the_other_colour_most_field_players_wear() -> None:
    roles = {"0": "keeper", "1": "kicker", "2": "referee_main", "21": "other"}
    roles.update({str(i): "other" for i in range(3, 12)})
    colours = {"0": _player(T.YELLOW), "1": _player(T.GREEN), "2": _player(T.ORANGE)}
    colours.update({str(i): _player(T.BLUE) for i in range(3, 12)})
    colours["21"] = _player(T.ORANGE)
    colours_by_team = T.choose_team_colours(colours, roles, team_a_colour=T.GREEN)
    assert colours_by_team == {"TEAM_A": T.GREEN, "TEAM_B": T.BLUE, "REF": T.ORANGE}


def test_naming_the_other_colour_for_team_a_swaps_team_b() -> None:
    roles = {"0": "keeper", "1": "kicker"}
    roles.update({str(i): "other" for i in range(2, 6)})
    colours = {"0": _player(T.YELLOW), "1": _player(T.BLUE)}
    colours.update({str(i): _player(T.GREEN) for i in range(2, 6)})
    colours_by_team = T.choose_team_colours(colours, roles, team_a_colour=T.BLUE)
    assert colours_by_team["TEAM_A"] == T.BLUE and colours_by_team["TEAM_B"] == T.GREEN


def test_the_goalkeeper_colour_is_not_counted_as_a_team() -> None:
    roles = {"0": "keeper", "1": "kicker", "2": "other"}
    colours = {"0": _player(T.BLUE, 200), "1": _player(T.GREEN), "2": _player(T.YELLOW)}
    colours_by_team = T.choose_team_colours(colours, roles, team_a_colour=T.GREEN)
    assert colours_by_team["TEAM_B"] == T.YELLOW  # the keeper's blue does not make a team
