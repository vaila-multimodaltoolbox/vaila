"""The report uses the player map: roles by player_id, labels by player_id (no pitch needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vaila.freekiki_report_i18n import tr
from vaila.freekiki_report_players import People, _name, load_camera_roles, people_note


def _write_map(folder: Path) -> None:
    rows = [
        "player_id,role,camera,sam_id,first_frame,last_frame,frames,partner,median_match_m,group",
        "0,keeper,c2,4,0,77,78,c1:18,0.4,0",
        "1,kicker,c2,11,0,77,78,c1:16,0.3,1",
        "2,other,c2,14,0,77,78,c1:21,0.5,2",
    ]
    (folder / "player_map.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_load_camera_roles_reads_keeper_kicker_and_the_names(tmp_path: Path) -> None:
    _write_map(tmp_path)
    roles = load_camera_roles(tmp_path)
    assert roles["keeper_id"] == 4 and roles["kicker_id"] == 11
    assert roles["names"] == {4: 0, 11: 1, 14: 2}


def test_load_camera_roles_is_empty_without_a_map(tmp_path: Path) -> None:
    assert load_camera_roles(tmp_path) == {}


def test_labels_show_the_player_id_and_the_note_names_the_source(tmp_path: Path) -> None:
    _write_map(tmp_path)
    roles = load_camera_roles(tmp_path)
    people = People(
        tracks={4: np.zeros((3, 2)), 11: np.zeros((3, 2)), 14: np.zeros((3, 2))},
        kicker=roles["kicker_id"],
        wall=[14],
        keeper=roles["keeper_id"],
        names=roles["names"],
        source="map",
    )
    assert _name(people, 11) == 1 and _name(people, 4) == 0
    assert _name(people, 99) == 99  # a person without a player id keeps its SAM id
    note = people_note(people, "pt")
    assert tr("pt", "top_how_map") in note
    assert "1" in note and "0" in note and "2" in note
