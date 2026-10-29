"""One player id across the cameras of the same kick (synthetic scenes, no model, no video).

Every person of the scene has a known identity; the cameras number them with their own ids, and
the cameras are shifted by a sync offset. The tests check that the same person gets the same
``player_id`` in both cameras, that the roles follow the rules, and that a person can never hold
two places at once.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from vaila.freekiki_player_ids import (
    PAIR_GATE_M,
    Track,
    assign_player_ids,
    link_tracks,
    pair_cost,
    run_player_ids,
)

FPS = 30.0
N = 40  # frames of the scene, in the index of the reference camera
CONTACT = 10  # the ball starts to move here
BALL_REST = (27.0, 0.0)
# the people: the ground position of each one (static), so the ordering by the kick is known
PEOPLE = {
    "keeper": (51.0, 0.8),  # 1.7 m from the centre of the goal being attacked (+x)
    "kicker": (26.8, 0.0),  # its foot is 0.2 m from the rest point
    "A": (24.0, 0.0),  # 3.0 m from the rest point -> id 2
    "B": (30.0, 4.0),  # 5.0 m -> id 3
    "C": (21.0, -6.0),  # 8.5 m -> id 4
    "D": (40.0, -12.0),  # 17.7 m, seen by camera 1 only -> id 5
}
EXPECTED = {"keeper": 0, "kicker": 1, "A": 2, "B": 3, "C": 4, "D": 5}


def _ball():
    """Ball pixels and metres of the reference camera: at rest, then 0.5 m per frame toward +x."""
    px: dict[int, tuple[float, float]] = {}
    metres: dict[int, dict[str, float]] = {}
    for f in range(N):
        x = BALL_REST[0] + (0.5 * (f - CONTACT + 1) if f >= CONTACT else 0.0)
        metres[f] = {"x_m": x, "y_m": BALL_REST[1]}
        px[f] = (1000.0 + 100.0 * x, 500.0 - 100.0 * BALL_REST[1])
    return px, metres


def _camera(name, offset, sam_ids, rng, absent=lambda person, f: False):
    """Tracks and feet of one camera. Frame ``f`` of the scene is index ``f + offset`` here."""
    tracks, feet = [], {}
    for person, sam in sam_ids.items():
        x0, y0 = PEOPLE[person]
        xy, fe = {}, {}
        for f in range(N):
            if absent(person, f):
                continue
            jx, jy = rng.normal(0.0, 0.05, 2)
            xy[f + offset] = (x0 + jx, y0 + jy)
            fe[f + offset] = np.array([[x0, y0 + 0.1], [x0, y0 - 0.1], [x0 + 0.1, y0]]) + [jx, jy]
        tracks.append(Track(camera=name, sam_id=sam, xy=xy))
        feet[(name, sam)] = fe
    return tracks, feet


def _scene(c2_numbers=None, extra=None, c1_sam=None, absent_a=False):
    rng = np.random.default_rng(7)
    c1_sam = c1_sam or {"keeper": 10, "kicker": 11, "A": 12, "B": 13, "C": 14, "D": 15}
    c2_sam = c2_numbers or {"keeper": 20, "kicker": 21, "A": 22, "B": 23, "C": 24}
    t1, f1 = _camera("c1", 0, c1_sam, rng)
    t2, f2 = _camera(
        "c2",
        2,  # the same instant is index f + 2 in camera 2
        c2_sam,
        rng,
        absent=(lambda p, f: p == "A" and 4 <= f < 9) if absent_a else (lambda p, f: False),
    )
    tracks = t1 + t2
    feet = {**f1, **f2}
    if extra:
        extra_tracks, extra_feet = extra(rng)
        tracks += extra_tracks
        feet.update(extra_feet)
    return tracks, feet


def _assign(tracks, feet, offsets=None):
    offsets = offsets or {"c1": 0, "c2": 2}
    px, metres = _ball()
    groups = link_tracks(tracks, offsets)
    ball_xy = {f: (r["x_m"], r["y_m"]) for f, r in metres.items()}
    ids, roles, info = assign_player_ids(
        groups,
        offsets=offsets,
        feet=feet,
        rest=BALL_REST,
        contact=CONTACT,
        side=1.0,
        ball_xy=ball_xy,
        fps=FPS,
        last_frame=N - 1,
    )
    player_of = {}
    for gi, g in enumerate(groups):
        for t in g.tracks:
            player_of[(t.camera, t.sam_id)] = ids[gi]
    return groups, ids, roles, info, player_of


def _check_identities(player_of, c1_sam, c2_sam):
    for person, expected in EXPECTED.items():
        assert player_of[("c1", c1_sam[person])] == expected, person
        if person in c2_sam:
            assert player_of[("c2", c2_sam[person])] == expected, person


def test_the_same_person_gets_the_same_id_in_both_cameras() -> None:
    c1_sam = {"keeper": 10, "kicker": 11, "A": 12, "B": 13, "C": 14, "D": 15}
    c2_sam = {"keeper": 20, "kicker": 21, "A": 22, "B": 23, "C": 24}
    tracks, feet = _scene()
    _, _, _, _, player_of = _assign(tracks, feet)
    _check_identities(player_of, c1_sam, c2_sam)


def test_the_numbering_of_the_second_camera_does_not_matter() -> None:
    c1_sam = {"keeper": 10, "kicker": 11, "A": 12, "B": 13, "C": 14, "D": 15}
    c2_sam = {"keeper": 34, "kicker": 33, "A": 32, "B": 31, "C": 30}  # reversed numbering
    tracks, feet = _scene(c2_numbers=c2_sam)
    _, _, _, _, player_of = _assign(tracks, feet)
    _check_identities(player_of, c1_sam, c2_sam)


def test_roles_are_keeper_zero_and_kicker_one_and_the_rest_by_the_foot_distance() -> None:
    tracks, feet = _scene()
    _, ids, roles, info, _ = _assign(tracks, feet)
    assert info["keeper_found"] and info["kicker_found"]
    assert info["reference_frame"] == CONTACT
    assert info["kicker_distance_m"] < 0.3  # the foot of the kicker is 0.2 m from the rest point
    assert sorted(roles.values()) == ["keeper", "kicker", "other", "other", "other", "other"]
    assert sorted(ids.values()) == [0, 1, 2, 3, 4, 5]


def test_a_person_seen_in_one_camera_keeps_an_id_of_its_own() -> None:
    tracks, feet = _scene()  # D is only in camera 1
    _, _, _, _, player_of = _assign(tracks, feet)
    assert player_of[("c1", 15)] == EXPECTED["D"]


def test_a_single_camera_gets_the_same_rules_without_a_partner() -> None:
    rng = np.random.default_rng(3)
    sam = {"keeper": 10, "kicker": 11, "A": 12, "B": 13, "C": 14, "D": 15}
    tracks, feet = _camera("c1", 0, sam, rng)
    _, _, _, _, player_of = _assign(tracks, feet, offsets={"c1": 0})
    assert {person: player_of[("c1", sam[person])] for person in sam} == EXPECTED


def test_two_tracks_alive_at_the_same_time_are_two_people() -> None:
    def ghost(rng):
        # a second track on the kicker's position in camera 1, alive during the whole clip
        pos = PEOPLE["kicker"]
        xy = dict.fromkeys(range(N), (pos[0] + 0.05, pos[1]))
        return [Track(camera="c1", sam_id=16, xy=xy)], {
            ("c1", 16): {f: np.array([[pos[0], pos[1] + 0.1]]) for f in range(N)}
        }

    tracks, feet = _scene(extra=ghost)
    _, _, _, _, player_of = _assign(tracks, feet)
    assert player_of[("c1", 16)] != player_of[("c1", 11)]


def test_fragments_of_one_person_join_the_same_group() -> None:
    tracks, feet = _scene()
    kicker = next(t for t in tracks if t.camera == "c1" and t.sam_id == 11)
    first = {f: p for f, p in kicker.xy.items() if f < 20}
    second = {f: p for f, p in kicker.xy.items() if f >= 20}
    tracks = [t for t in tracks if t is not kicker] + [
        Track(camera="c1", sam_id=11, xy=first),
        Track(camera="c1", sam_id=17, xy=second),
    ]
    feet[("c1", 17)] = {f: feet[("c1", 11)][f] for f in second}
    feet[("c1", 11)] = {f: feet[("c1", 11)][f] for f in first}
    _, _, _, _, player_of = _assign(tracks, feet)
    assert player_of[("c1", 11)] == player_of[("c1", 17)] == EXPECTED["kicker"]


def test_pair_cost_needs_enough_common_frames() -> None:
    a = Track("c1", 1, dict.fromkeys(range(30), (0.0, 0.0)))
    b = Track("c2", 2, dict.fromkeys(range(30), (0.5, 0.0)))
    cost, n = pair_cost(a, b, offset=0)
    assert cost == pytest.approx(0.5) and n == 30
    short = Track("c2", 3, dict.fromkeys(range(5), (0.5, 0.0)))
    assert pair_cost(a, short, offset=0)[0] == float("inf")


def test_pair_cost_reads_the_second_camera_at_the_offset_frame() -> None:
    # one person moving 1 m per frame in camera a; camera b sees it 3 frames later
    a = Track("c1", 1, {f: (float(f), 0.0) for f in range(30)})
    b = Track("c2", 2, {f + 3: (float(f), 0.0) for f in range(30)})
    assert pair_cost(a, b, offset=3)[0] == pytest.approx(0.0)
    assert pair_cost(a, b, offset=0)[0] > PAIR_GATE_M


def test_a_pair_farther_than_the_gate_is_not_linked() -> None:
    a = Track("c1", 1, dict.fromkeys(range(30), (0.0, 0.0)))
    b = Track("c2", 2, dict.fromkeys(range(30), (3.0, 0.0)))
    groups = link_tracks([a, b], {"c1": 0, "c2": 0})
    assert len(groups) == 2


def test_the_kicker_is_found_later_when_no_foot_is_within_reach_at_the_contact() -> None:
    # the kicker's foot is 4 m away at the contact and reaches the ball one frame later
    tracks = [Track("c1", 1, dict.fromkeys(range(N), (30.0, 0.0)))]
    feet = {("c1", 1): {f: np.array([[30.0, 0.0]]) for f in range(N)}}
    tracks.append(Track("c1", 2, dict.fromkeys(range(N), (40.0, 6.0))))
    feet[("c1", 2)] = {f: np.array([[40.0, 6.0]]) for f in range(N)}
    px, metres = _ball()
    groups = link_tracks(tracks, {"c1": 0})
    feet[("c1", 1)][CONTACT + 1] = np.array([[27.6, 0.0]])
    ids, roles, info = assign_player_ids(
        groups,
        offsets={"c1": 0},
        feet=feet,
        rest=BALL_REST,
        contact=CONTACT,
        side=1.0,
        ball_xy={f: (r["x_m"], r["y_m"]) for f, r in metres.items()},
        fps=FPS,
        last_frame=N - 1,
    )
    assert info["kicker_found"] and info["reference_frame"] == CONTACT + 1
    kicker_group = next(gi for gi, role in roles.items() if role == "kicker")
    assert groups[kicker_group].tracks[0].sam_id == 1


def test_run_player_ids_writes_the_three_tables_per_folder(tmp_path: Path) -> None:
    px, metres = _ball()
    c1_sam = {"keeper": 10, "kicker": 11, "A": 12, "B": 13, "C": 14, "D": 15}
    c2_sam = {"keeper": 20, "kicker": 21, "A": 22, "B": 23, "C": 24}
    rng = np.random.default_rng(11)
    folders = {
        "c1": tmp_path / "c1_main_sync_15_frames_2_to_79_3d_20990101_000000",
        "c2": tmp_path / "c2_16mm_sync_16_frames_1_to_78_3d_20990101_000000",
    }
    offset = 2  # (16 - 1) - (15 - 2)
    people = {"c1": c1_sam, "c2": c2_sam}
    for cam, folder in folders.items():
        folder.mkdir()
        shift = 0 if cam == "c1" else offset
        with (folder / "ground_tracks.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "person_id", "raw_x_m", "raw_y_m", "x_m", "y_m"])
            with (folder / "poses_3d_on_field.csv").open("w", newline="", encoding="utf-8") as ph:
                pw = csv.writer(ph)
                pw.writerow(["frame", "person_id", "kpt_idx", "kpt_name", "x_m", "y_m", "z_m"])
                for person, sam in people[cam].items():
                    x0, y0 = PEOPLE[person]
                    for f in range(N):
                        jx, jy = rng.normal(0.0, 0.05, 2)
                        w.writerow([f + shift, sam, x0, y0, x0 + jx, y0 + jy])
                        pw.writerow([f + shift, sam, 0, "nose", x0, y0, 1.6])
                        pw.writerow([f + shift, sam, 1, "left-ankle", x0 + jx, y0 + 0.1, 0.0])
                        pw.writerow([f + shift, sam, 2, "right-ankle", x0 + jx, y0 - 0.1, 0.0])
        with (folder / "ball.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "cx", "cy", "x1", "y1", "x2", "y2", "score"])
            for f in range(N):
                cx, cy = px[f]
                w.writerow([f + shift, cx, cy, cx - 5, cy - 5, cx + 5, cy + 5, 1.0])
    with (folders["c1"] / "ball_3d.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "cx", "cy", "score", "x_m", "y_m", "z_m", "on_field"])
        for f in range(N):
            w.writerow([f, "", "", "", metres[f]["x_m"], metres[f]["y_m"], 0.11, True])

    summary = run_player_ids([folders["c1"], folders["c2"]])

    assert summary["offsets"] == {folders["c1"].name: 0, folders["c2"].name: offset}
    assert summary["conflicts"] == 0
    assert summary["keeper"] and summary["kicker"]
    for folder in folders.values():
        assert (folder / "player_map.csv").is_file()
        assert (folder / "poses_3d_on_field_players.csv").is_file()
        assert (folder / "ground_tracks_players.csv").is_file()
    with (folders["c2"] / "player_map.csv").open(newline="", encoding="utf-8") as fh:
        rows = {int(r["sam_id"]): int(r["player_id"]) for r in csv.DictReader(fh)}
    assert rows[20] == 0 and rows[21] == 1  # keeper and kicker of camera 2
    with (folders["c1"] / "poses_3d_on_field_players.csv").open(newline="", encoding="utf-8") as fh:
        first = next(csv.DictReader(fh))
    assert "player_id" in first
