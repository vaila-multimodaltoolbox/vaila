"""Vertical scale of the pose tracks from the stature prior (synthetic skeletons, no model)."""

from __future__ import annotations

import numpy as np
import pytest

from vaila.freekiki_stature import (
    NOSE_BELOW_TOP_M,
    STATURE_M,
    apply_stature_scale,
    nose_heights,
)


def _track(pid: int, nose_z: float, frames: int = 5, feet: bool = True) -> list[dict]:
    """One person standing at x = 10 + pid, with a nose at ``nose_z`` and the feet on z = 0."""
    rows: list[dict] = []
    for f in range(frames):
        rows.append(
            {
                "frame": f,
                "person_id": pid,
                "kpt_name": "nose",
                "x_m": 10.0 + pid,
                "y_m": 2.0,
                "z_m": nose_z,
            }  # fmt: skip
        )
        if feet:
            rows.append(
                {
                    "frame": f,
                    "person_id": pid,
                    "kpt_name": "left-ankle",
                    "x_m": 10.0 + pid,
                    "y_m": 2.1,
                    "z_m": 0.0,
                }  # fmt: skip
            )
    return rows


def test_a_short_skeleton_is_scaled_to_the_stature_prior() -> None:
    rows = _track(1, nose_z=1.40)  # the raw skeleton of the SAM 3D Body, about 1.5 m tall
    [scale] = apply_stature_scale(rows)
    target = STATURE_M - NOSE_BELOW_TOP_M
    assert scale.scale == pytest.approx(target / 1.40)
    assert scale.stature_before_m == pytest.approx(1.50)
    assert scale.stature_after_m == pytest.approx(STATURE_M)
    assert scale.flag == "ok"
    nose = [r["z_m"] for r in rows if r["kpt_name"] == "nose"]
    assert np.allclose(nose, target)
    assert nose_heights(rows)[1][0] == pytest.approx(target)


def test_feet_stay_on_the_ground_and_the_ground_plane_does_not_move() -> None:
    rows = _track(2, nose_z=1.40)
    xy_before = [(r["x_m"], r["y_m"]) for r in rows]
    apply_stature_scale(rows)
    feet = [r["z_m"] for r in rows if r["kpt_name"] == "left-ankle"]
    assert np.allclose(feet, 0.0)
    assert [(r["x_m"], r["y_m"]) for r in rows] == xy_before  # only the vertical axis changes


def test_each_track_gets_its_own_factor() -> None:
    rows = _track(1, nose_z=1.40) + _track(2, nose_z=1.60)
    scales = {s.person_id: s for s in apply_stature_scale(rows)}
    assert scales[1].scale != pytest.approx(scales[2].scale)
    assert scales[1].stature_after_m == pytest.approx(scales[2].stature_after_m)


def test_a_track_that_never_stands_is_left_as_it_is() -> None:
    rows = _track(3, nose_z=0.15)  # lying down or a fallen player: no scale
    before = [r["z_m"] for r in rows]
    [scale] = apply_stature_scale(rows)
    assert scale.scale is None
    assert scale.flag == "not standing: no scale applied"
    assert [r["z_m"] for r in rows] == before


def test_a_factor_outside_the_plausible_range_is_flagged() -> None:
    rows = _track(4, nose_z=0.60)  # the factor is about 2.7: the raw skeleton is far off
    [scale] = apply_stature_scale(rows)
    assert scale.scale is not None and scale.scale > 1.6
    assert scale.flag == "check the track"


# --- known statures (penalty clip, 08 Nov 2026: the keeper is 2.04 m) ------------------------------


def test_a_stature_spec_reads_roles_ids_and_a_default() -> None:
    from vaila.freekiki_stature import parse_stature_spec

    spec = parse_stature_spec("keeper=2.04, kicker=1.82,sam:39=2.05,default=1.80")
    assert spec == {"keeper": 2.04, "kicker": 1.82, "sam:39": 2.05, "default": 1.80}
    assert parse_stature_spec("") == {}
    for bad in ("keeper", "keeper=tall", "coach=1.9", "keeper=0.5", "sam:x=1.9"):
        with pytest.raises(ValueError):
            parse_stature_spec(bad)


def test_statures_resolve_to_sam_ids_and_an_explicit_id_wins() -> None:
    from vaila.freekiki_stature import resolve_statures

    spec = {"keeper": 2.04, "kicker": 1.82, "sam:39": 2.05, "default": 1.80}
    by_person, default = resolve_statures(spec, {"keeper": 39, "kicker": 15})
    assert by_person == {39: (2.05, "sam:39"), 15: (1.82, "kicker")}
    assert default == 1.80
    by_person, default = resolve_statures({"keeper": 2.04}, {})  # no roles yet: nothing to resolve
    assert by_person == {} and default == STATURE_M


def _keeper(pid: int = 39, frames: int = 10, crouched: int = 6) -> list[dict]:
    """A raw keeper skeleton standing at (52, 0) with the nose at 1.50 m, crouched (1.05 m) in
    ``crouched`` frames, and a hand 0.6 m to its side at 1.0 m."""
    rows = []
    for f in range(frames):
        nose = 1.05 if f < crouched else 1.50
        for name, xyz in (("nose", (52.0, 0.0, nose)), ("left-ankle", (52.0, -0.2, 0.0)),
                          ("left-wrist", (52.0, -0.6, 1.0))):  # fmt: skip
            rows.append({"frame": f, "person_id": pid, "kpt_name": name,
                         "x_m": xyz[0], "y_m": xyz[1], "z_m": xyz[2]})  # fmt: skip
    return rows


def test_a_known_stature_scales_the_whole_body_about_its_ground_point() -> None:
    rows = _keeper()
    ground = {(f, 39): (52.0, 0.0) for f in range(10)}
    [scale] = apply_stature_scale(
        rows, by_person={39: (2.04, "keeper")}, ground=ground, height_quantile=0.9
    )
    # the standing frames set the raw height, not the crouched majority (median 1.05 m)
    expected = (2.04 - NOSE_BELOW_TOP_M) / 1.50
    assert scale.scale == pytest.approx(expected, rel=1e-6)
    assert (scale.stature_target_m, scale.stature_source, scale.isotropic) == (2.04, "keeper", True)
    wrist = next(r for r in rows if r["kpt_name"] == "left-wrist" and r["frame"] == 9)
    assert wrist["y_m"] == pytest.approx(-0.6 * expected)  # the arm grows sideways too
    assert wrist["z_m"] == pytest.approx(1.0 * expected)
    ankle = next(r for r in rows if r["kpt_name"] == "left-ankle" and r["frame"] == 0)
    assert ankle["z_m"] == pytest.approx(0.0)  # the feet stay on the ground


def test_the_default_prior_is_unchanged_without_known_statures() -> None:
    rows = _keeper()
    [scale] = apply_stature_scale(rows)  # median nose (1.05 m), vertical only, as before
    assert scale.scale == pytest.approx((STATURE_M - NOSE_BELOW_TOP_M) / 1.05)
    assert (scale.stature_source, scale.isotropic) == ("prior", False)
    wrist = next(r for r in rows if r["kpt_name"] == "left-wrist")
    assert wrist["y_m"] == pytest.approx(-0.6)
