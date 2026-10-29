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
