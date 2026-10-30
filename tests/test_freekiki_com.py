"""The centre of mass of a person from the MHR70 keypoints (vaila/freekiki_com.py).

The body is a symmetric stick figure in pitch metres: the checks are properties the CoM must have
(symmetry, translation, bounds, renormalisation and the cases that give NaN), not a copy of the formula.
"""

from __future__ import annotations

import numpy as np

from vaila import freekiki_com as com


def _body(n: int = 3) -> dict[str, np.ndarray]:
    """A person standing, facing +x, left side at +y, the same on every frame."""

    def side(sign: float) -> dict[str, tuple[float, float, float]]:
        s = "left" if sign > 0 else "right"
        return {
            f"{s}-ear": (0.0, 0.08 * sign, 1.70),
            f"{s}-shoulder": (0.0, 0.20 * sign, 1.50),
            f"{s}-elbow": (0.0, 0.30 * sign, 1.25),
            f"{s}-wrist": (0.0, 0.30 * sign, 1.00),
            f"{s}-hip": (0.0, 0.10 * sign, 1.00),
            f"{s}-knee": (0.0, 0.10 * sign, 0.50),
            f"{s}-ankle": (0.0, 0.10 * sign, 0.10),
            f"{s}-big-toe-tip": (0.20, 0.10 * sign, 0.02),
        }

    named = {**side(+1.0), **side(-1.0)}
    return {name: np.tile(np.array(xyz, dtype=float), (n, 1)) for name, xyz in named.items()}


def test_a_symmetric_body_has_its_centre_of_mass_on_the_midline() -> None:
    points = _body()
    centre = com.com_track(points)
    assert centre.shape == (3, 3)
    np.testing.assert_allclose(centre[:, 1], 0.0, atol=1e-9)  # left and right cancel
    assert 0.10 < centre[0, 2] < 1.70  # between the ankles and the head
    assert 0.0 <= centre[0, 0] <= 0.20  # between the body line and the toes


def test_moving_the_body_moves_the_centre_of_mass_by_the_same_vector() -> None:
    shift = np.array([10.0, -3.0, 0.0])
    base = com.com_track(_body())
    moved = com.com_track({k: v + shift for k, v in _body().items()})
    np.testing.assert_allclose(moved - base, np.tile(shift, (3, 1)), atol=1e-9)


def test_a_missing_segment_is_left_out_and_the_rest_is_renormalised() -> None:
    full = com.com_track(_body())
    points = _body()
    for name in ("left-wrist", "right-wrist"):  # no forearm and hand on either side
        points[name][:] = np.nan
    partial = com.com_track(points)
    assert np.isfinite(partial).all()
    assert not np.allclose(partial, full)
    assert abs(partial[0, 2] - full[0, 2]) < 0.1  # a small mass left out moves the CoM a little


def test_the_centre_of_mass_is_nan_without_the_trunk_or_with_too_little_mass() -> None:
    no_trunk = _body()
    no_trunk["left-hip"][:] = np.nan  # the trunk ends at the hips
    assert np.isnan(com.com_track(no_trunk)).all()

    legs_only = {
        k: v for k, v in _body().items() if k.endswith(("hip", "knee", "ankle", "big-toe-tip"))
    }
    assert np.isnan(com.com_track(legs_only)).all()  # no shoulders: no trunk


def test_a_frame_with_a_missing_point_is_nan_and_the_others_are_not() -> None:
    points = _body(n=3)
    for arr in points.values():
        arr[1] = np.nan  # the whole person is missing on frame 1
    centre = com.com_track(points)
    assert np.isnan(centre[1]).all()
    assert np.isfinite(centre[[0, 2]]).all()


def test_no_points_give_an_empty_track() -> None:
    assert com.com_track({}).shape == (0, 3)
