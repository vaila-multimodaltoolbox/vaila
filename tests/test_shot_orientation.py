"""Tests for :mod:`vaila.shot_orientation`.

Covers the four pure helpers used by the xGHub integration:
``body_orientation_2d``, ``body_orientation_3d``, ``goal_geometry`` and
``relative_orientation_to_goal``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vaila.shot_orientation import (
    FifaJointIndices,
    body_orientation_2d,
    body_orientation_3d,
    goal_geometry,
    relative_orientation_to_goal,
    wrap_angle_deg,
)


def _fifa_skel(
    rs, ls, rh, lh, *, dim: int = 2, joints: FifaJointIndices | None = None
) -> np.ndarray:
    j = joints or FifaJointIndices()
    n_joints = max(j.left_hip, j.right_hip, j.left_shoulder, j.right_shoulder) + 1
    skel = np.zeros((n_joints, dim), dtype=float)
    skel[j.right_shoulder] = rs
    skel[j.left_shoulder] = ls
    skel[j.right_hip] = rh
    skel[j.left_hip] = lh
    return skel


class TestWrapAngle:
    @pytest.mark.parametrize(
        "angle, expected",
        [
            (0.0, 0.0),
            (180.0, 180.0),
            (-180.0, 180.0),
            (270.0, -90.0),
            (-540.0, 180.0),
            (450.0, 90.0),
        ],
    )
    def test_wrap_scalar(self, angle: float, expected: float) -> None:
        assert wrap_angle_deg(angle) == pytest.approx(expected)

    def test_wrap_array(self) -> None:
        out = wrap_angle_deg(np.array([0.0, 360.0, -540.0, 270.0]))
        assert isinstance(out, np.ndarray)
        np.testing.assert_allclose(out, [0.0, 0.0, 180.0, -90.0])


class TestBodyOrientation2D:
    def test_facing_camera_image_y_down(self) -> None:
        skel = _fifa_skel(
            rs=[-0.2, 0.0],
            ls=[0.2, 0.0],
            rh=[-0.2, 1.0],
            lh=[0.2, 1.0],
        )
        assert body_orientation_2d(skel) == pytest.approx(-90.0, abs=1e-6)

    def test_facing_camera_math_axes(self) -> None:
        skel = _fifa_skel(rs=[-0.2, 0.0], ls=[0.2, 0.0], rh=[-0.2, -1.0], lh=[0.2, -1.0])
        assert body_orientation_2d(skel, image_y_down=False) == pytest.approx(-90.0, abs=1e-6)

    def test_facing_right_in_image(self) -> None:
        skel = _fifa_skel(rs=[0.0, 0.2], ls=[0.0, -0.2], rh=[0.0, 0.2], lh=[0.0, -0.2])
        assert body_orientation_2d(skel) == pytest.approx(0.0, abs=1e-6)

    def test_raises_on_degenerate(self) -> None:
        skel = _fifa_skel(rs=[0.0, 0.0], ls=[0.0, 0.0], rh=[0.0, 1.0], lh=[0.0, 1.0])
        with pytest.raises(ValueError, match="degenerate"):
            body_orientation_2d(skel)

    def test_raises_on_nan(self) -> None:
        skel = _fifa_skel(rs=[np.nan, 0.0], ls=[0.2, 0.0], rh=[-0.2, 1.0], lh=[0.2, 1.0])
        with pytest.raises(ValueError, match="NaN"):
            body_orientation_2d(skel)

    def test_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            body_orientation_2d(np.zeros((9, 3)))


class TestBodyOrientation3D:
    def test_upright_facing_plus_x(self) -> None:
        skel = _fifa_skel(
            rs=[0.0, -0.2, 1.5],
            ls=[0.0, 0.2, 1.5],
            rh=[0.0, -0.2, 0.9],
            lh=[0.0, 0.2, 0.9],
            dim=3,
        )
        out = body_orientation_3d(skel, up_axis="z")
        assert out["azimuth_deg"] == pytest.approx(0.0, abs=1e-6)
        assert out["elevation_deg"] == pytest.approx(0.0, abs=1e-6)

    def test_upright_facing_plus_y(self) -> None:
        skel = _fifa_skel(
            rs=[0.2, 0.0, 1.5],
            ls=[-0.2, 0.0, 1.5],
            rh=[0.2, 0.0, 0.9],
            lh=[-0.2, 0.0, 0.9],
            dim=3,
        )
        out = body_orientation_3d(skel, up_axis="z")
        assert out["azimuth_deg"] == pytest.approx(90.0, abs=1e-6)

    def test_lean_forward_positive_elevation(self) -> None:
        rs = [0.0, -0.2, 1.5]
        ls = [0.0, 0.2, 1.5]
        rh = [-0.5, -0.2, 0.9]
        lh = [-0.5, 0.2, 0.9]
        skel = _fifa_skel(rs=rs, ls=ls, rh=rh, lh=lh, dim=3)
        out = body_orientation_3d(skel, up_axis="z")
        expected = math.degrees(math.atan2(0.5, 0.6))
        assert out["azimuth_deg"] == pytest.approx(0.0, abs=1e-6)
        assert out["elevation_deg"] == pytest.approx(expected, abs=1e-6)

    def test_lean_backward_negative_elevation(self) -> None:
        rs = [-0.5, -0.2, 1.5]
        ls = [-0.5, 0.2, 1.5]
        rh = [0.0, -0.2, 0.9]
        lh = [0.0, 0.2, 0.9]
        skel = _fifa_skel(rs=rs, ls=ls, rh=rh, lh=lh, dim=3)
        out = body_orientation_3d(skel, up_axis="z")
        assert out["elevation_deg"] < 0.0

    def test_alternative_up_axis_y_consistency(self) -> None:
        skel_z = _fifa_skel(
            rs=[0.0, -0.2, 1.5],
            ls=[0.0, 0.2, 1.5],
            rh=[0.0, -0.2, 0.9],
            lh=[0.0, 0.2, 0.9],
            dim=3,
        )
        skel_y = _fifa_skel(
            rs=[0.0, 1.5, 0.2],
            ls=[0.0, 1.5, -0.2],
            rh=[0.0, 0.9, 0.2],
            lh=[0.0, 0.9, -0.2],
            dim=3,
        )
        az_z = body_orientation_3d(skel_z, up_axis="z")["azimuth_deg"]
        az_y = body_orientation_3d(skel_y, up_axis="y")["azimuth_deg"]
        assert az_z == pytest.approx(0.0, abs=1e-6)
        assert az_y == pytest.approx(0.0, abs=1e-6)
        elev_y = body_orientation_3d(skel_y, up_axis="y")["elevation_deg"]
        assert elev_y == pytest.approx(0.0, abs=1e-6)


class TestGoalGeometry:
    def test_centred_18m(self) -> None:
        out = goal_geometry(
            shooter_xy=[0.0, 0.0],
            goal_left_post_xy=[18.0, -3.66],
            goal_right_post_xy=[18.0, 3.66],
        )
        assert out["distance_m"] == pytest.approx(18.0, abs=1e-6)
        assert out["angle_to_goal_deg"] == pytest.approx(0.0, abs=1e-6)
        expected_aperture = 2.0 * math.degrees(math.atan2(3.66, 18.0))
        assert out["angle_to_posts_deg"] == pytest.approx(expected_aperture, abs=1e-6)

    def test_off_axis_left_of_goal(self) -> None:
        out = goal_geometry(
            shooter_xy=[10.0, 5.0],
            goal_left_post_xy=[18.0, -3.66],
            goal_right_post_xy=[18.0, 3.66],
        )
        expected = math.degrees(math.atan2(-5.0, 8.0))
        assert out["angle_to_goal_deg"] == pytest.approx(expected, abs=1e-6)
        assert out["angle_to_posts_deg"] > 0.0
        assert out["distance_m"] == pytest.approx(math.hypot(8.0, 5.0), abs=1e-6)

    def test_raises_on_zero_distance(self) -> None:
        with pytest.raises(ValueError, match="centre"):
            goal_geometry(
                shooter_xy=[18.0, 0.0],
                goal_left_post_xy=[18.0, -3.66],
                goal_right_post_xy=[18.0, 3.66],
            )

    def test_raises_on_post(self) -> None:
        with pytest.raises(ValueError, match="post"):
            goal_geometry(
                shooter_xy=[18.0, -3.66],
                goal_left_post_xy=[18.0, -3.66],
                goal_right_post_xy=[18.0, 3.66],
            )


class TestRelativeOrientation:
    def test_facing_goal_returns_zero(self) -> None:
        out = relative_orientation_to_goal(
            body_angle_deg=0.0,
            shooter_xy=[0.0, 0.0],
            goal_centre_xy=[18.0, 0.0],
        )
        assert out == pytest.approx(0.0, abs=1e-6)

    def test_back_to_goal_returns_180(self) -> None:
        out = relative_orientation_to_goal(
            body_angle_deg=180.0,
            shooter_xy=[0.0, 0.0],
            goal_centre_xy=[18.0, 0.0],
        )
        assert abs(out) == pytest.approx(180.0, abs=1e-6)

    def test_goal_on_right_returns_minus_90(self) -> None:
        out = relative_orientation_to_goal(
            body_angle_deg=0.0,
            shooter_xy=[0.0, 0.0],
            goal_centre_xy=[0.0, 18.0],
        )
        assert out == pytest.approx(-90.0, abs=1e-6)

    def test_wraps_correctly(self) -> None:
        out = relative_orientation_to_goal(
            body_angle_deg=170.0,
            shooter_xy=[0.0, 0.0],
            goal_centre_xy=[-1.0, 0.0],
        )
        assert out == pytest.approx(-10.0, abs=1e-6)

    def test_raises_on_zero_distance(self) -> None:
        with pytest.raises(ValueError, match="shooter"):
            relative_orientation_to_goal(
                body_angle_deg=0.0,
                shooter_xy=[1.0, 1.0],
                goal_centre_xy=[1.0, 1.0],
            )
