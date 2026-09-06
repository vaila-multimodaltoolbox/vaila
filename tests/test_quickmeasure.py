"""
Tests for vaila/quickmeasure.py — Kinovea-style quick on-image measurements
for getpixelvideo.py (Distance / Area / Velocity / Acceleration), with and
without a DLT2D calibration.

Fixtures live in tests/fixtures/quickmeasure/:
    - calib_pixels.csv + calib_ref.ref2d: a 4-point pixel/real square whose
      DLT2D fit is the well-known affine solution [100,0,100,0,100,100,0,0]
      (see tests/test_dlt_rec.py::test_dlt2d_basic / test_rec2d_basic for the
      same numbers, hand-derivable from pixel_u = 100*X + 100,
      pixel_v = 100*Y + 100).
    - sample.dlt2d: that same solved 8-parameter DLT2D file.
    - sample_bad_paramcount.dlt2d: malformed (6 params instead of 8).
    - calib_pixels_insufficient.csv + calib_ref_insufficient.ref2d: only 2
      common calibration points (need >= 4) -> NaN DLT2D parameters.
"""

import math
import os

import pytest

from vaila.quickmeasure import (
    MEASURE_TYPES,
    QuickMeasureCalibration,
    QuickMeasureError,
    QuickMeasureSession,
    format_result,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "quickmeasure")


def _fixture(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)


# ---------------------------------------------------------------------------
# QuickMeasureCalibration
# ---------------------------------------------------------------------------


def test_calibration_from_dlt2d_file():
    calib = QuickMeasureCalibration.from_dlt2d_file(_fixture("sample.dlt2d"))
    assert calib.dlt_params.shape == (8,)
    assert calib.unit_label == "m"
    x, y = calib.pixel_to_real(150, 150)
    assert math.isclose(x, 0.5, abs_tol=1e-3)
    assert math.isclose(y, 0.5, abs_tol=1e-3)


def test_calibration_from_dlt2d_file_missing_raises():
    with pytest.raises(QuickMeasureError):
        QuickMeasureCalibration.from_dlt2d_file(_fixture("does_not_exist.dlt2d"))


def test_calibration_from_dlt2d_file_bad_param_count_raises():
    with pytest.raises(QuickMeasureError):
        QuickMeasureCalibration.from_dlt2d_file(_fixture("sample_bad_paramcount.dlt2d"))


def test_calibration_from_calibration_points():
    calib = QuickMeasureCalibration.from_calibration_points(
        _fixture("calib_pixels.csv"), _fixture("calib_ref.ref2d")
    )
    # Matches the hand-derived affine solution from test_dlt_rec.py.
    expected = [100, 0, 100, 0, 100, 100, 0, 0]
    for got, want in zip(calib.dlt_params.tolist(), expected, strict=True):
        assert math.isclose(got, want, abs_tol=1e-3)
    x, y = calib.pixel_to_real(200, 200)
    assert math.isclose(x, 1.0, abs_tol=1e-3)
    assert math.isclose(y, 1.0, abs_tol=1e-3)


def test_calibration_from_calibration_points_missing_file_raises():
    with pytest.raises(QuickMeasureError):
        QuickMeasureCalibration.from_calibration_points(
            _fixture("does_not_exist.csv"), _fixture("calib_ref.ref2d")
        )
    with pytest.raises(QuickMeasureError):
        QuickMeasureCalibration.from_calibration_points(
            _fixture("calib_pixels.csv"), _fixture("does_not_exist.ref2d")
        )


def test_calibration_from_calibration_points_insufficient_points_raises():
    # Only 2 common points (< 4 required) -> dlt2d() fills NaN params.
    with pytest.raises(QuickMeasureError):
        QuickMeasureCalibration.from_calibration_points(
            _fixture("calib_pixels_insufficient.csv"),
            _fixture("calib_ref_insufficient.ref2d"),
        )


# ---------------------------------------------------------------------------
# QuickMeasureSession — uncalibrated (pixel units)
# ---------------------------------------------------------------------------


def test_session_distance_uncalibrated():
    session = QuickMeasureSession()
    session.add_point(0, 0, 0)
    session.add_point(0, 3, 4)
    result = session.measure("distance")
    assert result["unit"] == "px"
    assert math.isclose(result["value"], 5.0, abs_tol=1e-3)


def test_session_distance_needs_two_points():
    session = QuickMeasureSession()
    session.add_point(0, 0, 0)
    with pytest.raises(QuickMeasureError):
        session.measure("distance")


def test_session_area_uncalibrated_square():
    session = QuickMeasureSession()
    for x, y in [(0, 0), (4, 0), (4, 3), (0, 3)]:
        session.add_point(0, x, y)
    result = session.measure("area")
    assert result["unit"] == "px^2"
    assert math.isclose(result["value"], 12.0, abs_tol=1e-3)
    assert result["n_points"] == 4


def test_session_area_needs_three_points():
    session = QuickMeasureSession()
    session.add_point(0, 0, 0)
    session.add_point(0, 1, 0)
    with pytest.raises(QuickMeasureError):
        session.measure("area")


def test_session_velocity_uncalibrated():
    session = QuickMeasureSession(fps=10.0)
    session.add_point(0, 0, 0)
    session.add_point(10, 3, 4)
    result = session.measure("velocity")
    # dt = (10 - 0) / 10 fps = 1s; distance = 5 px -> 5 px/s.
    assert result["unit"] == "px/s"
    assert math.isclose(result["dt"], 1.0, abs_tol=1e-9)
    assert math.isclose(result["value"], 5.0, abs_tol=1e-3)


def test_session_velocity_requires_fps():
    session = QuickMeasureSession()
    session.add_point(0, 0, 0)
    session.add_point(1, 3, 4)
    with pytest.raises(QuickMeasureError):
        session.measure("velocity")


def test_session_velocity_requires_distinct_frames():
    session = QuickMeasureSession(fps=30.0)
    session.add_point(5, 0, 0)
    session.add_point(5, 3, 4)
    with pytest.raises(QuickMeasureError):
        session.measure("velocity")


def test_session_acceleration_uncalibrated():
    session = QuickMeasureSession(fps=1.0)
    session.add_point(0, 0, 0)
    session.add_point(1, 0, 5)
    session.add_point(2, 0, 15)
    result = session.measure("acceleration")
    # v1 = 5 px/s (frame 0->1), v2 = 10 px/s (frame 1->2), dt_avg = 1s
    # -> (10 - 5) / 1 = 5 px/s^2.
    assert result["unit"] == "px/s^2"
    assert math.isclose(result["value"], 5.0, abs_tol=1e-3)


def test_session_acceleration_requires_three_distinct_frames():
    session = QuickMeasureSession(fps=1.0)
    session.add_point(0, 0, 0)
    session.add_point(0, 0, 5)
    session.add_point(1, 0, 15)
    with pytest.raises(QuickMeasureError):
        session.measure("acceleration")


def test_session_undo_and_clear():
    session = QuickMeasureSession()
    session.add_point(0, 0, 0)
    session.add_point(0, 1, 1)
    assert session.undo_last() is True
    assert len(session.points) == 1
    session.clear()
    assert session.points == []
    assert session.undo_last() is False


def test_session_measure_unknown_kind_raises():
    session = QuickMeasureSession()
    session.add_point(0, 0, 0)
    session.add_point(0, 1, 1)
    with pytest.raises(QuickMeasureError):
        session.measure("banana")
    assert set(MEASURE_TYPES) == {"distance", "area", "velocity", "acceleration"}


# ---------------------------------------------------------------------------
# QuickMeasureSession — calibrated (DLT2D loaded)
# ---------------------------------------------------------------------------


def _calibrated_session(**kwargs) -> QuickMeasureSession:
    calib = QuickMeasureCalibration.from_dlt2d_file(_fixture("sample.dlt2d"))
    return QuickMeasureSession(calibration=calib, **kwargs)


def test_session_distance_calibrated():
    session = _calibrated_session()
    session.add_point(0, 100, 100)  # real (0, 0)
    session.add_point(0, 200, 200)  # real (1, 1)
    result = session.measure("distance")
    assert result["unit"] == "m"
    assert math.isclose(result["value"], math.sqrt(2), abs_tol=1e-3)


def test_session_area_calibrated_unit_square():
    session = _calibrated_session()
    for x, y in [(100, 100), (200, 100), (200, 200), (100, 200)]:
        session.add_point(0, x, y)
    result = session.measure("area")
    assert result["unit"] == "m^2"
    assert math.isclose(result["value"], 1.0, abs_tol=1e-3)


def test_session_velocity_calibrated():
    session = _calibrated_session(fps=5.0)
    session.add_point(0, 100, 100)  # real (0, 0)
    session.add_point(5, 200, 100)  # real (1, 0), 1s later
    result = session.measure("velocity")
    assert result["unit"] == "m/s"
    assert math.isclose(result["value"], 1.0, abs_tol=1e-3)


def test_session_acceleration_calibrated():
    session = _calibrated_session(fps=1.0)
    # Real world: (0,0)@f0, (0,1)@f1, (0,3)@f2 -> pixel_v = 100*Y + 100.
    session.add_point(0, 100, 100)
    session.add_point(1, 100, 200)
    session.add_point(2, 100, 400)
    result = session.measure("acceleration")
    # v1 = 1 m/s, v2 = 2 m/s, dt_avg = 1s -> 1 m/s^2.
    assert result["unit"] == "m/s^2"
    assert math.isclose(result["value"], 1.0, abs_tol=1e-3)


# ---------------------------------------------------------------------------
# format_result
# ---------------------------------------------------------------------------


def test_format_result():
    session = QuickMeasureSession()
    session.add_point(0, 0, 0)
    session.add_point(0, 3, 4)
    text = format_result(session.measure("distance"))
    assert text == "Distance: 5.0000 px"
