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

import pandas as pd
import pytest

from vaila.quickmeasure import (
    CALIBRATION_CLICKS,
    LIVE_MODE_KEYS,
    MEASURE_TYPES,
    CalibrationDraft,
    QuickMeasureCalibration,
    QuickMeasureError,
    QuickMeasureSession,
    Ref3dCalibrationDraft,
    finish_calibration_draft,
    format_result,
    main,
    measure_from_points_csv,
    needs_calibration,
    session_from_points_csv,
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
    assert set(MEASURE_TYPES) == {"distance", "area", "angle", "velocity", "acceleration"}


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


# ---------------------------------------------------------------------------
# Calibration-first: click-built calibrations (Kinovea style)
# ---------------------------------------------------------------------------


def test_line_calibration_from_clicks_scale_and_axes():
    # 100 px declared as 1 m -> 0.01 m/px, origin at the first click,
    # +x right and +y up (image rows grow downwards).
    calib = QuickMeasureCalibration.from_line_clicks((100, 100), (200, 100), 1.0)
    assert calib.kind == "line"
    assert math.isclose(calib.scale, 0.01, abs_tol=1e-12)
    assert calib.pixel_to_real(100, 100) == (0.0, 0.0)
    x, y = calib.pixel_to_real(200, 100)
    assert math.isclose(x, 1.0, abs_tol=1e-9)
    assert math.isclose(y, 0.0, abs_tol=1e-9)
    x, y = calib.pixel_to_real(100, 0)
    assert math.isclose(y, 1.0, abs_tol=1e-9)  # upwards in the image is +y


def test_line_calibration_rejects_degenerate_input():
    with pytest.raises(QuickMeasureError):
        QuickMeasureCalibration.from_line_clicks((10, 10), (10, 10), 1.0)
    with pytest.raises(QuickMeasureError):
        QuickMeasureCalibration.from_line_clicks((0, 0), (100, 0), 0.0)


def test_line_calibrated_session_distance():
    calib = QuickMeasureCalibration.from_line_clicks((0, 0), (100, 0), 2.0, unit_label="m")
    session = QuickMeasureSession(calibration=calib)
    session.add_point(0, 0, 0)
    session.add_point(0, 300, 400)  # 500 px -> 500 * 0.02 = 10 m
    result = session.measure("distance")
    assert result["unit"] == "m"
    assert math.isclose(result["value"], 10.0, abs_tol=1e-6)


def test_plane_calibration_from_clicks_matches_frozen_dlt2d():
    # Same square as the frozen fixture: pixel_u = 100X + 100, pixel_v = 100Y + 100.
    calib = QuickMeasureCalibration.from_plane_clicks(
        [(100, 100), (200, 100), (200, 200), (100, 200)], 1.0, 1.0
    )
    assert calib.kind == "plane"
    expected = [100, 0, 100, 0, 100, 100, 0, 0]
    for got, want in zip(calib.dlt_params.tolist(), expected, strict=True):
        assert math.isclose(got, want, abs_tol=1e-3)
    x, y = calib.pixel_to_real(150, 150)
    assert math.isclose(x, 0.5, abs_tol=1e-3)
    assert math.isclose(y, 0.5, abs_tol=1e-3)


def test_plane_calibration_rejects_bad_input():
    with pytest.raises(QuickMeasureError):
        QuickMeasureCalibration.from_plane_clicks([(0, 0), (1, 0), (1, 1)], 1.0, 1.0)
    with pytest.raises(QuickMeasureError):
        QuickMeasureCalibration.from_plane_clicks([(0, 0), (1, 0), (1, 1), (0, 1)], 1.0, -2.0)


# ---------------------------------------------------------------------------
# CalibrationDraft — the click-collection state machine
# ---------------------------------------------------------------------------


def test_calibration_draft_line_flow():
    draft = CalibrationDraft(mode="line", unit_label="cm")
    assert draft.required_points == CALIBRATION_CLICKS["line"] == 2
    assert draft.required_measures == ("length",)
    assert not draft.is_complete
    assert "CALIBRATION (line) 1/2" in draft.instructions()
    draft.add_point(0, 0)
    assert draft.remaining == 1
    draft.add_point(100, 0)
    assert draft.is_complete
    assert "type the real length" in draft.instructions()
    calib = draft.build([50.0])
    assert calib.unit_label == "cm"
    assert math.isclose(calib.scale, 0.5, abs_tol=1e-12)


def test_calibration_draft_plane_requires_four_points_and_two_measures():
    draft = CalibrationDraft(mode="plane")
    assert draft.required_points == 4
    for x, y in [(100, 100), (200, 100), (200, 200), (100, 200)]:
        draft.add_point(x, y)
    with pytest.raises(QuickMeasureError):
        draft.build([1.0])  # needs width AND height
    calib = draft.build([1.0, 1.0])
    assert calib.kind == "plane"
    assert calib.real_measures == {"width": 1.0, "height": 1.0}


def test_calibration_draft_undo_and_overflow_and_unknown_mode():
    draft = CalibrationDraft(mode="line")
    assert draft.undo_last() is False
    draft.add_point(1, 1)
    assert draft.undo_last() is True
    draft.add_point(0, 0)
    draft.add_point(10, 0)
    with pytest.raises(QuickMeasureError):
        draft.add_point(20, 0)
    with pytest.raises(QuickMeasureError):
        CalibrationDraft(mode="banana")


def test_finish_calibration_draft_prompts_and_builds():
    draft = CalibrationDraft(mode="line")
    draft.add_point(0, 0)
    draft.add_point(200, 0)
    calib, message = finish_calibration_draft(draft, lambda prompt, default: "2")
    assert calib is not None
    assert math.isclose(calib.scale, 0.01, abs_tol=1e-12)
    assert "Line calibration" in message


def test_finish_calibration_draft_cancel_and_invalid_and_incomplete():
    draft = CalibrationDraft(mode="line")
    draft.add_point(0, 0)
    calib, message = finish_calibration_draft(draft, lambda prompt, default: "1")
    assert calib is None and "more click" in message

    draft.add_point(100, 0)
    calib, message = finish_calibration_draft(draft, lambda prompt, default: None)
    assert calib is None and message == "Calibration cancelled."

    calib, message = finish_calibration_draft(draft, lambda prompt, default: "abc")
    assert calib is None and "Invalid length" in message


# ---------------------------------------------------------------------------
# CSV persistence and recomputation from the saved calibrated points file
# ---------------------------------------------------------------------------


def test_save_session_writes_points_calibration_and_results(tmp_path):
    calib = QuickMeasureCalibration.from_line_clicks((100, 100), (200, 100), 1.0)
    session = QuickMeasureSession(fps=10.0, calibration=calib)
    session.add_point(0, 100, 100)
    session.add_point(10, 400, 100)
    result = session.measure("distance")
    assert math.isclose(result["value"], 3.0, abs_tol=1e-9)
    assert result["point_ids"] == [1, 2]
    assert result["frames"] == [0, 10]

    paths = session.save_session(str(tmp_path), stem="clip.mp4")
    assert os.path.isdir(paths["dir"])
    points = pd.read_csv(paths["points"])
    assert list(points.columns) == [
        "point_id",
        "frame",
        "x_px",
        "y_px",
        "x_real",
        "y_real",
        "unit",
        "calibration_kind",
    ]
    assert points.loc[1, "x_real"] == pytest.approx(3.0)
    assert set(points["unit"]) == {"m"}
    calib_df = pd.read_csv(paths["calibration"])
    assert (calib_df["kind"] == "line").all()
    assert calib_df["measure_value"].dropna().tolist() == [1.0]
    results_df = pd.read_csv(paths["results"])
    assert results_df.loc[0, "type"] == "distance"
    assert results_df.loc[0, "value"] == pytest.approx(3.0)
    assert os.path.isfile(paths["report"])
    with open(paths["report"], encoding="utf-8") as handle:
        report = handle.read()
    assert "What each measurement means" in report
    assert "Results CSV column dictionary" in report
    assert "point_N_id" in report
    assert "No list, coordinate pair, or space-delimited sequence" in report
    assert "Distance" in report


def test_save_session_reuses_one_directory_and_refreshes_report(tmp_path):
    session = QuickMeasureSession(fps=10.0)
    session.calibration_skipped = True
    session.add_point(0, 0, 0)
    session.add_point(0, 3, 4)
    session.measure("distance")

    first = session.save_session(str(tmp_path), stem="clip.mp4")
    session.add_point(0, 0, 0)
    session.add_point(10, 0, 20)
    session.measure("velocity")
    second = session.save_session(str(tmp_path), stem="clip.mp4")

    assert second["dir"] == first["dir"]
    assert len(list(tmp_path.glob("processed_quickmeasure_*"))) == 1
    results = pd.read_csv(second["results"])
    assert results["type"].tolist() == ["distance", "velocity"]
    velocity = results.loc[results["type"] == "velocity"].iloc[0]
    assert velocity["elapsed_time_s"] == pytest.approx(1.0)
    assert velocity["point_1_frame"] == 0
    assert velocity["point_2_frame"] == 10
    with open(second["report"], encoding="utf-8") as handle:
        report = handle.read()
    assert "Completed results:</strong> 2" in report
    assert "Velocity" in report
    assert "distance / ((frame2-frame1)/fps)" in report


def test_save_session_refuses_empty_session(tmp_path):
    with pytest.raises(QuickMeasureError):
        QuickMeasureSession().save_session(str(tmp_path))


def test_measure_from_saved_points_csv_roundtrip(tmp_path):
    calib = QuickMeasureCalibration.from_line_clicks((0, 0), (100, 0), 1.0)
    session = QuickMeasureSession(fps=10.0, calibration=calib)
    session.add_point(0, 0, 0)
    session.add_point(10, 300, -400)  # real (3, 4) -> distance 5 m, 1 s apart
    paths = session.save_session(str(tmp_path), stem="clip")

    reloaded = session_from_points_csv(paths["points"], fps=10.0)
    assert reloaded.unit_label == "m"
    distance = reloaded.measure("distance")
    assert math.isclose(distance["value"], 5.0, abs_tol=1e-6)
    velocity = measure_from_points_csv(paths["points"], "velocity", fps=10.0)
    assert math.isclose(velocity["value"], 5.0, abs_tol=1e-6)
    assert velocity["unit"] == "m/s"


def test_measure_from_points_csv_fixture_and_point_ids():
    fixture = _fixture("points_calibrated.csv")
    result = measure_from_points_csv(fixture, "distance")
    assert result["unit"] == "m"
    assert math.isclose(result["value"], 5.0, abs_tol=1e-9)
    subset = measure_from_points_csv(fixture, "distance", point_ids=[1, 2])
    assert math.isclose(subset["value"], 5.0, abs_tol=1e-9)
    with pytest.raises(QuickMeasureError):
        measure_from_points_csv(fixture, "distance", point_ids=[1, 99])


def test_session_from_points_csv_errors(tmp_path):
    with pytest.raises(QuickMeasureError):
        session_from_points_csv(str(tmp_path / "missing.csv"))
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(bad, index=False)
    with pytest.raises(QuickMeasureError):
        session_from_points_csv(str(bad))


def test_cli_main_prints_measurement_and_appends(tmp_path, capsys):
    out_csv = tmp_path / "results.csv"
    code = main(
        [
            "--points-csv",
            _fixture("points_calibrated.csv"),
            "--measure",
            "distance",
            "--out",
            str(out_csv),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "Distance: 5.0000 m" in printed
    assert pd.read_csv(out_csv).loc[0, "value"] == pytest.approx(5.0)


def test_cli_main_reports_error_exit_code(tmp_path, capsys):
    code = main(["--points-csv", str(tmp_path / "nope.csv"), "--measure", "distance"])
    assert code == 2
    assert "error" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Calibration-first rule + end-to-end flow simulation (no pygame)
# ---------------------------------------------------------------------------


def test_needs_calibration_rule():
    assert needs_calibration(None) is True
    session = QuickMeasureSession()
    assert needs_calibration(session) is True  # brand new -> calibrate first
    session.calibration_skipped = True
    assert needs_calibration(session) is False  # user chose pixels explicitly
    calibrated = QuickMeasureSession(
        calibration=QuickMeasureCalibration.from_line_clicks((0, 0), (10, 0), 1.0)
    )
    assert needs_calibration(calibrated) is False


def test_calibration_first_then_free_measure_flow(tmp_path):
    """Simulates what Q / QMeas does: calibrate first, then measure, then save."""
    session = QuickMeasureSession(fps=30.0)
    assert needs_calibration(session)

    # Step 1: collect calibration clicks (a 200 px bar declared as 2 m).
    draft = CalibrationDraft(mode="line", unit_label="m")
    draft.add_point(100, 500)
    draft.add_point(300, 500)
    calib, message = finish_calibration_draft(draft, lambda prompt, default: "2")
    assert calib is not None and "Line calibration" in message
    session.calibration = calib
    assert not needs_calibration(session)

    # Step 2: free measuring in calibrated units (0.01 m/px).
    session.add_point(0, 100, 500)
    session.add_point(30, 700, 500)  # 600 px -> 6 m, 1 s later
    assert math.isclose(session.measure("distance")["value"], 6.0, abs_tol=1e-9)
    assert math.isclose(session.measure("velocity")["value"], 6.0, abs_tol=1e-9)

    # Step 3: persist, then recompute straight from the saved points CSV.
    paths = session.save_session(str(tmp_path), stem="clip.mp4")
    assert math.isclose(
        measure_from_points_csv(paths["points"], "distance")["value"], 6.0, abs_tol=1e-6
    )
    results_df = pd.read_csv(paths["results"])
    assert results_df["type"].tolist() == ["distance", "velocity"]


def test_getpixelvideo_toggle_is_calibration_first():
    """The host toggle must route an uncalibrated session into calibration."""
    source_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "vaila",
        "getpixelvideo.py",
    )
    with open(source_path, encoding="utf-8") as handle:
        source = handle.read()
    toggle_start = source.index("def _toggle_quick_measure_mode()")
    toggle_body = source[toggle_start : toggle_start + 2000]
    assert "quickmeasure.needs_calibration(quickmeasure_session)" in toggle_body
    assert "_start_quickmeasure_calibration()" in toggle_body
    # Calibration clicks must be routed before measure clicks.
    assert "quick_measure_calibrating" in source
    assert "quickmeasure_draft.add_point(video_x, video_y)" in source
    assert "quickmeasure_session.add_live_point" in source
    assert source.index("quickmeasure_draft.add_point(video_x, video_y)") < source.index(
        "quickmeasure_session.add_live_point"
    )
    assert "3=REF3D" in source or "REF3D file" in source
    assert "set_live_mode" in source
    assert "add_live_point" in source


def test_session_angle_three_points_right_angle():
    session = QuickMeasureSession()
    session.add_point(0, 1, 0)
    session.add_point(0, 0, 0)
    session.add_point(0, 0, 1)
    result = session.measure("angle")
    assert result["unit"] == "deg"
    assert math.isclose(result["value"], 90.0, abs_tol=1e-6)
    assert result["n_points"] == 3


def test_session_angle_four_points_two_lines():
    session = QuickMeasureSession()
    # Horizontal then vertical → 90 deg.
    session.add_point(0, 0, 0)
    session.add_point(0, 2, 0)
    session.add_point(0, 0, 0)
    session.add_point(0, 0, 3)
    result = session.measure("angle")
    assert math.isclose(result["value"], 90.0, abs_tol=1e-6)
    assert result["n_points"] == 4


def test_ref3d_modes_detected_and_mode2_matches_mode3():
    from vaila.dlt3d import detect_ref3d_format, normalize_ref3d_to_format1

    assert detect_ref3d_format(_fixture("ref3d_mode1.ref3d")) == 1
    assert detect_ref3d_format(_fixture("ref3d_mode2.ref3d")) == 4
    assert detect_ref3d_format(_fixture("ref3d_mode3.ref3d")) == 2
    df2 = normalize_ref3d_to_format1(_fixture("ref3d_mode2.ref3d"))
    df3 = normalize_ref3d_to_format1(_fixture("ref3d_mode3.ref3d"))
    assert df2 is not None and df3 is not None
    pd.testing.assert_frame_equal(df2, df3)


def test_calibration_from_ref3d_and_pixel_csv_drop_x():
    calib = QuickMeasureCalibration.from_ref3d_and_pixel_csv(
        _fixture("ref3d_mode1.ref3d"),
        _fixture("c1_cod_markers_1_line.csv"),
        drop_axis="x",
    )
    assert calib.kind == "ref3d"
    assert calib.drop_axis == "x"
    assert calib.dlt_params is not None
    assert len(calib.kept_point_indices) >= 4
    # p1 world (0,0,0.07) → YZ (0, 0.07); reconstructed from its pixel should be close.
    pix = pd.read_csv(_fixture("c1_cod_markers_1_line.csv")).iloc[0]
    y, z = calib.pixel_to_real(float(pix["p1_x"]), float(pix["p1_y"]))
    assert math.isclose(y, 0.0, abs_tol=0.15)
    assert math.isclose(z, 0.07, abs_tol=0.25)


def test_ref3d_calibration_draft_guide_build():
    draft = Ref3dCalibrationDraft.from_ref3d_file(_fixture("ref3d_mode1.ref3d"), drop_axis="x")
    assert draft.required_points >= 4
    assert not draft.is_complete
    pix = pd.read_csv(_fixture("c1_cod_markers_1_line.csv")).iloc[0]
    for label in draft.kept_labels:
        draft.add_point(float(pix[f"p{label}_x"]), float(pix[f"p{label}_y"]))
    assert draft.is_complete
    calib = draft.build()
    assert calib.kind == "ref3d"
    assert calib.drop_axis == "x"


def test_ref3d_save_session_writes_dlt2d_and_ref2d(tmp_path):
    calib = QuickMeasureCalibration.from_ref3d_and_pixel_csv(
        _fixture("ref3d_mode1.ref3d"),
        _fixture("c1_cod_markers_1_line.csv"),
        drop_axis="x",
    )
    session = QuickMeasureSession(fps=30.0, calibration=calib)
    session.add_point(0, 100, 100)
    session.add_point(0, 200, 100)
    session.measure("distance")
    paths = session.save_session(str(tmp_path), stem="c1_cod.mp4")
    assert os.path.isfile(paths["dlt2d"])
    assert os.path.isfile(paths["ref2d"])
    assert os.path.isfile(paths["ref3d_meta"])
    assert os.path.isfile(paths["readme"])
    assert os.path.isfile(paths["results"])


def test_live_mode_keys_mapping():
    assert LIVE_MODE_KEYS == {
        "1": "distance",
        "2": "area",
        "3": "angle",
        "4": "velocity",
        "5": "acceleration",
    }


def test_live_mode_distance_auto_finalizes_and_labels_geometry():
    session = QuickMeasureSession()
    msg = session.set_live_mode("1")
    assert "distance" in msg
    status, done = session.add_live_point(0, 0, 0)
    assert done is None
    assert "1/2" in status
    status, done = session.add_live_point(0, 3, 4)
    assert done is not None
    assert done["type"] == "distance"
    assert math.isclose(done["value"], 5.0, abs_tol=1e-9)
    assert done["pixels"] == [(0.0, 0.0), (3.0, 4.0)]
    assert done["result_frame"] == 0
    assert len(session.draft_points) == 0
    # Second pair accumulates another result.
    session.add_live_point(0, 10, 10)
    _, done2 = session.add_live_point(0, 10, 20)
    assert done2 is not None
    assert len(session.results) == 2


def test_live_mode_area_needs_enter():
    session = QuickMeasureSession()
    session.set_live_mode("2")
    session.add_live_point(0, 0, 0)
    session.add_live_point(0, 4, 0)
    session.add_live_point(0, 4, 3)
    assert len(session.results) == 0
    msg, done = session.finalize_live_area(0)
    assert done is not None
    assert math.isclose(done["value"], 6.0, abs_tol=1e-9)  # triangle 4*3/2
    assert "Area" in msg or "area" in msg.lower() or "6" in msg


def test_live_mode_velocity_requires_fps_and_two_frames():
    session = QuickMeasureSession()
    with pytest.raises(QuickMeasureError, match="FPS"):
        session.set_live_mode("4")
    session.fps = 10.0
    session.set_live_mode("4")
    session.add_live_point(0, 0, 0)
    with pytest.raises(QuickMeasureError, match="different frames"):
        session.add_live_point(0, 3, 4)
    session.draft_points.clear()
    session.add_live_point(0, 0, 0)
    _, done = session.add_live_point(10, 3, 4)
    assert done is not None
    assert math.isclose(done["value"], 5.0, abs_tol=1e-9)
    assert done["unit"] == "px/s"


def test_live_mode_reserved_key():
    session = QuickMeasureSession()
    msg = session.set_live_mode("9")
    assert "reserved" in msg.lower()
    assert session.active_mode is None


def test_live_mode_save_writes_per_type_results(tmp_path):
    session = QuickMeasureSession(fps=30.0)
    session.set_live_mode("1")
    session.add_live_point(5, 0, 0)
    session.add_live_point(5, 3, 4)
    session.set_live_mode("3")
    session.add_live_point(5, 1, 0)
    session.add_live_point(5, 0, 0)
    session.add_live_point(5, 0, 1)
    paths = session.save_session(str(tmp_path), stem="clip.mp4")
    assert os.path.isfile(paths["results"])
    assert os.path.isfile(paths["results_distance"])
    assert os.path.isfile(paths["results_angle"])
    df = pd.read_csv(paths["results"])
    assert "result_frame" in df.columns
    assert set(df["type"]) == {"distance", "angle"}
    assert {"frames", "point_ids", "pixels", "reals"}.isdisjoint(df.columns)
    assert {
        "point_1_id",
        "point_1_frame",
        "point_1_x_px",
        "point_1_y_px",
        "point_1_x_real",
        "point_1_y_real",
        "point_3_id",
    }.issubset(df.columns)
    distance_df = pd.read_csv(paths["results_distance"])
    angle_df = pd.read_csv(paths["results_angle"])
    assert distance_df.columns.tolist() == df.columns.tolist()
    assert angle_df.columns.tolist() == df.columns.tolist()
    matrix_columns = [
        column
        for column in df.columns
        if column.startswith("point_")
        or column in {"result_id", "value", "n_points", "result_frame", "fps", "elapsed_time_s"}
    ]
    assert all(pd.api.types.is_numeric_dtype(df[column]) for column in matrix_columns)
    assert pd.isna(distance_df.loc[0, "point_3_id"])
    with open(paths["results"], encoding="utf-8") as handle:
        csv_text = handle.read()
    assert '"' not in csv_text
    assert "0.000,0.000" not in csv_text


def test_results_dataframe_expands_area_vertices_as_scalar_columns():
    session = QuickMeasureSession()
    for x, y in [(0, 0), (4, 0), (5, 2), (2, 5), (0, 3)]:
        session.add_point(7, x, y)
    session.measure("area")

    df = session.results_dataframe()

    assert len(df) == 1
    assert "point_5_frame" in df.columns
    assert df.loc[0, "point_5_x_px"] == pytest.approx(0.0)
    assert df.loc[0, "point_5_y_px"] == pytest.approx(3.0)
    assert df.loc[0, "point_5_x_real"] == pytest.approx(0.0)
    assert df.loc[0, "point_5_y_real"] == pytest.approx(3.0)
    assert all(not isinstance(value, (list, tuple)) for value in df.iloc[0])
