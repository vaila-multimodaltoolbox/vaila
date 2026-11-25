"""
Version: 0.4.19
Update Date: 23 November 2026

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

import numpy as np
import pandas as pd
import pytest

from vaila.kick_measure import CameraTrack
from vaila.quickmeasure import (
    CALIBRATION_CLICKS,
    LIVE_MODE_KEYS,
    MEASURE_TYPES,
    CalibrationDraft,
    QuickMeasureCalibration,
    QuickMeasureError,
    QuickMeasurePoint,
    QuickMeasureSession,
    Ref3dCalibrationDraft,
    draw_calibration_overlay,
    finish_calibration_draft,
    format_hover_coords,
    format_result,
    main,
    measure_from_points_csv,
    needs_calibration,
    session_from_points_csv,
    snap_level_endpoint,
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
    assert set(MEASURE_TYPES) == {
        "distance",
        "area",
        "angle",
        "velocity",
        "acceleration",
        "level",
    }


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


def test_format_hover_coords_pixel_only_without_calibration():
    session = QuickMeasureSession()
    text = format_hover_coords(session, 320.4, 240.6)
    assert text == "Pix: (320, 241)"
    assert "Real:" not in text
    assert format_hover_coords(None, 10, 20) == "Pix: (10, 20)"


def test_format_hover_coords_includes_real_when_calibrated():
    # 100 px = 1 m → 0.01 m/px; origin at first click.
    calib = QuickMeasureCalibration.from_line_clicks((100, 100), (200, 100), 1.0, unit_label="m")
    session = QuickMeasureSession(calibration=calib)
    text = format_hover_coords(session, 200, 100)
    assert text.startswith("Pix: (200, 100)")
    assert "Real: (1.000, 0.000) m" in text


def test_format_hover_coords_respects_mm_unit():
    calib = QuickMeasureCalibration.from_line_clicks((0, 0), (100, 0), 100.0, unit_label="mm")
    session = QuickMeasureSession(calibration=calib)
    text = format_hover_coords(session, 50, 0)
    assert "Real: (50.000, 0.000) mm" in text


def test_format_hover_coords_uses_frame_override():
    default = QuickMeasureCalibration.from_line_clicks((0, 0), (100, 0), 1.0, unit_label="m")
    override = QuickMeasureCalibration.from_line_clicks((0, 0), (50, 0), 1.0, unit_label="m")
    session = QuickMeasureSession(calibration=default)
    session.set_calibration(override, frame=3)
    text = format_hover_coords(session, 50, 0, frame=3)
    assert "Real: (1.000, 0.000) m" in text
    text0 = format_hover_coords(session, 50, 0, frame=0)
    assert "Real: (0.500, 0.000) m" in text0


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


def test_calibration_draft_visual_semantics_and_review_feedback():
    line = CalibrationDraft(mode="line")
    assert line.point_roles == ("START", "END")
    assert line.edge_semantics == ((0, 1, "LENGTH"),)
    assert line.guide_lines()[1] == "Next: 1 START"
    line.add_point(0, 0)
    assert line.guide_lines()[1] == "Next: 2 END"
    line.add_point(10, 0)
    assert "Geometry complete" in line.guide_lines()[1]
    assert "press Enter" in line.instructions()

    plane = CalibrationDraft(mode="plane")
    assert plane.point_roles == ("ORIGIN", "WIDTH", "OPPOSITE", "CLOSE")
    assert plane.edge_semantics == (
        (0, 1, "WIDTH"),
        (1, 2, "HEIGHT"),
        (2, 3, "WIDTH"),
        (3, 0, "HEIGHT"),
    )


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


def test_finish_plane_calibration_prompts_name_the_labeled_edges():
    draft = CalibrationDraft(mode="plane", unit_label="m")
    for point in [(0, 0), (100, 0), (100, 50), (0, 50)]:
        draft.add_point(*point)
    prompts = []

    def answer(prompt, default):
        prompts.append(prompt)
        return "2" if len(prompts) == 1 else "1"

    calib, _message = finish_calibration_draft(draft, answer)
    assert calib is not None
    assert "WIDTH of edges 1-2 and 4-3" in prompts[0]
    assert "HEIGHT of edges 2-3 and 1-4" in prompts[1]


def test_draw_calibration_overlay_supports_every_line_and_plane_stage():
    pygame = pytest.importorskip("pygame")
    pygame.font.init()
    font = pygame.font.Font(None, 18)

    for mode, points in (
        ("line", [(100, 120), (300, 120)]),
        ("plane", [(100, 120), (300, 120), (300, 280), (100, 280)]),
    ):
        draft = CalibrationDraft(mode=mode)
        for stage in range(len(points) + 1):
            surface = pygame.Surface((800, 500))
            surface.fill((0, 0, 0))
            draw_calibration_overlay(
                surface,
                draft,
                1.0,
                0,
                0,
                font,
                cursor_pos=(420, 320),
            )
            assert surface.get_bounding_rect().width > 0
            if stage < len(points):
                draft.add_point(*points[stage])


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
        "calibration_frame",
    ]
    assert (points["calibration_frame"] == -1).all()
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
    assert "<svg" in report
    assert "Calibration figures" in report
    assert "Measurement figures" in report
    assert "Picture of this session" in report


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
    assert needs_calibration(session) is True  # brand new — no calib yet
    session.calibration_skipped = True
    # Skip alone does not count as calibrated; MEASURE may still run in px.
    assert needs_calibration(session) is True
    calibrated = QuickMeasureSession(
        calibration=QuickMeasureCalibration.from_line_clicks((0, 0), (10, 0), 1.0)
    )
    assert needs_calibration(calibrated) is False


def test_calibration_for_frame_fallback_and_override():
    default = QuickMeasureCalibration.from_line_clicks((0, 0), (100, 0), 1.0, unit_label="m")
    override = QuickMeasureCalibration.from_line_clicks((0, 0), (50, 0), 1.0, unit_label="m")
    session = QuickMeasureSession()
    session.set_calibration(default)  # default / all frames
    session.set_calibration(override, frame=10)
    assert session.calibration_for_frame(0) is default
    assert session.calibration_for_frame(10) is override
    # Measure on frame 10 uses override scale (0.02 m/px): 50 px → 1 m.
    session.add_point(10, 0, 0)
    session.add_point(10, 50, 0)
    result = session.measure("distance")
    assert math.isclose(result["value"], 1.0, abs_tol=1e-9)
    assert result["calibration_frame"] == 10


def test_save_session_writes_per_frame_calibration(tmp_path):
    default = QuickMeasureCalibration.from_line_clicks((0, 0), (100, 0), 1.0)
    frame_calib = QuickMeasureCalibration.from_line_clicks((0, 0), (200, 0), 2.0)
    session = QuickMeasureSession()
    session.set_calibration(default)
    session.set_calibration(frame_calib, frame=5)
    session.add_point(5, 0, 0)
    session.add_point(5, 200, 0)
    session.measure("distance")
    paths = session.save_session(str(tmp_path), stem="clip.mp4")
    calib_df = pd.read_csv(paths["calibration"])
    assert set(calib_df["frame"].unique()) == {-1, 5}
    points = pd.read_csv(paths["points"])
    assert (points["calibration_frame"] == 5).all()
    results = pd.read_csv(paths["results"])
    assert int(results.loc[0, "calibration_frame"]) == 5
    with open(paths["readme"], encoding="utf-8") as handle:
        readme = handle.read()
    assert "default (all frames)" in readme
    assert "frame 5" in readme


def test_calibration_first_then_free_measure_flow(tmp_path):
    """Simulates CALIB then MEASURE: calibrate, then measure, then save."""
    session = QuickMeasureSession(fps=30.0)
    assert needs_calibration(session)

    # Step 1: collect calibration clicks (a 200 px bar declared as 2 m).
    draft = CalibrationDraft(mode="line", unit_label="m")
    draft.add_point(100, 500)
    draft.add_point(300, 500)
    calib, message = finish_calibration_draft(draft, lambda prompt, default: "2")
    assert calib is not None and "Line calibration" in message
    session.set_calibration(calib)
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


def test_getpixelvideo_has_calib_and_measure_modes():
    """Host exposes separate CALIB (Shift+Q) and MEASURE (Q) toggles."""
    source_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "vaila",
        "getpixelvideo.py",
    )
    with open(source_path, encoding="utf-8") as handle:
        source = handle.read()
    assert "def _toggle_calib_mode()" in source
    assert "def _toggle_measure_mode()" in source
    assert "_start_quickmeasure_calibration()" in source
    assert (
        "needs_calibration(quickmeasure_session)"
        not in source.split("def _toggle_measure_mode()")[1][:1500]
    )
    # Calibration clicks must be routed before measure clicks.
    assert "quick_measure_calibrating" in source
    assert "quickmeasure_draft.add_point(video_x, video_y)" in source
    assert "quickmeasure_session.add_live_point" in source
    assert source.index("quickmeasure_draft.add_point(video_x, video_y)") < source.index(
        "quickmeasure_session.add_live_point"
    )
    assert "3=REF3D" in source or "REF3D" in source
    assert "4=DLT3D" in source or "DLT3D (11 params" in source
    assert "set_live_mode" in source
    assert "add_live_point" in source
    assert 'render("CALIB"' in source or "render(calib_label" in source
    assert "MEASURE" in source


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
        "6": "level",
        "7": "height",
        "8": "lineplane",
        "9": "event",
        "0": "spin",
        ",": "arc",
        ".": "heightline",
    }
    # the six original digit modes keep their keys
    assert [LIVE_MODE_KEYS[k] for k in "123456"] == [
        "distance",
        "area",
        "angle",
        "velocity",
        "acceleration",
        "level",
    ]


def test_snap_level_endpoint_horizontal_and_vertical():
    assert snap_level_endpoint(10.0, 10.0, 50.0, 12.0) == (50.0, 10.0, "H")
    assert snap_level_endpoint(10.0, 10.0, 12.0, 50.0) == (10.0, 50.0, "V")


def test_live_mode_level_snaps_horizontal_tape():
    session = QuickMeasureSession()
    session.set_live_mode("6")
    status, done = session.add_live_point(0, 10, 10)
    assert done is None
    assert "1/2" in status
    status, done = session.add_live_point(0, 50, 12)
    assert done is not None
    assert done["type"] == "level"
    assert done["axis"] == "H"
    assert done["pixels"] == [(10.0, 10.0), (50.0, 10.0)]
    assert math.isclose(done["value"], 40.0, abs_tol=1e-9)
    assert "Level (H)" in format_result(done)


def test_live_mode_level_snaps_vertical_tape():
    session = QuickMeasureSession()
    session.set_live_mode("6")
    session.add_live_point(0, 10, 10)
    _, done = session.add_live_point(0, 12, 50)
    assert done is not None
    assert done["axis"] == "V"
    assert done["pixels"] == [(10.0, 10.0), (10.0, 50.0)]
    assert math.isclose(done["value"], 40.0, abs_tol=1e-9)


def test_measure_undo_redo_and_clear_feedback():
    session = QuickMeasureSession()
    session.set_live_mode("1")
    session.add_live_point(0, 0, 0)
    _, done = session.add_live_point(0, 3, 4)
    assert done is not None
    assert len(session.results) == 1
    assert session.undo_measure() == "Measure: undo"
    assert len(session.results) == 0
    assert len(session.draft_points) == 1  # back to first click of the pair
    assert session.redo_measure() == "Measure: redo"
    assert len(session.results) == 1
    assert session.clear_measure_feedback() == "Measure: points and results cleared"
    assert session.results == []
    assert session.points == []
    assert session.draft_points == []
    assert session.undo_measure() == "Measure: nothing to undo"
    assert session.redo_measure() == "Measure: nothing to redo"


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


def test_live_mode_keys_7_to_0_are_no_longer_reserved():
    # 9 (event) and 0 (spin) need the FPS, 7 (height) and 8 (line-plane) need camera3d
    session = QuickMeasureSession()
    for key in "9", "0":
        with pytest.raises(QuickMeasureError, match="FPS"):
            session.set_live_mode(key)
    for key in "7", "8", ".":
        with pytest.raises(QuickMeasureError, match="camera3d"):
            session.set_live_mode(key)
    assert session.active_mode is None
    assert "event" in QuickMeasureSession(fps=30.0).set_live_mode("9").lower()
    with pytest.raises(QuickMeasureError):
        session.set_live_mode("x")


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


@pytest.mark.parametrize(
    "headers", [[f"L{i}" for i in range(1, 9)], [f"dlt_param_{i}" for i in range(1, 9)]]
)
def test_dlt_sequence_row_order_reconstruction_and_export(tmp_path, headers):
    import numpy as np

    from vaila.rec2d import rec2d

    path = tmp_path / "moving.dlt2d"
    rows = [[39 + i, 100, 0, 100 + i * 10, 0, 100, 100, 0, 0] for i in range(3)]
    pd.DataFrame(rows, columns=["frame", *headers]).to_csv(path, index=False)
    calibrations = QuickMeasureCalibration.from_dlt2d_sequence(str(path), total_frames=3)
    session = QuickMeasureSession()
    for frame, calib in enumerate(calibrations):
        session.set_calibration(calib, frame=frame)
        session.add_point(frame, 150, 150)
        expected = rec2d(np.asarray(rows[frame][1:]), np.array([[150, 150]]))[0]
        assert calib.pixel_to_real(150, 150) == pytest.approx(expected)
        assert calib.pixel_to_real(150, 150) == pytest.approx((0.5 - frame * 0.1, 0.5))
    assert session.points_dataframe()["x_real"].tolist() == pytest.approx([0.5, 0.4, 0.3])
    paths = session.save_session(str(tmp_path), stem="moving")
    exported = pd.read_csv(paths["dlt2d"])
    assert exported["frame"].tolist() == [1, 2, 3]
    loaded = QuickMeasureCalibration.from_dlt2d_sequence(paths["dlt2d"], total_frames=3)
    for original, restored in zip(calibrations, loaded, strict=True):
        assert restored.dlt_params == pytest.approx(original.dlt_params)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "invalid"])
def test_dlt_sequence_invalid_later_row(tmp_path, bad):
    path = tmp_path / "bad.dlt2d"
    rows = [[0, 100, 0, 100, 0, 100, 100, 0, 0], [1, 100, 0, bad, 0, 100, 100, 0, 0]]
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(QuickMeasureError):
        QuickMeasureCalibration.from_dlt2d_sequence(str(path), total_frames=2)


def test_dlt_sequence_count_and_empty(tmp_path):
    path = tmp_path / "counts.dlt2d"
    pd.DataFrame([[0, 100, 0, 100, 0, 100, 100, 0, 0]] * 2).to_csv(path, index=False)
    with pytest.raises(QuickMeasureError, match="2 rows; video has 3 frames"):
        QuickMeasureCalibration.from_dlt2d_sequence(str(path), total_frames=3)
    path.write_text("")
    with pytest.raises(QuickMeasureError):
        QuickMeasureCalibration.from_dlt2d_sequence(str(path), total_frames=3)
    assert (
        len(QuickMeasureCalibration.from_dlt2d_sequence(_fixture("sample.dlt2d"), total_frames=73))
        == 1
    )


@pytest.mark.parametrize("count", [1, 3, 2])
def test_dlt_dialog_fixed_sequence_and_mismatch(tmp_path, monkeypatch, count):
    from unittest.mock import Mock

    from vaila.quickmeasure import _ask_calibration_via_dialog

    path = tmp_path / "dialog.dlt2d"
    pd.DataFrame([[i, 100, 0, 100, 0, 100, 100, 0, 0] for i in range(count)]).to_csv(
        path, index=False
    )
    root = Mock()
    error = Mock()
    monkeypatch.setattr("tkinter.Tk", lambda: root)
    monkeypatch.setattr("tkinter.messagebox.askyesnocancel", lambda *a, **k: True)
    monkeypatch.setattr("tkinter.filedialog.askopenfilename", lambda **k: str(path))
    monkeypatch.setattr("tkinter.messagebox.showerror", error)
    result, _ = _ask_calibration_via_dialog(total_frames=3)
    if count == 1:
        assert isinstance(result, QuickMeasureCalibration)
    elif count == 3:
        assert isinstance(result, list) and len(result) == 3
    else:
        assert result is None
        error.assert_called_once()
    root.destroy.assert_called_once()


# ---------------------------------------------------------------------------
# Kick Measure: camera3d calibration and modes 7 / 8 / 9 / 0 / "," / "."
# (synthetic broadcast camera: fixed centre, pan and zoom change every frame)
# ---------------------------------------------------------------------------

_W, _H = 1920, 1080
_CENTRE = np.array([0.0, -75.0, 22.0])
_FPS = 30.0


def _true_camera(frame: int) -> np.ndarray:
    target = np.array([-30.0 + 2.0 * frame, 0.0, 0.0])
    zc = (target - _CENTRE) / np.linalg.norm(target - _CENTRE)
    xc = np.cross(zc, [0.0, 0.0, 1.0])
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.array([xc, yc, zc])
    f = 2200.0 + 40.0 * frame
    K = np.array([[f, 0, _W / 2], [0, f, _H / 2], [0, 0, 1.0]])
    return K @ np.c_[R, -R @ _CENTRE]


def _track(n_frames: int = 40) -> CameraTrack:
    return CameraTrack.from_matrices({f: _true_camera(f) for f in range(n_frames)})


def _px(frame: int, xyz) -> tuple[float, float]:
    """Pixel of a pitch point (x, y, z) m in ``frame`` of the synthetic camera."""
    X = np.array([*xyz, 1.0]) if len(xyz) == 3 else np.array([*xyz, 0.0, 1.0])
    h = _true_camera(frame) @ X
    return float(h[0] / h[2]), float(h[1] / h[2])


def _camera_session(fps: float | None = _FPS) -> QuickMeasureSession:
    session = QuickMeasureSession(fps=fps)
    session.set_calibration(QuickMeasureCalibration.from_camera_track(_track()))
    return session


def test_camera3d_calibration_pixel_to_real_at_each_frame():
    track = _track()
    calib = QuickMeasureCalibration.from_camera_track(track, source="synthetic.json")
    assert calib.kind == "camera3d" and calib.unit_label == "m"
    assert calib.real_measures["n_frames"] == 40
    assert calib.real_measures["frames_given"] == 40
    assert "Camera3D" in calib.describe()
    for frame, ground in ((0, (-20.0, 3.0)), (17, (4.0, -6.0)), (33, (25.0, 8.0))):
        u, v = _px(frame, ground)
        x, y = calib.pixel_to_real_at(frame, u, v)
        assert (x, y) == pytest.approx(ground, abs=1e-6)
    # without a frame the calibration cannot know which camera to use ...
    with pytest.raises(QuickMeasureError, match="frame"):
        calib.pixel_to_real(960.0, 540.0)
    # ... unless it was installed for one frame (per-frame override)
    session = QuickMeasureSession()
    session.set_calibration(calib, frame=17)
    u, v = _px(17, (4.0, -6.0))
    assert calib.pixel_to_real(u, v) == pytest.approx((4.0, -6.0), abs=1e-6)
    # a pixel above the horizon never reaches the grass: nan, no exception
    x, y = calib.pixel_to_real_at(0, 960.0, -5000.0)
    assert math.isnan(x) and math.isnan(y)


def test_camera3d_calibration_other_kinds_ignore_the_frame():
    calib = QuickMeasureCalibration.from_dlt2d_file(_fixture("sample.dlt2d"))
    assert calib.pixel_to_real_at(99, 200.0, 300.0) == calib.pixel_to_real(200.0, 300.0)


def test_camera3d_modes_1_to_6_give_metres_in_any_frame():
    session = _camera_session()
    assert session.unit_label == "m" and session.is_calibrated
    # distance: 10 m along the pitch seen at frame 25 (a different camera than frame 0)
    session.set_live_mode("1")
    session.add_live_point(25, *_px(25, (0.0, 0.0, 0.0)))
    _msg, done = session.add_live_point(25, *_px(25, (10.0, 0.0, 0.0)))
    assert done is not None
    assert done["value"] == pytest.approx(10.0, abs=1e-6)
    assert done["calibration"] == "camera3d" and done["unit"] == "m"
    # velocity across two frames with two different cameras: 2 m in 3 frames at 30 fps = 20 m/s
    session.set_live_mode("4")
    session.add_live_point(10, *_px(10, (0.0, 5.0, 0.0)))
    _msg, done = session.add_live_point(13, *_px(13, (2.0, 5.0, 0.0)))
    assert done is not None
    assert done["value"] == pytest.approx(2.0 / (3 / _FPS), rel=1e-6)
    assert done["unit"] == "m/s"
    # angle at a vertex: 90 degrees on the grass
    session.set_live_mode("3")
    session.add_live_point(5, *_px(5, (3.0, 0.0, 0.0)))
    session.add_live_point(5, *_px(5, (0.0, 0.0, 0.0)))
    _msg, done = session.add_live_point(5, *_px(5, (0.0, 4.0, 0.0)))
    assert done is not None and done["value"] == pytest.approx(90.0, abs=1e-6)
    # area of a 4 x 3 m rectangle
    session.set_live_mode("2")
    for xy in ((0, 0), (4, 0), (4, 3), (0, 3)):
        session.add_live_point(8, *_px(8, (*xy, 0.0)))
    _msg, done = session.finalize_live_area(8)
    assert done is not None and done["value"] == pytest.approx(12.0, abs=1e-6)


def test_camera3d_hover_text_and_approximate_flag():
    track = CameraTrack.from_matrices({0: _true_camera(0), 10: _true_camera(10)}, n_frames=20)
    session = QuickMeasureSession()
    session.set_calibration(QuickMeasureCalibration.from_camera_track(track))
    u, v = _px(0, (-20.0, 3.0))
    text = format_hover_coords(session, u, v, frame=0)
    assert "Real: (-20.000, 3.000) m" in text and "approx" not in text
    assert "[approx. camera]" in format_hover_coords(session, u, v, frame=5)
    # nan (above the horizon) and a missing frame fall back to the pixel text
    assert "Real" not in format_hover_coords(session, 960.0, -5000.0, frame=0)
    assert "Real" not in format_hover_coords(session, u, v)


def test_mode_7_height_base_top():
    session = _camera_session()
    session.set_live_mode("7")
    _msg, none_yet = session.add_live_point(12, *_px(12, (6.0, -3.0, 0.0)))
    assert none_yet is None
    msg, done = session.add_live_point(12, *_px(12, (6.0, -3.0, 1.0)))
    assert done is not None and done["type"] == "height"
    assert done["value"] == pytest.approx(1.0, abs=0.01)
    assert done["unit"] == "m" and done["n_points"] == 2 and done["mode_key"] == "7"
    extra = done["extra"]
    assert 0.0 < extra["uncertainty"] < 0.2  # +-1 px on either click
    assert extra["camera_source"] == "given" and extra["approximate"] == 0
    assert "Height: 1.00" in msg and "±" in msg
    assert done["frames"] == [12, 12]


def test_mode_7_needs_the_same_frame_and_flags_an_approximate_camera():
    session = _camera_session()
    session.set_live_mode("7")
    session.add_live_point(3, *_px(3, (6.0, 0.0, 0.0)))
    with pytest.raises(QuickMeasureError, match="same frame"):
        session.add_live_point(4, *_px(4, (6.0, 0.0, 1.0)))
    # a frame whose camera is a copy of a neighbour is reported, not hidden
    track = CameraTrack.from_matrices({0: _true_camera(0)}, n_frames=5)
    session = QuickMeasureSession(fps=_FPS)
    session.set_calibration(QuickMeasureCalibration.from_camera_track(track))
    session.set_live_mode("7")
    session.add_live_point(3, *_px(0, (6.0, 0.0, 0.0)))
    msg, done = session.add_live_point(3, *_px(0, (6.0, 0.0, 2.0)))
    assert done is not None and done["extra"]["approximate"] == 1
    assert done["extra"]["camera_source"] == "approximate" and "approximate camera" in msg


def _plane_session() -> tuple[QuickMeasureSession, np.ndarray, float]:
    """Session with a line kick -> arrival on the grass marked in mode 8 (frame 0)."""
    session = _camera_session()
    session.set_live_mode("8")
    start, end = (-20.0, -5.0), (25.0, 10.0)
    session.add_live_point(0, *_px(0, (*start, 0.0)))
    msg, line = session.add_live_point(0, *_px(0, (*end, 0.0)))
    assert line is not None and line["type"] == "line_plane", msg
    direction = np.array(end) - np.array(start)
    length = float(np.hypot(*direction))
    assert line["value"] == pytest.approx(length, abs=1e-6)
    return session, direction / length, length


def test_mode_8_line_then_ball_positions_in_the_plane():
    session, heading, _length = _plane_session()
    assert session.plane_ready
    start = np.array([-20.0, -5.0])
    # ball in the vertical plane of the line: s = 12 m along it, 3.5 m high, seen at frame 7
    xy = start + 12.0 * heading
    msg, done = session.add_live_point(7, *_px(7, (xy[0], xy[1], 3.5)))
    assert done is not None and done["type"] == "plane_point"
    assert done["value"] == pytest.approx(3.5, abs=1e-6)
    assert done["extra"]["s"] == pytest.approx(12.0, abs=1e-6)
    assert done["extra"]["x"] == pytest.approx(xy[0], abs=1e-6)
    assert done["reals"][0] == pytest.approx((xy[0], xy[1]), abs=1e-6)
    assert done["n_points"] == 1 and len(done["pixels"]) == 1
    # one camera cannot know the lateral position: the bound is reported and is a few dm
    assert 0.0 < done["extra"]["uncertainty"] < 1.5
    assert "s 12.000" in msg and "±" in msg
    assert math.isnan(done["extra"]["speed"])  # one position: no speed yet


def test_mode_8_speed_and_acceleration_of_the_sequence():
    session, heading, _length = _plane_session()
    start = np.array([-20.0, -5.0])
    # uniform motion 1.2 m along the line per frame at z = 2 (36 m/s), then it speeds up
    results = []
    for frame, s_m in ((2, 4.0), (3, 5.2), (4, 6.4), (5, 7.9)):
        xy = start + s_m * heading
        _msg, done = session.add_live_point(frame, *_px(frame, (xy[0], xy[1], 2.0)))
        assert done is not None
        results.append(done)
    assert results[1]["extra"]["speed"] == pytest.approx(1.2 * _FPS, rel=1e-5)
    assert results[2]["extra"]["speed"] == pytest.approx(1.2 * _FPS, rel=1e-5)
    assert results[2]["extra"]["accel"] == pytest.approx(0.0, abs=1e-3)
    assert results[3]["extra"]["speed"] == pytest.approx(1.5 * _FPS, rel=1e-5)
    assert results[3]["extra"]["accel"] == pytest.approx(0.3 * _FPS / (1 / _FPS), rel=1e-4)
    # the series of the clicked frames (consecutive frames, so every row is a click)
    series = session.lineplane_series()
    assert series["frame"].tolist() == [2, 3, 4, 5]
    assert series["clicked"].all()
    assert series["z"].to_numpy() == pytest.approx([2.0] * 4, abs=1e-6)
    assert series["speed"].iloc[1] == pytest.approx(1.2 * _FPS, rel=1e-4)
    assert series.columns.tolist()[:5] == ["frame", "time_s", "clicked", "x", "y"]


def test_mode_8_series_interpolates_between_sparse_clicks_and_needs_two():
    session, heading, _length = _plane_session()
    with pytest.raises(QuickMeasureError, match="at least 2"):
        session.lineplane_series()
    start = np.array([-20.0, -5.0])
    for frame, s_m in ((2, 4.0), (6, 8.0)):
        xy = start + s_m * heading
        session.add_live_point(frame, *_px(frame, (xy[0], xy[1], 2.0)))
    series = session.lineplane_series()
    assert series["frame"].tolist() == [2, 3, 4, 5, 6]
    assert series["clicked"].tolist() == [True, False, False, False, True]
    assert series["s"].to_numpy() == pytest.approx([4.0, 5.0, 6.0, 7.0, 8.0], abs=1e-5)


def test_mode_8_without_fps_still_gives_position_and_series_refuses():
    session = _camera_session(fps=None)
    session.set_live_mode("8")
    session.add_live_point(0, *_px(0, (-20.0, -5.0, 0.0)))
    session.add_live_point(0, *_px(0, (25.0, 10.0, 0.0)))
    xy = np.array([-20.0, -5.0]) + 5.0 * np.array([45.0, 15.0]) / np.hypot(45.0, 15.0)
    _msg, done = session.add_live_point(3, *_px(3, (xy[0], xy[1], 1.0)))
    assert done is not None and done["value"] == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(QuickMeasureError, match="FPS"):
        session.lineplane_series()


def test_mode_8_keeps_the_line_when_you_return_and_redefines_on_second_press():
    session, heading, _length = _plane_session()
    # leaving to another mode and coming back keeps the line: one click = one ball position
    session.set_live_mode("6")
    msg = session.set_live_mode("8")
    assert session.plane_ready and "click the ball" in msg
    xy = np.array([-20.0, -5.0]) + 6.0 * heading
    _msg, done = session.add_live_point(4, *_px(4, (xy[0], xy[1], 1.0)))
    assert done is not None and done["type"] == "plane_point"
    # pressing 8 while already in mode 8 starts a new line (2 clicks again)
    msg = session.set_live_mode("8")
    assert not session.plane_ready and "kick point" in msg
    session.add_live_point(0, *_px(0, (0.0, 0.0, 0.0)))
    _msg, line = session.add_live_point(0, *_px(0, (10.0, 0.0, 0.0)))
    assert line is not None and line["type"] == "line_plane"
    assert line["value"] == pytest.approx(10.0, abs=1e-6)


def test_mode_8_undo_redoes_the_line_state():
    session, heading, _length = _plane_session()
    assert session.plane_ready
    session.undo_measure()  # undoes the arrival click: the line result is gone
    assert not session.plane_ready
    assert session._active_line_plane() is None
    session.redo_measure()
    assert session.plane_ready and session._active_line_plane() is not None


def test_mode_8_rejects_a_line_on_two_frames_and_a_point_without_a_line():
    session = _camera_session()
    session.set_live_mode("8")
    session.add_live_point(0, *_px(0, (0.0, 0.0, 0.0)))
    with pytest.raises(QuickMeasureError, match="same frame"):
        session.add_live_point(1, *_px(1, (10.0, 0.0, 0.0)))
    fresh = _camera_session()
    fresh.points.append(QuickMeasurePoint(0, 960.0, 540.0))
    with pytest.raises(QuickMeasureError, match="line"):
        fresh.measure("plane_point")


def test_mode_9_event_duration_and_mean_speed():
    session = _camera_session()
    session.set_live_mode("9")
    session.add_live_point(10, *_px(10, (0.0, 0.0, 0.0)))
    msg, done = session.add_live_point(25, *_px(25, (18.0, 0.0, 0.0)))
    assert done is not None and done["type"] == "event"
    assert done["value"] == pytest.approx(0.5, abs=1e-9)  # 15 frames at 30 fps
    assert done["unit"] == "s" and done["dt"] == pytest.approx(0.5)
    assert done["extra"]["speed"] == pytest.approx(36.0, rel=1e-6)
    assert done["extra"]["uncertainty"] == pytest.approx(1.0 / _FPS)
    assert (done["extra"]["frame_start"], done["extra"]["frame_end"]) == (10, 25)
    assert "mean speed 36.000 m/s" in msg and "±" in msg
    # clicking the end first gives the same event (start = earlier frame)
    session.set_live_mode("9")
    session.add_live_point(25, *_px(25, (18.0, 0.0, 0.0)))
    _msg, done2 = session.add_live_point(10, *_px(10, (0.0, 0.0, 0.0)))
    assert done2 is not None and done2["value"] == pytest.approx(0.5)


def test_mode_9_needs_distinct_frames_and_fps():
    session = QuickMeasureSession(fps=25.0)
    session.set_live_mode("9")
    session.add_live_point(5, 0.0, 0.0)
    with pytest.raises(QuickMeasureError, match="different frames"):
        session.add_live_point(5, 3.0, 4.0)
    with pytest.raises(QuickMeasureError, match="FPS"):
        QuickMeasureSession().set_live_mode("9")
    # uncalibrated: the speed is in px/s
    s2 = QuickMeasureSession(fps=10.0)
    s2.set_live_mode("9")
    s2.add_live_point(0, 0.0, 0.0)
    _msg, done = s2.add_live_point(10, 30.0, 40.0)
    assert done is not None and done["value"] == pytest.approx(1.0)
    assert done["extra"]["speed"] == pytest.approx(50.0)
    assert done["extra"]["length_unit"] == "px"


def test_mode_0_spin_rate_updates_on_every_click():
    session = QuickMeasureSession(fps=_FPS)
    session.set_live_mode("0")
    msg, done = session.add_live_point(0, 100.0, 100.0)
    assert done is None and "1 mark" in msg
    msg, done = session.add_live_point(6, 101.0, 100.0)  # 5 rps = one turn every 6 frames
    assert done is not None and done["type"] == "spin"
    assert done["value"] == pytest.approx(5.0) and done["extra"]["turns"] == 1
    assert done["running"] is True
    session.add_live_point(12, 102.0, 100.0)
    _msg, done = session.add_live_point(18, 103.0, 100.0)
    assert len(session.results) == 1  # the same result is replaced, not stacked
    assert done is session.results[0]
    assert done["extra"]["turns"] == 3 and done["extra"]["rps"] == pytest.approx(5.0)
    assert done["extra"]["rpm"] == pytest.approx(300.0)
    assert done["extra"]["uncertainty"] == pytest.approx(5.0 * (1 / _FPS) / 0.6)
    assert done["extra"]["rpm_uncertainty"] == pytest.approx(60.0 * done["extra"]["uncertainty"])
    assert done["n_points"] == 4 and done["frames"] == [0, 6, 12, 18]
    assert done["point_ids"] == [1, 2, 3, 4] and len(session.points) == 4
    text = format_result(done)
    assert "Spin: 5.0000 rps" in text and "3 turn(s)" in text and "300.0 rpm" in text
    # Enter closes it; the next click starts a new count
    msg, closed = session.finalize_live_area(18)
    assert "closed" in msg and closed is done and "running" not in done
    assert not session.draft_points
    session.add_live_point(30, 100.0, 100.0)
    _msg, second = session.add_live_point(40, 100.0, 100.0)
    assert len(session.results) == 2 and second is session.results[1]
    assert second["extra"]["rps"] == pytest.approx(3.0)


def test_mode_0_spin_errors_and_aliasing_note():
    session = QuickMeasureSession(fps=30.0)
    session.set_live_mode("0")
    session.add_live_point(4, 0.0, 0.0)
    with pytest.raises(QuickMeasureError, match="different frames"):
        session.add_live_point(4, 1.0, 1.0)
    # the failed click was not kept
    assert len(session.draft_points) == 1 and len(session.points) == 1
    _msg, done = session.add_live_point(6, 0.0, 0.0)  # 1 turn in 2 frames = 15 rps = fps / 2
    assert done is not None and "aliases" in done["extra"]["note"]
    assert "aliases" in format_result(done)


def test_mode_0_spin_undo_removes_the_last_mark():
    session = QuickMeasureSession(fps=_FPS)
    session.set_live_mode("0")
    for frame in (0, 6, 12):
        session.add_live_point(frame, 1.0, 1.0)
    assert session.results[0]["extra"]["turns"] == 2
    session.undo_measure()
    assert len(session.draft_points) == 2 and session.results[0]["extra"]["turns"] == 1


def _arc_clicks(radius: float, centre, angles_deg, frame: int = 20):
    return [
        _px(frame, (centre[0] + radius * math.cos(math.radians(a)),
                    centre[1] + radius * math.sin(math.radians(a)), 0.0))
        for a in angles_deg
    ]  # fmt: skip


def test_arc_mode_comma_fits_a_circle_on_the_grass():
    session = _camera_session()
    session.set_live_mode(",")
    angles = (10, 25, 40, 55, 70)
    done = None
    for i, (u, v) in enumerate(_arc_clicks(12.0, (0.0, 0.0), angles)):
        msg, done = session.add_live_point(20, u, v)
        if i < 2:
            assert done is None and "point(s)" in msg
    assert done is not None and done["type"] == "arc" and len(session.results) == 1
    chord_len = 2 * 12.0 * math.sin(math.radians(30))
    assert done["extra"]["radius"] == pytest.approx(12.0, abs=1e-5)
    assert done["extra"]["angle_deg"] == pytest.approx(60.0, abs=1e-4)
    assert done["extra"]["sagitta"] == pytest.approx(
        12.0 * (1 - math.cos(math.radians(30))), abs=1e-5
    )
    assert done["extra"]["arc_length"] == pytest.approx(12.0 * math.radians(60), abs=1e-5)
    # the polyline through 5 clicks is a bit shorter than the arc and longer than the chord
    assert chord_len < done["value"] < done["extra"]["arc_length"] + 1e-9
    assert done["extra"]["uncertainty"] < 1e-5
    text = format_result(done)
    assert "R 12.000 m" in text and "60.0 deg" in text and "sagitta" in text
    # a straight line has no finite radius but is not an error
    session.set_live_mode(",")
    for x in (0.0, 5.0, 10.0):
        done = session.add_live_point(20, *_px(20, (x, 3.0, 0.0)))[1]
    assert done is not None and math.isinf(done["extra"]["radius"])
    assert done["value"] == pytest.approx(10.0, abs=1e-6)


def test_arc_mode_works_in_pixels_without_calibration():
    session = QuickMeasureSession()
    session.set_live_mode(",")
    for a in (0, 30, 60, 90):
        done = session.add_live_point(
            0, 100 + 50 * math.cos(math.radians(a)), 100 + 50 * math.sin(math.radians(a))
        )[1]
    assert done is not None and done["unit"] == "px"
    assert done["extra"]["radius"] == pytest.approx(50.0, abs=1e-6)
    assert done["extra"]["angle_deg"] == pytest.approx(90.0, abs=1e-4)


def test_height_line_mode_point_draws_guides_over_the_video():
    session, heading, length = _plane_session()
    session.set_live_mode(".")
    assert session.active_mode == "heightline"
    xy = np.array([-20.0, -5.0]) + 10.0 * heading
    msg, done = session.add_live_point(9, *_px(9, (xy[0], xy[1], 2.44)))
    assert done is not None and done["type"] == "height_line"
    assert done["value"] == pytest.approx(2.44, abs=1e-6) and "Height line" in msg
    assert done["extra"]["z"] == pytest.approx(2.44)
    # the guide is the horizontal line at 2.44 m over the whole line, in any frame
    plane = session._active_line_plane()
    assert plane is not None
    for frame in (0, 9, 30):
        guides = session.height_line_guides(frame)
        assert len(guides) == 1 and guides[0][0] == pytest.approx(2.44)
        uv = guides[0][1]
        assert uv.shape == (2, 2)
        s_end, z_end = plane.locate(frame, uv[1])
        assert s_end == pytest.approx(length, abs=1e-5) and z_end == pytest.approx(2.44, abs=1e-5)
    # a typed height adds another guide; undo takes it away
    typed = session.add_height_line(1.0, frame=9)
    assert typed["n_points"] == 0 and len(session.height_line_guides(0)) == 2
    session.undo_measure()
    assert len(session.height_line_guides(0)) == 1


def test_height_line_mode_needs_a_line_first():
    session = _camera_session()
    with pytest.raises(QuickMeasureError, match="line"):
        session.set_live_mode(".")
    with pytest.raises(QuickMeasureError, match="line"):
        session.add_height_line(2.0)
    assert session.height_line_guides(0) == []


def test_new_results_export_columns_files_and_report(tmp_path):
    session, heading, _length = _plane_session()
    start = np.array([-20.0, -5.0])
    for frame, s_m in ((2, 4.0), (3, 5.2), (4, 6.4)):
        xy = start + s_m * heading
        session.add_live_point(frame, *_px(frame, (xy[0], xy[1], 2.0)))
    session.set_live_mode("7")
    session.add_live_point(12, *_px(12, (6.0, -3.0, 0.0)))
    session.add_live_point(12, *_px(12, (6.0, -3.0, 1.0)))
    session.set_live_mode("9")
    session.add_live_point(10, *_px(10, (0.0, 0.0, 0.0)))
    session.add_live_point(25, *_px(25, (18.0, 0.0, 0.0)))
    session.set_live_mode("0")
    for frame in (0, 6, 12):
        session.add_live_point(frame, 5.0, 5.0)
    session.set_live_mode(",")
    for u, v in _arc_clicks(10.0, (0.0, 0.0), (0, 30, 60)):
        session.add_live_point(20, u, v)
    session.set_live_mode("1")
    session.add_live_point(5, *_px(5, (0.0, 0.0, 0.0)))
    session.add_live_point(5, *_px(5, (3.0, 4.0, 0.0)))

    paths = session.save_session(str(tmp_path), stem="kick.mp4")
    df = pd.read_csv(paths["results"])
    assert {"line_plane", "plane_point", "height", "event", "spin", "arc", "distance"} == set(
        df["type"]
    )
    for column in (
        "uncertainty",
        "length_unit",
        "camera_source",
        "approximate",
        "plane",
        "frame_start",
        "frame_end",
        "s",
        "z",
        "speed",
        "accel",
        "turns",
        "rps",
        "rpm",
        "radius",
        "angle_deg",
        "sagitta",
        "note",
    ):
        assert column in df.columns, column
    assert (df["calibration"].isin({"camera3d", "none"})).all()
    height = df[df["type"] == "height"].iloc[0]
    assert height["value"] == pytest.approx(1.0, abs=0.01) and height["unit"] == "m"
    assert height["camera_source"] == "given" and height["uncertainty"] > 0
    event = df[df["type"] == "event"].iloc[0]
    assert event["value"] == pytest.approx(0.5) and event["speed"] == pytest.approx(36.0, rel=1e-6)
    assert event["elapsed_time_s"] == pytest.approx(0.5)
    spin = df[df["type"] == "spin"].iloc[0]
    assert (
        spin["turns"] == 2
        and spin["rps"] == pytest.approx(5.0)
        and spin["rpm"] == pytest.approx(300)
    )
    # old modes leave the new columns blank
    distance = df[df["type"] == "distance"].iloc[0]
    assert distance["value"] == pytest.approx(5.0, abs=1e-6) and pd.isna(distance["uncertainty"])
    # per-type files share the schema; camera and series files are written
    for kind in ("height", "line_plane", "plane_point", "event", "spin", "arc"):
        sub = pd.read_csv(paths[f"results_{kind}"])
        assert sub.columns.tolist() == df.columns.tolist() and set(sub["type"]) == {kind}
    cams = pd.read_csv(paths["camera3d"])
    assert len(cams) == 40 and {"frame", "source", "approximate", "P11", "P34"} <= set(cams.columns)
    series = pd.read_csv(paths["lineplane_series"])
    assert series["frame"].tolist() == [2, 3, 4] and series["clicked"].all()
    calib = pd.read_csv(paths["calibration"])
    assert set(calib["kind"]) == {"camera3d"}
    assert {"n_frames", "frames_given"} <= set(calib["measure_name"])
    with open(paths["report"], encoding="utf-8") as handle:
        report = handle.read()
    for word in ("Camera3D", "Line plane", "Event time", "Spin", "Arc / curve", "Height line"):
        assert word in report
    with open(paths["readme"], encoding="utf-8") as handle:
        assert "Camera3D" in handle.read()


def test_format_result_of_the_old_types_is_unchanged():
    assert (
        format_result({"type": "distance", "value": 2.0, "unit": "m", "extra": {}})
        == "Distance: 2.0000 m"
    )
    assert (
        format_result({"type": "level", "value": 1.0, "unit": "px", "axis": "H"})
        == "Level (H): 1.0000 px"
    )
    assert format_result({"type": "height_line", "value": 2.44, "unit": "m"}) == (
        "Height line: 2.4400 m"
    )


def test_draw_quickmeasure_overlay_new_modes_and_guides():
    pygame = pytest.importorskip("pygame")
    pygame.font.init()
    font = pygame.font.Font(None, 18)
    from vaila.quickmeasure import draw_quickmeasure_overlay

    session, heading, _length = _plane_session()
    xy = np.array([-20.0, -5.0]) + 10.0 * heading
    session.add_live_point(9, *_px(9, (xy[0], xy[1], 2.0)))
    session.set_live_mode(".")
    session.add_live_point(9, *_px(9, (xy[0], xy[1], 2.44)))
    session.set_live_mode("0")
    session.add_live_point(1, 400.0, 300.0)
    session.add_live_point(7, 410.0, 300.0)
    surface = pygame.Surface((_W, _H))
    surface.fill((0, 0, 0))
    draw_quickmeasure_overlay(surface, session, 1.0, 0, 0, font, frame=9)
    assert surface.get_bounding_rect().width > 0
    plain = pygame.Surface((_W, _H))
    plain.fill((0, 0, 0))
    draw_quickmeasure_overlay(plain, session, 1.0, 0, 0, font)  # no frame: no guide, no crash
    guided = pygame.image.tobytes(surface, "RGB")
    assert guided != pygame.image.tobytes(plain, "RGB")


def test_overlay_badges_use_portuguese_names_when_lang_is_pt() -> None:
    """The result badge on the video says ``altura`` in PT and ``height`` in EN (default)."""
    pygame = pytest.importorskip("pygame")
    pygame.font.init()
    from vaila.quickmeasure import (
        RESULT_TYPE_LABELS_PT,
        draw_quickmeasure_overlay,
        result_type_label,
    )

    session = QuickMeasureSession(fps=30.0)
    session.results.append(
        {"type": "height", "value": 1.4996, "unit": "m", "pixels": [(100.0, 200.0), (100.0, 90.0)]}
    )
    seen: dict[str, list[str]] = {"en": [], "pt": []}

    class _Font:
        def __init__(self, lang: str) -> None:
            self.lang = lang
            self._f = pygame.font.Font(None, 16)

        def render(self, text, *a, **k):
            seen[self.lang].append(str(text))
            return self._f.render(text, *a, **k)

    for lang in ("en", "pt"):
        surface = pygame.Surface((_W, _H))
        if lang == "en":
            draw_quickmeasure_overlay(surface, session, 1.0, 0, 0, _Font(lang))
        else:
            draw_quickmeasure_overlay(surface, session, 1.0, 0, 0, _Font(lang), lang="pt")
    assert "#1 · height · 1.500 m" in seen["en"]
    assert "#1 · altura · 1.500 m" in seen["pt"]
    assert result_type_label("spin", "pt") == "giro" and result_type_label("spin") == "spin"
    assert set(RESULT_TYPE_LABELS_PT) >= {
        "distance",
        "area",
        "angle",
        "velocity",
        "acceleration",
        "level",
        "height",
        "line_plane",
        "plane_point",
        "height_line",
        "event",
        "spin",
        "arc",
    }
