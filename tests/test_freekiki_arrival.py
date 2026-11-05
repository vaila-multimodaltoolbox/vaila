"""Observed entry near a post, frame truncation and calibration audit regressions.

Version: 0.4.14
Update Date: 02 November 2026
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from tests.test_freekiki_auto_flight import FPS, L, N, _ball_rows, _flight, _pix, _pixels
from vaila import freekiki_arrival as ar
from vaila import freekiki_auto_flight as af
from vaila.freekiki_detect3d import BallArrivalObservation, ball_anchor_spec_from_json
from vaila.freekiki_field3d import ball_3d
from vaila.freekiki_report import arrival_note, confidence
from vaila.monocular_dlt_align import dlt_projection_matrix


def observed(frame=45):
    _, _, world = ar.load_field_points()
    corners = {f"p{i}": _pix(world[i]).tolist() for i in ar.GOAL_IDS[1]}
    return {
        "type": "goal_entry",
        "frame": frame,
        "pixel": _pix([52.45, 3.4, 1.2]).tolist(),
        "corners_px": corners,
        "goal_side": 1,
        "sigma_m": 0.2,
    }


def test_entry_uses_confirmed_corners_instead_of_projected_corners():
    obs = ar.validate_observation(observed())
    P = dlt_projection_matrix(L)
    wrong = P.copy()
    wrong[0] += 20 * wrong[2]  # a camera 20 px out; the local observation must not move
    point = ar.entry_anchor(obs, wrong)
    assert point["xyz"] == pytest.approx([52.45, 3.4, 1.2], abs=1e-6)
    assert all(
        e == pytest.approx(20) for e in point["diagnostics"]["goal_corner_errors_px"].values()
    )
    assert point["diagnostics"]["global_local_gap_m"] > 0.2
    assert point["sigma"] >= 0.2


@pytest.mark.parametrize(
    "changes",
    [
        {"frame": 1.5},
        {"frame": -1},
        {"frame": 60},
        {"frame": True},
        {"sigma_m": float("nan")},
        {"pixel_sigma": 0},
        {"goal_side": True},
        {"pixel": [float("nan"), 10]},
        {"corners_px": {}},
        {"corners_px": {f"p{i}": [1, 1] for i in (40, 41, 42, 43)}},
    ],
)
def test_invalid_observations_fail_before_output_mutation(changes):
    with pytest.raises(ValueError):
        ar.validate_observation(observed() | changes, 60)


def test_keeper_confirmation_is_mandatory_and_point_anchors_accept_observations():
    hand = {"type": "keeper_hand", "frame": 12, "xyz": [51, 1, 1]}
    with pytest.raises(ValueError, match="confirmation|confirmed"):
        ar.validate_observation(hand)
    assert ar.validate_observation(hand | {"confirmed": True})["provenance"] == "manual"
    spec = ball_anchor_spec_from_json(json.dumps({"arrival": observed()}))
    assert isinstance(spec, BallArrivalObservation)
    with pytest.raises(ValueError, match="combined"):
        ball_anchor_spec_from_json(json.dumps({"arrival": observed(), "points": []}))


def test_saving_keeps_backup_and_camera_key_excludes_only_ball_coordinates(tmp_path):
    path = tmp_path / "corrections" / "arrival_observation.json"
    first = ar.validate_observation(observed())
    ar.save_observation(path, first)
    moved = first | {"pixel": [900, 400]}
    ar.save_observation(path, moved)
    backups = list(path.parent.glob("arrival_observation_*.json"))
    assert len(backups) == 1 and ar.load_observation(backups[0]) == first
    assert ar.calibration_key(first) == ar.calibration_key(moved)
    changed = first | {"corners_px": first["corners_px"] | {"p40": [1, 2]}}
    assert ar.calibration_key(changed) != ar.calibration_key(first)


def test_manual_near_post_entry_ignores_later_net_pixels_and_never_reads_keeper(monkeypatch):
    entry = 45
    theta = np.array([30.0, 3.4, 22.0, 0.0, 7.0, 0.4, 0.0, 0.0, 0.0])
    theta[0] += 52.45 - _flight(theta)[entry, 0]
    truth = _flight(theta)
    pixels = _pixels(truth)
    for f in range(entry + 1, N):
        pixels[f] = (1800, 1000)  # net/bounce observations do not fit the free flight
    cameras = dict.fromkeys(range(N), L)
    obs = observed(entry) | {"pixel": _pix(truth[entry]).tolist()}

    def forbidden(*_args):
        raise AssertionError("no automatic keeper anchor")

    monkeypatch.setattr(af, "keeper_hand", forbidden)
    result = af.auto_flight(
        ball_3d(_ball_rows(pixels), cameras), pixels, cameras, [], FPS, N, arrival_observation=obs
    )
    rows, record = af.auto_rows_and_record(result, pixels, N)
    assert record["arrival_status"] == "entry" and record["arrival"]["provenance"] == "manual"
    assert record["anchor_points"]["goal_entry"]["frame"] == entry
    assert "p_arrival" not in record["anchor_points"]
    assert max(map(int, record["per_frame_px"])) == entry
    assert (
        np.linalg.norm(np.array([rows[entry][k] for k in ("x_m", "y_m", "z_m")]) - truth[entry])
        < 0.1
    )
    assert all(np.isnan(r["x_m"]) for r in rows[entry + 1 :])
    assert record["arrival"]["diagnostics"]["fit_residual_m"] < 0.1
    assert record["rms_px"] < 1
    assert record["quality"] == "ok"


def test_refit_preserves_reference_and_records_every_used_point(tmp_path):
    obs = ar.validate_observation(observed(0))
    _, _, world = ar.load_field_points()
    before = np.asarray(world).copy()
    pixel_csv = tmp_path / "field_kps_getpixelvideo.csv"
    # Only front-of-camera points, with broad pitch spread, plus all confirmed corners.
    P = dlt_projection_matrix(L)
    h = np.c_[world, np.ones(len(world))] @ P.T
    xy = h[:, :2] / h[:, 2:3]
    xy[h[:, 2] <= 0] = np.nan
    with pixel_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame"] + [f"p{i}_{c}" for i in range(49) for c in "xy"])
        writer.writerow([0, *xy.ravel()])
    path = tmp_path / "test.dlt3d"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame"] + [f"L{i}" for i in range(1, 12)])
        writer.writerow([0, *L])
    result = ar.refit_goal_calibration(tmp_path, obs, 1920, 1080)
    assert result["frames"][0]["goal_max_px"] < 1e-4
    with (tmp_path / "goal_calibration_residuals.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert {r["point"] for r in rows if r["confirmed"] == "True"} == {"p40", "p41", "p42", "p43"}
    assert all(r["used"] == "True" for r in rows if r["confirmed"] == "True")
    assert (tmp_path / "before_goal_refit" / "test.dlt3d").is_file()
    np.testing.assert_array_equal(ar.load_field_points()[2], before)
    camera_before_failure = path.read_bytes()
    bad = obs | {
        "corners_px": obs["corners_px"]
        | {"p42": (np.asarray(obs["corners_px"]["p42"]) + [35, 0]).tolist()}
    }
    with pytest.raises(ValueError, match="point review"):
        ar.refit_goal_calibration(tmp_path, bad, 1920, 1080)
    assert path.read_bytes() == camera_before_failure
    assert (tmp_path / "goal_calibration_residuals.csv").is_file()


def test_reports_distinguish_undetermined_observed_and_confirmed_arrivals():
    metrics = {"arrival_status": "undetermined"}
    assert "indeterminada" in arrival_note(metrics, "pt")
    assert "undetermined" in arrival_note(metrics, "en")
    obs = ar.validate_observation(observed())
    arrival = ar.entry_anchor(obs, dlt_projection_matrix(L))
    arrival["diagnostics"]["fit_residual_m"] = 0.1
    for lang, term in (("pt", "Entrada observada"), ("en", "Observed entry")):
        text = arrival_note({"arrival_status": "entry", "arrival": arrival}, lang)
        assert term in text and "45" in text and "0.100 m" in text
    assert (
        confidence({"anchor_residuals": {"p0_m": 0.1}, "quality": "review calibration"})[0]
        == "baixa"
    )


def test_topview_retains_the_same_valid_goal_crossing_as_csv_and_c3d(tmp_path):
    from vaila.freekiki_topview import _read_ball

    path = tmp_path / "ball_3d_on_field.csv"
    path.write_text(
        "frame,x_m,y_m,z_m,on_field\n0,52.4,3.0,1.2,True\n1,52.6,3.0,1.2,False\n2,nan,nan,nan,False\n"
    )
    assert _read_ball(path) == {0: (52.4, 3.0), 1: (52.6, 3.0)}


def test_goal_entry_is_never_reported_as_the_keepers_hands():
    from vaila.freekiki_report import event_frames
    from vaila.freekiki_report_i18n import localize_anchor_label

    anchors = {"goal_entry": {"frame": 14, "xyz": [52.45, 1, 1], "type": "entry"}}
    for wall in (True, False):
        _wall, _keeper, hands = event_frames(anchors, 14, wall=wall)
        assert np.isnan(hands).all()
    assert "hands" not in localize_anchor_label(
        "en", "chegada: entrada no gol (observada; plano da baliza)"
    )


def test_an_observation_is_never_silently_ignored_when_no_kick_can_be_fitted():
    cameras = dict.fromkeys(range(N), L)
    observation = observed(N - 1)
    pixels = dict.fromkeys(range(N), tuple(observation["pixel"]))
    with pytest.raises(ValueError, match="cannot fit the observed arrival"):
        af.auto_flight(
            ball_3d(_ball_rows(pixels), cameras),
            pixels,
            cameras,
            [],
            FPS,
            N,
            arrival_observation=observation,
        )
