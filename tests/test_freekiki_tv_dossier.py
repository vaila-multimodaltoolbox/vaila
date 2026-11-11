"""The TV dossier of a saved penalty (vaila/freekiki_tv_dossier.py): cards, talking points, technical dossier.

The numbers come from a synthetic ``data/penalty_metrics.json`` that has the shape of a real one
(Lescano x Carlos Miguel, 08 Nov 2026). The tests check that every number of the text is the file's,
that the three counts of the prompt are reconciled by their own rates, that the two wrong speeds the
prompt names (118 and 80 km/h) are explained with the file's numbers, and that the files are written.

Version: 0.4.15
Update Date: 09 November 2026
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaila import freekiki_tv_dossier as dos

SPEED = {
    "fps": 30000 / 1001,
    "frame_ms": 33.37,
    "kick": {"frame": 79.23, "time_s": 2.6438, "first_moved_frame": 80, "counted_frame": 79,
             "pixel_check": {"frame": 79.237, "low": 79.189, "high": 79.272, "variants": 4},
             "agrees_with_pixels": True},
    "save": {"frame": 90, "time_s": 3.003},
    "flight_time_s": 0.3592,
    "flight_frames_counted": 11,
    "flight_time_counted_s": 0.367,
    "distance": {"spot_to_hands_m": 10.667, "path_m": 10.732, "to_line_m": 11.059, "nominal_m": 11.0},
    "mean": {"ms": 29.7, "kmh": 106.9, "kmh_counted": 104.6, "uncertainty_kmh": 4.6,
             "range_kmh": [102.3, 111.5]},
    "exit_model": {"launch_ms": 33.19, "launch_kmh": 119.5, "peak_ms": 32.31, "peak_kmh": 116.3,
                   "range_kmh": [117.3, 121.8]},
    "at_save_model": {"ms": 27.25, "kmh": 98.1},
    "sigma": {"kick_frames": 0.25, "save_frames": 0.25, "distance_m": 0.3},
}  # fmt: skip

METRICS = {
    "available": True,
    "event": "save",
    "goal_x_m": 52.45,
    "kick_frame": 80,
    "save_frame": 90,
    "speed": SPEED,
    "interception": {"frame": 90, "xyz": [52.0604, -0.5814, 1.0354], "in_front_of_line_m": 0.39,
                     "height_m": 1.035, "lateral_m": -0.581, "time_s_after_kick": 0.3592,
                     "distance_from_spot_m": 10.667, "mean_speed_mps": 29.7},
    "counterfactual": {"y_m": -0.6598, "z_m": 1.0375, "inside_goal_mouth": True,
                       "dist_to_nearest_post_m": 3.0, "hwr_accuracy_pct": 18.0,
                       "zone_label": "Middle Centre"},
    "uncertainty": {"along_camera_ray": [
        {"shift_m": -0.4, "interception_xyz": [51.8358, -0.8902, 1.1083]},
        {"shift_m": 0.4, "interception_xyz": [52.2852, -0.2729, 0.9626]},
    ]},
    "keeper": {"onset_frame": 74, "onset_ms_before_kick": 174.7, "movement_time_s": 0.5339,
               "dive_side": "left", "lateral_displacement_m": 0.357, "peak_lateral_speed_mps": 2.327},
    "anthropometrics": {"gk_height_m": 2.04, "gk_arm_span_m": 2.09, "gk_mass_kg": 94.0},
}  # fmt: skip

FILE_FPS = 30000 / 1001
CAMERAS = [
    {"name": "Câmera 1 (principal)", "file_fps": FILE_FPS, "capture_fps": FILE_FPS, "frames": 11,
     "source": "measured"},
    {"name": "Câmera 2 (atrás do gol)", "file_fps": FILE_FPS, "capture_fps": 50.0, "frames": 19,
     "source": "declared"},
    {"name": "Câmera 3 (câmera lenta)", "file_fps": FILE_FPS, "capture_fps": 119.88, "frames": 42,
     "source": "declared"},
]  # fmt: skip


def _content(**kw):
    return dos.dossier_content(METRICS, names={"keeper": "Carlos Miguel", "kicker": "Alan Lescano"},
                               match="Vasco x Palmeiras", cameras=kw.get("cameras"))  # fmt: skip


def test_numbers_are_written_with_a_decimal_comma() -> None:
    assert dos.pt(106.9, 0) == "107" and dos.pt(0.3592, 3) == "0,359" and dos.pt(29.7, 1) == "29,7"
    assert dos.pt(None) == "—"
    assert (
        dos.pt(1.035, 2) == "1,04" and dos.pt(0.125, 2) == "0,13"
    )  # half up, as the other reports
    assert dos.frames_text(17.5) == "17,5" and dos.frames_text(11.0) == "11"


def test_the_three_cards_carry_the_files_numbers() -> None:
    cards = _content()["cards"]
    assert [c["id"] for c in cards] == ["speed", "decision", "interception"]
    speed, decision, hit = cards
    assert speed["headline"] == "≈ 107 km/h" and "29,7 m/s" in speed["sub"]
    assert "102 a 112" in speed["note"]
    assert (
        "120" in speed["aside"] and "estimativa" in speed["aside"]
    )  # the exit speed is not the headline
    assert decision["headline"] == "0,36 s" and "36 centésimos" in decision["sub"]
    assert "11 quadros" in decision["note"] and "29,97" in decision["note"]
    assert hit["headline"].startswith("≈ 0,4 m") and "à frente da linha" in hit["headline"]
    assert "0,2" in hit["note"] and "0,6" in hit["note"]  # the spread along the camera ray


def test_the_talking_points_are_four_short_plain_statements() -> None:
    points = _content()["points"]
    assert len(points) == 4 and all(len(p["text"]) < 700 for p in points)
    text = " ".join(p["text"] for p in points)
    assert "107 km/h" in text and "175 ms" in text and "Carlos Miguel" in text
    assert (
        "valor do prompt" in text
    )  # the visuomotor latencies come from the prompt, not from a source
    assert "0,53 s" in text or "534 ms" in text  # the dive lasted 0.534 s, the ball took 0.359 s


def test_the_main_camera_alone_gives_a_table_of_one_row() -> None:
    tech = _content()["technical"]
    assert len(tech["cameras"]) == 1 and tech["cameras"][0]["frames"] == 11
    assert tech["reconciliation"] is None


def test_the_three_cameras_are_reconciled_by_their_own_capture_rates() -> None:
    tech = _content(cameras=CAMERAS)["technical"]
    rows = {r["name"]: r for r in tech["cameras"]}
    assert rows["Câmera 2 (atrás do gol)"]["dt_s"] == pytest.approx(0.380, abs=1e-3)
    assert rows["Câmera 3 (câmera lenta)"]["tol_s"] == pytest.approx(0.0083, abs=1e-4)
    rec = tech["reconciliation"]
    assert rec["max_gap_ms"] == pytest.approx(1.4, abs=0.3)
    text = " ".join(tech["reconciliation_text"])
    assert "não são exatamente iguais" in text or "dentro da contagem" in text
    assert "0,63" in text  # 19 frames take 0.634 s to play at 29.97 fps: why the count is not time


def test_the_two_wrong_speeds_are_explained_with_the_files_numbers() -> None:
    why = _content()["technical"]["why_not"]
    by = {w["id"]: w for w in why}
    assert by["118"]["kmh"] == pytest.approx(119.2, abs=0.6)  # 11.06 m over 10 counted frames
    assert by["24fps"]["kmh"] == pytest.approx(83.8, abs=0.6)  # 10.667 m over 11 frames at 24 fps
    assert "0,77 quadro" in by["118"]["text"] and "29,97" in by["24fps"]["text"]


def test_the_sensitivity_grid_holds_the_prompts_scenario_and_the_files() -> None:
    grid = _content()["technical"]["grid"]
    assert grid["distances_m"][0] == 10.2 and 10.667 in grid["distances_m"]
    assert 0.35 in grid["times_s"] and 0.359 in grid["times_s"] and 0.367 in grid["times_s"]
    cell = {(r["d"], r["t"]): r["kmh"] for r in grid["cells"]}
    assert cell[(10.2, 0.35)] == pytest.approx(104.9, abs=0.1)  # the prompt's own number
    assert cell[(10.667, 0.359)] == pytest.approx(106.9, abs=0.3)
    assert grid["range_kmh"][0] < 102.0 and grid["range_kmh"][1] > 111.0


def test_a_penalty_that_was_not_saved_is_refused() -> None:
    with pytest.raises(ValueError, match="save"):
        dos.dossier_content({"available": True, "event": "goal"}, names={})


def test_build_writes_markdown_and_html_next_to_the_reports(tmp_path: Path) -> None:
    run = tmp_path / "c1_main_3d_20261109_000000"
    (run / "data").mkdir(parents=True)
    (run / "data" / "penalty_metrics.json").write_text(json.dumps(METRICS), encoding="utf-8")
    cams = tmp_path / "cameras.json"
    cams.write_text(json.dumps({"cameras": CAMERAS}), encoding="utf-8")
    out = dos.build(run, names={"keeper": "Carlos Miguel", "kicker": "Alan Lescano"},
                    match="Vasco x Palmeiras", cameras_file=cams, pdf=False)  # fmt: skip
    md = Path(out["md"]).read_text(encoding="utf-8")
    html = Path(out["html"]).read_text(encoding="utf-8")
    assert Path(out["md"]).parent == run / "report_tv" / "pt" and out["pdf"] is None
    for text in (md, html):
        assert "{{" not in text and "nan" not in text.lower().replace("finance", "")
        assert "BLOCO 1" in text.upper() and "BLOCO 2" in text.upper() and "BLOCO 3" in text.upper()
        assert "≈ 107 km/h" in text and "Câmera 3 (câmera lenta)" in text
    assert "<table" in html and "Navarro" in html


def test_a_run_without_the_metrics_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="penalty_metrics.json"):
        dos.build(tmp_path, names={})


def test_frame_check_finds_a_repeated_frame_in_a_video(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 29.97, (64, 48))  # ty: ignore[unresolved-attribute]
    for i in range(12):
        frame = np.full((48, 64, 3), 10 * i, dtype=np.uint8)
        frame[:, : 4 * i] = 200  # a bar that grows: every frame differs from the last...
        writer.write(frame)
        if i == 5:
            writer.write(frame)  # ...but this one is written twice
    writer.release()
    check = dos.frame_check(video)
    assert check is not None and check["frames"] == 13 and check["repeated"] == 1
    assert (
        check["diff_min"] < 0.3 * check["diff_median"] <= check["diff_median"] <= check["diff_max"]
    )
    assert dos.frame_check(tmp_path / "missing.avi") is None
    # a clean file puts the evidence in the text of the 24 fps explanation
    ev = {"frames": 91, "diff_min": 2.58, "diff_median": 3.41, "diff_max": 5.68, "repeated": 0}
    content = dos.dossier_content(METRICS, names={}, evidence=ev)
    text = content["technical"]["why_not"][1]["text"]
    assert "91 quadros" in text and "2,6" in text and "5,7" in text and "0 repetem" in text


def test_the_cli_writes_the_dossier_of_a_run(tmp_path: Path, capsys) -> None:
    from vaila import freekiki as fk

    run = tmp_path / "c1_main_3d_20261109_000000"
    (run / "data").mkdir(parents=True)
    (run / "data" / "penalty_metrics.json").write_text(json.dumps(METRICS), encoding="utf-8")
    args = fk.build_parser().parse_args(
        ["tv-dossier", "--run", str(run), "--goalkeeper", "Carlos Miguel", "--no-pdf"]
    )
    assert args.command == "tv-dossier" and args.goalkeeper == "Carlos Miguel" and args.no_pdf
    assert fk.main(["tv-dossier", "--run", str(run), "--kicker", "Alan Lescano", "--no-pdf"]) == 0
    seen = capsys.readouterr()
    assert "dossier written" in seen.out + seen.err
    html = (run / "report_tv" / "pt" / "dossie_tv.html").read_text(encoding="utf-8")
    assert "Alan Lescano" in html and "≈ 107 km/h" in html


def test_a_camera_with_events_gets_its_effective_rate_and_an_estimated_one_is_not_a_check() -> None:
    fps = SPEED["fps"]
    events = [  # onset, contact (a fraction of a frame) and save: the same events on both clocks
        {"name": "início do goleiro", "main_frame": 74, "frame": 32.0},
        {"name": "contato", "main_frame": 79.2348, "frame": 40.7},
        {"name": "defesa", "main_frame": 90, "frame": 59.0},
    ]
    cams = [
        CAMERAS[0],
        {**CAMERAS[1], "frames": 17.5, "events": events},
        {"name": "Câmera 4 (replay)", "file_fps": fps, "capture_fps": 59.94, "frames": 21.5,
         "source": "estimated"},
    ]  # fmt: skip
    tech = _content(cameras=cams)["technical"]
    assert {r["name"] for r in tech["cameras"]} == {c["name"] for c in cams}
    rec = tech["reconciliation"]
    assert [r["name"] for r in rec["cameras"]] == [CAMERAS[0]["name"], CAMERAS[1]["name"]]
    fit = tech["clock_fits"][CAMERAS[1]["name"]]
    assert 48.0 < fit["rate_hz"] < 51.0 and fit["n"] == 3
    text = " ".join(tech["reconciliation_text"])
    assert "taxa efetiva" in text and "não como checagem independente" in text
    row = {r["name"]: r for r in tech["cameras"]}[CAMERAS[1]["name"]]
    assert row["frames"] == 17.5
    cams[1]["note"] = "A contagem da equipe foi de 19 quadros."
    again = " ".join(_content(cameras=cams)["technical"]["reconciliation_text"])
    assert "A contagem da equipe foi de 19 quadros." in again
    md = dos.render_markdown(_content(cameras=cams))
    assert "17,5" in md and "Origem da taxa" in md
