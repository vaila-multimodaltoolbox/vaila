"""Synthetic tests for the Carlos Miguel comparative keeper report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vaila import freekiki_keeper_story as story
from vaila.freekiki_keeper_story import (
    build_compare_html,
    inject_keeper_link,
    load_penalty_card,
    write_compare_reports,
)

cv2 = pytest.importorskip("cv2")


def _png(path: Path, color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((60, 80, 3), color, dtype=np.uint8))


def _mp4(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(6):
        w.write(np.full((48, 64, 3), 40 * i, dtype=np.uint8))
    w.release()


def _cam(tmp: Path, *, name: str, centre: float, lat: float, vel_ms: float, onset: float) -> Path:
    cam = tmp / name
    payload = {
        "available": True,
        "event": "save",
        "anthropometrics": {"gk_height_m": 2.04, "gk_arm_span_m": 2.09},
        "metrics": {
            "gk_height_m": 2.04,
            "gk_arm_span_m": 2.09,
            "vel_ms": vel_ms,
            "dist": 10.7,
            "flight_time_s": 0.36,
            "ball_entry_x_from_centre_m": centre,
            "gk_response_time": -onset / 1000.0,
            "zone_label": "Middle Centre",
            "shot_outcome_label": "Save",
            "launch_elevation_deg": 8.0,
            "reach_dive_m": 2.065,
            "gk_standing_reach_m": 2.55,
            "kicker_height_m": 1.82,
        },
        "keeper": {
            "onset_ms_before_kick": onset,
            "onset_auto_ms_before_kick": onset + 30,
            "dive_side": "left",
            "lateral_displacement_m": lat,
            "com_drop_m": 0.4,
            "peak_lateral_speed_mps": 2.3,
            "hands_ball_gap_m": 0.1,
        },
        "counterfactual": {"y_m": -centre, "z_m": 1.04, "dist_to_nearest_post_m": 3.0,
                           "hwr_accuracy_pct": 18.0, "zone_label": "Middle Centre"},
    }  # fmt: skip
    (cam / "data").mkdir(parents=True)
    (cam / "data" / "penalty_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    (cam / "data" / "kicker_metrics.json").write_text(
        json.dumps({"events": {"kick_foot": "left"},
                    "metrics": {"kick_foot": "left", "approach_angle_deg": 2.17,
                                "v_foot_pre_mps": 8.35}}),
        encoding="utf-8",
    )  # fmt: skip
    (cam / "ball_flight.json").write_text(
        json.dumps({"peak_speed_mps": 32.31, "spin_rps": 12.16, "rms_px": 1.6, "quality": "ok"}),
        encoding="utf-8",
    )
    for lang, html in (("pt", "relatorio_bola.html"), ("en", "ball_report.html")):
        p = cam / "report_tv" / lang / html
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("<html><body><h1>bola</h1></body></html>", encoding="utf-8")
        _png(cam / "report_tv" / lang / "figuras" / "fig_perfil.png", 90)
    pen = cam / "report_tv" / "penalty"
    for k in ("kick", "gk_move", "goal"):
        _png(pen / "snapshots" / f"snapshot_{k}.png", 120)
    _mp4(pen / "media" / "animacao_lance.mp4")
    (pen / "report_pt.html").write_text("<html><body>p</body></html>", encoding="utf-8")
    (pen / "report.html").write_text("<html><body>p</body></html>", encoding="utf-8")
    return cam


def _tras(tmp: Path, name: str) -> Path:
    tras = tmp / name
    _png(tras / "WhatsApp Image 1.jpeg", 200)
    _mp4(tras / "c2_tras.mp4")
    qm = tras / "processed_quickmeasure_1" / "quickmeasure_report.html"
    qm.parent.mkdir(parents=True)
    qm.write_text("<html></html>", encoding="utf-8")
    return tras


def test_load_card_reads_all_sources(tmp_path: Path) -> None:
    cam = _cam(tmp_path, name="p1", centre=0.66, lat=0.357, vel_ms=29.7, onset=174.7)
    card = load_penalty_card(cam, tras=_tras(tmp_path, "t1"))
    assert card["source"] == "penalty_metrics"
    assert abs(card["peak_kmh"] - 32.31 * 3.6) < 1e-6
    assert abs(card["mean_kmh"] - 29.7 * 3.6) < 1e-6
    assert abs(card["onset_ms"] - 174.7) < 1e-6
    assert abs(card["lat_m"] - 0.357) < 1e-6
    assert card["kick_foot_raw"] == "left" and abs(card["approach_deg"] - 2.17) < 1e-6
    assert set(card["images"]) >= {"snap_kick", "snap_gk_move", "snap_goal"}
    assert "rear" in card["videos"] and card["quickmeasure"]


def test_flight_only_ignores_rest_frames(tmp_path: Path) -> None:
    cam = tmp_path / "p2_zeros"
    cam.mkdir()
    (cam / "ball_flight.json").write_text(
        json.dumps({"peak_speed_mps": 29.84, "speed_mps": [0.0, 0.1, 1.0, 20.0, 28.0, 27.0]}),
        encoding="utf-8",
    )
    card = load_penalty_card(cam)
    assert card["source"] == "ball_flight"
    assert card["mean_kmh"] is not None and card["mean_kmh"] > 80


def test_compare_html_has_metrics_and_embedded_images(tmp_path: Path) -> None:
    c1 = load_penalty_card(
        _cam(tmp_path, name="p1", centre=0.66, lat=0.357, vel_ms=29.7, onset=174.7)
    )
    c2 = load_penalty_card(
        _cam(tmp_path, name="p2", centre=3.594, lat=1.864, vel_ms=30.63, onset=166.9)
    )
    html = build_compare_html(c1, c2, lang="pt", out_dir=tmp_path / "out")
    assert "Carlos Miguel" in html and "2.04" in html
    for value in ("106.9", "110.3", "175", "167", "0.66", "3.59", "1.86", "18.0"):
        assert value in html, value
    assert "data:image/jpeg;base64," in html
    assert "Por que importa" in html
    assert "narrativa" not in html


def test_the_speed_block_sets_the_mean_and_the_exit_is_an_estimate(tmp_path: Path) -> None:
    """The mean speed (distance over the sub-frame flight time) is the headline; the exit is the model's."""
    cams = []
    for name, centre, lat in (("p1", 0.66, 0.357), ("p2", 3.594, 1.864)):
        cam = _cam(tmp_path, name=name, centre=centre, lat=lat, vel_ms=29.7, onset=174.7)
        f = cam / "data" / "penalty_metrics.json"
        d = json.loads(f.read_text(encoding="utf-8"))
        d["speed"] = {"flight_time_s": 0.3592,
                      "mean": {"ms": 29.7, "kmh": 106.9, "uncertainty_kmh": 4.6},
                      "exit_model": {"launch_ms": 33.19, "launch_kmh": 119.5}}  # fmt: skip
        d["counterfactual"]["dist_to_nearest_post_m"] = -0.031  # a few centimetres outside the post
        f.write_text(json.dumps(d), encoding="utf-8")
        cams.append(load_penalty_card(cam))
    card = cams[0]
    assert abs(card["mean_kmh"] - 29.7 * 3.6) < 1e-6 and abs(card["peak_kmh"] - 33.19 * 3.6) < 1e-6
    assert abs(card["flight_ms"] - 359.2) < 1e-6
    html = build_compare_html(cams[0], cams[1], lang="pt", out_dir=tmp_path / "out")
    assert "106.9" in html and "estimativa" in html and "muito melhor" not in html
    assert (
        "não dá para dizer se entraria" in html
    )  # the post is inside the measurement's uncertainty
    assert (
        "−" not in html.split("dentro da incerteza da medida")[0][-80:]
    )  # a distance, not a signed number
    assert "Bola (pico)" not in html and "Ângulo de aproximação" not in html


def test_write_reports_media_names_and_inject(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(story, "_print_pdf", lambda h, p: None)
    c1 = _cam(tmp_path, name="p1", centre=0.66, lat=0.357, vel_ms=29.7, onset=174.7)
    c2 = _cam(tmp_path, name="p2", centre=3.594, lat=1.864, vel_ms=30.63, onset=166.9)
    out = tmp_path / "out"
    written = write_compare_reports(
        out, c1, c2, tras1=_tras(tmp_path, "t1"), tras2=_tras(tmp_path, "t2")
    )
    assert Path(written["pt"]).is_file() and Path(written["en"]).is_file()
    media = {p.name for p in (out / "media").iterdir()}
    assert {"p1_animacao_lance.mp4", "p2_animacao_lance.mp4"} <= media
    assert {"p1_tras_c2_tras.mp4", "p2_tras_c2_tras.mp4"} <= media
    body = Path(written["pt"]).read_text(encoding="utf-8")
    assert "src='media/p1_animacao_lance.mp4'" in body
    assert "src='media/p2_animacao_lance.mp4'" in body
    assert "quickmeasure_report.html" in body
    target = c1 / "report_tv" / "pt" / "relatorio_bola.html"
    assert inject_keeper_link(target, written["pt"], lang="pt")
    injected = target.read_text(encoding="utf-8")
    assert "vaila-keeper-story" in injected and "file://" not in injected
    assert "../" in injected and "relatorio_carlos_miguel.html" in injected


def test_pdf_written_when_chrome_present(tmp_path: Path) -> None:
    from vaila.freekiki_report import find_chrome

    if find_chrome() is None:
        pytest.skip("no Chrome")
    c1 = _cam(tmp_path, name="p1", centre=0.66, lat=0.357, vel_ms=29.7, onset=174.7)
    c2 = _cam(tmp_path, name="p2", centre=3.594, lat=1.864, vel_ms=30.63, onset=166.9)
    written = write_compare_reports(tmp_path / "out", c1, c2)
    assert Path(written["pt_pdf"]).stat().st_size > 1000
