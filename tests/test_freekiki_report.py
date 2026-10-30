"""Tests for the TV report of a 3D run (report_tv/): a small synthetic run, no GPU, no model.

The report is written in Portuguese (``report_tv/pt/``) and in English (``report_tv/en/``). The
interactive editor of the HTML runs :file:`vaila/freekiki_report_app.js`; the parity test
compares that JavaScript with the Python physics when ``node`` is installed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from vaila import freekiki_report as rep
from vaila.fifa_to_dlt import compute_dlt3d_from_KRt
from vaila.freekiki_detect3d import _project_point
from vaila.freekiki_physics import BALL_RADIUS, simulate_flight
from vaila.freekiki_report_i18n import fmt_num, localize_anchor_label
from vaila.freekiki_report_players import load_ground_tracks

W, H = 1920, 1080
N = 6
FPS = 30.0
APP_JS = Path(rep.APP_JS)


def _camera_coefficients(pan: float = 0.0) -> np.ndarray:
    """DLT3D coefficients of a camera looking at the kick from behind the pitch."""
    centre = np.array([-60.0 + pan, -45.0, 25.0])
    target = np.array([-30.0, -10.0, 0.0])
    z = target - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    t = -R @ centre
    K = np.array([[2400.0, 0.0, W / 2.0], [0.0, 2400.0, H / 2.0], [0.0, 0.0, 1.0]])
    return compute_dlt3d_from_KRt(K, R, t)


def _project(L: np.ndarray, xyz) -> tuple[float, float]:
    P = np.array([[L[0], L[1], L[2], L[3]], [L[4], L[5], L[6], L[7]], [L[8], L[9], L[10], 1.0]])
    h = P @ np.append(np.asarray(xyz, dtype=float), 1.0)
    return float(h[0] / h[2]), float(h[1] / h[2])


def _make_run(
    root: Path,
    *,
    residual: float = 0.1,
    with_anchors: bool = True,
    moving: bool = False,
    with_people: bool = False,
) -> Path:
    """A 3D folder: the README with the video, ball_flight.json, the track, the markers and work/."""
    video = root / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))  # ty: ignore[unresolved-attribute]
    for _ in range(N):
        writer.write(np.full((H, W, 3), 60, dtype=np.uint8))
    writer.release()
    run = root / "clip_3d_20260101_000000"
    detect = run / "work" / "freekiki_predict_clip_1"
    detect.mkdir(parents=True)
    # the broadcast camera pans from frame to frame when ``moving``: one DLT per frame
    Ls = [_camera_coefficients(pan=0.8 * f if moving else 0.0) for f in range(N)]
    (detect / "clip.dlt3d").write_text(
        "frame," + ",".join(f"L{i}" for i in range(1, 12)) + "\n"
        + "".join(f"{f}," + ",".join(f"{v:.12g}" for v in Ls[f]) + "\n" for f in range(N)),
        encoding="utf-8",
    )  # fmt: skip
    (run / "README.txt").write_text(
        f"FreeKiki 3D output (vailá)\nvideo: {video}\n", encoding="utf-8"
    )
    # A ball that rises and falls over the first frames, on the pitch.
    track = []
    for f in range(N):
        t = f / FPS
        track.append((-30.0 + 20.0 * t, -20.0 + 8.0 * t, 0.11 + 6.0 * t - 4.9 * t * t))
    with (run / "ball_3d_on_field.csv").open("w", encoding="utf-8") as fh:
        fh.write("frame,cx,cy,score,x_m,y_m,z_m,on_field\n")
        for f, (x, y, z) in enumerate(track):
            fh.write(f"{f},,,,{x},{y},{z},True\n")
    with (run / "ball.csv").open("w", encoding="utf-8") as fh:
        fh.write("frame,cx,cy,x1,y1,x2,y2,score\n")
        for f, (x, y, z) in enumerate(track):
            u, v = _project(Ls[f], (x, y, z))
            fh.write(f"{f},{u:.2f},{v:.2f},,,,,1.0\n")
    if with_people:
        _write_people(run, track)
    speed = [0.0] + [
        float(np.linalg.norm(np.subtract(track[i], track[i - 1])) * FPS) for i in range(1, N)
    ]
    anchors = {}
    if with_anchors:
        anchors = {
            "kick_foot": {"frame": 0, "xyz": [-30.0, -20.0, 0.11]},
            "barrier_shoulder": {"frame": 3, "xyz": [-24.0, -12.0, 1.2]},
            "barrier_target": {"frame": 3, "xyz": [-24.0, -12.0, 1.8]},
            "keeper_hands": {"frame": N - 1, "xyz": list(track[-1])},
        }
    flight = {
        "theta": [-30.0, -20.0, 20.0, 8.0, 6.0, 0.0, 0.0, 0.0, -30.0],
        "kick_time_s": 0.0,
        "spin_rps": 4.8,
        "spin_axis": [0.0, 0.0, -1.0],
        "rms_px": 1.5,
        "max_px": 3.0,
        "anchor_residuals": {"rest_m": residual, "barrier_distance_m": residual, "barrier_height_m": residual, "keeper_m": residual},
        "speed_mps": speed,
        "peak_speed_mps": max(speed),
        "above_fit_range": False,
        "success": True,
        "per_frame_px": {str(f): 1.0 + 0.1 * f for f in range(N)},
        "anchor_points": anchors,
    }  # fmt: skip
    (run / "ball_flight.json").write_text(json.dumps(flight), encoding="utf-8")
    return run


def _write_people(run: Path, track) -> None:
    """Four people on the ground: the kicker (id 1) at the kick, the wall (ids 2 and 3) at the
    wall frame, one on each side of the ball, and the keeper (id 4) at the entry."""
    spots = {
        1: (track[0][0], track[0][1]),
        2: (track[3][0], track[3][1] + 0.6),
        3: (track[3][0], track[3][1] - 0.6),
        4: (track[-1][0], track[-1][1]),
    }
    with (run / "poses_3d_on_field.csv").open("w", encoding="utf-8") as fh:
        fh.write("frame,person_id,kpt_idx,kpt_name,x_m,y_m,z_m\n")
        for f in range(N):
            for pid, (x, y) in spots.items():
                for k, name in enumerate(("left-ankle", "right-ankle")):
                    fh.write(f"{f},{pid},{k},{name},{x:.4f},{y:.4f},0.0\n")


def _html(result: dict, lang: str) -> str:
    return Path(result["languages"][lang]["html"]).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("worst", "tier"),
    [(0.1, "alta"), (0.6, "média"), (1.5, "baixa")],
)
def test_confidence_tier_follows_the_worst_anchor(worst: float, tier: str) -> None:
    flight = {
        "anchor_residuals": {"rest_m": worst, "keeper_m": worst / 2, "barrier_height_m": None}
    }
    assert rep.confidence(flight)[0] == tier


def test_load_and_metrics_read_the_run_files(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    run = rep.load_run(run_dir)
    assert run.n == N and run.projections
    m = rep.compute_metrics(run)
    assert m["kick_frame"] == 0 and m["end_frame"] == N - 1
    assert m["distance_m"] > 0 and m["flight_s"] == pytest.approx((N - 1) / FPS, rel=0.01)
    assert m["tier"] == "alta"  # the fixture's anchors are 0.1 m off
    assert np.isfinite(m["re"][1:]).all() and np.isfinite(m["cd"][1:]).all()
    assert m["wall_z"] == pytest.approx(run.track[3, 2])


def test_both_languages_are_written_without_placeholders(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, residual=1.5)
    result = rep.build_report(run_dir, animation=False, names="Fulano", authors="Equipe")
    assert set(result["languages"]) == {"pt", "en"}
    for lang in ("pt", "en"):
        html = _html(result, lang)
        assert "{{" not in html and "}}" not in html
        assert "Fulano" in html and "Equipe" in html
        figs = Path(result["languages"][lang]["dir"]) / "figuras"
        assert (figs / "fig_perfil.png").is_file() and (figs / "fig_tempo.png").is_file()
        assert result["languages"][lang]["animation"] is None
    assert "Confiança baixa" in _html(
        result, "pt"
    )  # the warning appears when a point misses by 1 m
    assert "Fit confidence: low" in _html(result, "en")


def test_portuguese_and_english_texts_do_not_mix(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    result = rep.build_report(run_dir, animation=False, lang="both")
    pt, en = _html(result, "pt"), _html(result, "en")
    assert "A bola percorreu" in pt and "The ball travelled" not in pt
    assert "The ball travelled" in en and "A bola percorreu" not in en
    assert "relatório para imprensa" in pt and "press report" in en
    # numbers: a decimal comma in Portuguese, a decimal point in English
    assert re.search(r"\d+,\d+ m\b", re.sub(r"<[^>]+>", " ", pt))
    assert re.search(r"\d+\.\d+ m\b", re.sub(r"<[^>]+>", " ", en))


def test_single_language_writes_one_folder(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    result = rep.build_report(run_dir, animation=False, lang="en")
    assert list(result["languages"]) == ["en"]
    assert (Path(result["dir"]) / "en" / "ball_report.html").is_file()
    assert not (Path(result["dir"]) / "pt").exists()
    with pytest.raises(ValueError, match="lang must be"):
        rep.build_report(run_dir, animation=False, lang="fr")


def test_report_without_anchor_points_still_builds(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, residual=0.1, with_anchors=False)
    result = rep.build_report(run_dir, animation=False)
    for lang in ("pt", "en"):
        html = _html(result, lang)
        assert "{{" not in html
    assert "Confiança baixa no ajuste" not in _html(
        result, "pt"
    )  # no warning when anchors are close
    assert "Fit confidence: low" not in _html(result, "en")


def test_animation_is_written_in_each_language(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    result = rep.build_report(run_dir, animation=True)
    for lang, name in (("pt", "animacao_lance.mp4"), ("en", "kick_animation.mp4")):
        mp4 = Path(result["languages"][lang]["animation"])
        assert mp4.name == name and mp4.is_file() and mp4.stat().st_size > 0
        cap = cv2.VideoCapture(str(mp4))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # every source frame is repeated (slow motion) and the key moments are held longer
        assert count >= N * 5


def test_editor_payload_is_embedded_and_valid(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    result = rep.build_report(run_dir, animation=False, lang="en")
    html = _html(result, "en")
    match = re.search(r'<script type="application/json" id="ed-data">(.*?)</script>', html, re.S)
    assert match, "the editor data must be in the HTML"
    payload = json.loads(match.group(1))
    assert payload["lang"] == "en" and payload["n"] == N
    assert len(payload["frames"]) == N and len(payload["frames"][0]["P"]) == 12
    # the fixture has the anchors of the first version (no labels): they come out translated
    labels = [p["label"] for p in payload["points"]]
    assert "barrier: ball over the wall" in labels and "end: keeper's hands" in labels
    assert all(p["sigma"] == pytest.approx(0.3) for p in payload["points"])
    assert "fitFlight" in html  # the script of the editor is inline
    assert "Recompute" in html and "ed-recalc" in html
    assert "</script" not in match.group(1)  # the JSON cannot close its own script element


def test_english_labels_of_the_dialog_are_recognised() -> None:
    # the label keeps no height: the barrier point of a run is placed by its own z
    assert localize_anchor_label("pt", "barrier: gap midpoint 15-2 at 1.13 m") == (
        "barreira: bola sobre a barreira"
    )
    assert localize_anchor_label("en", "barreira: bola a 1,8 m") == "barrier: ball over the wall"
    assert localize_anchor_label("en", "chegada: mãos do goleiro") == "end: keeper's hands"
    assert localize_anchor_label("pt", "meu ponto") == "meu ponto"
    assert fmt_num(2.44, 2, "pt") == "2,44" and fmt_num(2.44, 2, "en") == "2.44"
    assert fmt_num(float("nan"), 1, "en") == "—"


def test_event_frames_read_labelled_points_and_legacy_names() -> None:
    labelled = {
        "p0": {"label": "início: bola parada no pé", "frame": 0, "xyz": [-30.0, -20.0, 0.11]},
        "p1": {"label": "barreira: bola a 1,8 m", "frame": 9, "xyz": [-38.0, -14.0, 1.8]},
        "p2": {"label": "chegada: gol", "frame": 30, "xyz": [-52.45, 1.2, 0.11]},
    }
    wall, keeper, keeper_xyz = rep.event_frames(labelled, end_f=30)
    assert (wall, keeper) == (9, 30)
    assert keeper_xyz == pytest.approx([-52.45, 1.2, 0.11])
    english = {
        "p0": {"label": "start: ball on the kick foot", "frame": 0, "xyz": [-31.0, -19.0, 0.11]},
        "p1": {"label": "barrier: ball at 1.8 m (hypothesis)", "frame": 9, "xyz": [-38.5, -13.5, 1.8]},
        "p2": {"label": "end: keeper's hands", "frame": 30, "xyz": [-52.0, -1.7, 0.93]},
    }  # fmt: skip
    assert rep.event_frames(english, end_f=30)[:2] == (9, 30)
    legacy = {
        "kick_foot": {"frame": 0, "xyz": [-30.0, -20.0, 0.04]},
        "barrier_target": {"frame": 12, "xyz": [-30.0, -15.0, 1.8]},
        "keeper_hands": {"frame": 40, "xyz": [-52.0, -1.7, 0.9]},
    }
    assert rep.event_frames(legacy, end_f=40)[:2] == (12, 40)
    # no labels at all: the middle anchor is the wall, the last one the keeper
    unlabelled = {f"p{j}": {"frame": f, "xyz": [0.0, 0.0, 0.1]} for j, f in enumerate([0, 5, 20])}
    assert rep.event_frames(unlabelled, end_f=20)[:2] == (5, 20)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is needed for the JS parity test")
def test_editor_javascript_matches_the_python_physics(tmp_path: Path) -> None:
    """The browser editor simulates the same flight as freekiki_physics.simulate_flight."""
    run_dir = _make_run(tmp_path)
    run = rep.load_run(run_dir)
    payload = rep.editor_payload(run, rep.compute_metrics(run), "en")
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    theta = np.array(payload["theta"]) + np.array([0.3, -0.2, 0.5, 0.4, 0.2, 0.01, 2.0, -1.0, 0.5])
    driver = tmp_path / "parity.js"
    driver.write_text(
        "const fs = require('fs');\n"
        f"const app = require({json.dumps(str(APP_JS))});\n"
        f"const data = JSON.parse(fs.readFileSync({json.dumps(str(payload_path))}, 'utf8'));\n"
        f"const theta = {json.dumps(theta.tolist())};\n"
        "const times = Array.from({length: data.n}, (_, f) => f / data.fps);\n"
        "console.log(JSON.stringify(app.simulate(theta, times, data.rest_height)));\n",
        encoding="utf-8",
    )
    node = shutil.which("node") or "node"
    out = subprocess.run(
        [node, str(driver)], check=True, capture_output=True, text=True, timeout=120
    )
    js = np.array(json.loads(out.stdout))
    times = np.arange(payload["n"]) / payload["fps"]
    py = simulate_flight(theta, times, BALL_RADIUS)
    assert np.abs(js - py).max() < 1e-4  # metres: the RK4 step is far below the fit's noise


def test_overlay_points_use_the_camera_of_their_own_frame(tmp_path: Path) -> None:
    """The start anchor and the start of the path are the same 3D point: both are drawn with the
    camera of the frame shown, so they land on the same spot, and that spot moves with the camera."""
    from vaila.freekiki_report_video import overlay_geometry

    run = rep.load_run(_make_run(tmp_path, moving=True))
    m = rep.compute_metrics(run)
    last = N - 1
    geo = overlay_geometry(run, m, last)
    start = np.array(run.anchors["kick_foot"]["xyz"])
    now = np.array(_project_point(run.projections[last], start))
    then = np.array(_project_point(run.projections[0], start))
    assert np.allclose(geo["anchors"]["kick_foot"], now)
    assert np.linalg.norm(now - then) > 20  # the camera moved: the same point is elsewhere now
    assert np.allclose(geo["trail"][0], geo["anchors"]["kick_foot"])
    assert len(geo["trail"]) == last + 1 and geo["rest"] == []


def test_ground_tracks_are_the_mean_of_the_ankles(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, with_people=True)
    tracks = load_ground_tracks(run_dir, N)
    assert set(tracks) == {1, 2, 3, 4}
    run = rep.load_run(run_dir)
    assert tracks[2][3] == pytest.approx([run.track[3][0], run.track[3][1] + 0.6], abs=1e-3)


def test_people_are_found_by_the_nearest_point_and_can_be_given(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, with_people=True)
    result = rep.build_report(run_dir, animation=True, lang="en")
    html = _html(result, "en")
    assert "kicker id 1, wall ids 2, 3, goalkeeper id 4" in html
    assert "Found by the nearest point" in html
    assert Path(
        result["languages"]["en"]["animation"]
    ).is_file()  # the people are drawn in the video
    manual = rep.build_report(
        run_dir, animation=False, lang="en",
        players={"kicker_id": 1, "wall_ids": [3, 2], "keeper_id": 4},
    )  # fmt: skip
    manual_html = _html(manual, "en")
    assert "wall ids 3, 2" in manual_html and "Given by" in manual_html


def test_a_penalty_has_no_wall_and_the_keeper_is_never_a_wall_player() -> None:
    from vaila.freekiki_report_players import find_people

    n = 30
    ball = np.tile([10.0, 0.0], (n, 1))
    tracks = {
        0: np.tile([50.0, 0.0], (n, 1)),  # the keeper, at the goal
        1: np.tile([10.0, 0.0], (n, 1)),  # the kicker, at the ball
        2: np.tile([10.5, 0.5], (n, 1)),  # a player within 2 m of the ball at the wall frame
    }
    common = {
        "kick_f": 5,
        "barrier_f": 10,
        "end_f": 20,
        "keeper_xy": np.array([50.0, 0.0]),
        "kicker_id": 1,
        "keeper_id": 0,
    }
    penalty = find_people(tracks, ball, penalty=True, **common)
    assert penalty.wall == []
    assert penalty.kicker == 1 and penalty.keeper == 0
    free_kick = find_people(tracks, ball, penalty=False, **common)
    assert 2 in free_kick.wall and 0 not in free_kick.wall


# --------------------------------------------------------------------------- #
# Penalty kick: no wall, the goalkeeper's hands instead
# --------------------------------------------------------------------------- #
KEEPER_SAM, KICKER_SAM = 4, 1
HANDS_START = np.array([-5.0, -1.0, 1.0])  # the keeper's hands at the kick (pitch metres)
HANDS_END = np.array([-5.0, 0.5, 1.4])  # ... and in the last frame


def _make_penalty(root: Path) -> Path:
    """The synthetic run of a penalty: the anchors the pipeline writes (start, the wall plane, the
    keeper's hand), ``data/setpiece.json``, the player map and the keeper's wrists in the poses."""
    run = _make_run(root, with_people=True)
    flight_path = run / "ball_flight.json"
    flight = json.loads(flight_path.read_text(encoding="utf-8"))
    flight["anchor_points"] = {
        "p0": {"label": "start: contato do pé", "frame": 0, "xyz": [-30.0, -20.0, 0.11]},
        "p_wall": {
            "label": "barreira: plano a 9,15 m do chute (regra FIFA)",
            "frame": 3,
            "xyz": [-24.0, -12.0, 1.1],
        },
        "p_arrival": {
            "label": "chegada: mão do goleiro no último quadro",
            "frame": N - 1,
            "xyz": [-5.0, 0.0, 1.5],
        },
    }
    flight_path.write_text(json.dumps(flight), encoding="utf-8")
    (run / "data").mkdir()
    (run / "data" / "setpiece.json").write_text(
        json.dumps({"kind": "penalty", "goal_x_m": 0.0, "wall_ids": []}), encoding="utf-8"
    )
    (run / "player_map.csv").write_text(
        f"player_id,role,camera,sam_id\n0,keeper,cam,{KEEPER_SAM}\n1,kicker,cam,{KICKER_SAM}\n",
        encoding="utf-8",
    )
    with (run / "poses_3d_on_field.csv").open("a", encoding="utf-8") as fh:
        for f in range(N):
            hands = HANDS_START + (HANDS_END - HANDS_START) * f / (N - 1)
            for k, (name, dy) in enumerate((("left-wrist", 0.1), ("right-wrist", -0.1))):
                x, y, z = hands + np.array([0.0, dy, 0.0])
                fh.write(f"{f},{KEEPER_SAM},{9 + k},{name},{x:.4f},{y:.4f},{z:.4f}\n")
    return run


def test_the_penalty_is_read_from_the_setpiece_and_the_hands_from_the_wrists(
    tmp_path: Path,
) -> None:
    run = rep.load_run(_make_penalty(tmp_path))
    assert run.penalty and run.goal_x == 0.0
    assert run.keeper_hands is not None and run.keeper_hands.shape == (N, 3)
    np.testing.assert_allclose(run.keeper_hands[0], HANDS_START, atol=1e-3)  # the two wrists' mean


def test_a_free_kick_is_not_a_penalty(tmp_path: Path) -> None:
    run = rep.load_run(_make_run(tmp_path, with_people=True))
    assert not run.penalty and run.keeper_hands is None
    m = rep.compute_metrics(run)
    assert not m["penalty"] and m["hands_start"] is None and m["hands_travel_m"] is None
    assert set(m["anchors"]) == set(run.anchors)  # a free kick draws every anchor


def test_penalty_metrics_drop_the_wall_and_measure_the_hands(tmp_path: Path) -> None:
    run = rep.load_run(_make_penalty(tmp_path))
    m = rep.compute_metrics(run)
    assert m["penalty"] and m["barrier_f"] == m["end_frame"]
    assert set(m["anchors"]) == {"p0", "keeper_hands_start", "keeper_hands"}  # no wall, no arrival
    np.testing.assert_allclose(m["hands_start"], HANDS_START, atol=1e-3)
    np.testing.assert_allclose(m["hands_end"], HANDS_END, atol=1e-3)
    assert m["hands_travel_m"] == pytest.approx(np.linalg.norm(HANDS_END - HANDS_START), abs=1e-3)
    assert m["keeper_gap_m"] == pytest.approx(
        np.linalg.norm(HANDS_END - run.track[m["end_frame"]]), abs=1e-3
    )
    assert m["anchors"]["keeper_hands_start"]["frame"] == m["kick_frame"]
    assert m["anchors"]["keeper_hands"]["frame"] == m["end_frame"]
    for key in ("barrier_t", "barrier_kmh", "after_wall_s", "wall_z", "wall_clear_cm"):
        assert np.isnan(m[key]), key


def test_a_penalty_without_the_keepers_wrists_keeps_the_anchors_of_the_fit(tmp_path: Path) -> None:
    run_dir = _make_penalty(tmp_path)
    poses = run_dir / "poses_3d_on_field.csv"
    poses.write_text(
        "\n".join(ln for ln in poses.read_text(encoding="utf-8").splitlines() if "wrist" not in ln)
        + "\n",
        encoding="utf-8",
    )
    run = rep.load_run(run_dir)
    assert run.penalty and run.keeper_hands is None
    m = rep.compute_metrics(run)
    assert m["hands_start"] is None and m["hands_end"] is None
    assert set(m["anchors"]) == {"p0", "p_arrival"}  # only the wall is dropped
    for lang in ("pt", "en"):
        html = _html(rep.build_report(run_dir, animation=False, lang=lang), lang)
        assert "{{" not in html
    assert "In the last frame the hands were" in html  # the hand of the arrival anchor
    flight_path = run_dir / "ball_flight.json"
    flight = json.loads(flight_path.read_text(encoding="utf-8"))
    del flight["anchor_points"]["p_arrival"]  # no hand at all now
    flight_path.write_text(json.dumps(flight), encoding="utf-8")
    html = _html(rep.build_report(run_dir, animation=False, lang="en"), "en")
    assert "could not be measured" in html


def test_hands_missing_on_a_frame_are_taken_from_the_nearest_one() -> None:
    hands = np.full((8, 3), np.nan)
    hands[2] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(rep._hands_at(hands, 3), [1.0, 2.0, 3.0])  # one frame away
    assert rep._hands_at(hands, 7) is None  # further than HANDS_REACH_FRAMES
    assert rep._hands_at(None, 0) is None


def test_keeper_wrists_of_an_unknown_person_is_none(tmp_path: Path) -> None:
    run_dir = _make_penalty(tmp_path)
    assert rep.keeper_wrists(run_dir, 99, N) is None
    assert rep.keeper_wrists(tmp_path / "nothing", KEEPER_SAM, N) is None


@pytest.mark.parametrize(
    ("lang", "gone", "present"),
    [
        ("pt", ("Por cima da barreira", "O desafio da falta", "A altura da barreira"), ("As mãos do goleiro", "No pênalti não há barreira", "O voo da bola")),
        ("en", ("Over the wall", "The free kick is a geometry problem", "The height of the wall"), ("The goalkeeper's hands", "A penalty kick has no wall", "The flight of the ball")),
    ],
)  # fmt: skip
def test_the_penalty_report_has_no_wall_text_and_shows_the_hands(
    tmp_path: Path, lang: str, gone: tuple[str, ...], present: tuple[str, ...]
) -> None:
    result = rep.build_report(_make_penalty(tmp_path), animation=False, lang=lang)
    html = _html(result, lang)
    assert "{{" not in html and "<!--pk" not in html and "<!--fk" not in html
    for text in gone:
        assert text not in html, text
    for text in present:
        assert text in html, text
    assert "fig_wall" not in html
    figs = Path(result["languages"][lang]["dir"]) / "figuras"
    assert (figs / "fig_perfil.png").is_file() and (figs / "fig_vista_de_cima.png").is_file()


def test_the_free_kick_report_keeps_the_wall_text(tmp_path: Path) -> None:
    result = rep.build_report(_make_run(tmp_path), animation=False, lang="en")
    html = _html(result, "en")
    assert "Over the wall" in html and "The free kick is a geometry problem" in html
    assert "The goalkeeper's hands" not in html and "<!--pk" not in html


def test_the_penalty_animation_shows_the_hands_and_no_wall(tmp_path: Path) -> None:
    from vaila.freekiki_report_video import _event, overlay_geometry

    run = rep.load_run(_make_penalty(tmp_path))
    m = rep.compute_metrics(run)
    kf, ef = m["kick_frame"], m["end_frame"]
    assert _event(kf, m, "en")[0] == "KICK"
    assert _event(kf + 1, m, "en")[0] == "Ball in flight"
    assert _event(ef, m, "en")[0] == "Last frame"
    assert _event(ef, m, "pt")[0] == "Último quadro"
    mid = overlay_geometry(run, m, kf + 1)["anchors"]
    last = overlay_geometry(run, m, ef)["anchors"]
    assert "p_wall" not in mid and mid["keeper_hands_start"] is not None
    assert (
        mid["keeper_hands"] is None and last["keeper_hands"] is not None
    )  # the end appears at the end
    result = rep.build_report(run.run_dir, animation=True, lang="en")
    assert Path(result["languages"]["en"]["animation"]).is_file()


def test_the_penalty_note_names_the_kicker_and_the_keeper_only(tmp_path: Path) -> None:
    result = rep.build_report(_make_penalty(tmp_path), animation=False, lang="en")
    html = _html(result, "en")
    assert "penalty taker id 1 and goalkeeper id 0" in html  # ids of the player map
    assert "wall ids" not in html
