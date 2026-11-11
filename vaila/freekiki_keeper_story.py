"""
================================================================================
Script: freekiki_keeper_story.py - comparative goalkeeper coaching report (HTML + PDF)
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.16
Created: 09 November 2026
Update Date: 11 November 2026

Description:
    Builds a side-by-side TV report for two penalties faced by the same
    goalkeeper (e.g. Carlos Miguel, Palmeiras) from the FreeKiki + Pynalty
    save-3D outputs of each run: ball (speed, spin, crossing point, HWR),
    goalkeeper (anticipation, centre-of-mass travel and speed, step, hands gap),
    kicker (foot, approach, foot speed) and fit quality. Snapshots, figures and
    rear-camera photos are embedded in the HTML; videos are copied next to it
    with a poster frame. A PDF of each language is printed with headless Chrome.

    CLI::

        uv run python -m vaila.freekiki_keeper_story \\
            --penalti1 <cam3d> --penalti2 <cam3d> \\
            --tras1 <tras_dir> --tras2 <tras_dir> \\
            -o <output_dir> --inject [--no-pdf]
"""

from __future__ import annotations

import argparse
import base64
import html as html_lib
import json
import math
import mimetypes
import os
import re
import shutil
from datetime import datetime as dt
from pathlib import Path
from typing import Any
from urllib.parse import quote

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - optional
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Narrative (match-specific copy; every number is filled from the runs)
# ---------------------------------------------------------------------------

STORY: dict[str, Any] = {
    "keeper_name": "Carlos Miguel",
    "team": "Palmeiras",
    "match": "Vasco × Palmeiras",
    "stature_m": 2.04,
    "arm_span_m": 2.09,
    "mass_kg": 94.0,
    "pt": {
        "title": "Carlos Miguel nos pênaltis — leitura do goleiro",
        "subtitle": "Dois chutes, duas defesas: o quadril andou o mesmo, mas no segundo a bola foi mais para o canto e as mãos foram mais longe",
        "p1_title": "Pênalti 1 — defesa a {hit_lat} m do centro",
        "p2_title": "Pênalti 2 — defesa a {hit_lat} m do centro",
        "p1_body": (
            "Com {stature} m e {span} m de envergadura, Carlos Miguel encontrou a bola {hit_front} m à "
            "frente da linha, {hit_lat} m do centro do gol e {hit_z} m de altura. A bola, chutada com o pé "
            "{foot}, chegou às mãos a {mean_kmh} km/h de média em {flight_ms} ms (a saída do pé, estimada "
            "pelo modelo de voo em cerca de {peak_kmh} km/h, não é medida); sem a defesa, cruzaria a linha "
            "a {centre} m do centro ({zone}; {post} m da trave mais próxima; HWR {hwr}%). Ele antecipou "
            "o contato em {onset} ms, deslocou o quadril cerca de {lat} m (faixa de {lat_lo} a {lat_hi} m, "
            "medido na câmera de trás) para a {side} e levou as mãos {hands_lat} m para o lado, baixando o "
            "centro de massa {drop} m."
        ),
        "p2_body": (
            "No segundo pênalti a bola chegou às mãos a {mean_kmh} km/h de média em {flight_ms} ms "
            "(saída estimada em cerca de {peak_kmh} km/h), {hit_front} m à frente da linha, {hit_lat} m do "
            "centro e {hit_z} m de altura; sem a defesa, cruzaria a linha a {centre} m do centro ({zone}; "
            "{post} m da trave mais próxima; HWR {hwr}%). Carlos Miguel antecipou {onset} ms, deslocou o "
            "quadril cerca de {lat} m (faixa de {lat_lo} a {lat_hi} m, câmera de trás) para a {side} e levou "
            "as mãos {hands_lat} m para o lado, com o centro de massa {drop} m mais baixo."
        ),
        "compare_lead": (
            "Mesmo goleiro, a mesma antecipação (cerca de {o1} e {o2} ms antes do chute). O quadril andou "
            "praticamente o mesmo nos dois lances ({l1} e {l2} m, medido na câmera de trás), mas as mãos foram "
            "mais longe no pênalti 2: {hl2} m para o lado, contra {hl1} m no pênalti 1, porque a bola foi "
            "parada mais para o canto ({h2} m do centro, contra {h1} m). Nos dois, a defesa foi a mais de um "
            "metro à frente da linha."
        ),
        "kpi_peak": "Velocidade média",
        "kpi_onset": "Antecipação",
        "kpi_centre": "Mãos, para o lado",
        "kpi_lat": "Quadril, para o lado",
        "sec_ball": "Bola",
        "sec_keeper": "Goleiro",
        "sec_kicker": "Batedor",
        "sec_quality": "Qualidade da reconstrução",
        "col_metric": "Métrica",
        "col_why": "Por que importa",
        "sec_snapshots": "Momentos-chave (câmera principal)",
        "sec_figures": "Figuras da física da bola",
        "sec_videos": "Vídeos do lance",
        "sec_rear": "Câmera de trás",
        "snap_kick": "Contato pé–bola",
        "snap_gk_move": "Início do movimento do goleiro",
        "snap_goal": "Defesa",
        "quickmeasure": "Medidas da câmera de trás (Quick Measure)",
        "video_print": "Vídeo (ver no HTML): ",
        "method": (
            "Os dois pênaltis foram reconstruídos com FreeKiki 3D (voo da bola, poses 3D "
            "do goleiro e do batedor) e Pynalty (métricas do pênalti). Antecipação = início "
            "do movimento do goleiro antes do contato pé–bola. O ponto em que a bola chegou às mãos "
            "combina as duas câmeras: o lado vem da câmera de trás (plano do gol) e a profundidade e a "
            "altura, da câmera principal. O deslocamento do quadril é medido na câmera de trás. O "
            "centro de massa (CM) e a queda vêm das poses 3D de uma câmera. HWR = índice de precisão "
            "de Attaallah et al. (2021)."
        ),
        "inject_heading": "Leitura do goleiro",
        "inject_link": "Abrir o relatório comparativo Carlos Miguel",
        "footer": "Gerado por vailá FreeKiki — {date}",
        "side": {"left": "esquerda", "right": "direita"},
        "foot": {"left": "esquerdo", "right": "direito"},
    },
    "en": {
        "title": "Carlos Miguel on the penalties — keeper reading",
        "subtitle": "Two kicks, two saves: the hip travelled the same, but on the second the ball went wider and the hands went further",
        "p1_title": "Penalty 1 — save {hit_lat} m from the centre",
        "p2_title": "Penalty 2 — save {hit_lat} m from the centre",
        "p1_body": (
            "At {stature} m with a {span} m arm span, Carlos Miguel met the ball {hit_front} m in front of "
            "the line, {hit_lat} m from the goal centre and {hit_z} m high. The ball, kicked with the "
            "{foot} foot, reached the hands at a mean {mean_kmh} km/h in {flight_ms} ms (the exit speed, "
            "about {peak_kmh} km/h from the flight model, is an estimate, not a measurement); without the "
            "save it would have crossed the line {centre} m from the centre ({zone}; {post} m from the "
            "nearest post; HWR {hwr}%). He anticipated contact by {onset} ms and moved his hip about {lat} m "
            "(range {lat_lo} to {lat_hi} m, measured on the rear camera) to his {side} and took his hands "
            "{hands_lat} m to the side, lowering his centre of mass {drop} m."
        ),
        "p2_body": (
            "On the second kick the ball reached the hands at a mean {mean_kmh} km/h in {flight_ms} ms "
            "(exit estimated at about {peak_kmh} km/h), {hit_front} m in front of the line, {hit_lat} m from "
            "the centre and {hit_z} m high; without the save it would have crossed the line {centre} m from "
            "the centre ({zone}; {post} m from the nearest post; HWR {hwr}%). Carlos Miguel anticipated by "
            "{onset} ms, moved his hip about {lat} m (range {lat_lo} to {lat_hi} m, rear camera) to his "
            "{side} and took his hands {hands_lat} m to the side, his centre of mass {drop} m lower."
        ),
        "compare_lead": (
            "Same keeper, same anticipation (about {o1} and {o2} ms before the kick). The hip travelled "
            "practically the same on both kicks ({l1} and {l2} m, measured on the rear camera), but the hands "
            "went further on penalty 2: {hl2} m to the side, against {hl1} m on penalty 1, because the ball was "
            "stopped wider ({h2} m from the centre, against {h1} m). On both, the save was more than a metre "
            "in front of the line."
        ),
        "kpi_peak": "Mean speed",
        "kpi_onset": "Anticipation",
        "kpi_centre": "Hands, to the side",
        "kpi_lat": "Hip, to the side",
        "sec_ball": "Ball",
        "sec_keeper": "Goalkeeper",
        "sec_kicker": "Kicker",
        "sec_quality": "Reconstruction quality",
        "col_metric": "Metric",
        "col_why": "Why it matters",
        "sec_snapshots": "Key moments (main camera)",
        "sec_figures": "Ball physics figures",
        "sec_videos": "Clip videos",
        "sec_rear": "Rear camera",
        "snap_kick": "Foot–ball contact",
        "snap_gk_move": "Keeper movement onset",
        "snap_goal": "Save",
        "quickmeasure": "Rear-camera measures (Quick Measure)",
        "video_print": "Video (see the HTML): ",
        "method": (
            "Both penalties were reconstructed with FreeKiki 3D (ball flight, 3D poses of "
            "keeper and kicker) and Pynalty (penalty metrics). Anticipation = keeper "
            "movement onset before foot–ball contact. The point where the ball reached the hands "
            "combines both cameras: the side comes from the rear camera (goal plane), the depth and "
            "height from the main camera. The hip travel is measured on the rear camera. The centre of "
            "mass (CoM) and its drop come from the one-camera 3D poses. HWR = accuracy index of "
            "Attaallah et al. (2021)."
        ),
        "inject_heading": "Keeper reading",
        "inject_link": "Open the Carlos Miguel comparative report",
        "footer": "Generated by vailá FreeKiki — {date}",
        "side": {"left": "left", "right": "right"},
        "foot": {"left": "left", "right": "right"},
    },
}

# (label_pt, label_en, card key, digits, unit, why_pt, why_en)
_ROWS: dict[str, list[tuple[str, str, str, int, str, str, str]]] = {
    "ball": [
        ("Velocidade média, marca até as mãos", "Mean speed, spot to the hands", "mean_kmh", 1, "km/h",
         "Distância ÷ tempo de voo (Navarro et al. 2013); ±5 km/h.",
         "Distance over flight time (Navarro et al. 2013); ±5 km/h."),
        ("Saída do pé (estimativa do modelo)", "Exit speed (model estimate)", "peak_kmh", 1, "km/h",
         "Depende do modelo de arrasto; não é medida.", "Depends on the drag model; not measured."),
        ("Tempo de voo até as mãos", "Flight time to the hands", "flight_ms", 0, "ms",
         "Janela total da defesa.", "Total window for the save."),
        ("Distância percorrida", "Distance travelled", "dist_m", 2, "m",
         "Da marca até as mãos.", "From the spot to the hands."),
        ("Rotação (spin)", "Spin", "spin_rpm", 0, "rpm",
         "Curva e estabilidade do voo.", "Curve and flight stability."),
        ("Elevação de saída", "Launch elevation", "launch_elev_deg", 1, "°",
         "Altura da trajetória.", "Trajectory height."),
        ("Azimute de saída", "Launch azimuth", "launch_az_deg", 1, "°",
         "Quanto a bola abre para o canto.", "How much the ball opens to the corner."),
        ("Defesa: à frente da linha", "Save: in front of the line", "hit_front_m", 2, "m",
         "Onde as mãos encontraram a bola (duas câmeras).", "Where the hands met the ball (two cameras)."),
        ("Defesa: distância ao centro do gol", "Save: distance from goal centre", "hit_lat_m", 2, "m",
         "Lado da câmera de trás.", "Side from the rear camera."),
        ("Defesa: altura", "Save: height", "hit_z_m", 2, "m",
         "Altura da bola nas mãos.", "Ball height at the hands."),
        ("Sem a defesa, cruzaria a linha a (do centro)", "Without the save, would cross the line at (from centre)", "centre_m", 2, "m",
         "Pelo voo ajustado até a defesa.", "From the flight fitted up to the save."),
        ("Altura na linha", "Height at the line", "cross_z_m", 2, "m",
         "Bola alta, média ou rasteira.", "High, mid or low ball."),
        ("Distância à trave mais próxima", "Distance to nearest post", "post_m", 3, "m",
         "Colocação da cobrança.", "Placement of the kick."),
        ("Zona do gol", "Goal zone", "zone", 0, "",
         "Região de 3×3 do gol.", "3×3 goal region."),
        ("Índice HWR", "HWR index", "hwr_pct", 1, "%",
         "Precisão: velocidade × colocação.", "Accuracy: speed × placement."),
        ("Resultado", "Outcome", "outcome", 0, "", "", ""),
    ],
    "keeper": [
        ("Antecipação (visual)", "Anticipation (visual)", "onset_ms", 0, "ms",
         "Início do movimento antes do chute.", "Movement onset before the kick."),
        ("Antecipação (automática, 3D de uma câmera)", "Anticipation (automatic, one-camera 3D)", "onset_auto_ms", 0, "ms",
         "Confirmação pelo detector cinemático.", "Kinematic detector check."),
        ("Lado do salto", "Dive side", "side", 0, "",
         "Escolha do canto.", "Corner choice."),
        ("Deslocamento lateral do quadril (câmera de trás)", "Hip lateral travel (rear camera)", "lat_m", 2, "m",
         "Demanda espacial da defesa; faixa de 90 % no texto.", "Spatial demand of the save; 90 % range in the text."),
        ("Avanço do quadril até a defesa", "Hip forward travel to the save", "hip_forward_m", 2, "m",
         "Duas câmeras.", "Two cameras."),
        ("Deslocamento lateral das mãos, início → defesa", "Hands lateral travel, onset → save", "hands_lat_m", 2, "m",
         "Média das palmas no início até o ponto da defesa.", "Mean of the palms at onset to the save point."),
        ("Percurso das mãos, início → defesa", "Hands travel, onset → save", "hands_travel_m", 2, "m",
         "Em linha reta, com o avanço e a subida.", "Straight line, with the forward and upward travel."),
        ("Abaixamento do CM", "CoM drop", "drop_m", 2, "m",
         "Queda para a altura da bola.", "Drop to the ball height."),
        ("Velocidade lateral média do quadril", "Mean hip lateral speed", "mean_lat_mps", 1, "m/s",
         "Deslocamento ÷ duração do movimento.", "Travel over movement duration."),
        ("Velocidade lateral de pico do CM", "Peak CoM lateral speed", "peak_lat_mps", 2, "m/s",
         "Explosão do salto.", "Explosiveness of the jump."),
        ("Velocidade resultante de pico", "Peak resultant speed", "peak_res_mps", 2, "m/s",
         "Velocidade total do corpo.", "Total body speed."),
        ("Passo preparatório", "Preparatory step", "step_m", 2, "m",
         "Pré-carga antes do salto.", "Pre-load before the jump."),
        ("Pé do passo", "Step foot", "step_foot", 0, "", "", ""),
        ("Duração do movimento", "Movement duration", "move_ms", 0, "ms",
         "Do início do movimento à defesa.", "From onset to the save."),
        ("Distância mãos–bola", "Hands–ball gap", "hands_gap_m", 2, "m",
         "Margem no contato.", "Margin at contact."),
        ("Alcance lateral em pé / mergulho", "Lateral reach standing / dive", "reach_txt", 0, "",
         "Envelope do goleiro (estatura e envergadura).", "Keeper envelope (stature, span)."),
    ],
    "kicker": [
        ("Pé de chute", "Kicking foot", "kick_foot", 0, "", "", ""),
        ("Estatura do batedor", "Kicker stature", "kicker_h_m", 2, "m", "", ""),
        ("Ângulo do joelho no contato", "Knee angle at contact", "knee_deg", 0, "°", "", ""),
    ],
    "quality": [
        ("Erro de reprojeção RMS", "RMS reprojection error", "rms_px", 2, "px",
         "Quanto o modelo segue a bola na imagem.", "How well the fit follows the image."),
        ("Erro máximo", "Max error", "max_px", 2, "px", "", ""),
        ("Qualidade do ajuste", "Fit quality", "quality", 0, "", "", ""),
        ("Método", "Method", "method", 0, "", "", ""),
    ],
}  # fmt: skip

_IMG_MAX_W = 1280
_POSTER_MAX_W = 960


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _num(value: Any, digits: int = 2) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(f):
        return "—"
    return f"{f:.{digits}f}"


def _fnum(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _esc(text: Any) -> str:
    return html_lib.escape(str(text), quote=True)


def _rel(path: Path, base: Path) -> str:
    """Relative path from ``base`` directory to ``path`` (works when target is outside base)."""
    return Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix()


def _href(path: Path, base: Path) -> str:
    return quote(_rel(path, base), safe="/.-_~")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _b64_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _b64_jpeg(frame: Any, max_w: int) -> str | None:
    if cv2 is None or frame is None:
        return None
    h, w = frame.shape[:2]
    if w > max_w:
        frame = cv2.resize(frame, (max_w, int(round(h * max_w / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _image_uri(path: Path, max_w: int = _IMG_MAX_W) -> str | None:
    """Downscaled JPEG data URI of an image (raw bytes when OpenCV is missing)."""
    if not path.is_file():
        return None
    if cv2 is not None and np is not None:
        frame = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        uri = _b64_jpeg(frame, max_w)
        if uri:
            return uri
    return _b64_file(path)


def _video_poster(path: Path) -> str | None:
    """Data URI of a frame from the middle of the video."""
    if cv2 is None or not path.is_file():
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n > 2:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, frame = cap.read()
    finally:
        cap.release()
    return _b64_jpeg(frame, _POSTER_MAX_W) if ok else None


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_penalty_card(cam3d: Path | str, *, tras: Path | str | None = None) -> dict[str, Any]:
    """Collect the measured numbers and the media of one FreeKiki camera 3D folder."""
    cam3d = Path(cam3d)
    card: dict[str, Any] = {
        "cam3d": str(cam3d),
        "available": False,
        "source": "none",
        "videos": {},
        "images": {},
        "figures": {"pt": [], "en": []},
        "report_tv": {},
        "tras": str(tras) if tras else None,
    }
    penalty = _read_json(cam3d / "data" / "penalty_metrics.json")
    try:
        from .freekiki_keeper import apply_rear_to_metrics
    except ImportError:
        from freekiki_keeper import apply_rear_to_metrics  # ty: ignore[unresolved-import]
    penalty = apply_rear_to_metrics(
        penalty, cam3d / "data"
    )  # the rear camera's lateral travel, if measured
    flight = _read_json(cam3d / "ball_flight.json")
    kicker = _read_json(cam3d / "data" / "kicker_metrics.json")
    m = penalty.get("metrics") or {}
    keeper = penalty.get("keeper") or {}
    counter = penalty.get("counterfactual") or {}
    anthro = penalty.get("anthropometrics") or {}
    km = kicker.get("metrics") or {}
    ke = kicker.get("events") or {}

    if penalty.get("available"):
        card["available"] = True
        card["source"] = "penalty_metrics"
    elif flight:
        card["available"] = True
        card["source"] = "ball_flight"

    peak = _fnum(flight.get("peak_speed_mps"))
    mean = _fnum(m.get("vel_ms"))
    if mean is None:
        series = flight.get("speed_mps")
        if isinstance(series, list):
            vals = [v for v in (_fnum(x) for x in series) if v is not None and v >= 5.0]
            if vals:
                mean = sum(vals) / len(vals)
    if mean is None and peak is not None:
        mean = peak
    flight_s = _fnum(m.get("flight_time_s"))
    if flight_s is None and _fnum(keeper.get("save_ms_after_kick")) is not None:
        flight_s = float(keeper["save_ms_after_kick"]) / 1000.0
    speed = penalty.get("speed") or {}  # the one definition (freekiki_penalty.speed_summary)
    if speed.get("mean") and speed.get("exit_model"):
        mean = _fnum(speed["mean"].get("ms"))
        flight_s = _fnum(speed.get("flight_time_s"))
        peak = _fnum(speed["exit_model"].get("launch_ms"))  # the model's exit speed: an estimate
    centre = _fnum(m.get("ball_entry_x_from_centre_m"))
    if centre is None and _fnum(counter.get("y_m")) is not None:
        centre = float(counter["y_m"])
    resp = _fnum(m.get("gk_response_time"))
    onset = _fnum(keeper.get("onset_ms_before_kick"))
    if onset is None and resp is not None:
        onset = abs(resp) * 1000.0
    spin = _fnum(flight.get("spin_rps"))
    stand = _fnum(m.get("reach_standing_m"))
    dive = _fnum(m.get("reach_dive_m"))

    hit_lat = _fnum((penalty.get("interception") or {}).get("lateral_m"))
    move_s = _fnum(keeper.get("movement_time_s"))
    card.update(
        {
            "stature_m": _fnum(m.get("gk_height_m")) or _fnum(anthro.get("gk_height_m")),
            "arm_span_m": _fnum(m.get("gk_arm_span_m")) or _fnum(anthro.get("gk_arm_span_m")),
            # Ball
            "peak_mps": peak,
            "peak_kmh": peak * 3.6 if peak is not None else None,
            "mean_mps": mean,
            "mean_kmh": mean * 3.6 if mean is not None else None,
            "vel_kmh": mean * 3.6 if mean is not None else None,
            "flight_ms": flight_s * 1000.0 if flight_s is not None else None,
            "dist_m": _fnum(m.get("dist")),
            "spin_rps": spin,
            "spin_rpm": spin * 60.0 if spin is not None else None,
            "launch_elev_deg": _fnum(m.get("launch_elevation_deg")),
            "launch_az_deg": _fnum(m.get("launch_azimuth_deg")),
            "centre_m": abs(centre) if centre is not None else None,
            "ball_from_centre_m": abs(centre) if centre is not None else None,
            "cross_z_m": _fnum(counter.get("z_m")) or _fnum(m.get("ball_entry_z_m")),
            "post_m": _fnum(counter.get("dist_to_nearest_post_m"))
            or _fnum(m.get("dist_to_nearest_post_m")),
            "zone": counter.get("zone_label") or m.get("zone_label"),
            "hwr_pct": _fnum(counter.get("hwr_accuracy_pct")) or _fnum(m.get("hwr_accuracy_pct")),
            "outcome": m.get("shot_outcome_label")
            or penalty.get("event")
            or flight.get("shot_outcome"),
            "event": penalty.get("event") or m.get("shot_outcome"),
            # Keeper
            "onset_ms": onset,
            "anticipation_ms": onset,
            "onset_auto_ms": _fnum(keeper.get("onset_auto_ms_before_kick")),
            "side_raw": keeper.get("dive_side"),
            "lat_m": _fnum(keeper.get("lateral_displacement_m")),
            "lat_range_m": keeper.get("lateral_displacement_range_m") or [None, None],
            "lat_source": keeper.get("lateral_source"),
            "mean_lat_mps": _fnum(keeper.get("mean_lateral_speed_mps")),
            "hip_forward_m": _fnum(keeper.get("hip_forward_m")),
            "hands_lat_m": _fnum(keeper.get("hands_lateral_m")),
            "hands_travel_m": _fnum(keeper.get("hands_travel_m")),
            "hit_front_m": _fnum((penalty.get("interception") or {}).get("in_front_of_line_m")),
            "hit_lat_m": abs(hit_lat) if hit_lat is not None else None,
            "hit_z_m": _fnum((penalty.get("interception") or {}).get("height_m")),
            "dive_m": _fnum(keeper.get("lateral_displacement_m")) or _fnum(m.get("gk_dist")),
            "drop_m": _fnum(keeper.get("com_drop_m")),
            "peak_lat_mps": _fnum(keeper.get("peak_lateral_speed_mps")),
            "peak_res_mps": _fnum(keeper.get("peak_resultant_speed_mps")),
            "step_m": _fnum(keeper.get("frontal_step_m")),
            "step_foot_raw": keeper.get("frontal_step_foot"),
            "move_ms": (move_s * 1000.0 if move_s is not None else None),
            "hands_gap_m": _fnum(keeper.get("hands_ball_gap_m")),
            "reach_standing_m": stand,
            "reach_dive_m": dive,
            # Kicker
            "kick_foot_raw": km.get("kick_foot") or ke.get("kick_foot"),
            "kicker_h_m": _fnum(m.get("kicker_height_m")),
            "approach_deg": _fnum(km.get("approach_angle_deg")),
            "v_foot_mps": _fnum(km.get("v_foot_pre_mps")),
            "knee_deg": _fnum(km.get("knee_angle_contact_deg")),
            # Quality
            "rms_px": _fnum(flight.get("rms_px")),
            "max_px": _fnum(flight.get("max_px")),
            "quality": flight.get("quality"),
            "method": flight.get("method"),
        }
    )

    # Media
    snaps = cam3d / "report_tv" / "penalty" / "snapshots"
    for key in ("kick", "gk_move", "goal"):
        p = snaps / f"snapshot_{key}.png"
        if p.is_file():
            card["images"][f"snap_{key}"] = str(p)
    topview = next(iter(sorted(cam3d.glob("*_topview.png"))), None)
    if topview is not None:
        card["images"]["topview"] = str(topview)
    for lang in ("pt", "en"):
        fig_dir = cam3d / "report_tv" / lang / "figuras"
        for name in ("fig_vista_de_cima.png", "fig_perfil.png", "fig_tempo.png", "fig_curva.png"):
            if (fig_dir / name).is_file():
                card["figures"][lang].append(str(fig_dir / name))

    media_dir = cam3d / "report_tv" / "penalty" / "media"
    candidates = {
        "animacao_lance": ["animacao_lance.mp4"],
        "kick_animation": ["kick_animation.mp4"],
        "animacao_3d": ["animacao_3d.mp4"],
        "overlay": [],
        "topview": [],
    }
    for key, names in candidates.items():
        for name in names:
            for folder in (media_dir, cam3d / "report_tv" / "pt", cam3d / "report_tv" / "en"):
                if (folder / name).is_file() and key not in card["videos"]:
                    card["videos"][key] = str(folder / name)
    overlay = next(iter(sorted(media_dir.glob("*_3d_overlay.mp4"))), None) or next(
        iter(sorted(cam3d.glob("*_3d_overlay.mp4"))), None
    )
    if overlay is not None:
        card["videos"]["overlay"] = str(overlay)
    top_mp4 = next(iter(sorted(media_dir.glob("*_topview.mp4"))), None) or next(
        iter(sorted(cam3d.glob("*_topview.mp4"))), None
    )
    if top_mp4 is not None:
        card["videos"]["topview"] = str(top_mp4)

    if tras and Path(tras).is_dir():
        tras_p = Path(tras)
        photos: list[str] = []
        for pattern in ("*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG"):
            photos.extend(str(p) for p in sorted(tras_p.glob(pattern)))
        card["tras_photos"] = photos
        rear = next(iter(sorted(tras_p.glob("*.mp4"))), None)
        if rear is not None:
            card["videos"]["rear"] = str(rear)
        qm = next(
            iter(sorted(tras_p.glob("processed_quickmeasure_*/quickmeasure_report.html"))), None
        )
        if qm is not None:
            card["quickmeasure"] = str(qm)

    for lang, name in (("pt", "relatorio_bola.html"), ("en", "ball_report.html")):
        p = cam3d / "report_tv" / lang / name
        if p.is_file():
            card["report_tv"][lang] = str(p)
    penalty_html = cam3d / "report_tv" / "penalty" / "report_pt.html"
    if penalty_html.is_file():
        card["report_tv"]["penalty_pt"] = str(penalty_html)
        en_html = cam3d / "report_tv" / "penalty" / "report.html"
        if en_html.is_file():
            card["report_tv"]["penalty_en"] = str(en_html)
    return card


def _localized(card: dict[str, Any], lang: str) -> dict[str, Any]:
    """Card values with language-dependent text (sides, feet, reach)."""
    t = STORY[lang]
    out = dict(card)
    out["side"] = t["side"].get(str(card.get("side_raw")), card.get("side_raw") or "—")
    out["step_foot"] = t["foot"].get(
        str(card.get("step_foot_raw")), card.get("step_foot_raw") or "—"
    )
    out["kick_foot"] = t["foot"].get(
        str(card.get("kick_foot_raw")), card.get("kick_foot_raw") or "—"
    )
    if card.get("reach_standing_m") is not None or card.get("reach_dive_m") is not None:
        out["reach_txt"] = (
            f"{_num(card.get('reach_standing_m'))} / {_num(card.get('reach_dive_m'))} m"
        )
    return out


def _fmt_cell(card: dict[str, Any], key: str, digits: int, unit: str) -> str:
    value = card.get(key)
    if value is None or value == "":
        return "—"
    if isinstance(value, str):
        return _esc(value)
    s = _num(value, digits)
    return f"{s} {unit}".strip() if s != "—" else "—"


def _story_values(card: dict[str, Any], lang: str) -> dict[str, str]:
    c = _localized(card, lang)
    return {
        "stature": _num(card.get("stature_m") or STORY["stature_m"]),
        "span": _num(card.get("arm_span_m") or STORY["arm_span_m"]),
        "foot": str(c["kick_foot"]),
        "peak_kmh": _num(card.get("peak_kmh"), 0),
        "mean_kmh": _num(card.get("mean_kmh"), 0),
        "flight_ms": _num(card.get("flight_ms"), 0),
        "centre": _num(card.get("centre_m")),
        "zone": _esc(card.get("zone") or "—"),
        "post": _num(abs(card["post_m"]) if card.get("post_m") is not None else None),
        "hwr": _num(card.get("hwr_pct"), 0),
        "onset": _num(card.get("onset_ms"), 0),
        "lat": _num(card.get("lat_m")),
        "side": str(c["side"]),
        "drop": _num(card.get("drop_m")),
        "gap": _num(card.get("hands_gap_m")),
        "plat": _num(card.get("peak_lat_mps"), 1),
        "lat_lo": _num((card.get("lat_range_m") or [None, None])[0]),
        "lat_hi": _num((card.get("lat_range_m") or [None, None])[1]),
        "hit_front": _num(card.get("hit_front_m")),
        "hit_lat": _num(card.get("hit_lat_m")),
        "hit_z": _num(card.get("hit_z_m")),
        "hands_lat": _num(card.get("hands_lat_m")),
    }


POST_UNCERTAINTY_M = 0.3  # the distance sigma of the mean speed (freekiki_penalty.SIGMA_DISTANCE_M)


def _post_note(card: dict[str, Any], lang: str) -> str:
    """Say that the crossing cannot be called when the post is inside the uncertainty."""
    post = card.get("post_m")
    if post is None or abs(post) >= POST_UNCERTAINTY_M:
        return ""
    if lang == "en":
        return (
            " The post is inside the measurement's uncertainty (±0.3 m): it is not possible to say"
            " whether the ball would have gone in."
        )
    return " A trave está dentro da incerteza da medida (±0,3 m): não dá para dizer se entraria."


# ---------------------------------------------------------------------------
# Media rendering
# ---------------------------------------------------------------------------


class _MediaSink:
    """Copies videos into ``<out>/media/p<N>_<name>`` once and caches posters."""

    def __init__(self, out_dir: Path | None) -> None:
        self.out_dir = out_dir
        self._posters: dict[str, str | None] = {}
        self._images: dict[tuple[str, int], str | None] = {}

    def image(self, path: str | Path, max_w: int = _IMG_MAX_W) -> str | None:
        key = (str(path), max_w)
        if key not in self._images:
            self._images[key] = _image_uri(Path(path), max_w)
        return self._images[key]

    def video(self, path: str | Path, prefix: str) -> tuple[str, str | None] | None:
        src = Path(path)
        if not src.is_file():
            return None
        if self.out_dir is None:
            href = src.resolve().as_uri()
        else:
            dest = self.out_dir / "media" / f"{prefix}_{_safe_name(src.name)}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or dest.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dest)
            href = _href(dest, self.out_dir)
        if str(src) not in self._posters:
            self._posters[str(src)] = _video_poster(src)
        return href, self._posters[str(src)]


def _figure(uri: str | None, caption: str) -> str:
    if not uri:
        return ""
    return (
        f"<figure><img src='{uri}' alt='{_esc(caption)}'>"
        f"<figcaption>{_esc(caption)}</figcaption></figure>"
    )


def _video_html(sink: _MediaSink, path: str | None, prefix: str, label: str, lang: str) -> str:
    if not path:
        return ""
    got = sink.video(path, prefix)
    if got is None:
        return ""
    href, poster = got
    poster_attr = f" poster='{poster}'" if poster else ""
    print_img = f"<img class='print-only' src='{poster}' alt=''>" if poster else ""
    name = Path(href).name
    return (
        f"<div class='video-card'><div class='vlabel'>{_esc(label)}</div>"
        f"<video class='screen-only' controls preload='metadata' src='{href}'{poster_attr}></video>"
        f"{print_img}<div class='print-only vlabel'>{STORY[lang]['video_print']}"
        f"media/{_esc(name)}</div></div>"
    )


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def _metric_table(lang: str, section: str, c1: dict[str, Any], c2: dict[str, Any]) -> str:
    t = STORY[lang]
    p1 = "Pênalti 1" if lang == "pt" else "Penalty 1"
    p2 = "Pênalti 2" if lang == "pt" else "Penalty 2"
    rows = []
    for lab_pt, lab_en, key, digits, unit, why_pt, why_en in _ROWS[section]:
        v1 = _fmt_cell(c1, key, digits, unit)
        v2 = _fmt_cell(c2, key, digits, unit)
        if v1 == "—" and v2 == "—":
            continue
        label = lab_pt if lang == "pt" else lab_en
        why = why_pt if lang == "pt" else why_en
        rows.append(
            f"<tr><th>{_esc(label)}</th><td>{v1}</td><td>{v2}</td>"
            f"<td class='why'>{_esc(why)}</td></tr>"
        )
    if not rows:
        return ""
    return (
        f"<h3>{t['sec_' + section]}</h3><table class='metrics'><thead><tr>"
        f"<th>{t['col_metric']}</th><th>{p1}</th><th>{p2}</th><th>{t['col_why']}</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _kpis(lang: str, card: dict[str, Any], title: str) -> str:
    t = STORY[lang]
    tiles = [
        (t["kpi_peak"], f"{_num(card.get('mean_kmh'), 0)} km/h"),
        (t["kpi_onset"], f"{_num(card.get('onset_ms'), 0)} ms"),
        (t["kpi_centre"], f"{_num(card.get('hands_lat_m'))} m"),
        (t["kpi_lat"], f"{_num(card.get('lat_m'))} m"),
    ]
    body = "".join(
        f"<div class='kpi'><div class='kv'>{v}</div><div class='kl'>{_esc(k)}</div></div>"
        for k, v in tiles
    )
    return f"<div class='kpi-block'><div class='kpi-title'>{_esc(title)}</div><div class='kpis'>{body}</div></div>"


_CSS = """
  :root { --bg:#14181d; --card:#1e242c; --ink:#eceff1; --muted:#98a4b3; --line:#313c48;
          --accent:#4caf50; --blue:#4fc3f7; --amber:#ffb74d; --violet:#e040fb; }
  * { box-sizing:border-box; }
  body { font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif; background:var(--bg);
         color:var(--ink); margin:0; padding:24px; line-height:1.55; }
  .wrap { max-width:1150px; margin:0 auto; }
  header { background:var(--card); border-radius:12px; padding:24px 28px; margin-bottom:20px;
           border-left:6px solid var(--accent); }
  h1 { margin:0 0 8px; color:var(--accent); font-size:1.7em; }
  h2 { color:var(--accent); border-bottom:2px solid #35414e; padding-bottom:8px; margin:30px 0 14px; }
  h3 { color:var(--blue); margin:20px 0 8px; }
  .meta { color:var(--muted); }
  .lead { background:var(--card); border-left:6px solid var(--violet); border-radius:10px;
          padding:16px 20px; margin:18px 0; font-size:1.05em; }
  .grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
  .panel { background:var(--card); border-radius:10px; padding:18px; }
  .kpi-block { background:var(--card); border-radius:10px; padding:14px 16px; }
  .kpi-title { color:var(--blue); font-weight:600; margin-bottom:8px; }
  .kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }
  .kpi { background:#151a20; border:1px solid var(--line); border-radius:8px; padding:10px; text-align:center; }
  .kv { font-size:1.35em; font-weight:700; color:var(--amber); }
  .kl { font-size:.78em; color:var(--muted); }
  table.metrics { width:100%; border-collapse:collapse; font-size:.9em; background:var(--card);
                  border-radius:10px; overflow:hidden; }
  table.metrics th, table.metrics td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }
  table.metrics thead th { color:var(--accent); background:#192028; }
  table.metrics tbody th { color:var(--ink); font-weight:500; width:30%; }
  table.metrics td { color:var(--amber); font-variant-numeric:tabular-nums; white-space:nowrap; }
  table.metrics td.why { color:var(--muted); white-space:normal; font-size:.92em; }
  .prose { margin-top:6px; }
  .note { color:var(--muted); font-size:.88em; }
  .gallery { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; }
  figure { margin:0; }
  figure img { width:100%; border-radius:8px; border:1px solid #3a4655; display:block; }
  figcaption { color:var(--muted); font-size:.8em; margin-top:4px; }
  .video-card { margin:12px 0; }
  .video-card video, .video-card img { width:100%; border-radius:8px; background:#000; display:block; }
  .vlabel { color:var(--muted); font-size:.85em; margin:4px 0; }
  .print-only { display:none; }
  footer { margin-top:36px; text-align:center; color:var(--muted); font-size:.85em; }
  a { color:var(--blue); }
  @page { size:A4; margin:12mm; }
  @media print {
    :root { --bg:#ffffff; --card:#f5f7fa; --ink:#111827; --muted:#4b5563; --line:#d1d5db;
            --accent:#047857; --blue:#0369a1; --amber:#b45309; --violet:#7e22ce; }
    body { padding:0; font-size:11px; -webkit-print-color-adjust:exact; print-color-adjust:exact; }
    .screen-only { display:none !important; }
    .print-only { display:block; }
    table.metrics thead th { background:#e5e7eb; }
    .kpi { background:#ffffff; padding:8px 4px; }
    .kv { font-size:1.1em; white-space:nowrap; }
    .panel, .kpi-block, figure, .video-card, tr { break-inside:avoid; }
    h2 { break-after:avoid; }
    .pagebreak { break-before:page; }
  }
"""


def build_compare_html(
    card1: dict[str, Any],
    card2: dict[str, Any],
    *,
    lang: str = "pt",
    tras1: Path | None = None,
    tras2: Path | None = None,
    out_dir: Path | None = None,
    compare_href: str | None = None,
    sink: _MediaSink | None = None,
) -> str:
    """Render one language of the comparative report as an HTML string."""
    t = STORY[lang]
    sink = sink or _MediaSink(out_dir)
    media: _MediaSink = sink  # the nested helpers below close over a non-optional sink
    if tras1 and not card1.get("tras"):
        card1 = load_penalty_card(card1["cam3d"], tras=tras1)
    if tras2 and not card2.get("tras"):
        card2 = load_penalty_card(card2["cam3d"], tras=tras2)
    l1, l2 = _localized(card1, lang), _localized(card2, lang)
    stature = _num(card1.get("stature_m") or STORY["stature_m"])
    p1_body = t["p1_body"].format(**_story_values(card1, lang)) + _post_note(card1, lang)
    p2_body = t["p2_body"].format(**_story_values(card2, lang)) + _post_note(card2, lang)
    p1_title = t["p1_title"].format(**_story_values(card1, lang))
    p2_title = t["p2_title"].format(**_story_values(card2, lang))
    lead = t["compare_lead"].format(
        o1=_num(card1.get("onset_ms"), 0),
        o2=_num(card2.get("onset_ms"), 0),
        c1=_num(card1.get("centre_m")),
        c2=_num(card2.get("centre_m")),
        h1=_num(card1.get("hit_lat_m")),
        h2=_num(card2.get("hit_lat_m")),
        hl1=_num(card1.get("hands_lat_m")),
        hl2=_num(card2.get("hands_lat_m")),
        l1=_num(card1.get("lat_m")),
        l2=_num(card2.get("lat_m")),
        s2=_num(card2.get("peak_lat_mps"), 1),
    )
    pen = ("Pênalti 1", "Pênalti 2") if lang == "pt" else ("Penalty 1", "Penalty 2")

    tables = "".join(
        _metric_table(lang, s, l1, l2) for s in ("ball", "keeper", "kicker", "quality")
    )

    def snaps(card: dict[str, Any]) -> str:
        figs = [
            _figure(media.image(card["images"][f"snap_{k}"]), t[f"snap_{k}"])
            for k in ("kick", "gk_move", "goal")
            if f"snap_{k}" in card["images"]
        ]
        return "".join(figs)

    def figures(card: dict[str, Any]) -> str:
        out = []
        if "topview" in card["images"]:
            out.append(_figure(media.image(card["images"]["topview"]), "Top view"))
        for p in card["figures"].get(lang) or card["figures"].get("pt") or []:
            out.append(_figure(media.image(p), Path(p).stem.replace("fig_", "").replace("_", " ")))
        return "".join(out)

    def videos(card: dict[str, Any], prefix: str) -> str:
        v = card.get("videos") or {}
        main = v.get("animacao_lance") if lang == "pt" else v.get("kick_animation")
        main = main or v.get("animacao_lance") or v.get("kick_animation")
        items = [
            (
                main,
                "Câmera lenta com a bola ajustada"
                if lang == "pt"
                else "Slow motion with fitted ball",
            ),
            (v.get("overlay"), "Overlay 3D" if lang == "pt" else "3D overlay"),
            (v.get("topview"), "Vista de cima" if lang == "pt" else "Top view"),
            (v.get("animacao_3d"), "Animação 3D" if lang == "pt" else "3D animation"),
        ]
        return "".join(_video_html(media, p, prefix, lab, lang) for p, lab in items)

    def rear(card: dict[str, Any], prefix: str) -> str:
        parts = [_figure(media.image(p), Path(p).stem) for p in (card.get("tras_photos") or [])]
        out = f"<div class='gallery'>{''.join(parts)}</div>" if parts else ""
        rv = (card.get("videos") or {}).get("rear")
        label = "Vídeo da câmera de trás" if lang == "pt" else "Rear-camera video"
        out += _video_html(media, rv, prefix + "_tras", label, lang)
        qm = card.get("quickmeasure")
        if qm and out_dir is not None:
            out += f"<p><a href='{_href(Path(qm), out_dir)}'>{t['quickmeasure']}</a></p>"
        return out or f"<p class='note'>{'Sem mídia.' if lang == 'pt' else 'No media.'}</p>"

    links = []
    for card, label in ((card1, pen[0]), (card2, pen[1])):
        rt = card.get("report_tv") or {}
        for key, name in ((lang, "TV"), (f"penalty_{lang}", "Pynalty")):
            if rt.get(key) and out_dir is not None:
                links.append(f"<a href='{_href(Path(rt[key]), out_dir)}'>{label} — {name}</a>")
    if compare_href:
        links.append(f"<a href='{compare_href}'>{t['inject_link']}</a>")
    date = dt.now().strftime("%Y-%m-%d %H:%M")
    span_word = "envergadura" if lang == "pt" else "arm span"

    return f"""<!DOCTYPE html>
<html lang="{"pt-BR" if lang == "pt" else "en"}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{t["title"]}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>{t["title"]}</h1>
  <div class="meta"><b>{STORY["match"]}</b> · {STORY["keeper_name"]} ({STORY["team"]}) ·
    {stature} m · {_num(card1.get("arm_span_m") or STORY["arm_span_m"])} m {span_word} ·
    {_num(STORY["mass_kg"], 0)} kg</div>
  <div class="meta">{t["subtitle"]}</div>
</header>

<div class="grid">
  {_kpis(lang, card1, pen[0])}
  {_kpis(lang, card2, pen[1])}
</div>

<div class="lead">{lead}</div>

<div class="grid">
  <div class="panel"><h2>{p1_title}</h2><p class="prose">{p1_body}</p></div>
  <div class="panel"><h2>{p2_title}</h2><p class="prose">{p2_body}</p></div>
</div>

<h2>{"Métricas que fundamentam a leitura" if lang == "pt" else "Metrics behind the reading"}</h2>
{tables}

<h2 class="pagebreak">{t["sec_snapshots"]}</h2>
<h3>{pen[0]}</h3><div class="gallery">{snaps(card1)}</div>
<h3>{pen[1]}</h3><div class="gallery">{snaps(card2)}</div>

<h2>{t["sec_rear"]}</h2>
<div class="grid">
  <div class="panel"><h3>{pen[0]}</h3>{rear(card1, "p1")}</div>
  <div class="panel"><h3>{pen[1]}</h3>{rear(card2, "p2")}</div>
</div>

<h2>{t["sec_videos"]}</h2>
<div class="grid">
  <div class="panel"><h3>{pen[0]}</h3>{videos(card1, "p1")}</div>
  <div class="panel"><h3>{pen[1]}</h3>{videos(card2, "p2")}</div>
</div>

<h2 class="pagebreak">{t["sec_figures"]}</h2>
<h3>{pen[0]}</h3><div class="gallery">{figures(card1)}</div>
<h3>{pen[1]}</h3><div class="gallery">{figures(card2)}</div>

<h2>{"Método" if lang == "pt" else "Methods"}</h2>
<div class="panel"><p class="note">{t["method"]}</p>
<p>{" · ".join(links)}</p></div>

<footer>{t["footer"].format(date=date)}</footer>
</div>
</body>
</html>
"""


def _print_pdf(html_path: Path, pdf_path: Path) -> Path | None:
    try:
        from .freekiki_report import print_pdf
    except ImportError:
        from freekiki_report import print_pdf  # ty: ignore[unresolved-import]
    return print_pdf(html_path, pdf_path)


def write_compare_reports(
    out_dir: Path | str,
    cam3d_1: Path | str,
    cam3d_2: Path | str,
    *,
    tras1: Path | str | None = None,
    tras2: Path | str | None = None,
    pdf: bool = True,
) -> dict[str, str]:
    """Write ``relatorio_carlos_miguel.html`` / ``_en.html`` (and the PDFs when Chrome exists)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "media").mkdir(exist_ok=True)
    card1 = load_penalty_card(cam3d_1, tras=tras1)
    card2 = load_penalty_card(cam3d_2, tras=tras2)
    sink = _MediaSink(out)
    written: dict[str, str] = {}
    for lang, stem in (("pt", "relatorio_carlos_miguel"), ("en", "relatorio_carlos_miguel_en")):
        path = out / f"{stem}.html"
        path.write_text(
            build_compare_html(card1, card2, lang=lang, out_dir=out, sink=sink),
            encoding="utf-8",
        )
        written[lang] = str(path)
        if pdf:
            got = _print_pdf(path, out / f"{stem}.pdf")
            if got is not None:
                written[f"{lang}_pdf"] = str(got)
    return written


_INJECT_MARK = "<!-- vaila-keeper-story -->"


def inject_keeper_link(
    html_path: Path | str,
    compare_html: Path | str,
    *,
    lang: str = "pt",
) -> bool:
    """Insert a short ‘keeper reading’ block with a relative link into an existing HTML."""
    html_path = Path(html_path)
    compare_html = Path(compare_html)
    if not html_path.is_file():
        return False
    text = html_path.read_text(encoding="utf-8")
    href = _href(compare_html, html_path.parent)
    block = _inject_block(lang, href)
    if _INJECT_MARK in text:
        text = re.sub(
            r"<!-- vaila-keeper-story -->.*?</div>",
            lambda _m: block,
            text,
            count=1,
            flags=re.S,
        )
    elif "</body>" in text:
        text = text.replace("</body>", block + "\n</body>", 1)
    else:
        text = text + "\n" + block
    html_path.write_text(text, encoding="utf-8")
    return True


def _inject_block(lang: str, href: str) -> str:
    t = STORY[lang]
    return (
        f"{_INJECT_MARK}\n"
        f"<div class='panel' style='margin:24px auto;max-width:1100px;background:#1e242c;"
        f"border-radius:10px;padding:16px 20px;border-left:5px solid #e040fb;color:#eceff1'>"
        f"<h2 style='color:#4caf50;margin:0 0 8px'>{t['inject_heading']}</h2>"
        f"<p style='margin:0'><a href='{href}' style='color:#4fc3f7'>{t['inject_link']}</a></p>"
        f"</div>"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Comparative Carlos Miguel keeper report (HTML + PDF)."
    )
    ap.add_argument("--penalti1", required=True, help="Camera 3D folder of penalty 1")
    ap.add_argument("--penalti2", required=True, help="Camera 3D folder of penalty 2")
    ap.add_argument("--tras1", default=None, help="Rear-camera folder of penalty 1")
    ap.add_argument("--tras2", default=None, help="Rear-camera folder of penalty 2")
    ap.add_argument("-o", "--output", required=True, help="Output directory for the reports")
    ap.add_argument("--no-pdf", action="store_true", help="Skip the PDF (headless Chrome)")
    ap.add_argument(
        "--inject",
        action="store_true",
        help="Insert a link block into each run's ball / penalty HTML",
    )
    args = ap.parse_args(argv)
    written = write_compare_reports(
        args.output,
        args.penalti1,
        args.penalti2,
        tras1=args.tras1,
        tras2=args.tras2,
        pdf=not args.no_pdf,
    )
    for key, path in written.items():
        print(f">> vaila/freekiki_keeper_story: {key}: {path}")
    if args.inject:
        for cam in (args.penalti1, args.penalti2):
            card = load_penalty_card(cam)
            for key, lang in (
                ("pt", "pt"),
                ("en", "en"),
                ("penalty_pt", "pt"),
                ("penalty_en", "en"),
            ):
                path = (card.get("report_tv") or {}).get(key)
                if path:
                    ok = inject_keeper_link(path, written[lang], lang=lang)
                    print(
                        f">> vaila/freekiki_keeper_story: inject {'ok' if ok else 'skip'}: {path}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
