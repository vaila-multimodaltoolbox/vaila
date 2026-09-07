"""
Project: vailá Multimodal Toolbox
Script: pynalty_report.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 07 September 2026
Update Date: 07 September 2026
Version: 0.3.129

Description:
    Reporting layer for the Pynalty penalty analysis.

    Produces a self-contained HTML report (every image base64-embedded, the
    goal diagrams drawn as inline SVG and the trajectory replay as a plain
    canvas animation, so the file opens offline with no CDN), in English and
    Portuguese, plus the CSV artefacts: a tidy per-variable ``results.csv``, a
    one-row-per-penalty ``pynalty_summary.csv``, and an appendable
    ``pynalty_database.csv`` that accumulates every analysed penalty for
    cross-session comparison.

License:
    This project is licensed under the terms of AGPLv3.0.
"""

from __future__ import annotations

import base64
import csv
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path

import numpy as np

try:
    from .pynalty_analysis import GoalGeometry
    from .pynalty_vision import LANDMARK_NAMES
except ImportError:
    from pynalty_analysis import GoalGeometry
    from pynalty_vision import LANDMARK_NAMES

RESULTS_CSV = "results.csv"
SUMMARY_CSV = "pynalty_summary.csv"
DATABASE_CSV = "pynalty_database.csv"
BALL_PATH_PIXEL_CSV = "ball_path_pixel.csv"
BALL_PATH_3D_CSV = "ball_path_3d.csv"

REPORT_HTML = {"en": "report.html", "pt": "report_pt.html"}


@dataclass
class ReportContext:
    """Everything the report writers need for one analysed penalty."""

    video_path: str
    fps: float
    metrics: dict
    goal: GoalGeometry
    snapshots: dict[str, str] = field(default_factory=dict)
    ball_rows: list[dict] = field(default_factory=list)
    flight_path: np.ndarray | None = None
    pose_metrics: dict = field(default_factory=dict)
    pose_files: dict[str, str] = field(default_factory=dict)
    videos: dict[str, str] = field(default_factory=dict)
    anthro: dict = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: dt.now().strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def video_name(self) -> str:
        return os.path.basename(self.video_path or "")


# ==============================================================================
# Localisation
# ==============================================================================

_T: dict[str, dict[str, str]] = {
    "title": {"en": "Pynalty Penalty Analysis", "pt": "Análise de Pênalti - Pynalty"},
    "video": {"en": "Video", "pt": "Vídeo"},
    "date": {"en": "Generated", "pt": "Gerado em"},
    "fps": {"en": "Frame rate", "pt": "Taxa de quadros"},
    "summary": {"en": "Executive summary", "pt": "Resumo executivo"},
    "kicker": {"en": "Kicker", "pt": "Cobrador"},
    "goalkeeper": {"en": "Goalkeeper", "pt": "Goleiro"},
    "reach": {"en": "Reach and time to the ball", "pt": "Alcance e tempo até a bola"},
    "trajectory": {"en": "Ball trajectory", "pt": "Trajetória da bola"},
    "pose": {"en": "Pose kinematics", "pt": "Cinemática por pose"},
    "snapshots": {"en": "Event snapshots", "pt": "Imagens dos eventos"},
    "methods": {"en": "Methods and limitations", "pt": "Métodos e limitações"},
    "files": {"en": "Files in this analysis", "pt": "Arquivos desta análise"},
    "ball_speed": {"en": "Ball speed", "pt": "Velocidade da bola"},
    "flight_time": {"en": "Flight time", "pt": "Tempo de voo"},
    "flight_distance": {"en": "Flight distance", "pt": "Distância percorrida"},
    "entry_point": {"en": "Entry point (X, Z)", "pt": "Ponto de entrada (X, Z)"},
    "target_zone": {"en": "Target zone", "pt": "Zona do gol"},
    "placement": {"en": "Placement index", "pt": "Índice de colocação"},
    "launch_angle": {"en": "Launch elevation", "pt": "Ângulo de saída"},
    "apex": {"en": "Apex height", "pt": "Altura máxima"},
    "nearest_post": {"en": "Distance to nearest post", "pt": "Distância à trave mais próxima"},
    "to_crossbar": {"en": "Distance to crossbar", "pt": "Distância ao travessão"},
    "reaction": {"en": "Reaction", "pt": "Reação"},
    "dive_side": {"en": "Dive side", "pt": "Lado da defesa"},
    "dive_distance": {"en": "Dive distance", "pt": "Deslocamento no salto"},
    "dive_speed": {"en": "Dive speed", "pt": "Velocidade do salto"},
    "gap": {"en": "Distance to the ball", "pt": "Distância até a bola"},
    "travel_needed": {"en": "Body travel needed", "pt": "Deslocamento necessário"},
    "available_time": {"en": "Time available", "pt": "Tempo disponível"},
    "required_speed": {"en": "Required dive speed", "pt": "Velocidade necessária"},
    "time_to_reach": {"en": "Time to reach the ball", "pt": "Tempo para alcançar a bola"},
    "time_margin": {"en": "Time margin", "pt": "Margem de tempo"},
    "reach_standing": {"en": "Standing reach radius", "pt": "Raio de alcance em pé"},
    "reach_dive": {"en": "Diving reach radius", "pt": "Raio de alcance em salto"},
    "verdict": {"en": "Verdict", "pt": "Veredito"},
    "gk_height": {"en": "Goalkeeper stature", "pt": "Estatura do goleiro"},
    "gk_span": {"en": "Arm span", "pt": "Envergadura"},
    "gk_reach": {"en": "Standing overhead reach", "pt": "Alcance vertical em pé"},
    "kicker_height": {"en": "Kicker stature", "pt": "Estatura do cobrador"},
    "goal_map": {"en": "Goal map", "pt": "Mapa do gol"},
    "play": {"en": "Play", "pt": "Reproduzir"},
    "pause": {"en": "Pause", "pt": "Pausar"},
    "restart": {"en": "Restart", "pt": "Reiniciar"},
    "front_view": {"en": "Keeper's view", "pt": "Visão do goleiro"},
    "plan_view": {"en": "Pitch plan", "pt": "Vista superior"},
    "side_view": {"en": "Flight profile", "pt": "Perfil do voo"},
    "time_label": {"en": "Time", "pt": "Tempo"},
    "ball": {"en": "Ball", "pt": "Bola"},
    "no_pose": {
        "en": "No pose data was produced for this penalty.",
        "pt": "Nenhum dado de pose foi gerado para este pênalti.",
    },
    "no_anthro": {
        "en": "Goalkeeper anthropometrics were not provided, so the reach and time analysis could not be computed. Fill in step 7 of the marking workflow to enable it.",
        "pt": "As medidas antropométricas do goleiro não foram informadas, portanto a análise de alcance e tempo não pôde ser calculada. Preencha o passo 7 do fluxo de marcação para habilitá-la.",
    },
    "kicking_leg": {"en": "Kicking leg", "pt": "Perna de chute"},
    "support_knee": {"en": "Support knee angle", "pt": "Ângulo do joelho de apoio"},
    "support_hip": {"en": "Support hip angle", "pt": "Ângulo do quadril de apoio"},
    "trunk_lean": {"en": "Trunk lean", "pt": "Inclinação do tronco"},
    "foot_speed": {"en": "Kicking foot speed", "pt": "Velocidade do pé de chute"},
    "hand_span": {"en": "Widest hand span", "pt": "Maior abertura das mãos"},
    "hip_travel": {"en": "Hip displacement", "pt": "Deslocamento do quadril"},
    "first_hand_move": {"en": "First hand movement", "pt": "Primeiro movimento das mãos"},
    "left": {"en": "left", "pt": "esquerda"},
    "right": {"en": "right", "pt": "direita"},
    "footer": {
        "en": "Generated by <em>vailá</em> - Pynalty module",
        "pt": "Gerado por <em>vailá</em> - módulo Pynalty",
    },
    "calibration": {"en": "Calibration", "pt": "Calibração"},
    "calib_residual": {"en": "Corner reprojection RMS", "pt": "RMS de reprojeção dos cantos"},
    "snapshot_gk_move": {"en": "1. Goalkeeper starts moving", "pt": "1. Goleiro inicia movimento"},
    "snapshot_kick": {"en": "2. Ball contact", "pt": "2. Contato com a bola"},
    "snapshot_goal": {"en": "3. Ball crosses the goal line", "pt": "3. Bola cruza a linha"},
    "snapshot_calibration": {"en": "4. Goal calibration", "pt": "4. Calibração do gol"},
    "snapshot_ball_path": {"en": "5. Ball path composite", "pt": "5. Composição da trajetória"},
    "zone_grid_note": {
        "en": "Zones are named from the camera's point of view. The dashed circles are the goalkeeper's standing and diving reach around the position marked at ball contact.",
        "pt": "As zonas são nomeadas do ponto de vista da câmera. Os círculos tracejados são os alcances em pé e em salto do goleiro em torno da posição marcada no contato com a bola.",
    },
    "anim_note": {
        "en": "The replay is a model reconstruction: horizontal motion is uniform and the vertical component is a gravity parabola solved to hit the measured entry point at the measured flight time. Drag and spin are neglected.",
        "pt": "A reprodução é uma reconstrução por modelo: o movimento horizontal é uniforme e a componente vertical é uma parábola de gravidade ajustada para atingir o ponto de entrada medido no tempo de voo medido. Arrasto e efeito são desprezados.",
    },
}

_ZONE_LABELS = {
    "en": {
        "Low": "Low",
        "Middle": "Middle",
        "High": "High",
        "Left": "Left",
        "Centre": "Centre",
        "Right": "Right",
    },
    "pt": {
        "Low": "Baixo",
        "Middle": "Meio",
        "High": "Alto",
        "Left": "Esquerda",
        "Centre": "Centro",
        "Right": "Direita",
    },
}

_VERDICT_LABELS = {
    "reachable_standing": {
        "en": "Within standing reach, no dive required",
        "pt": "Dentro do alcance em pé, sem necessidade de salto",
    },
    "unsaveable": {
        "en": "Unsaveable: beyond an elite dive within the flight time",
        "pt": "Indefensável: além de um salto de elite no tempo de voo",
    },
    "saveable_late": {
        "en": "Saveable, but the keeper arrived late",
        "pt": "Defensável, mas o goleiro chegou atrasado",
    },
    "saveable": {
        "en": "Saveable: reach and timing were both sufficient",
        "pt": "Defensável: alcance e tempo foram suficientes",
    },
    "unknown": {
        "en": "Not assessed (goalkeeper anthropometrics missing)",
        "pt": "Não avaliado (faltam medidas do goleiro)",
    },
}

_REACTION_LABELS = {
    "anticipation": {
        "en": "Anticipated contact by {ms} ms",
        "pt": "Antecipou o contato em {ms} ms",
    },
    "simultaneous": {
        "en": "Moved with contact (within 50 ms)",
        "pt": "Moveu junto com o contato (até 50 ms)",
    },
    "fast_reactive": {"en": "Fast reaction ({ms} ms)", "pt": "Reação rápida ({ms} ms)"},
    "reactive": {"en": "Reactive ({ms} ms)", "pt": "Reativo ({ms} ms)"},
    "late": {"en": "Late reaction ({ms} ms)", "pt": "Reação tardia ({ms} ms)"},
}

_SIDE_LABELS = {
    "correct_side": {"en": "Dived to the correct side", "pt": "Defendeu para o lado correto"},
    "wrong_side": {"en": "Dived to the wrong side", "pt": "Defendeu para o lado errado"},
    "stayed_central": {
        "en": "Stayed central (no committed dive)",
        "pt": "Permaneceu no centro (sem salto definido)",
    },
    "ball_central": {
        "en": "Ball came at the keeper's body",
        "pt": "A bola veio no corpo do goleiro",
    },
}

_VERDICT_TONE = {
    "saveable": "good",
    "reachable_standing": "good",
    "saveable_late": "warn",
    "unsaveable": "bad",
    "unknown": "neutral",
}


def t(lang: str, key: str) -> str:
    entry = _T.get(key)
    if not entry:
        return key
    return entry.get(lang, entry.get("en", key))


def _zone_label(lang: str, raw: str) -> str:
    table = _ZONE_LABELS.get(lang, _ZONE_LABELS["en"])
    return " ".join(table.get(part, part) for part in str(raw).split())


def _verdict_label(lang: str, key: str, fallback: str) -> str:
    entry = _VERDICT_LABELS.get(key)
    return entry.get(lang, entry.get("en")) if entry else fallback


def _reaction_label(lang: str, key: str, reaction_time_s: float, fallback: str) -> str:
    entry = _REACTION_LABELS.get(key)
    if not entry:
        return fallback
    return entry.get(lang, entry["en"]).format(ms=f"{abs(reaction_time_s) * 1000:.0f}")


def _side_label(lang: str, key: str, fallback: str) -> str:
    entry = _SIDE_LABELS.get(key)
    return entry.get(lang, entry.get("en")) if entry else fallback


# ==============================================================================
# Formatting helpers
# ==============================================================================


def _num(value, nd: int = 2, dash: str = "n/a") -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return dash
    if not math.isfinite(f):
        return "&infin;" if f > 0 else dash
    return f"{f:.{nd}f}"


#: Longest edge, in pixels, of an image embedded in the HTML report. Full-HD
#: snapshots stay on disk at native resolution; embedding them raw would push a
#: single report past 10 MB, which makes it slow to open and awkward to email.
EMBED_MAX_EDGE = 1400
EMBED_JPEG_QUALITY = 86


def _b64_img(path: str | None) -> str | None:
    """Read an image and return a downscaled JPEG data URI, or None if missing."""
    if not path or not os.path.exists(path):
        return None
    try:
        import cv2

        img = cv2.imread(path)
        if img is None:
            raise ValueError("unreadable")
        h, w = img.shape[:2]
        scale = min(1.0, EMBED_MAX_EDGE / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, EMBED_JPEG_QUALITY])
        if ok:
            return f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"
    except Exception as exc:
        print(f">> vaila/pynalty_report: could not downscale {path}: {exc}")

    ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
    mime = "jpeg" if ext in {"jpg", "jpeg"} else ext
    with open(path, "rb") as fh:
        return f"data:image/{mime};base64,{base64.b64encode(fh.read()).decode('ascii')}"


# ==============================================================================
# CSV writers
# ==============================================================================

#: Ordered, stable column set for the summary and the accumulating database.
SUMMARY_FIELDS: tuple[str, ...] = (
    "analysis_datetime",
    "video_name",
    "video_path",
    "fps",
    "gk_move_frame",
    "kick_frame",
    "goal_frame",
    "flight_frames",
    "flight_time_s",
    "dist",
    "vel_ms",
    "vel_kmh",
    "ball_entry_x_m",
    "ball_entry_z_m",
    "ball_entry_x_from_centre_m",
    "ball_inside_goal",
    "shot_outcome",
    "shot_outcome_label",
    "anthro_source",
    "gap_source",
    "gap_landmark",
    "zone_index",
    "zone_label",
    "placement_index",
    "dist_to_nearest_post_m",
    "dist_to_crossbar_m",
    "launch_elevation_deg",
    "launch_azimuth_deg",
    "apex_height_m",
    "gk_response_frames",
    "gk_response_time",
    "reaction_class",
    "gk_start_x_m",
    "gk_start_z_m",
    "gk_end_x_m",
    "gk_end_z_m",
    "gk_dist",
    "gk_dive_time_s",
    "gk_vel_ms",
    "gk_vel_kmh",
    "dive_side_class",
    "gap_m",
    "gap_horizontal_m",
    "gap_vertical_m",
    "reach_standing_m",
    "reach_dive_m",
    "travel_dive_m",
    "available_time_s",
    "required_dive_speed_ms",
    "time_to_reach_s",
    "time_margin_s",
    "reachable_standing",
    "reachable_dive",
    "verdict_class",
    "gk_height_m",
    "gk_arm_span_m",
    "gk_standing_reach_m",
    "kicker_height_m",
    "goal_width_m",
    "goal_height_m",
    "penalty_distance_m",
    "calibration_residual_px",
    "calibration_convex",
    "ball_path_points",
    "kicker_kicking_leg",
    "kicker_support_knee_angle_deg",
    "kicker_support_hip_angle_deg",
    "kicker_trunk_lean_deg",
    "kicker_foot_speed_px_s",
    "gk_max_hand_span_px",
    "gk_max_hand_span_m",
    "gk_hip_travel_px",
    "gk_first_hand_move_rel_kick_s",
)

#: Grouping and units for the tidy per-variable results file.
_RESULTS_ROWS: tuple[tuple[str, str, str, int], ...] = (
    ("ball", "dist", "m", 3),
    ("ball", "vel_ms", "m/s", 3),
    ("ball", "vel_kmh", "km/h", 2),
    ("ball", "flight_frames", "frames", 0),
    ("ball", "flight_time_s", "s", 4),
    ("ball", "ball_entry_x_m", "m", 3),
    ("ball", "ball_entry_z_m", "m", 3),
    ("ball", "ball_entry_x_from_centre_m", "m", 3),
    ("ball", "ball_inside_goal", "-", -1),
    ("ball", "shot_outcome", "-", -1),
    ("ball", "shot_outcome_label", "-", -1),
    ("ball", "zone_index", "-", 0),
    ("ball", "zone_label", "-", -1),
    ("ball", "placement_index", "0-100", 1),
    ("ball", "dist_to_nearest_post_m", "m", 3),
    ("ball", "dist_to_crossbar_m", "m", 3),
    ("ball", "launch_elevation_deg", "deg", 2),
    ("ball", "launch_azimuth_deg", "deg", 2),
    ("ball", "apex_height_m", "m", 3),
    ("goalkeeper", "gk_response_frames", "frames", 0),
    ("goalkeeper", "gk_response_time", "s", 4),
    ("goalkeeper", "reaction_class", "-", -1),
    ("goalkeeper", "reaction_label", "-", -1),
    ("goalkeeper", "gk_dist", "m", 3),
    ("goalkeeper", "gk_dive_time_s", "s", 4),
    ("goalkeeper", "gk_vel_ms", "m/s", 3),
    ("goalkeeper", "gk_vel_kmh", "km/h", 2),
    ("goalkeeper", "dive_side_class", "-", -1),
    ("goalkeeper", "gk_start_x_m", "m", 3),
    ("goalkeeper", "gk_start_z_m", "m", 3),
    ("goalkeeper", "gk_end_x_m", "m", 3),
    ("goalkeeper", "gk_end_z_m", "m", 3),
    ("reach", "gap_m", "m", 3),
    ("reach", "gap_source", "-", -1),
    ("reach", "gap_landmark", "-", -1),
    ("reach", "gap_ref_x_m", "m", 3),
    ("reach", "gap_ref_z_m", "m", 3),
    ("reach", "gap_horizontal_m", "m", 3),
    ("reach", "gap_vertical_m", "m", 3),
    ("reach", "reach_standing_m", "m", 3),
    ("reach", "reach_dive_m", "m", 3),
    ("reach", "travel_dive_m", "m", 3),
    ("reach", "available_time_s", "s", 4),
    ("reach", "required_dive_speed_ms", "m/s", 3),
    ("reach", "time_to_reach_s", "s", 4),
    ("reach", "time_margin_s", "s", 4),
    ("reach", "verdict_class", "-", -1),
    ("reach", "verdict_label", "-", -1),
    ("anthropometrics", "gk_height_m", "m", 3),
    ("anthropometrics", "gk_arm_span_m", "m", 3),
    ("anthropometrics", "gk_standing_reach_m", "m", 3),
    ("anthropometrics", "kicker_height_m", "m", 3),
    ("setup", "fps", "Hz", 4),
    ("setup", "kick_frame", "frame", 0),
    ("setup", "goal_frame", "frame", 0),
    ("setup", "gk_move_frame", "frame", 0),
    ("setup", "goal_width_m", "m", 2),
    ("setup", "goal_height_m", "m", 2),
    ("setup", "penalty_distance_m", "m", 2),
    ("setup", "calibration_residual_px", "px", 4),
    ("pose", "kicker_kicking_leg", "-", -1),
    ("pose", "kicker_support_knee_angle_deg", "deg", 2),
    ("pose", "kicker_support_hip_angle_deg", "deg", 2),
    ("pose", "kicker_trunk_lean_deg", "deg", 2),
    ("pose", "kicker_foot_speed_px_s", "px/s", 1),
    ("pose", "gk_max_hand_span_px", "px", 1),
    ("pose", "gk_max_hand_span_m", "m", 3),
    ("pose", "gk_hip_travel_px", "px", 1),
    ("pose", "gk_first_hand_move_rel_kick_s", "s", 4),
)


def _cell(value, nd: int) -> str:
    """Format one CSV cell. Undefined values become empty, not a fake number."""
    if value is None:
        return ""
    if nd < 0 or isinstance(value, str | bool):
        return str(value)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return ""
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    return f"{f:.{nd}f}" if nd > 0 else str(int(round(f)))


def write_results_csv(out_dir: str, ctx: ReportContext) -> str:
    """Write the tidy per-variable results table."""
    merged = {**ctx.metrics, **ctx.pose_metrics}
    path = os.path.join(out_dir, RESULTS_CSV)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["category", "variable", "value", "unit"])
        writer.writerow(["setup", "video_path", ctx.video_path, ""])
        writer.writerow(["setup", "analysis_datetime", ctx.generated_at, ""])
        for category, key, unit, nd in _RESULTS_ROWS:
            if key not in merged:
                continue
            writer.writerow([category, key, _cell(merged[key], nd), unit])

        writer.writerow([])
        writer.writerow(["raw_pixel", "point", "x_px", "y_px"])
        for name, key in (
            ("kick_ball", "kick_ball_px"),
            ("kick_gk", "kick_gk_px"),
            ("goal_ball", "goal_ball_px"),
            ("goal_gk", "goal_gk_px"),
        ):
            p = ctx.metrics.get(key)
            if p:
                writer.writerow(["raw_pixel", name, _cell(p[0], 2), _cell(p[1], 2)])
        for i, p in enumerate(ctx.metrics.get("calib_pixels") or []):
            writer.writerow(["raw_pixel", f"calib_{i + 1}", _cell(p[0], 2), _cell(p[1], 2)])
    return path


def summary_row(ctx: ReportContext) -> dict:
    """Flatten one penalty into the stable wide schema."""
    merged = {**ctx.metrics, **ctx.pose_metrics}
    row = {
        "analysis_datetime": ctx.generated_at,
        "video_name": ctx.video_name,
        "video_path": ctx.video_path,
        "ball_path_points": len(ctx.ball_rows),
    }
    for key in SUMMARY_FIELDS:
        if key in row:
            continue
        value = merged.get(key)
        if isinstance(value, float) and not math.isfinite(value):
            value = "" if math.isnan(value) else ("inf" if value > 0 else "-inf")
        row[key] = "" if value is None else value
    return {k: row.get(k, "") for k in SUMMARY_FIELDS}


def write_summary_csv(out_dir: str, ctx: ReportContext) -> str:
    """Write the one-row-per-penalty wide summary."""
    path = os.path.join(out_dir, SUMMARY_CSV)
    row = summary_row(ctx)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SUMMARY_FIELDS))
        writer.writeheader()
        writer.writerow(row)
    return path


def append_database(db_path: str, ctx: ReportContext) -> str:
    """Add this penalty to the accumulating database, replacing a prior run.

    Rows are keyed on ``(video_name, kick_frame)`` so re-analysing the same
    clip updates its row instead of duplicating it. The file is rewritten in
    full rather than appended to, which keeps the header consistent when the
    schema grows; these files hold one row per penalty, so the cost is
    negligible.
    """
    row = summary_row(ctx)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if os.path.exists(db_path):
        try:
            with open(db_path, newline="", encoding="utf-8") as fh:
                existing = [r for r in csv.DictReader(fh) if any(v for v in r.values())]
        except Exception as exc:
            print(f">> vaila/pynalty_report: could not read {db_path}: {exc}")
            existing = []

    key = (str(row["video_name"]), str(row["kick_frame"]))
    kept = [
        r for r in existing if (str(r.get("video_name", "")), str(r.get("kick_frame", ""))) != key
    ]
    kept.append(row)

    with open(db_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SUMMARY_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for r in kept:
            writer.writerow({k: r.get(k, "") for k in SUMMARY_FIELDS})
    return db_path


def write_ball_path_csvs(
    out_dir: str,
    ball_rows: list[dict],
    flight_path: np.ndarray | None,
) -> dict[str, str]:
    """Write the measured pixel path and the modelled 3D flight path."""
    written: dict[str, str] = {}

    if ball_rows:
        path = os.path.join(out_dir, BALL_PATH_PIXEL_CSV)
        cols = [
            "frame",
            "time_s",
            "x_px",
            "y_px",
            "source",
            "plane_x_m",
            "plane_z_m",
            "speed_px_s",
            "plane_speed_ms",
        ]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for r in ball_rows:
                writer.writerow(
                    {
                        c: ("" if r.get(c) is None else _cell(r.get(c), 4 if c != "frame" else 0))
                        for c in cols
                    }
                    | {"source": r.get("source", "")}
                )
        written["ball_path_pixel"] = path

    if flight_path is not None and len(flight_path):
        path = os.path.join(out_dir, BALL_PATH_3D_CSV)
        arr = np.asarray(flight_path, dtype=float)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time_s", "X_m", "Y_m", "Z_m", "speed_ms"])
            for i in range(arr.shape[0]):
                speed = ""
                if i > 0:
                    dt_s = arr[i, 0] - arr[i - 1, 0]
                    if dt_s > 0:
                        speed = f"{np.linalg.norm(arr[i, 1:4] - arr[i - 1, 1:4]) / dt_s:.4f}"
                writer.writerow(
                    [
                        f"{arr[i, 0]:.4f}",
                        f"{arr[i, 1]:.4f}",
                        f"{arr[i, 2]:.4f}",
                        f"{arr[i, 3]:.4f}",
                        speed,
                    ]
                )
        written["ball_path_3d"] = path

    return written


def write_pose_csv(out_dir: str, sequence, filename: str) -> str | None:
    """Write one pose sequence as pixel landmarks in named and vailá columns."""
    if sequence is None or not getattr(sequence, "landmarks", None):
        return None
    path = os.path.join(out_dir, filename)
    n = len(LANDMARK_NAMES)
    header = ["frame"]
    for name in LANDMARK_NAMES:
        header += [f"{name}_x", f"{name}_y", f"{name}_vis"]
    header += [f"p{i + 1}_{ax}" for i in range(n) for ax in ("x", "y")]

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for frame in sorted(sequence.landmarks):
            pts = sequence.landmarks[frame]
            row: list = [frame]
            for i in range(n):
                row += [_cell(pts[i, 0], 3), _cell(pts[i, 1], 3), _cell(pts[i, 2], 4)]
            for i in range(n):
                row += [_cell(pts[i, 0], 3), _cell(pts[i, 1], 3)]
            writer.writerow(row)
    return path


# ==============================================================================
# SVG goal map
# ==============================================================================


def build_goal_svg(ctx: ReportContext, lang: str) -> str:
    """Inline SVG of the goal mouth: zone grid, reach circles, entry point.

    Drawn to scale in metres and mapped onto a fixed viewBox, so the aspect
    ratio of a non-standard goal is preserved.
    """
    m = ctx.metrics
    goal = ctx.goal
    pad = 0.6
    vb_w = goal.width + 2 * pad
    vb_h = goal.height + 2 * pad
    scale = 120.0

    def px(x: float) -> float:
        return (x + pad) * scale

    def py(z: float) -> float:
        return (goal.height - z + pad) * scale

    w_px, h_px = vb_w * scale, vb_h * scale
    parts: list[str] = [
        f'<svg viewBox="0 0 {w_px:.0f} {h_px:.0f}" class="goal-svg" '
        'xmlns="http://www.w3.org/2000/svg" role="img">',
        f'<rect x="0" y="0" width="{w_px:.0f}" height="{h_px:.0f}" fill="#20262e"/>',
        # pitch surface below the goal line
        f'<rect x="0" y="{py(0):.1f}" width="{w_px:.0f}" height="{h_px - py(0):.1f}" fill="#27412c"/>',
    ]

    # zone grid
    for c in range(3):
        for r in range(3):
            x0, z0 = c * goal.width / 3.0, r * goal.height / 3.0
            fill = "#2f3b47" if (r + c) % 2 == 0 else "#333f4c"
            if m.get("zone_index") == r * 3 + c:
                fill = "#7a3b3b"
            parts.append(
                f'<rect x="{px(x0):.1f}" y="{py(z0 + goal.height / 3.0):.1f}" '
                f'width="{goal.width / 3.0 * scale:.1f}" height="{goal.height / 3.0 * scale:.1f}" '
                f'fill="{fill}" stroke="#4b5a6a" stroke-width="1"/>'
            )

    # posts and crossbar
    parts.append(
        f'<rect x="{px(0):.1f}" y="{py(goal.height):.1f}" '
        f'width="{goal.width * scale:.1f}" height="{goal.height * scale:.1f}" '
        'fill="none" stroke="#f5f5f5" stroke-width="6"/>'
    )

    # goalkeeper reach circles
    gk_x, gk_z = m.get("gk_start_x_m"), m.get("gk_start_z_m")
    if gk_x is not None and gk_z is not None:
        for radius_key, color in (("reach_dive_m", "#ffb74d"), ("reach_standing_m", "#4fc3f7")):
            radius = m.get(radius_key)
            if radius and math.isfinite(radius):
                parts.append(
                    f'<circle cx="{px(gk_x):.1f}" cy="{py(gk_z):.1f}" r="{radius * scale:.1f}" '
                    f'fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="10 8"/>'
                )
        parts.append(
            f'<circle cx="{px(gk_x):.1f}" cy="{py(gk_z):.1f}" r="10" fill="#4fc3f7"/>'
            f'<text x="{px(gk_x) + 16:.1f}" y="{py(gk_z) - 12:.1f}" class="svg-lbl">GK</text>'
        )

    gk_ex, gk_ez = m.get("gk_end_x_m"), m.get("gk_end_z_m")
    if gk_x is not None and gk_ex is not None:
        parts.append(
            f'<line x1="{px(gk_x):.1f}" y1="{py(gk_z):.1f}" x2="{px(gk_ex):.1f}" '
            f'y2="{py(gk_ez):.1f}" stroke="#e040fb" stroke-width="4"/>'
            f'<circle cx="{px(gk_ex):.1f}" cy="{py(gk_ez):.1f}" r="8" fill="#e040fb"/>'
        )

    # ball entry and the gap line to the keeper
    bx, bz = m.get("ball_entry_x_m"), m.get("ball_entry_z_m")
    if bx is not None and bz is not None:
        if gk_x is not None:
            parts.append(
                f'<line x1="{px(gk_x):.1f}" y1="{py(gk_z):.1f}" x2="{px(bx):.1f}" '
                f'y2="{py(bz):.1f}" stroke="#ff5252" stroke-width="2" stroke-dasharray="4 6"/>'
            )
            mid_x, mid_z = (gk_x + bx) / 2.0, (gk_z + bz) / 2.0
            parts.append(
                f'<text x="{px(mid_x):.1f}" y="{py(mid_z) - 10:.1f}" class="svg-lbl">'
                f"{_num(m.get('gap_m'))} m</text>"
            )
        parts.append(
            f'<circle cx="{px(bx):.1f}" cy="{py(bz):.1f}" r="13" fill="#ffee58" '
            'stroke="#212121" stroke-width="3"/>'
        )

    # axis ticks
    for x in (0.0, goal.mid_x, goal.width):
        parts.append(
            f'<text x="{px(x):.1f}" y="{py(0) + 34:.1f}" class="svg-tick" text-anchor="middle">'
            f"{x:.2f} m</text>"
        )
    parts.append(
        f'<text x="{px(0) - 14:.1f}" y="{py(goal.height):.1f}" class="svg-tick" '
        f'text-anchor="end">{goal.height:.2f} m</text>'
    )
    parts.append(
        f'<text x="{px(0.05):.1f}" y="{py(goal.height) - 14:.1f}" class="svg-lbl">'
        f"{t(lang, 'goal_map')}</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


# ==============================================================================
# HTML report
# ==============================================================================

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="{{LANG}}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{TITLE}} - {{VIDEO_NAME}}</title>
<style>
  :root {
    --bg: #14181d; --card: #1e242c; --card2: #262e38; --ink: #eceff1;
    --muted: #98a4b3; --accent: #4caf50; --blue: #4fc3f7; --amber: #ffb74d;
    --red: #ff5252; --violet: #e040fb;
  }
  * { box-sizing: border-box; }
  body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: var(--bg);
         color: var(--ink); margin: 0; padding: 24px; line-height: 1.55; }
  .container { max-width: 1100px; margin: 0 auto; }
  header.head { background: var(--card); border-radius: 12px; padding: 24px 28px; margin-bottom: 22px; }
  h1 { margin: 0 0 8px; font-size: 1.7em; color: var(--accent); }
  h2 { color: var(--accent); border-bottom: 2px solid #35414e; padding-bottom: 8px;
       margin: 34px 0 16px; font-size: 1.25em; }
  h3 { margin: 0 0 10px; font-size: 1.02em; color: var(--blue); }
  .meta { color: var(--muted); font-size: .92em; }
  .meta b { color: var(--ink); font-weight: 600; }
  .verdict { display: flex; align-items: center; gap: 16px; background: var(--card);
             border-radius: 12px; padding: 20px 24px; border-left: 6px solid var(--muted); }
  .verdict.good { border-left-color: var(--accent); }
  .verdict.warn { border-left-color: var(--amber); }
  .verdict.bad { border-left-color: var(--red); }
  .verdict.neutral { border-left-color: var(--muted); }
  .verdict-title { font-size: 1.25em; font-weight: 700; }
  .verdict-body { color: var(--muted); font-size: .95em; margin-top: 4px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; }
  .card { background: var(--card); border-radius: 10px; padding: 14px 16px;
          border-left: 4px solid var(--blue); }
  .card.amber { border-left-color: var(--amber); }
  .card.red { border-left-color: var(--red); }
  .card.violet { border-left-color: var(--violet); }
  .card.green { border-left-color: var(--accent); }
  .k { font-size: .82em; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
  .v { font-size: 1.42em; font-weight: 700; margin-top: 4px; }
  .u { font-size: .85em; color: var(--muted); }
  .two { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: 20px;
         align-items: start; }
  .panel { background: var(--card); border-radius: 10px; padding: 18px; }
  .goal-svg { width: 100%; height: auto; border-radius: 8px; background: #20262e; }
  .svg-lbl { fill: #eceff1; font: 600 15px 'Segoe UI', sans-serif; }
  .svg-tick { fill: #98a4b3; font: 400 13px 'Segoe UI', sans-serif; }
  table { width: 100%; border-collapse: collapse; font-size: .93em; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #313c48; }
  th { color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: .8em; }
  td.n { text-align: right; font-variant-numeric: tabular-nums; }
  .note { color: var(--muted); font-size: .88em; margin-top: 10px; }
  .snaps { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
  .snap { background: var(--card); border-radius: 10px; padding: 12px; }
  .snap img { width: 100%; border-radius: 6px; border: 1px solid #3a4655; display: block; }
  .anim-wrap { background: var(--card); border-radius: 10px; padding: 18px; }
  .canvases { display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 16px; }
  .canvas-box { background: #20262e; border-radius: 8px; padding: 10px; }
  canvas { width: 100%; height: auto; display: block; background: #20262e; border-radius: 6px; }
  .controls { display: flex; align-items: center; gap: 14px; margin-top: 16px; flex-wrap: wrap; }
  button { background: var(--accent); color: #11151a; border: 0; border-radius: 6px;
           padding: 8px 18px; font-size: .95em; font-weight: 600; cursor: pointer; }
  button.ghost { background: #35414e; color: var(--ink); }
  input[type=range] { flex: 1 1 220px; accent-color: var(--accent); }
  .tval { font-variant-numeric: tabular-nums; color: var(--muted); min-width: 130px; }
  ul.files { list-style: none; padding: 0; margin: 0; columns: 2; column-gap: 26px; }
  ul.files li { padding: 3px 0; color: var(--muted); font-size: .9em; }
  ul.files code { color: var(--ink); }
  footer { margin-top: 42px; text-align: center; color: var(--muted); font-size: .85em; }
  em { font-style: italic; }
  @media (max-width: 820px) { .two { grid-template-columns: 1fr; } ul.files { columns: 1; } }
</style>
</head>
<body>
<div class="container">

<header class="head">
  <h1>{{TITLE}}</h1>
  <div class="meta">
    <b>{{L_VIDEO}}:</b> {{VIDEO_NAME}} &nbsp;|&nbsp;
    <b>{{L_DATE}}:</b> {{DATE}} &nbsp;|&nbsp;
    <b>{{L_FPS}}:</b> {{FPS}} Hz
  </div>
</header>

<h2>{{L_SUMMARY}}</h2>
<div class="verdict {{VERDICT_TONE}}">
  <div>
    <div class="verdict-title">{{VERDICT_LABEL}}</div>
    <div class="verdict-body">{{SUMMARY_TEXT}}</div>
  </div>
</div>

<h2>{{L_KICKER}}</h2>
<div class="grid">{{KICKER_CARDS}}</div>

<h2>{{L_GK}}</h2>
<div class="grid">{{GK_CARDS}}</div>

<h2>{{L_REACH}}</h2>
<div class="two">
  <div class="panel">
    {{GOAL_SVG}}
    <div class="note">{{ZONE_NOTE}}</div>
  </div>
  <div class="panel">
    <h3>{{L_REACH}}</h3>
    <table>{{REACH_TABLE}}</table>
    {{REACH_NOTE}}
  </div>
</div>

<h2>{{L_TRAJECTORY}}</h2>
<div class="anim-wrap">
  <div class="canvases">
    <div class="canvas-box">
      <h3>{{L_FRONT}}</h3>
      <canvas id="cvFront" width="520" height="300"></canvas>
    </div>
    <div class="canvas-box">
      <h3>{{L_PLAN}}</h3>
      <canvas id="cvPlan" width="520" height="300"></canvas>
    </div>
    <div class="canvas-box">
      <h3>{{L_SIDE}}</h3>
      <canvas id="cvSide" width="520" height="300"></canvas>
    </div>
  </div>
  <div class="controls">
    <button id="btnPlay">{{L_PLAY}}</button>
    <button id="btnReset" class="ghost">{{L_RESTART}}</button>
    <input id="scrub" type="range" min="0" max="1000" value="0">
    <span class="tval" id="tval"></span>
  </div>
  <div class="note">{{ANIM_NOTE}}</div>
</div>
{{BALL_TABLE_BLOCK}}

{{POSE_BLOCK}}

<h2>{{L_SNAPSHOTS}}</h2>
<div class="snaps">{{SNAPSHOTS}}</div>

<h2>{{L_METHODS}}</h2>
<div class="panel">{{METHODS}}</div>

<h2>{{L_FILES}}</h2>
<div class="panel"><ul class="files">{{FILES}}</ul></div>

<footer>{{FOOTER}}</footer>
</div>

<script>
const DATA = {{ANIM_DATA}};
const LBL = {{ANIM_LABELS}};

function mk(canvas) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width, h = canvas.height;
  // Scale the backing store for crisp lines but keep the intrinsic aspect
  // ratio, so the CSS `width:100%; height:auto` rule cannot squash the drawing.
  canvas.width = w * dpr; canvas.height = h * dpr;
  ctx.scale(dpr, dpr);
  return {ctx: ctx, w: w, h: h};
}

const front = mk(document.getElementById('cvFront'));
const plan  = mk(document.getElementById('cvPlan'));
const side  = mk(document.getElementById('cvSide'));

const GW = DATA.goal.width, GH = DATA.goal.height, PD = DATA.goal.penalty_distance;
const T  = DATA.flight_time > 0 ? DATA.flight_time : 1e-6;

function lerp(a, b, u) { return a + (b - a) * u; }

function sampleAt(u) {
  const p = DATA.samples;
  if (!p.length) return {t: 0, x: 0, y: 0, z: 0};
  const pos = u * (p.length - 1);
  const i = Math.min(p.length - 1, Math.max(0, Math.floor(pos)));
  const j = Math.min(p.length - 1, i + 1);
  const f = pos - i;
  return {
    t: lerp(p[i][0], p[j][0], f),
    x: lerp(p[i][1], p[j][1], f),
    y: lerp(p[i][2], p[j][2], f),
    z: lerp(p[i][3], p[j][3], f)
  };
}

// Goalkeeper travels from the set position toward the marked end position,
// starting only after the measured reaction delay.
function gkAt(t) {
  const g = DATA.gk;
  if (g.start_x === null) return null;
  const t0 = Math.max(0, g.reaction_s);
  const span = Math.max(1e-6, T - t0);
  const u = Math.min(1, Math.max(0, (t - t0) / span));
  return {x: lerp(g.start_x, g.end_x, u), z: lerp(g.start_z, g.end_z, u)};
}

function drawFront(s, gk) {
  const g = front, p = {l: 46, r: 18, t: 18, b: 34};
  const iw = g.w - p.l - p.r, ih = g.h - p.t - p.b;
  const sc = Math.min(iw / (GW + 1.2), ih / (GH + 1.0));
  const ox = p.l + (iw - GW * sc) / 2, oy = p.t + (ih - GH * sc) / 2;
  const X = (x) => ox + x * sc;              // x in [0, GW]
  const Z = (z) => oy + (GH - z) * sc;       // z in [0, GH]

  g.ctx.clearRect(0, 0, g.w, g.h);
  g.ctx.fillStyle = '#27412c';
  g.ctx.fillRect(0, Z(0), g.w, g.h - Z(0));

  for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) {
    g.ctx.fillStyle = ((r + c) % 2 === 0) ? '#2f3b47' : '#333f4c';
    g.ctx.fillRect(X(c * GW / 3), Z((r + 1) * GH / 3), GW / 3 * sc, GH / 3 * sc);
    g.ctx.strokeStyle = '#46545f'; g.ctx.lineWidth = 1;
    g.ctx.strokeRect(X(c * GW / 3), Z((r + 1) * GH / 3), GW / 3 * sc, GH / 3 * sc);
  }
  g.ctx.strokeStyle = '#f5f5f5'; g.ctx.lineWidth = 4;
  g.ctx.strokeRect(X(0), Z(GH), GW * sc, GH * sc);

  if (gk) {
    const rd = DATA.gk.reach_dive, rs = DATA.gk.reach_standing;
    g.ctx.setLineDash([7, 6]); g.ctx.lineWidth = 2;
    if (rd) { g.ctx.strokeStyle = '#ffb74d'; g.ctx.beginPath();
              g.ctx.arc(X(gk.x), Z(gk.z), rd * sc, 0, 6.2832); g.ctx.stroke(); }
    if (rs) { g.ctx.strokeStyle = '#4fc3f7'; g.ctx.beginPath();
              g.ctx.arc(X(gk.x), Z(gk.z), rs * sc, 0, 6.2832); g.ctx.stroke(); }
    g.ctx.setLineDash([]);
    g.ctx.fillStyle = '#4fc3f7'; g.ctx.beginPath();
    g.ctx.arc(X(gk.x), Z(gk.z), 7, 0, 6.2832); g.ctx.fill();
  }

  // Perspective: the ball grows as it closes on the goal line.
  const centreX = GW / 2 + s.x;
  const depth = Math.max(0.04, 1 - s.y / PD);
  const rad = 4 + 11 * (1 - depth);
  g.ctx.strokeStyle = 'rgba(255,238,88,.45)'; g.ctx.lineWidth = 2;
  g.ctx.beginPath();
  g.ctx.moveTo(X(GW / 2), Z(DATA.goal.ball_radius));
  g.ctx.lineTo(X(centreX), Z(Math.max(s.z, 0)));
  g.ctx.stroke();
  g.ctx.fillStyle = '#ffee58';
  g.ctx.beginPath(); g.ctx.arc(X(centreX), Z(Math.max(s.z, 0)), rad, 0, 6.2832); g.ctx.fill();
  g.ctx.strokeStyle = '#212121'; g.ctx.lineWidth = 2; g.ctx.stroke();

  g.ctx.fillStyle = '#98a4b3'; g.ctx.font = '11px Segoe UI, sans-serif';
  g.ctx.fillText('0', X(0) - 4, Z(0) + 16);
  g.ctx.fillText(GW.toFixed(2) + ' m', X(GW) - 26, Z(0) + 16);
}

function drawPlan(s, gk) {
  const g = plan, p = {l: 46, r: 18, t: 18, b: 30};
  const iw = g.w - p.l - p.r, ih = g.h - p.t - p.b;
  const halfW = GW / 2 + 1.6;
  const sc = Math.min(iw / (2 * halfW), ih / (PD + 1.4));
  const ox = p.l + iw / 2, oy = p.t + ih - 0.7 * sc;
  const X = (x) => ox + x * sc;      // x in metres from the centre line
  const Y = (y) => oy - y * sc;      // y toward the goal

  g.ctx.clearRect(0, 0, g.w, g.h);
  g.ctx.fillStyle = '#27412c'; g.ctx.fillRect(0, 0, g.w, g.h);
  g.ctx.strokeStyle = '#f5f5f5'; g.ctx.lineWidth = 3;
  g.ctx.beginPath(); g.ctx.moveTo(X(-halfW), Y(PD)); g.ctx.lineTo(X(halfW), Y(PD)); g.ctx.stroke();
  g.ctx.lineWidth = 5;
  g.ctx.beginPath(); g.ctx.moveTo(X(-GW / 2), Y(PD)); g.ctx.lineTo(X(GW / 2), Y(PD)); g.ctx.stroke();
  g.ctx.strokeStyle = 'rgba(245,245,245,.35)'; g.ctx.lineWidth = 1;
  g.ctx.beginPath(); g.ctx.moveTo(X(0), Y(0)); g.ctx.lineTo(X(0), Y(PD)); g.ctx.stroke();

  g.ctx.fillStyle = '#f5f5f5';
  g.ctx.beginPath(); g.ctx.arc(X(0), Y(0), 4, 0, 6.2832); g.ctx.fill();

  if (gk) {
    const rd = DATA.gk.reach_dive;
    if (rd) {
      g.ctx.setLineDash([7, 6]); g.ctx.strokeStyle = '#ffb74d'; g.ctx.lineWidth = 2;
      g.ctx.beginPath(); g.ctx.arc(X(gk.x - GW / 2), Y(PD), rd * sc, 0, 6.2832); g.ctx.stroke();
      g.ctx.setLineDash([]);
    }
    g.ctx.fillStyle = '#4fc3f7';
    g.ctx.beginPath(); g.ctx.arc(X(gk.x - GW / 2), Y(PD), 7, 0, 6.2832); g.ctx.fill();
  }

  g.ctx.strokeStyle = 'rgba(255,238,88,.55)'; g.ctx.lineWidth = 2;
  g.ctx.beginPath(); g.ctx.moveTo(X(0), Y(0)); g.ctx.lineTo(X(s.x), Y(s.y)); g.ctx.stroke();
  g.ctx.fillStyle = '#ffee58';
  g.ctx.beginPath(); g.ctx.arc(X(s.x), Y(s.y), 6, 0, 6.2832); g.ctx.fill();

  g.ctx.fillStyle = '#cfd8dc'; g.ctx.font = '11px Segoe UI, sans-serif';
  g.ctx.fillText(PD.toFixed(1) + ' m', X(halfW) - 44, Y(PD / 2));
}

function drawSide(s) {
  const g = side, p = {l: 44, r: 18, t: 18, b: 30};
  const iw = g.w - p.l - p.r, ih = g.h - p.t - p.b;
  const zMax = Math.max(GH + 0.5, DATA.apex_z + 0.4);
  const scx = iw / (PD + 0.8), scz = ih / zMax;
  const ox = p.l, oy = p.t + ih;
  const X = (y) => ox + y * scx;
  const Z = (z) => oy - z * scz;

  g.ctx.clearRect(0, 0, g.w, g.h);
  g.ctx.fillStyle = '#20262e'; g.ctx.fillRect(0, 0, g.w, g.h);
  g.ctx.fillStyle = '#27412c'; g.ctx.fillRect(0, Z(0), g.w, g.h - Z(0));
  g.ctx.strokeStyle = '#f5f5f5'; g.ctx.lineWidth = 4;
  g.ctx.beginPath(); g.ctx.moveTo(X(PD), Z(0)); g.ctx.lineTo(X(PD), Z(GH)); g.ctx.stroke();
  g.ctx.beginPath(); g.ctx.moveTo(X(PD), Z(GH)); g.ctx.lineTo(X(PD) + 14, Z(GH)); g.ctx.stroke();

  g.ctx.strokeStyle = 'rgba(255,238,88,.4)'; g.ctx.lineWidth = 2;
  g.ctx.beginPath();
  DATA.samples.forEach(function (q, i) {
    if (i === 0) g.ctx.moveTo(X(q[2]), Z(q[3])); else g.ctx.lineTo(X(q[2]), Z(q[3]));
  });
  g.ctx.stroke();

  g.ctx.fillStyle = '#ffee58';
  g.ctx.beginPath(); g.ctx.arc(X(s.y), Z(s.z), 6, 0, 6.2832); g.ctx.fill();

  g.ctx.fillStyle = '#98a4b3'; g.ctx.font = '11px Segoe UI, sans-serif';
  g.ctx.fillText(zMax.toFixed(1) + ' m', 4, Z(zMax) + 12);
  g.ctx.fillText('0 m', 4, Z(0) - 4);
}

const scrub = document.getElementById('scrub');
const tval = document.getElementById('tval');
const btnPlay = document.getElementById('btnPlay');
let playing = false, u = 0, last = 0;

function render() {
  const s = sampleAt(u);
  const gk = gkAt(s.t);
  drawFront(s, gk);
  drawPlan(s, gk);
  drawSide(s);
  tval.textContent = LBL.time + ': ' + (s.t * 1000).toFixed(0) + ' ms / ' +
                     (T * 1000).toFixed(0) + ' ms';
  scrub.value = Math.round(u * 1000);
}

function tick(ts) {
  if (!playing) return;
  if (!last) last = ts;
  // Replay at one fifth of real time: a penalty lasts under half a second.
  u += (ts - last) / 1000 / (T * 5);
  last = ts;
  if (u >= 1) { u = 1; playing = false; btnPlay.textContent = LBL.play; }
  render();
  if (playing) requestAnimationFrame(tick);
}

btnPlay.addEventListener('click', function () {
  if (u >= 1) u = 0;
  playing = !playing;
  btnPlay.textContent = playing ? LBL.pause : LBL.play;
  last = 0;
  if (playing) requestAnimationFrame(tick);
});
document.getElementById('btnReset').addEventListener('click', function () {
  playing = false; btnPlay.textContent = LBL.play; u = 0; render();
});
scrub.addEventListener('input', function () {
  playing = false; btnPlay.textContent = LBL.play;
  u = parseFloat(scrub.value) / 1000; render();
});
render();
</script>
</body>
</html>
"""


def _card(label: str, value: str, unit: str = "", tone: str = "") -> str:
    cls = f"card {tone}".strip()
    unit_html = f'<div class="u">{unit}</div>' if unit else ""
    return f'<div class="{cls}"><div class="k">{label}</div><div class="v">{value}</div>{unit_html}</div>'


def _row(label: str, value: str, unit: str = "") -> str:
    unit_html = f" {unit}" if unit else ""
    return f'<tr><th>{label}</th><td class="n">{value}{unit_html}</td></tr>'


def _kicker_cards(ctx: ReportContext, lang: str) -> str:
    m = ctx.metrics
    cards = [
        _card(
            t(lang, "ball_speed"),
            _num(m.get("vel_kmh")),
            f"km/h &middot; {_num(m.get('vel_ms'))} m/s",
            "green",
        ),
        _card(
            t(lang, "flight_time"),
            _num(m.get("flight_time_s"), 3),
            f"s &middot; {m.get('flight_frames', '?')} frames",
        ),
        _card(t(lang, "flight_distance"), _num(m.get("dist")), "m"),
        _card(
            t(lang, "target_zone"),
            _zone_label(lang, m.get("zone_label", "")),
            f"X {_num(m.get('ball_entry_x_m'))} m &middot; Z {_num(m.get('ball_entry_z_m'))} m",
            "amber",
        ),
        _card(t(lang, "placement"), _num(m.get("placement_index"), 0), "0-100"),
        _card(t(lang, "launch_angle"), _num(m.get("launch_elevation_deg"), 1), "&deg;"),
        _card(t(lang, "apex"), _num(m.get("apex_height_m")), "m"),
        _card(t(lang, "nearest_post"), _num(m.get("dist_to_nearest_post_m")), "m"),
        _card(t(lang, "to_crossbar"), _num(m.get("dist_to_crossbar_m")), "m"),
    ]
    return "".join(cards)


def _gk_cards(ctx: ReportContext, lang: str) -> str:
    m = ctx.metrics
    reaction = _reaction_label(
        lang,
        m.get("reaction_class", ""),
        m.get("gk_response_time", 0.0),
        m.get("reaction_label", ""),
    )
    side = _side_label(lang, m.get("dive_side_class", ""), m.get("dive_side_label", ""))
    tone = "green" if m.get("dive_side_class") == "correct_side" else "amber"
    cards = [
        _card(t(lang, "reaction"), reaction, f"{_num(m.get('gk_response_time'), 3)} s", "violet"),
        _card(t(lang, "dive_side"), side, "", tone),
        _card(t(lang, "dive_distance"), _num(m.get("gk_dist")), "m"),
        _card(
            t(lang, "dive_speed"),
            _num(m.get("gk_vel_ms")),
            f"m/s &middot; {_num(m.get('gk_vel_kmh'))} km/h",
        ),
    ]
    if m.get("gk_height_m"):
        cards.append(_card(t(lang, "gk_height"), _num(m.get("gk_height_m")), "m"))
        cards.append(_card(t(lang, "gk_span"), _num(m.get("gk_arm_span_m")), "m"))
        cards.append(_card(t(lang, "gk_reach"), _num(m.get("gk_standing_reach_m")), "m"))
    return "".join(cards)


def _reach_table(ctx: ReportContext, lang: str) -> str:
    m = ctx.metrics
    rows = [
        _row(t(lang, "gap"), _num(m.get("gap_m")), "m"),
        _row(t(lang, "reach_standing"), _num(m.get("reach_standing_m")), "m"),
        _row(t(lang, "reach_dive"), _num(m.get("reach_dive_m")), "m"),
        _row(t(lang, "travel_needed"), _num(m.get("travel_dive_m")), "m"),
        _row(t(lang, "available_time"), _num(m.get("available_time_s"), 3), "s"),
        _row(t(lang, "required_speed"), _num(m.get("required_dive_speed_ms")), "m/s"),
        _row(t(lang, "dive_speed"), _num(m.get("gk_vel_ms")), "m/s"),
        _row(t(lang, "time_to_reach"), _num(m.get("time_to_reach_s"), 3), "s"),
        _row(t(lang, "time_margin"), _num(m.get("time_margin_s"), 3), "s"),
        _row(t(lang, "calib_residual"), _num(m.get("calibration_residual_px"), 3), "px"),
    ]
    return "".join(rows)


def _summary_text(ctx: ReportContext, lang: str) -> str:
    m = ctx.metrics
    zone = _zone_label(lang, m.get("zone_label", ""))
    reaction = _reaction_label(
        lang,
        m.get("reaction_class", ""),
        float(m.get("gk_response_time", 0) or 0),
        m.get("reaction_label", ""),
    )
    side = _side_label(lang, m.get("dive_side_class", ""), m.get("dive_side_label", ""))
    outcome = m.get("shot_outcome_label") or m.get("shot_outcome") or ""
    gap_note = ""
    if m.get("gap_source") == "nearest_body_point" and m.get("gap_landmark"):
        gap_note = (
            f" Distância até a bola medida do ponto mais próximo ({m['gap_landmark']})."
            if lang == "pt"
            else f" Gap to the ball measured from the nearest body point ({m['gap_landmark']})."
        )
    elif m.get("gap_source") == "defending_part" and m.get("gap_landmark"):
        gap_note = (
            f" Distância até a bola medida da parte que defendeu ({m['gap_landmark']})."
            if lang == "pt"
            else f" Gap to the ball measured from the defending part ({m['gap_landmark']})."
        )
    anthro_note = ""
    if m.get("anthro_source") == "default":
        anthro_note = (
            " Medidas do goleiro: valores padrão."
            if lang == "pt"
            else " Goalkeeper body measures: default values."
        )

    if lang == "pt":
        text = (
            f"Resultado: {outcome or 'n/a'}. "
            f"A bola saiu a {_num(m.get('vel_kmh'))} km/h e cruzou a linha em "
            f"{_num(m.get('flight_time_s'), 3)} s, na zona {zone} "
            f"(X {_num(m.get('ball_entry_x_m'))} m, Z {_num(m.get('ball_entry_z_m'))} m). "
            f"{reaction}; {side.lower()}.{gap_note}{anthro_note} "
        )
        if m.get("gk_height_m"):
            text += (
                f"O goleiro precisava cobrir {_num(m.get('travel_dive_m'))} m em "
                f"{_num(m.get('available_time_s'), 3)} s, ou seja "
                f"{_num(m.get('required_dive_speed_ms'))} m/s, contra "
                f"{_num(m.get('gk_vel_ms'))} m/s medidos."
            )
    else:
        text = (
            f"Result: {outcome or 'n/a'}. "
            f"The ball left at {_num(m.get('vel_kmh'))} km/h and crossed the line in "
            f"{_num(m.get('flight_time_s'), 3)} s, in the {zone} zone "
            f"(X {_num(m.get('ball_entry_x_m'))} m, Z {_num(m.get('ball_entry_z_m'))} m). "
            f"{reaction}; {side.lower()}.{gap_note}{anthro_note} "
        )
        if m.get("gk_height_m"):
            text += (
                f"The keeper had to cover {_num(m.get('travel_dive_m'))} m in "
                f"{_num(m.get('available_time_s'), 3)} s, which needs "
                f"{_num(m.get('required_dive_speed_ms'))} m/s against the "
                f"{_num(m.get('gk_vel_ms'))} m/s measured."
            )
    return text


def _pose_block(ctx: ReportContext, lang: str) -> str:
    p = ctx.pose_metrics
    if not p:
        return ""
    cards: list[str] = []
    if p.get("kicker_kicking_leg"):
        leg = t(lang, p["kicker_kicking_leg"])
        cards.append(_card(t(lang, "kicking_leg"), leg, "", "green"))
    for key, label, unit, nd in (
        ("kicker_support_knee_angle_deg", "support_knee", "&deg;", 1),
        ("kicker_support_hip_angle_deg", "support_hip", "&deg;", 1),
        ("kicker_trunk_lean_deg", "trunk_lean", "&deg;", 1),
        ("kicker_foot_speed_px_s", "foot_speed", "px/s", 0),
        ("gk_max_hand_span_m", "hand_span", "m", 2),
        ("gk_hip_travel_px", "hip_travel", "px", 0),
        ("gk_first_hand_move_rel_kick_s", "first_hand_move", "s", 3),
    ):
        if key in p:
            cards.append(_card(t(lang, label), _num(p[key], nd), unit))
    if not cards:
        return ""
    return f'<h2>{t(lang, "pose")}</h2><div class="grid">{"".join(cards)}</div>'


def _ball_table_block(ctx: ReportContext, lang: str) -> str:
    """Frame-by-frame table of the marked ball path.

    Positions are measured in pixels; the speed column comes from the fitted
    3D flight, because a pixel displacement can only be converted to metres on
    the goal plane and the ball spends most of its flight well in front of it.
    """
    rows = ctx.ball_rows
    if len(rows) < 2:
        return ""
    head = (
        "<tr><th>frame</th><th>t (s)</th><th>x (px)</th><th>y (px)</th>"
        "<th>v model (m/s)</th><th>source</th></tr>"
    )
    body = "".join(
        f'<tr><td class="n">{int(r["frame"])}</td>'
        f'<td class="n">{_num(r.get("flight_time_s", r.get("time_s")), 3)}</td>'
        f'<td class="n">{_num(r["x_px"], 1)}</td><td class="n">{_num(r["y_px"], 1)}</td>'
        f'<td class="n">{_num(r.get("model_speed_ms"), 2)}</td><td>{r.get("source", "")}</td></tr>'
        for r in rows
    )
    caption = (
        "Posições em pixels marcadas no vídeo (<code>manual</code>), detectadas por YOLO "
        "(<code>auto</code>) ou preenchidas por interpolação (<code>interp</code>). A velocidade "
        "vem do modelo de voo 3D: converter deslocamento em pixels para metros só é válido sobre "
        "o plano do gol."
        if lang == "pt"
        else "Positions are pixels marked on the video (<code>manual</code>), detected by YOLO "
        "(<code>auto</code>) or filled by interpolation (<code>interp</code>). The speed comes "
        "from the 3D flight model: converting pixel displacement to metres is only valid on the "
        "goal plane."
    )
    return (
        f'<div class="panel" style="margin-top:16px"><h3>{t(lang, "ball")}</h3>'
        f"<table>{head}{body}</table>"
        f'<div class="note">{caption}</div></div>'
    )


def _snapshots_html(ctx: ReportContext, lang: str) -> str:
    order = (
        ("snapshot_gk_move", "snapshot_gk_move"),
        ("snapshot_kick", "snapshot_kick"),
        ("snapshot_goal", "snapshot_goal"),
        ("snapshot_calibration", "snapshot_calibration"),
        ("ball_path_composite", "snapshot_ball_path"),
    )
    out: list[str] = []
    for key, label_key in order:
        uri = _b64_img(ctx.snapshots.get(key))
        if not uri:
            continue
        out.append(
            f'<div class="snap"><h3>{t(lang, label_key)}</h3>'
            f'<img src="{uri}" alt="{t(lang, label_key)}"></div>'
        )
    return "".join(out) or f'<div class="note">{t(lang, "no_pose")}</div>'


def _methods_html(ctx: ReportContext, lang: str) -> str:
    m = ctx.metrics
    if lang == "pt":
        items = [
            f"Calibração por DLT2D a partir dos quatro cantos do gol ({_num(m.get('goal_width_m'))} x "
            f"{_num(m.get('goal_height_m'))} m). Só pontos sobre o plano do gol são reconstruídos com exatidão.",
            f"A distância de voo é a reta 3D entre a marca do pênalti ({_num(m.get('penalty_distance_m'))} m) "
            "e o ponto de entrada, portanto a velocidade é uma média do percurso, não a velocidade instantânea de saída.",
            "O tempo de reação é assinado: valores negativos indicam que o goleiro se moveu antes do contato.",
            "O modelo de alcance trata os raios em pé e em salto como círculos isotrópicos em torno do centro marcado "
            "do goleiro; envelopes reais são mais amplos na horizontal do que na vertical.",
            "Ângulos por pose são medidos no plano da imagem e dependem do ponto de vista da câmera.",
            "Câmera única, sem correção de distorção da lente nem de rolling shutter.",
        ]
    else:
        items = [
            f"DLT2D calibration from the four goal corners ({_num(m.get('goal_width_m'))} x "
            f"{_num(m.get('goal_height_m'))} m). Only points on the goal plane reconstruct exactly.",
            f"Flight distance is the 3D straight line from the penalty mark ({_num(m.get('penalty_distance_m'))} m) "
            "to the entry point, so the speed is a path average rather than the instantaneous launch speed.",
            "Reaction time is signed: negative values mean the keeper moved before ball contact.",
            "The reach model treats the standing and diving radii as isotropic circles around the marked keeper "
            "centre; real envelopes are wider laterally than vertically.",
            "Pose angles are measured in the image plane and therefore depend on the camera viewpoint.",
            "Single camera, with no lens distortion or rolling-shutter correction.",
        ]
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def _files_html(ctx: ReportContext) -> str:
    names = [RESULTS_CSV, SUMMARY_CSV, "data.toml"]
    if ctx.ball_rows:
        names += [BALL_PATH_PIXEL_CSV, BALL_PATH_3D_CSV]
    names += [os.path.basename(p) for p in ctx.pose_files.values() if p]
    names += [os.path.basename(p) for p in ctx.videos.values() if p]
    names += [os.path.basename(p) for p in ctx.snapshots.values() if p]
    seen: list[str] = []
    for n in names:
        if n and n not in seen:
            seen.append(n)
    return "".join(f"<li><code>{n}</code></li>" for n in seen)


def _anim_data(ctx: ReportContext) -> str:
    m = ctx.metrics
    goal = ctx.goal
    samples: list[list[float]] = []
    if ctx.flight_path is not None and len(ctx.flight_path):
        arr = np.asarray(ctx.flight_path, dtype=float)
        step = max(1, arr.shape[0] // 120)
        samples = [[round(float(v), 5) for v in arr[i, :4]] for i in range(0, arr.shape[0], step)]
        if samples and samples[-1][0] != round(float(arr[-1, 0]), 5):
            samples.append([round(float(v), 5) for v in arr[-1, :4]])

    def clean(value):
        if value is None:
            return None
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return round(f, 5) if math.isfinite(f) else None

    payload = {
        "flight_time": clean(m.get("flight_time_s")) or 0.0,
        "apex_z": clean(m.get("apex_height_m")) or goal.height,
        "samples": samples,
        "goal": {
            "width": goal.width,
            "height": goal.height,
            "penalty_distance": goal.penalty_distance,
            "ball_radius": goal.ball_radius,
        },
        "entry": [clean(m.get("ball_entry_x_m")), clean(m.get("ball_entry_z_m"))],
        "gk": {
            "start_x": clean(m.get("gk_start_x_m")),
            "start_z": clean(m.get("gk_start_z_m")),
            "end_x": clean(m.get("gk_end_x_m")),
            "end_z": clean(m.get("gk_end_z_m")),
            "reach_standing": clean(m.get("reach_standing_m")),
            "reach_dive": clean(m.get("reach_dive_m")),
            "reaction_s": clean(m.get("gk_response_time")) or 0.0,
        },
    }
    return json.dumps(payload)


def build_report_html(ctx: ReportContext, lang: str = "en") -> str:
    """Render the full self-contained report for one language."""
    m = ctx.metrics
    verdict_key = m.get("verdict_class", "unknown")
    replacements = {
        "{{LANG}}": "pt-BR" if lang == "pt" else "en",
        "{{TITLE}}": t(lang, "title"),
        "{{VIDEO_NAME}}": ctx.video_name,
        "{{DATE}}": ctx.generated_at,
        "{{FPS}}": f"{ctx.fps:.3f}",
        "{{L_VIDEO}}": t(lang, "video"),
        "{{L_DATE}}": t(lang, "date"),
        "{{L_FPS}}": t(lang, "fps"),
        "{{L_SUMMARY}}": t(lang, "summary"),
        "{{L_KICKER}}": t(lang, "kicker"),
        "{{L_GK}}": t(lang, "goalkeeper"),
        "{{L_REACH}}": t(lang, "reach"),
        "{{L_TRAJECTORY}}": t(lang, "trajectory"),
        "{{L_SNAPSHOTS}}": t(lang, "snapshots"),
        "{{L_METHODS}}": t(lang, "methods"),
        "{{L_FILES}}": t(lang, "files"),
        "{{L_FRONT}}": t(lang, "front_view"),
        "{{L_PLAN}}": t(lang, "plan_view"),
        "{{L_SIDE}}": t(lang, "side_view"),
        "{{L_PLAY}}": t(lang, "play"),
        "{{L_RESTART}}": t(lang, "restart"),
        "{{VERDICT_TONE}}": _VERDICT_TONE.get(verdict_key, "neutral"),
        "{{VERDICT_LABEL}}": _verdict_label(lang, verdict_key, m.get("verdict_label", "")),
        "{{SUMMARY_TEXT}}": _summary_text(ctx, lang),
        "{{KICKER_CARDS}}": _kicker_cards(ctx, lang),
        "{{GK_CARDS}}": _gk_cards(ctx, lang),
        "{{GOAL_SVG}}": build_goal_svg(ctx, lang),
        "{{ZONE_NOTE}}": t(lang, "zone_grid_note"),
        "{{REACH_TABLE}}": _reach_table(ctx, lang),
        "{{REACH_NOTE}}": ""
        if m.get("gk_height_m")
        else f'<div class="note">{t(lang, "no_anthro")}</div>',
        "{{ANIM_NOTE}}": t(lang, "anim_note"),
        "{{ANIM_DATA}}": _anim_data(ctx),
        "{{ANIM_LABELS}}": json.dumps(
            {
                "play": t(lang, "play"),
                "pause": t(lang, "pause"),
                "time": t(lang, "time_label"),
            }
        ),
        "{{BALL_TABLE_BLOCK}}": _ball_table_block(ctx, lang),
        "{{POSE_BLOCK}}": _pose_block(ctx, lang),
        "{{SNAPSHOTS}}": _snapshots_html(ctx, lang),
        "{{METHODS}}": _methods_html(ctx, lang),
        "{{FILES}}": _files_html(ctx),
        "{{FOOTER}}": t(lang, "footer"),
    }
    html = _HTML_TEMPLATE
    for key, value in replacements.items():
        html = html.replace(key, str(value))
    return html


def write_html_reports(out_dir: str, ctx: ReportContext, lang: str = "both") -> dict[str, str]:
    """Write the report in the requested language(s), returning the paths."""
    langs = ("en", "pt") if lang == "both" else (lang,)
    written: dict[str, str] = {}
    for code in langs:
        if code not in REPORT_HTML:
            continue
        path = os.path.join(out_dir, REPORT_HTML[code])
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(build_report_html(ctx, code))
        written[code] = path
    return written
