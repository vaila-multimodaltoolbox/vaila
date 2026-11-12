"""
Project: vailá
Script: freekiki_tv_dossier.py - TV dossier of a saved penalty: cards, talking points, technical dossier

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 09 November 2026
Update Date: 12 November 2026
Version: 0.4.17

Description:
    The editorial companion of the TV report of a saved penalty, in three blocks:

    1. cards for the big screen and the graphics (GC): the speed of the kick, the time to decide and
       the point of interception;
    2. a guide for the commentator: four plain statements;
    3. the technical dossier for production: the cameras (capture rate, frames counted, time and
       tolerance), why those counts are one interval, the speed from the spot and from the line, and
       why the wrong speeds (118 km/h, 80 km/h) are wrong.

    Every number comes from ``data/penalty_metrics.json`` of a run (the ``speed`` block of
    :func:`vaila.freekiki_penalty.speed_summary`) and from the camera counts given in a JSON file;
    nothing is typed by hand. The mean speed is the distance from the penalty mark to the hands over
    the flight time (the ball speed of Navarro et al. 2013); the exit speed is the flight model's and
    always appears as an estimate.

Usage:
    uv run vaila/freekiki.py tv-dossier --run <3D folder> [--cameras cameras.json]
        [--goalkeeper "Carlos Miguel"] [--kicker "Alan Lescano"] [--match "Vasco x Palmeiras"]

    cameras.json: {"cameras": [{"name": "...", "file_fps": 29.97, "capture_fps": 50.0,
                                 "frames": 19, "source": "declared"}, ...]}

License: AGPLv3
"""

from __future__ import annotations

import html as _html
import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

try:
    from .freekiki_keeper import apply_rear_to_metrics
    from .freekiki_timing import CameraSpan, fit_clock, reconcile
except ImportError:
    from freekiki_keeper import apply_rear_to_metrics  # ty: ignore[unresolved-import]
    from freekiki_timing import CameraSpan, fit_clock, reconcile  # ty: ignore[unresolved-import]

REACTION_MS = (180.0, 240.0)  # the visuomotor latency of the prompt: given, with no source cited
PROMPT_SCENARIO = {"distance_m": 10.2, "time_s": 0.350}  # the production's own counts (cam 3)
SIDE_PT = {"left": "esquerda", "right": "direita"}
SOURCE_PT = {"measured": "medida", "declared": "informada", "estimated": "estimada"}
NAVARRO = (
    "Navarro, M., van der Kamp, J., Ranvaud, R. & Savelsbergh, G. J. P. (2013). The mere presence of "
    "a goalkeeper affects the accuracy of penalty kicks. Journal of Sports Sciences, 31(9), 921–929. "
    "doi:10.1080/02640414.2012.762602"
)


def pt(value, nd: int = 1) -> str:
    """A number with the Portuguese decimal comma, rounded half up ("—" for a missing one).

    Half up from the number's own decimal text, so 1.035 reads 1,04 as in the other reports (a binary
    float would read 1,03).
    """
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    exact = Decimal(str(float(value))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return f"{exact:.{nd}f}".replace(".", ",")


def frames_text(frames: float) -> str:
    """A frame count: whole when it is whole, else one decimal (17.5 frames stay 17,5)."""
    return pt(frames, 0) if abs(frames - round(frames)) < 0.05 else pt(frames, 1)


def frame_check(video: Path | str) -> dict | None:
    """Whether a video holds repeated frames, from the change between consecutive frames.

    A file converted from another rate repeats frames (24 fps film shown at 29.97 fps repeats one in
    five). Returns the frame count, the rate and the minimum, median and maximum mean absolute
    change, with the number of frames that repeat the previous one (``repeated``); None when the
    video cannot be read.
    """
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    previous, steps = None, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if previous is not None:
            steps.append(float(np.mean(np.abs(gray - previous))))
        previous = gray
    cap.release()
    if not steps:
        return None
    median = float(np.median(steps))
    return {
        "frames": len(steps) + 1,
        "fps": fps,
        "diff_min": round(min(steps), 2),
        "diff_median": round(median, 2),
        "diff_max": round(max(steps), 2),
        "repeated": int(sum(1 for s in steps if s < 0.3 * median)),
    }


def _main_camera(speed: dict) -> dict:
    fps = float(speed["fps"])
    span = CameraSpan(
        "Câmera 1 (principal)", fps, fps, float(speed["flight_frames_counted"]), "measured"
    )
    return span.row()


def _grid(speed: dict) -> dict:
    """Mean speed for the distances and times in play: the production's scenario, the file's, the line."""
    d_file = float(speed["distance"]["spot_to_hands_m"])
    t_file = round(float(speed["flight_time_s"]), 3)
    t_count = round(float(speed["flight_time_counted_s"]), 3)
    distances = sorted(
        {PROMPT_SCENARIO["distance_m"], round(d_file, 3), float(speed["distance"]["nominal_m"])}
    )
    times = sorted({PROMPT_SCENARIO["time_s"], t_file, t_count})
    cells = [{"d": d, "t": t, "kmh": round(d / t * 3.6, 1)} for d in distances for t in times]
    kmh = [c["kmh"] for c in cells]
    return {
        "distances_m": distances,
        "times_s": times,
        "cells": cells,
        "range_kmh": [min(kmh), max(kmh)],
    }


def _in_front_range(metrics: dict) -> tuple[float, float] | None:
    """How far in front of the line the hands met the ball, over the refits along the camera ray."""
    goal_x = float(metrics["goal_x_m"])
    sign = math.copysign(1.0, goal_x)
    fronts = [float(metrics["interception"]["in_front_of_line_m"])]
    for row in (metrics.get("uncertainty") or {}).get("along_camera_ray", []):
        xyz = row.get("interception_xyz")
        if xyz:
            fronts.append(sign * (goal_x - float(xyz[0])))
    return (min(fronts), max(fronts)) if len(fronts) > 1 else None


def _cards(metrics: dict, speed: dict) -> list[dict]:
    mean, exit_model = speed["mean"], speed["exit_model"]
    sigma_t = (
        math.hypot(speed["sigma"]["kick_frames"], speed["sigma"]["save_frames"]) / speed["fps"]
    )
    front = _in_front_range(metrics)
    hit = metrics["interception"]
    rear = metrics.get("rear_override")
    if rear and rear.get("interception"):
        # the point was fixed by two cameras: the rear one gives the side, the main one the depth and height
        note_hit = (
            "duas câmeras: o lado vem da câmera de trás e a profundidade e a altura, da câmera principal "
            f"(±{pt(rear['interception'].get('sigma_m', 0.2), 1)} m)"
        )
    else:
        note_hit = (
            f"faixa de {pt(front[0])} a {pt(front[1])} m pela incerteza de profundidade de uma só câmera "
            f"(±{pt(speed['sigma']['distance_m'])} m no ponto de defesa)"
            if front
            else f"incerteza de profundidade de uma só câmera (±{pt(speed['sigma']['distance_m'])} m)"
        )
    return [
        {
            "id": "speed",
            "title": "A velocidade da batida",
            "headline": f"≈ {pt(mean['kmh'], 0)} km/h",
            "sub": f"{pt(mean['ms'])} m/s · velocidade média, da marca do pênalti às mãos do goleiro",
            "note": f"faixa de {pt(mean['range_kmh'][0], 0)} a {pt(mean['range_kmh'][1], 0)} km/h "
            f"(±{pt(mean['uncertainty_kmh'], 0)} km/h)",
            "aside": f"Saída do pé, estimada pelo modelo de voo: ≈ {pt(exit_model['launch_kmh'], 0)} km/h"
            + (
                f" (faixa {pt(exit_model['range_kmh'][0], 0)} a {pt(exit_model['range_kmh'][1], 0)})"
                if round(exit_model["range_kmh"][0]) != round(exit_model["range_kmh"][1])
                else ""
            )
            + ": estimativa, não medida.",
        },
        {
            "id": "decision",
            "title": "O tempo de decisão",
            "headline": f"{pt(speed['flight_time_s'], 2)} s",
            "sub": f"{pt(round(speed['flight_time_s'] * 100), 0)} centésimos de segundo, do contato do pé às mãos",
            "note": f"{speed['flight_frames_counted']} quadros a {pt(speed['fps'], 2)} quadros/s = "
            f"{pt(speed['flight_time_counted_s'], 3)} s contados; {pt(speed['flight_time_s'], 3)} s com o "
            f"contato entre dois quadros (±{pt(sigma_t, 3)} s)",
            "aside": "",
        },
        {
            "id": "interception",
            "title": "O ponto de interceptação",
            "headline": f"≈ {pt(hit['in_front_of_line_m'])} m à frente da linha",
            "sub": f"a {pt(hit['height_m'], 2)} m de altura e {pt(abs(hit['lateral_m']), 2)} m do centro do gol",
            "note": note_hit,
            "aside": "",
        },
    ]


def _points(metrics: dict, speed: dict, names: dict) -> list[dict]:
    gk = names.get("keeper") or "o goleiro"
    mean, exit_model = speed["mean"], speed["exit_model"]
    flight = speed["flight_time_s"]
    keeper = metrics.get("keeper") or {}
    hit = metrics["interception"]
    end_ms = (speed["at_save_model"] or {}).get("ms")
    front = float(hit["in_front_of_line_m"])
    gain = f"{pt(front / end_ms * 1000.0, 0)} ms" if end_ms else "—"
    scenario = ""
    if end_ms and front < 0.8:  # the production's eye estimate only matters when ours is shorter
        lo, hi = 0.8 / end_ms * 1000.0, 1.0 / end_ms * 1000.0
        scenario = (
            f" Se a defesa tivesse sido a 0,8–1,0 m da linha (estimativa visual da equipe), o ganho seria "
            f"de {pt(lo, 0)} a {pt(hi, 0)} ms."
        )
    onset = keeper.get("onset_ms_before_kick")
    move = keeper.get("movement_time_s")
    flight_ms = flight * 1000.0
    points = [
        {
            "title": "A bola parecia indefensável",
            "text": f"Da marca do pênalti às mãos de {gk}, a bola levou {pt(flight, 2)} s, "
            f"{pt(round(flight * 100), 0)} centésimos de segundo, a cerca de {pt(mean['kmh'], 0)} km/h em "
            f"média. É a velocidade média no trajeto, medida a {pt(speed['fps'], 2)} quadros por segundo. "
            f"A velocidade no instante do chute foi estimada pelo modelo em torno de "
            f"{pt(exit_model['launch_kmh'], 0)} km/h, mas isso é estimativa e não deve ir ao ar como número "
            "medido.",
        },
        {
            "title": "Atacar a bola à frente da linha",
            "text": f"{gk} tocou a bola cerca de {pt(front)} m à frente da linha. Ir ao encontro da bola "
            f"encurta o caminho dela: com a bola a {pt(end_ms, 0)} m/s no fim do voo, esses {pt(front)} m "
            f"valem cerca de {gain}, o equivalente a {pt(front / end_ms * 1000.0 / speed['frame_ms'], 1)} "
            f"quadros de transmissão ({pt(speed['frame_ms'], 0)} ms).{scenario}"
            if end_ms
            else f"{gk} tocou a bola cerca de {pt(front)} m à frente da linha.",
        },
    ]
    if onset is not None and move is not None:
        late = [flight_ms - REACTION_MS[1], flight_ms - REACTION_MS[0]]
        points.append(
            {
                "title": "Ele saiu antes do chute",
                "text": f"{gk} começou a se mover {pt(onset, 0)} ms antes do contato do pé na bola. O "
                f"movimento dele, do primeiro passo até a mão na bola, durou {pt(move, 2)} s "
                f"({pt(move * 1000.0, 0)} ms); a bola levou {pt(flight, 2)} s. Se esperasse o chute e "
                f"somasse uma latência de reação de {pt(REACTION_MS[0], 0)} a {pt(REACTION_MS[1], 0)} ms "
                f"(valor do prompt, sem fonte citada), teria só {pt(late[0], 0)} a {pt(late[1], 0)} ms para "
                f"um movimento que levou {pt(move * 1000.0, 0)} ms. A defesa só é viável porque ele partiu "
                "na leitura do movimento do batedor, antes do impacto.",
            }
        )
    anthro = metrics.get("anthropometrics") or {}
    if keeper.get("lateral_displacement_m") is not None:
        side = SIDE_PT.get(str(keeper.get("dive_side")), str(keeper.get("dive_side")))
        height = anthro.get("gk_height_m")
        span = anthro.get("gk_arm_span_m")
        body = (
            f"{pt(height, 2)} m de altura e envergadura estimada em {pt(span, 2)} m"
            if height
            else ""
        )
        if keeper.get("lateral_source") == "rear_camera":
            lo, hi = (keeper.get("lateral_displacement_range_m") or [None, None])[:2]
            span_txt = (
                f" (faixa de {pt(lo, 2)} a {pt(hi, 2)} m)"
                if lo is not None and hi is not None
                else ""
            )
            text = (
                f"O quadril se deslocou cerca de {pt(keeper['lateral_displacement_m'], 1)} m{span_txt} para a "
                f"{side}, do início do movimento até a defesa, medido na câmera de trás do gol"
                + (
                    f", e as mãos foram {pt(keeper['hands_lateral_m'], 1)} m para o lado, até a bola"
                    if keeper.get("hands_lateral_m") is not None
                    else ""
                )
                + (f", com {body}" if body else "")
                + ". Não dizer centímetros: os postes e o quadril foram marcados à mão e o tempo tem a "
                f"incerteza de um quadro ({pt(speed['frame_ms'], 0)} ms)."
            )
        else:
            text = (
                f"Deslocou o centro de massa {pt(keeper['lateral_displacement_m'], 2)} m para a "
                f"{side}, com pico de {pt(keeper.get('peak_lateral_speed_mps'), 1)} m/s"
                + (f", com {body}" if body else "")
                + f". Não dizer centímetros: a medida tem incerteza de ±{pt(speed['sigma']['distance_m'])} m "
                f"na profundidade e de um quadro ({pt(speed['frame_ms'], 0)} ms) no tempo."
            )
        points.append({"title": "O que ele fez com o corpo", "text": text})
    return points


def _reconciliation(rows: list[dict], rec: dict) -> list[str]:
    """The sentences that make the counts one interval."""
    replay = next((r for r in rows if r["slowdown"] > 1.05), None)
    out = []
    if replay is not None:
        out.append(
            "Contar quadros só vira tempo na taxa em que os quadros foram capturados. O arquivo de um replay "
            f"sempre diz {pt(replay['file_fps'], 2)} quadros/s, mas ele foi capturado a "
            f"{pt(replay['capture_fps'], 2)} quadros/s: os {frames_text(replay['frames'])} quadros de "
            f"{replay['name']} levam {pt(replay['shown_s'], 2)} s para tocar no arquivo, mas aconteceram em "
            f"{frames_text(replay['frames'])} ÷ {pt(replay['capture_fps'], 2)} = {pt(replay['dt_s'], 3)} s."
        )
    counts = "; ".join(f"{r['name']}: {pt(r['dt_s'], 3)} s (±{pt(r['tol_s'], 3)})" for r in rows)
    out.append(
        f"As contagens medem o mesmo intervalo: {counts}. A maior diferença entre os valores centrais é de "
        f"{pt(rec['span_s'] * 1000.0, 0)} ms."
    )
    if rec["common_interval_s"] is not None:
        lo, hi = rec["common_interval_s"]
        out.append(f"Todos os intervalos se sobrepõem entre {pt(lo, 3)} s e {pt(hi, 3)} s.")
    else:
        miss = [p for p in rec["pairs"] if not p["consistent"]]
        pairs = "; ".join(
            f"{p['a']} e {p['b']} (a {pt(p['gap_ms'], 1)} ms de se tocarem)" for p in miss
        )
        out.append(
            f"Os intervalos não são exatamente iguais: {pairs}. Isso cabe dentro da contagem de um quadro e meio "
            "e das fontes não resolvidas: o instante do evento muda com o ângulo de cada câmera e a taxa de "
            "captura das câmeras de replay foi informada, não medida no arquivo."
        )
    out.append(
        f"Média ponderada pela resolução de cada relógio: {pt(rec['weighted_mean_s'], 3)} s "
        f"(±{pt(rec['weighted_sigma_s'], 3)} s)."
    )
    return out


def _clock_text(name: str, fit: dict, file_fps: float) -> str:
    """The effective capture rate of a replay, from the events both cameras show."""
    check = (
        "com dois eventos o ajuste não tem resíduo para conferir"
        if fit["n"] == 2
        else f"com resíduo de {pt(fit['rms_ms'], 1)} ms (rms)"
    )
    return (
        f"A taxa efetiva de {name}, ajustada pelos {fit['n']} eventos que as duas câmeras mostram "
        f"(t = a · quadro + b), é de {pt(fit['rate_hz'], 1)} quadros/s, {pt(fit['rate_hz'] / file_fps, 2)} "
        f"vez a taxa do arquivo; {check}."
    )


def _why_not(speed: dict, evidence: dict | None) -> list[dict]:
    """The two speeds the production names as wrong, reproduced from this file's numbers."""
    fps = float(speed["fps"])
    kick = speed["kick"]
    moved = kick["first_moved_frame"]
    fraction = moved - float(kick["time_s"]) * fps  # the part of a frame the ball had flown by then
    ten = speed["save"]["frame"] - moved
    to_line = speed["distance"]["to_line_m"] or speed["distance"]["nominal_m"]
    raw = to_line / (ten / fps) * 3.6
    d_hands = speed["distance"]["spot_to_hands_m"]
    n = speed["flight_frames_counted"]
    at24 = d_hands / (n / 24.0) * 3.6
    proof = ""
    if evidence:
        proof = (
            f" Neste arquivo, {evidence['frames']} quadros mudam entre {pt(evidence['diff_min'])} e "
            f"{pt(evidence['diff_max'])} (mediana {pt(evidence['diff_median'])}) de um quadro para o "
            f"seguinte e {evidence['repeated']} repetem o anterior."
        )
    return [
        {
            "id": "118",
            "kmh": round(raw, 1),
            "title": f"Por que não ≈ {pt(raw, 0)} km/h",
            "text": f"Esse número sai de contar {ten} quadros, do primeiro quadro com a bola andando "
            f"({moved}) até a defesa ({speed['save']['frame']}), e de usar a distância até a linha "
            f"({pt(to_line, 2)} m): {pt(to_line, 2)} ÷ {pt(ten / fps, 3)} s. Está errado por dois motivos. No "
            f"quadro {moved} a bola já tinha voado {pt(fraction, 2)} quadro (o contato do pé foi em "
            f"{pt(kick['frame'], 2)}), então o tempo de voo é {pt(speed['flight_time_s'], 3)} s e não "
            f"{pt(ten / fps, 3)} s. E o goleiro defende à frente da linha, a {pt(d_hands, 2)} m da marca e não "
            f"a {pt(to_line, 2)} m.",
        },
        {
            "id": "24fps",
            "kmh": round(at24, 1),
            "title": f"Por que não ≈ {pt(at24, 0)} km/h",
            "text": f"Seria a velocidade de {n} quadros a 24 quadros/s: {pt(n / 24.0, 3)} s e "
            f"{pt(d_hands, 2)} m. A hipótese não se sustenta: o arquivo declara {pt(fps, 2)} quadros/s e uma "
            f"gravação de 24 quadros mostrada a {pt(fps, 2)} repete um quadro a cada cinco.{proof}",
        },
    ]


def dossier_content(
    metrics: dict,
    *,
    names: dict,
    match: str = "",
    cameras: list[dict] | None = None,
    evidence: dict | None = None,
) -> dict:
    """The three blocks of the dossier of a saved penalty, as data (see :func:`render_markdown`)."""
    if not metrics.get("available") or metrics.get("event") != "save":
        raise ValueError("the dossier needs the metrics of a saved penalty (event 'save')")
    speed = metrics.get("speed")
    if not speed:
        raise ValueError("penalty_metrics.json has no 'speed' block: run the penalty stage again")
    rows = [_main_camera(speed)]
    rec = None
    text: list[str] = []
    fits: dict[str, dict] = {}
    if cameras:
        spans, estimated = [], []
        for c in cameras:
            span = CameraSpan(
                c["name"],
                float(c["file_fps"]),
                float(c["capture_fps"]),
                float(c["frames"]),
                c.get("source", "declared"),
                float(c.get("frames_sigma", 1.0)),
            )
            # a rate estimated from the main camera's own clock cannot check that clock
            (estimated if span.source == "estimated" else spans).append(span)
            events = c.get("events") or []
            if len(events) >= 2:
                fits[c["name"]] = fit_clock(
                    [float(e["main_frame"]) / float(speed["fps"]) for e in events],
                    [float(e["frame"]) for e in events],
                )
        if len(spans) >= 2:
            rec = reconcile(spans)
            rows = rec["cameras"] + [sp.row() for sp in estimated]
            text = _reconciliation(rec["cameras"], rec)
        else:
            rows = [sp.row() for sp in spans + estimated]
        for name, fit in fits.items():
            text.append(_clock_text(name, fit, float(speed["fps"])))
        text += [str(c["note"]) for c in cameras if c.get("note")]
        for sp in estimated:
            text.append(
                f"A taxa de {sp.name} ({pt(sp.capture_fps, 2)} quadros/s) foi estimada pelos eventos da "
                "câmera principal: serve para colocar a câmera na linha do tempo, não como checagem "
                "independente do intervalo."
            )
    return {
        "title": f"Dossiê de TV: {names.get('keeper') or 'o goleiro'} × {names.get('kicker') or 'o batedor'}",
        "match": match,
        "cards": _cards(metrics, speed),
        "points": _points(metrics, speed, names),
        "technical": {
            "cameras": rows,
            "reconciliation": rec,
            "reconciliation_text": text,
            "clock_fits": fits,
            "grid": _grid(speed),
            "why_not": _why_not(speed, evidence),
            "definition": (
                "Velocidade média = distância da marca do pênalti até as mãos do goleiro ÷ tempo de voo "
                "(do contato do pé até a defesa): a velocidade da bola de Navarro et al. (2013), distância "
                "dividida pelo tempo de voo. O contato do pé cai entre dois quadros e é lido pelo ajuste de "
                "voo, conferido pelos pixels da bola."
            ),
            "speed": speed,
        },
        "references": [NAVARRO],
    }


def _md_vaila(text: str) -> str:
    """The project name in italics, lowercase, as in every vailá text."""
    return text.replace("vailá", "*vailá*")


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines


def _camera_rows(tech: dict) -> list[list[str]]:
    return [
        [
            r["name"],
            SOURCE_PT.get(r["source"], r["source"]),
            pt(r["capture_fps"], 2),
            frames_text(r["frames"]),
            f"{pt(r['dt_s'], 3)} s",
            f"±{pt(r['tol_s'] * 1000.0, 1)} ms",
        ]
        for r in tech["cameras"]
    ]


CAMERA_HEADERS = [
    "Câmera",
    "Origem da taxa",
    "Taxa de captura (quadros/s)",
    "Quadros contados (N)",
    "Tempo Δt = N ÷ taxa",
    "Incerteza (±1 quadro)",
]


def render_markdown(c: dict) -> str:
    """The dossier as Markdown."""
    out = [f"# {c['title']}"]
    if c["match"]:
        out.append(f"\n**{c['match']}**")
    out += ["", "## BLOCO 1: cartões para o telão e a computação gráfica (GC / Chyron)", ""]
    for i, card in enumerate(c["cards"], 1):
        out += [
            f"### Cartão {i}: {card['title']}",
            "",
            f"**{card['headline']}**",
            "",
            card["sub"],
            "",
        ]
        out += [f"_{card['note']}_", ""]
        if card["aside"]:
            out += [card["aside"], ""]
    out += ["## BLOCO 2: guia rápido para o comentarista", ""]
    for p in c["points"]:
        out += [f"- **{p['title']}.** {p['text']}"]
    tech = c["technical"]
    out += ["", "## BLOCO 3: dossiê técnico e metodológico (produção / editorial)", ""]
    out += _md_table(CAMERA_HEADERS, _camera_rows(tech)) + [""]
    out += tech["reconciliation_text"] + [""]
    out += ["### Definição da velocidade", "", tech["definition"], ""]
    grid = tech["grid"]
    out += ["### Sensibilidade da velocidade média (km/h)", ""]
    heads = ["Distância (m) \\ tempo (s)"] + [pt(t, 3) for t in grid["times_s"]]
    body = []
    for d in grid["distances_m"]:
        body.append(
            [pt(d, 3)]
            + [
                pt(next(x["kmh"] for x in grid["cells"] if x["d"] == d and x["t"] == t), 1)
                for t in grid["times_s"]
            ]
        )
    out += _md_table(heads, body)
    out += [
        "",
        f"Faixa: {pt(grid['range_kmh'][0], 0)} a {pt(grid['range_kmh'][1], 0)} km/h. "
        "A distância de 10,200 m e o tempo de 0,350 s são os da equipe de produção (câmera lenta).",
        "",
    ]
    for w in tech["why_not"]:
        out += [f"### {w['title']}", "", w["text"], ""]
    out += ["## Referências", ""] + [f"- {r}" for r in c["references"]]
    return _md_vaila("\n".join(out) + "\n")


CSS = """
:root{--ink:#14202e;--mut:#5b6b7b;--bg:#fff;--card:#f2f6fa;--acc:#0b5fa5;--est:#b45309}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--ink:#e8eef5;--mut:#9fb0c1;--bg:#0f1720;--card:#18232f;--acc:#6db3f2;--est:#f0a35e}}
body{font:16px/1.5 system-ui,Segoe UI,Roboto,sans-serif;color:var(--ink);background:var(--bg);margin:0 auto;max-width:980px;padding:20px 16px 48px}
h1{font-size:1.7rem;margin:.2em 0}h2{font-size:1.25rem;margin:1.6em 0 .6em;border-bottom:2px solid var(--acc);padding-bottom:.2em}
h3{font-size:1.05rem;margin:1.2em 0 .3em}.match{color:var(--mut);margin:0 0 1em}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.card{background:var(--card);border-radius:10px;padding:14px 16px}.card .t{color:var(--mut);font-size:.85rem;text-transform:uppercase;letter-spacing:.04em}
.card .h{font-size:2rem;font-weight:700;color:var(--acc);line-height:1.15;margin:.2em 0}.card .s{margin:.2em 0}.card .n{color:var(--mut);font-size:.9rem}
.card .a{color:var(--est);font-size:.9rem;margin-top:.5em}
table{border-collapse:collapse;width:100%;margin:.6em 0;font-size:.92rem}th,td{border:1px solid var(--mut);padding:6px 8px;text-align:left}
th{background:var(--card)}ul{padding-left:1.2em}li{margin:.5em 0}.ref{color:var(--mut);font-size:.88rem}
@media print{body{max-width:none}.card,tr{break-inside:avoid}h2,h3{break-after:avoid}}
"""


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{_html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_html.escape(v)}</td>" for v in r) + "</tr>" for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(c: dict) -> str:
    """The dossier as one self-contained HTML page."""
    esc = _html.escape
    tech = c["technical"]
    parts = [f"<h1>{esc(c['title'])}</h1>"]
    if c["match"]:
        parts.append(f'<p class="match">{esc(c["match"])}</p>')
    parts.append("<h2>BLOCO 1: cartões para o telão e a computação gráfica (GC / Chyron)</h2>")
    cards = []
    for i, card in enumerate(c["cards"], 1):
        aside = f'<div class="a">{esc(card["aside"])}</div>' if card["aside"] else ""
        cards.append(
            f'<div class="card"><div class="t">Cartão {i}: {esc(card["title"])}</div>'
            f'<div class="h">{esc(card["headline"])}</div><div class="s">{esc(card["sub"])}</div>'
            f'<div class="n">{esc(card["note"])}</div>{aside}</div>'
        )
    parts.append(f'<div class="cards">{"".join(cards)}</div>')
    parts.append("<h2>BLOCO 2: guia rápido para o comentarista</h2><ul>")
    parts += [f"<li><b>{esc(p['title'])}.</b> {esc(p['text'])}</li>" for p in c["points"]]
    parts.append("</ul>")
    parts.append("<h2>BLOCO 3: dossiê técnico e metodológico (produção / editorial)</h2>")
    parts.append(_html_table(CAMERA_HEADERS, _camera_rows(tech)))
    parts += [
        f"<p>{esc(t).replace('vailá', '<i>vailá</i>')}</p>" for t in tech["reconciliation_text"]
    ]
    parts += ["<h3>Definição da velocidade</h3>", f"<p>{esc(tech['definition'])}</p>"]
    grid = tech["grid"]
    heads = ["Distância (m) \\ tempo (s)"] + [pt(t, 3) for t in grid["times_s"]]
    body = [
        [pt(d, 3)]
        + [
            pt(next(x["kmh"] for x in grid["cells"] if x["d"] == d and x["t"] == t), 1)
            for t in grid["times_s"]
        ]
        for d in grid["distances_m"]
    ]
    parts += ["<h3>Sensibilidade da velocidade média (km/h)</h3>", _html_table(heads, body)]
    parts.append(
        f"<p>Faixa: {pt(grid['range_kmh'][0], 0)} a {pt(grid['range_kmh'][1], 0)} km/h. A distância de "
        "10,200 m e o tempo de 0,350 s são os da equipe de produção (câmera lenta).</p>"
    )
    for w in tech["why_not"]:
        parts += [f"<h3>{esc(w['title'])}</h3>", f"<p>{esc(w['text'])}</p>"]
    parts.append("<h2>Referências</h2>")
    parts += [f'<p class="ref">{esc(r)}</p>' for r in c["references"]]
    return (
        '<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(c['title'])}</title><style>{CSS}</style></head><body>"
        + "\n".join(parts)
        + "</body></html>"
    )


def build(
    run_dir: Path | str,
    *,
    names: dict,
    match: str = "",
    cameras_file: Path | str | None = None,
    video: Path | str | None = None,
    out_dir: Path | str | None = None,
    pdf: bool = True,
) -> dict:
    """Write ``dossie_tv.md``, ``.html`` and (with Chrome) ``.pdf`` in ``<run>/report_tv/pt/``."""
    run = Path(run_dir)
    metrics_file = run / "data" / "penalty_metrics.json"
    if not metrics_file.is_file():
        raise FileNotFoundError(
            f"{metrics_file}: run the penalty analysis first (penalty_metrics.json)"
        )
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    metrics = apply_rear_to_metrics(
        metrics, run / "data"
    )  # the rear camera's lateral travel, if measured
    cameras = None
    if cameras_file:
        cameras = json.loads(Path(cameras_file).read_text(encoding="utf-8"))["cameras"]
    evidence = frame_check(video) if video else None
    content = dossier_content(metrics, names=names, match=match, cameras=cameras, evidence=evidence)
    out = Path(out_dir) if out_dir else run / "report_tv" / "pt"
    out.mkdir(parents=True, exist_ok=True)
    md_path, html_path = out / "dossie_tv.md", out / "dossie_tv.html"
    md_path.write_text(render_markdown(content), encoding="utf-8")
    html_path.write_text(render_html(content), encoding="utf-8")
    pdf_path = None
    if pdf:
        try:
            from .freekiki_report import print_pdf
        except ImportError:
            from freekiki_report import print_pdf  # ty: ignore[unresolved-import]
        pdf_path = print_pdf(html_path, out / "dossie_tv.pdf")
    return {
        "md": str(md_path),
        "html": str(html_path),
        "pdf": str(pdf_path) if pdf_path else None,
        "content": content,
    }
