"""Original-frame dribble TV animation and auditable Portuguese report.

Version: 0.4.14
Created: 02 November 2026
Update Date: 02 November 2026

With manifest ``turns`` the 28–40 s chapter becomes one scene per change of direction (angle,
time, entry/exit speed, slow motion over the turn) and the report gains a "Mudanças de direção"
section with ``figuras/turns.png``.
"""

from __future__ import annotations

import html
import json
import shlex
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .freekiki_field3d import dlt_for_frames
from .freekiki_openplay_manifest import (
    analyze_originals,
    fingerprint_inputs,
    frames,
    load_manifest,
    prepare_camera,
    read_csv,
    rerun_body,
    transform,
    video_info,
    write_csv,
    write_json,
)
from .freekiki_openplay_report import make_figures
from .freekiki_report import print_pdf
from .monocular_dlt_align import dlt_projection_matrix

REFERENCES = [
    (
        "Wilson et al. (2019; publicado online em 2018). Dribbling speed along curved paths predicts attacking performance in match-realistic one vs. one soccer games.",
        "https://doi.org/10.1080/02640414.2018.1544110",
    ),
    (
        "Wilson et al. (2026; online em 2025). Dribbling and passing performances predict individual success in small-sided soccer games.",
        "https://journals.sagepub.com/doi/10.1177/17479541251346902",
    ),
    (
        "Santiago / Jornal da USP (2022). O futebol de um contra um é fonte para análises científicas.",
        "https://jornal.usp.br/radio-usp/o-futebol-de-um-para-um-e-fonte-para-analises-cientificas/",
    ),
    (
        "CBN Ribeirão (2018). Pesquisador encontra relação entre o reino animal e o futebol.",
        "https://cbnribeirao.com.br/pesquisador-encontra-relacao-entre-o-reino-animal-e-o-futebol/",
    ),
]

# Each caption is also exported as an editable, timed TV script and SRT.
SCENES = [
    (
        0,
        4,
        "O LANCE",
        "Um driblador, três defensores",
        "Um driblador e três defensores, vistos por duas câmeras.",
    ),
    (
        4,
        16,
        "REPLAY LENTO",
        "Corpo e bola em movimento",
        "O esqueleto é uma estimativa do SAM 3D Body. A bola só é marcada quando visível.",
    ),
    (
        16,
        28,
        "DISPUTA PELO ESPAÇO",
        "Os defensores encurtam os caminhos",
        "A distância ao driblador muda enquanto cada defensor ajusta sua trajetória.",
    ),
    (
        28,
        40,
        "VELOCIDADE E DIREÇÃO",
        "Mudar de rumo também muda o espaço",
        "Velocidade e distância são estimadas nos originais. Picos instáveis são omitidos.",
    ),
    (
        40,
        52,
        "PRESA–PREDADOR",
        "Uma analogia para entender o drible",
        "O driblador busca espaço; os defensores procuram fechar os caminhos. O mapa usa este lance.",
    ),
    (
        52,
        57,
        "DESFECHO DISPONÍVEL",
        "O recorte termina em disputa",
        "O trecho termina em disputa. Não é possível concluir o resultado da jogada.",
    ),
    (
        57,
        60,
        "CIÊNCIA E FUTEBOL",
        "Velocidade, curvas e habilidade",
        "Estudos: Wilson et al. (2019; 2026). Contexto: USP e CBN.",
    ),
]


TURN_REFERENCES = [
    (
        "Monteiro, Santiago et al. (2025). Kinematic demands of FIFA's EPTS validation circuit compared to a sport-specific alternative. Scientific Reports.",
        "https://doi.org/10.1038/s41598-025-31587-w",
    ),
    (
        "Marques, …, Santiago (2025). Wearable technology identifies differences in change of direction kinetics and kinematics in soccer players with a history of ACL reconstruction. KSSTA 34:402–414.",
        "https://doi.org/10.1002/ksa.12679",
    ),
    (
        "Agência FAPESP (2025). Tecnologia vestível pode ser aliada na recuperação de lesões no joelho em atletas.",
        "https://agencia.fapesp.br/tecnologia-vestivel-pode-ser-aliada-na-recuperacao-de-lesoes-no-joelho-em-atletas/55866",
    ),
]
ZONE_PT = {"very_low": "muito baixa", "low": "baixa", "moderate": "moderada", "high": "alta"}


def _br(value, digits=0):
    return f"{value:.{digits}f}".replace(".", ",")


def turn_summary(turn):
    """The numbers of one manifest turn for the card, the script and the report (None if missing).

    A path turn takes the mean over the 5/7/9 filters; a body turn the mean over the cameras,
    with the path entry/exit speeds next to it. Ranges keep the spread visible.
    """
    path = [r for r in turn["path"].values() if r is not None]
    body = [b for b in (turn.get("body") or {}).values() if b is not None]
    if turn["primary"] == "body":
        if not body:
            return None
        angles = [b["abs_deg"] for b in body]
        duration = float(np.mean([b["duration_s"] for b in body]))
        source = "giro do corpo (linha do quadril, SAM 3D Body), " + " e ".join(
            sorted((turn.get("body") or {}).keys())
        )
    else:
        if not path:
            return None
        angles = [r["abs_deg"] for r in path]
        duration = float(np.mean([r["duration_s"] for r in path]))
        source = "direção de deslocamento (trajetória no campo), filtros 5/7/9"
    angle = float(np.mean(angles))
    rate = angle / duration if duration > 0 else None
    v_in = float(np.mean([r["v_in_mps"] for r in path])) * 3.6 if path else None
    v_out = float(np.mean([r["v_out_mps"] for r in path])) * 3.6 if path else None
    mins = [r["v_min_mps"] * 3.6 for r in path if r["v_min_mps"] is not None]
    t0 = min(r["t_start_s"] for r in (body if turn["primary"] == "body" else path))
    t1 = max(r["t_end_s"] for r in (body if turn["primary"] == "body" else path))
    from .freekiki_openplay_kinematics import cod_zone

    return {
        "angle_deg": angle,
        "angle_range_deg": [min(angles), max(angles)],
        "duration_s": duration,
        "rate_deg_s": rate,
        "zone": ZONE_PT.get(cod_zone(rate) or "", "—"),
        "v_in_kmh": v_in,
        "v_out_kmh": v_out,
        "v_min_kmh_range": [min(mins), max(mins)] if mins else None,
        "t_start_s": t0,
        "t_end_s": t1,
        "source": source,
        "stable": bool(turn.get("stable")),
    }


def turn_text(summary):
    """Headline and detail line of a turn card, in Brazilian Portuguese."""
    if summary is None or not summary["stable"]:
        return "Ângulo: sem valor estável", ""
    head = f"≈{_br(summary['angle_deg'])}° em {_br(summary['duration_s'], 2)} s"
    parts = []
    if summary["v_in_kmh"] is not None:
        parts.append(
            f"entra a {_br(summary['v_in_kmh'])} km/h, sai a {_br(summary['v_out_kmh'])} km/h"
        )
    if summary["rate_deg_s"] is not None:
        parts.append(f"~{_br(summary['rate_deg_s'])} °/s")
    return head, " · ".join(parts)


def build_scenes(analysis):
    """The editorial scenes: the 28–40 s chapter becomes one scene per manifest turn (up to two)."""
    turns = list(analysis.get("turns") or [])[:2]
    if not turns:
        return list(SCENES)
    scenes = []
    for scene in SCENES:
        if scene[0] == 28:
            step = 12 / len(turns)
            for i, turn in enumerate(turns):
                s = turn_summary(turn)
                head, detail = turn_text(s)
                if s is not None and s["stable"] and turn["primary"] == "body":
                    caption = (
                        f"O corpo gira {head}, quase meia-volta, e sai conduzindo a "
                        f"{_br(s['v_out_kmh'])} km/h. Medido nas duas câmeras."
                    )
                elif s is not None and s["stable"]:
                    low = s["v_min_kmh_range"]
                    caption = (
                        f"A direção muda {head}: entra a {_br(s['v_in_kmh'])} km/h"
                        + (f", quase para ({_br(low[0])}–{_br(low[1])} km/h)" if low else "")
                        + f" e sai a {_br(s['v_out_kmh'])} km/h."
                    )
                else:
                    caption = (
                        "Sem valor estável entre filtros ou câmeras; o número fica fora da tela."
                    )
                start = 28 + i * step
                scenes.append(
                    (
                        round(start, 3),
                        round(start + step, 3),
                        f"MUDANÇA DE DIREÇÃO {i + 1}/{len(turns)}",
                        turn.get("name") or f"Mudança {i + 1}",
                        caption,
                        i,
                    )
                )
        elif scene[0] == 57:
            scenes.append(
                (
                    *scene[:4],
                    "Estudos: Wilson et al. (2019; 2026); Monteiro et al. (2025). Contexto: USP, CBN e FAPESP.",
                )
            )
        else:
            scenes.append(scene)
    return scenes


def turn_frames(turn):
    """First and last original pair shown in a turn scene."""
    spans = [turn["before"], turn["after"]] + (
        [turn["body_frames"]] if turn.get("body_frames") else []
    )
    return min(s[0] for s in spans), max(s[1] for s in spans)


def frame_at_time(seconds, fps, n_frames, start, end, natural=False):
    """Display sampling never enters measurement calculations."""
    if natural:
        return min(n_frames - 1, int((seconds - start) * float(fps)))
    return min(n_frames - 1, max(0, int((seconds - start) / (end - start) * n_frames)))


def _font(size, bold=False):
    suffix = "-Bold" if bold else ""
    choices = [
        Path(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf"),
        Path(f"C:/Windows/Fonts/{'arialbd' if bold else 'arial'}.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for path in choices:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def _text(draw, text, xy, size=28, color="#f0f5fb", bold=False, width=None):
    font = _font(size, bold)
    if width is None:
        draw.text(xy, text, font=font, fill=color)
        return
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) > width and line:
            lines.append(line)
            line = word
        else:
            line = trial
    lines.append(line)
    for i, value in enumerate(lines):
        draw.text((xy[0], xy[1] + i * (size + 9)), value, font=font, fill=color)


def stable_speed(analysis, frame):
    values = [p["dribbler"]["speed_mps"][frame] for p in analysis["filter_sensitivity"].values()]
    if any(v is None for v in values):
        return None
    if max(values) - min(values) > max(0.7, np.mean(values) * 0.25):
        return None
    return analysis["dribbler"]["speed_mps"][frame]


class DribbleRenderer:
    def __init__(self, manifest, analysis, poses, out):
        self.manifest, self.analysis, self.poses, self.out = manifest, analysis, poses, out
        self.n = analysis["n_frames"]
        self.cameras = manifest["cameras"]
        self.people = manifest["participants"]
        self.dribbler = next(p for p in self.people if p["role"] == "dribbler")
        self.main = next(i for i, c in enumerate(self.cameras) if c["main"])
        self.open = 1 - self.main
        self.images, self.dlts, self.balls = [], [], []
        for cam in self.cameras:
            start, end = cam["range_1based"]
            self.images.append(
                [
                    cv2.resize(im, (1280, 720))
                    for f, im in enumerate(frames(cam["video"]))
                    if start - 1 <= f < end
                ]
            )
            folder = out / "cameras" / cam["id"]
            self.dlts.append(dlt_for_frames(folder / "camera.dlt3d"))
            self.balls.append(
                {
                    int(r["frame"]): np.array([float(r["p0_x"]), float(r["p0_y"])])
                    for r in read_csv(folder / "ball_markers.csv")
                    if r["p0_x"] and r["p0_y"]
                }
            )
        # Use the same skeleton topology as the existing SAM3D renderer.
        from .sam3dinov3 import SKELETON_EDGE_NAMES

        self.edges = SKELETON_EDGE_NAMES
        self.paths = {self.dribbler["id"]: np.array(analysis["dribbler"]["xy_m"], dtype=float)}
        self.paths.update(
            {
                p["person_id"]: np.array(p["metrics"]["xy_m"], dtype=float)
                for p in analysis["markers"]
            }
        )
        valid = np.concatenate(list(self.paths.values()))
        valid = valid[np.isfinite(valid).all(axis=1)]
        if not len(valid):
            raise ValueError("no valid ground paths for the animation")
        self.centre = (valid.min(axis=0) + valid.max(axis=0)) / 2
        self.span = max(np.ptp(valid[:, 0]), np.ptp(valid[:, 1]), 6) + 3

    def view(self, ci, f, overlay=True):
        image = self.images[ci][f].copy()
        cam = self.cameras[ci]
        original = cam["range_1based"][0] - 1 + f
        sx, sy = 1280 / video_info_cached(cam)[0], 720 / video_info_cached(cam)[1]
        for p in self.people:
            kp = self.poses[ci].get((original, int(p["camera_ids"][cam["id"]])), {})
            if not kp:
                continue
            color = tuple(int(p["color"][j : j + 2], 16) for j in (5, 3, 1))
            for a, b in self.edges if overlay else ():
                if a in kp and b in kp:
                    axy, bxy = (np.rint(kp[k] * [sx, sy]).astype(int) for k in (a, b))
                    cv2.line(image, tuple(axy), tuple(bxy), color, 2, cv2.LINE_AA)
            anchor = kp.get("nose", next(iter(kp.values()))) * [sx, sy]
            x, y = np.rint(anchor + [0, -18]).astype(int)
            x, y = max(0, min(1140, x)), max(22, min(696, y))
            cv2.putText(
                image,
                p["name"],
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (10, 20, 35),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                image, p["name"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA
            )
        # Project trail using the CURRENT camera, so camera motion is respected.
        xyz = np.array(self.analysis["ball_raw_xyz_m"], dtype=float)
        if original in self.dlts[ci]:
            uv = transform(
                dlt_projection_matrix(self.dlts[ci][original]), xyz[max(0, f - 16) : f + 1]
            ) * [sx, sy]
            for a, b in zip(uv[:-1], uv[1:], strict=True):
                if np.isfinite([a, b]).all() and np.max(np.abs([a, b])) < 10000:
                    cv2.line(
                        image,
                        tuple(np.rint(a).astype(int)),
                        tuple(np.rint(b).astype(int)),
                        (100, 235, 255),
                        2,
                        cv2.LINE_AA,
                    )
        point = self.balls[ci].get(original)
        if point is not None:
            centre = tuple(np.rint(point * [sx, sy]).astype(int))
            cv2.circle(image, centre, 9, (50, 50, 250), 2, cv2.LINE_AA)
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def map(self, size, f, analogy=False, turn=None):
        image = Image.new("RGB", size, "#142e35")
        draw = ImageDraw.Draw(image)
        w, h = size
        side = min(w - 60, h - 110)
        origin = np.array([w / 2, (h + 35) / 2])

        def pixel(point):
            return tuple(
                origin + (np.array(point) - self.centre) * [side / self.span, -side / self.span]
            )

        for offset in range(-8, 9, 2):
            x = self.centre[0] + offset
            draw.line(
                [pixel([x, self.centre[1] - 12]), pixel([x, self.centre[1] + 12])],
                fill="#22424a",
                width=1,
            )
        _text(
            draw,
            "PRESA–PREDADOR" if analogy else "TRAJETÓRIAS NO CAMPO",
            (22, 16),
            24 if analogy else 18,
            bold=True,
        )
        for p in self.people:
            path = self.paths[p["id"]]
            for a, b in zip(path[:f], path[1 : f + 1], strict=True):
                if np.isfinite([a, b]).all():
                    draw.line([pixel(a), pixel(b)], fill=p["color"], width=4 if analogy else 2)
            if np.isfinite(path[f]).all():
                x, y = pixel(path[f])
                radius = 10 if analogy else 6
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=p["color"])
                if analogy:
                    _text(draw, p["name"], (x + 14, y - 15), 23, p["color"])
                if (
                    p["role"] != "dribbler"
                    and np.isfinite(self.paths[self.dribbler["id"]][f]).all()
                ):
                    draw.line(
                        [pixel(path[f]), pixel(self.paths[self.dribbler["id"]][f])],
                        fill="#50666b",
                        width=1,
                    )
        if turn is not None:
            # The two straight stretches the angle is measured between, on the dribbler's path.
            path = self.paths[self.dribbler["id"]]
            for (a, b), colour in ((turn["before"], "#ffffff"), (turn["after"], "#ffe25c")):
                if np.isfinite(path[[a, b]]).all():
                    p0, p1 = np.array(pixel(path[a])), np.array(pixel(path[b]))
                    draw.line([tuple(p0), tuple(p1)], fill=colour, width=5)
                    d = p1 - p0
                    if np.linalg.norm(d) > 1:
                        u = d / np.linalg.norm(d)
                        left = p1 - 14 * u + 8 * np.array([-u[1], u[0]])
                        right = p1 - 14 * u - 8 * np.array([-u[1], u[0]])
                        draw.polygon([tuple(p1), tuple(left), tuple(right)], fill=colour)
            brk = turn.get("break_frame")
            if brk is not None and np.isfinite(path[brk]).all():
                x, y = pixel(path[brk])
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline="#ff5d5d", width=3)
        # Explicit scale, and the same coordinates and time as the series CSV.
        draw.line([(28, h - 44), (28 + 2 * side / self.span, h - 44)], fill="#dfebef", width=3)
        _text(draw, "2 m · estimativa no plano", (28, h - 32), 16)
        return image

    def render(self, f, scene=None, elapsed=0):
        canvas = Image.new("RGB", (1920, 1080), "#0b1728")
        draw = ImageDraw.Draw(canvas)
        editorial = scene is not None
        chapter = scene[2] if editorial else "SAM 3D BODY · VELOCIDADE NATURAL"
        title = scene[3] if editorial else "O drible e a disputa pelo espaço"
        _text(draw, "LANCE DECIFRADO  /  FREEKIKI + VAILÁ", (55, 29), 20, "#8da9c2", bold=True)
        _text(draw, title, (55, 70), 43, bold=True)
        _text(draw, chapter, (1375, 45), 20, "#ffe25c", bold=True, width=485)
        _text(draw, f"Tempo do lance: {f / self.analysis['fps']:.2f} s", (1375, 108), 22)
        analogy = editorial and scene[0] == 40
        if analogy:
            canvas.paste(self.map((1280, 720), f, True), (55, 170))
            canvas.paste(self.view(self.main, f).resize((500, 281)), (1370, 170))
        else:
            canvas.paste(self.view(self.main, f, overlay=not editorial or scene[0] != 0), (55, 170))
            canvas.paste(
                self.view(self.open, f, overlay=not editorial or scene[0] != 0).resize((500, 281)),
                (1370, 170),
            )
        draw = ImageDraw.Draw(canvas)
        _text(
            draw,
            "CÂMERA PRINCIPAL" if analogy else "CÂMERA ABERTA · SINCRONIZADA",
            (1380, 463),
            17,
            "#8da9c2",
            bold=True,
        )
        turn = None
        if editorial and len(scene) > 5 and scene[5] is not None:
            turn = (self.analysis.get("turns") or [])[scene[5]]
        if not analogy:
            canvas.paste(self.map((500, 300), f, turn=turn), (1370, 507))
        else:
            _text(draw, "Trajetórias do próprio lance", (1370, 510), 26, bold=True)
            _text(
                draw,
                "Analogia didática de perseguição e criação de espaço. Sem inferir intenção ou tempo de reação neural.",
                (1370, 560),
                25,
                width=490,
            )
        draw = ImageDraw.Draw(canvas)
        # Keep numbers off uncertain filter-sensitive samples.
        if turn is not None:
            # Turn card: the measured angle and time, never a filter-sensitive peak.
            head, detail = turn_text(turn_summary(turn))
            _text(draw, head, (1370, 818), 30, "#ffe25c", bold=True)
            _text(draw, detail, (1370, 862), 19)
        else:
            speed = stable_speed(self.analysis, f)
            number = (
                f"{speed * 3.6:.0f} km/h estimados"
                if speed is not None
                else "Velocidade: sem valor estável"
            )
            _text(draw, number, (1370, 825), 23, "#ffe25c", bold=True)
            distances = []
            for row in self.analysis["markers"]:
                value = row["vs_dribbler"]["distance_m"][f]
                if value is not None:
                    distances.append(f"D{row['person_id']}: {value:.1f} m")
            _text(draw, "  ·  ".join(distances), (1370, 862), 21)
        draw.rectangle((55, 914, 1870, 1031), fill="#152b43")
        caption = (
            scene[4]
            if editorial
            else "SAM 3D Body: poses estimadas nos originais. Mesmas cores nas duas câmeras."
        )
        _text(draw, caption, (78, 933), 29, width=1755)
        for i, p in enumerate(self.people):
            _text(draw, p["name"], (60 + i * 310, 1044), 19, p["color"], bold=True)
        label = (
            "30 fps · SEM NARRAÇÃO" if editorial else f"{self.analysis['fps']:.3f} fps · ORIGINAIS"
        )
        _text(draw, label, (1440, 1044), 18, "#8da9c2")
        progress = elapsed / 60 if editorial else (f + 1) / self.n
        draw.rectangle((0, 1074, int(1920 * progress), 1079), fill="#ffe25c")
        return np.asarray(canvas)


def video_info_cached(cam):
    if "_dimensions" not in cam:
        _, w, h = video_info(cam["video"])
        cam["_dimensions"] = (w, h)
    return cam["_dimensions"]


def encode_frames(path, rate, pictures):
    """Shared ffmpeg export contract, with exact rational timestamps and H.264."""
    rate = Fraction(rate)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1920x1080",
        "-r",
        str(rate),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-enc_time_base",
        f"{rate.denominator}/{rate.numerator}",
        "-video_track_timescale",
        str(rate.numerator),
        "-movflags",
        "+faststart",
        str(path),
    ]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        assert proc.stdin is not None
        try:
            for picture in pictures:
                proc.stdin.write(np.ascontiguousarray(picture).tobytes())
        finally:
            proc.stdin.close()
        code = proc.wait()
        if code:
            raise RuntimeError(f"ffmpeg failed ({code}): {path}")


def write_videos(renderer, out):
    rate = Fraction(renderer.analysis["fps_rational"])
    encode_frames(out / "lance_sam3body.mp4", rate, (renderer.render(f) for f in range(renderer.n)))

    scenes = build_scenes(renderer.analysis)

    def tv_frames():
        for k in range(1800):
            seconds = k / 30
            scene = next(s for s in scenes if s[0] <= seconds < s[1])
            f = frame_at_time(
                seconds,
                rate,
                renderer.n,
                scene[0],
                scene[1],
                natural=scene[0] == 0 or scene[0] == 52,
            )
            if len(scene) > 5 and scene[5] is not None:
                # Slow motion over the turn window, then hold its last frame for the card.
                first, last = turn_frames(renderer.analysis["turns"][scene[5]])
                play = 0.8 * (scene[1] - scene[0])
                share = min(1.0, (seconds - scene[0]) / play)
                f = min(last, first + int(share * (last - first + 1)))
            if scene[0] == 57:
                f = renderer.n - 1
            if k % 300 == 0:
                print(f">> TV: {k}/1800 quadros", flush=True)
            yield renderer.render(f, scene, seconds)

    encode_frames(out / "explicacao_tv.mp4", Fraction(30), tv_frames())
    return ["lance_sam3body.mp4", "explicacao_tv.mp4"]


def _srt_time(seconds):
    ms = int(round(seconds * 1000))
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def write_script(out, scenes=None):
    rows, srt = [], []
    for i, (start, end, chapter, title, caption, *_) in enumerate(scenes or SCENES, 1):
        rows.append(
            {"start_s": start, "end_s": end, "chapter": chapter, "title": title, "caption": caption}
        )
        srt.append(f"{i}\n{_srt_time(start)} --> {_srt_time(end)}\n{title}\n{caption}\n")
    write_csv(out / "roteiro_legendas.csv", rows)
    (out / "explicacao_tv.srt").write_text("\n".join(srt), encoding="utf-8")
    (out / "roteiro_tv.md").write_text(
        "# Roteiro de legendas — sem narração\n\n"
        + "\n\n".join(
            f"**{r['start_s']:g}–{r['end_s']:g} s · {r['chapter']}**\n\n{r['title']}\n\n{r['caption']}"
            for r in rows
        ),
        encoding="utf-8",
    )


def turn_figure(analysis, path):
    """Top view of the dribbler's path with the stretches each turn is measured between."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drib = np.array(analysis["dribbler"]["xy_m"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot(drib[:, 0], drib[:, 1], color="#c9a800", lw=2, label="driblador")
    colours = ("#14588d", "#c0392b")
    for i, turn in enumerate(analysis.get("turns") or []):
        s = turn_summary(turn)
        c = colours[i % 2]
        for (a, b), ls in ((turn["before"], "-"), (turn["after"], "--")):
            if np.isfinite(drib[[a, b]]).all():
                ax.annotate(
                    "",
                    xy=drib[b],
                    xytext=drib[a],
                    arrowprops={"arrowstyle": "-|>", "color": c, "lw": 2.4, "ls": ls},
                )
        head, _ = turn_text(s)
        ax.plot([], [], color=c, lw=2.4, label=f"{i + 1}. {turn.get('name', '')}: {head}")
        brk = turn.get("break_frame")
        if brk is not None and np.isfinite(drib[brk]).all():
            ax.plot(
                *drib[brk],
                "o",
                mfc="none",
                mec="#c0392b",
                ms=11,
                mew=2,
                label=f"quebra (par {brk})",
            )
    ax.set_aspect("equal")
    ax.set_xlabel("x no campo (m)")
    ax.set_ylabel("y no campo (m)")
    ax.set_title("Mudanças de direção: trechos de entrada (—) e saída (- -)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def turns_section(analysis):
    """HTML section with each turn, its numbers, method and filter / camera spread."""
    turns = analysis.get("turns") or []
    if not turns:
        return ""
    rows = []
    for i, turn in enumerate(turns, 1):
        s = turn_summary(turn)
        if s is None:
            rows.append(
                f"<tr><td>{i}. {html.escape(turn.get('name', ''))}</td><td colspan='5'>sem medida</td></tr>"
            )
            continue
        lo, hi = s["angle_range_deg"]
        low = s["v_min_kmh_range"]
        speeds = (
            f"{_br(s['v_in_kmh'])} → {_br(s['v_out_kmh'])}"
            + (
                f" (mínima {_br(low[0])}–{_br(low[1])})"
                if low and turn["primary"] == "path"
                else ""
            )
            if s["v_in_kmh"] is not None
            else "—"
        )
        rows.append(
            f"<tr><td>{i}. {html.escape(turn.get('name', ''))}<br><small>{html.escape(s['source'])}</small></td>"
            f"<td><b>≈{_br(s['angle_deg'])}°</b><br><small>{_br(lo)}–{_br(hi)}°</small></td>"
            f"<td>{_br(s['t_start_s'], 2)}–{_br(s['t_end_s'], 2)} s<br><small>{_br(s['duration_s'], 2)} s</small></td>"
            f"<td>{_br(s['rate_deg_s'])} °/s<br><small>zona {s['zone']}</small></td>"
            f"<td>{speeds}</td><td>{'sim' if s['stable'] else 'não'}</td></tr>"
        )
    side = []
    for turn in turns:
        if turn["primary"] == "body":
            path = [r for r in turn["path"].values() if r is not None]
            if path:
                side.append(
                    f"<li>{html.escape(turn.get('name', ''))}: o deslocamento, quase parado antes do giro, "
                    f"sai {_br(float(np.mean([r['abs_deg'] for r in path])))}° da direção em que vinha; "
                    "o número da meia-volta é o giro do corpo, não o do caminho.</li>"
                )
    return f"""<h2>Mudanças de direção</h2>
<p>Cada mudança foi escolhida no vídeo e medida entre um trecho de entrada e um de saída (cordas da trajetória, que fazem a média do ruído de cada passo). O ângulo é a diferença entre as duas direções; o tempo vai do fim da entrada ao início da saída; a velocidade angular média é o ângulo dividido pelo tempo. A meia-volta inicial é medida pelo giro do corpo: a linha do quadril do SAM 3D Body, em cada câmera, no referencial da câmera.</p>
<table><tr><th>Mudança</th><th>Ângulo</th><th>Quando / duração</th><th>Velocidade angular</th><th>Velocidade (km/h) entrada → saída</th><th>Estável</th></tr>{"".join(rows)}</table>
<ul>{"".join(side)}</ul>
<figure><img src='figuras/turns.png'><figcaption>Trajetória do driblador com os trechos de entrada (linha cheia) e saída (tracejada) de cada mudança; círculo: quadro da quebra marcado no vídeo.</figcaption></figure>
<p class='note'>Zonas de velocidade angular (&lt;30, 30–60, 60–90 e ≥90 °/s) e de velocidade (&lt;6, 6–15, 15–20, 20–25 e ≥25 km/h) de Monteiro, Santiago et al. (2025), medidas com GPS a 10 Hz filtrado a 1 Hz. Aqui a fonte é vídeo a {analysis["fps"]:.3f} fps; a zona é uma leitura qualitativa, não uma comparação numérica. O corte de 90° de Marques et al. (2025) é o mesmo tipo de gesto, estudado com sensores vestíveis em atletas com reconstrução do LCA; o paralelo é só do gesto, sem nenhuma conclusão sobre lesão ou carga. Estável: ângulo com variação ≤15° entre filtros (trajetória) ou ≤20° entre câmeras (corpo), e velocidades dentro da regra dos vídeos. Esquerda/direita não são afirmadas.</p>"""


def build_html(manifest, analysis, out, videos):
    figs = make_figures(analysis, out / "figuras", "pt")
    if analysis.get("turns"):
        turn_figure(analysis, out / "figuras" / "turns.png")
    n, fps = analysis["n_frames"], analysis["fps"]
    rows = []
    for p in manifest["participants"]:
        rows.append(
            f"<tr><td>{html.escape(p['name'])}</td>"
            + "".join(f"<td>{p['camera_ids'][c['id']]}</td>" for c in manifest["cameras"])
            + "</tr>"
        )
    sensitivities = analysis["filter_sensitivity"]
    sensitivity = []
    for window, a in sensitivities.items():
        m = a["dribbler"]

        def num(v):
            return "ausente" if v is None else f"{v:.1f}"

        sensitivity.append(
            f"<tr><td>{window} quadros</td><td>{num(m['v_max_mps'] * 3.6 if m['v_max_mps'] is not None else None)}</td><td>{num(m['a_peak_mps2'])}</td><td>{num(m['decel_peak_mps2'])}</td></tr>"
        )
    closest = []
    for p in analysis["markers"]:
        a = p["vs_dribbler"]
        if a["min_distance_m"] is not None:
            closest.append(
                f"<li>{html.escape(p['name'])}: aproximação mínima estimada de {a['min_distance_m']:.1f} m, em {a['min_t_s']:.2f} s.</li>"
            )
    checks = []
    for cam in manifest["cameras"]:
        records = read_csv(out / "cameras" / cam["id"] / "original_frames.csv")
        values = [float(r["field_rmse_px"]) for r in records if r["field_rmse_px"]]
        checks.append(
            f"<li>{cam['id']}: {len(records)} originais, RMSE médio dos pontos de campo {np.mean(values):.1f} px (máximo {max(values):.1f} px).</li>"
        )
    links = "".join(
        f"<figure><video controls preload='metadata' src='{v}'></video><figcaption><a href='{v}'>{v}</a></figcaption></figure>"
        for v in videos
    )
    captions = {
        "top": "Trajetórias estimadas no plano do campo.",
        "speed": "Velocidades estimadas, janela de 7 quadros; veja a sensibilidade dos filtros abaixo.",
        "dist": "Distâncias estimadas ao driblador, incluindo bola e defensores.",
    }
    images = "".join(
        f"<figure><img src='figuras/{p.name}'><figcaption>{captions.get(label, label)}</figcaption></figure>"
        for label, p in figs.items()
    )
    qa_images = "".join(
        f"<figure><img src='{p.name}'><figcaption>Conferência visual: sincronismo, identidades, poses e bola nos originais.</figcaption></figure>"
        for p in sorted(out.glob("conferencia_*.jpg"))
    )
    references = "".join(
        f"<li><a href='{url}'>{html.escape(title)}</a></li>"
        for title, url in REFERENCES + (TURN_REFERENCES if analysis.get("turns") else [])
    )
    markup = f"""<!doctype html><html lang='pt-BR'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>O drible e a disputa pelo espaço</title><style>
@page{{size:A4;margin:15mm}}*{{box-sizing:border-box}}body{{font:16px/1.6 system-ui,sans-serif;color:#14213d;margin:0;background:#eef2f7}}main{{max-width:1060px;margin:auto;padding:42px;background:white}}h1{{font-size:42px;line-height:1.12}}h2{{margin-top:32px}}.eyebrow{{color:#31577c;letter-spacing:.12em;font-weight:bold}}.lead{{font-size:21px;padding:20px;background:#edf4f9;border-left:5px solid #14213d}}.note{{padding:16px;background:#fff6df}}table{{width:100%;border-collapse:collapse}}td,th{{border-bottom:1px solid #d7e0eb;text-align:left;padding:10px}}video,img{{width:100%;height:auto}}figure{{margin:25px 0;break-inside:avoid}}figcaption{{font-size:13px;color:#53657a}}a{{color:#14588d}}code{{overflow-wrap:anywhere}}@media print{{body{{background:white;font-size:11px}}main{{padding:0}}h1{{font-size:28px}}video{{display:none}}figure img{{max-height:155mm;object-fit:contain}}}}</style>
<main><div class='eyebrow'>LANCE DECIFRADO · FREEKIKI / VAILÁ · 02 NOVEMBRO 2026</div><h1>O drible e a disputa pelo espaço</h1>
<p class='lead'>O driblador busca criar espaço enquanto três defensores ajustam seus caminhos. A analogia presa–predador ajuda a explicar essa disputa; o recorte não estabelece o sucesso do drible.</p>
<p>{n} pares originais · {fps:.6f} fps ({analysis["fps_rational"]}) · intervalo observado {(n - 1) / fps:.3f} s · vídeo natural {n / fps:.3f} s.</p>
<h2>Os vídeos</h2>{links}<p>Legendas já desenhadas nos vídeos; também disponíveis em <a href='explicacao_tv.srt'>SRT</a>, <a href='roteiro_tv.md'>roteiro</a> e <a href='roteiro_legendas.csv'>CSV</a>. Sem voz.</p>
<h2>Identidades entre câmeras</h2><table><tr><th>Participante</th><th>Aberta</th><th>Principal</th></tr>{"".join(rows)}</table>
<p>Aberta 4–81 corresponde à principal 1–78, sem esticar o tempo. IDs 41 e 35 são a mesma pessoa. IDs globais 0, 1, 2 e 3 identificam driblador e defensores 1, 2 e 3.</p>
{turns_section(analysis)}
<h2>Medidas e momentos de aproximação</h2><ul>{"".join(closest)}</ul>{images}
<h2>Estabilidade das derivadas</h2><p>Posições suavizadas por trechos válidos, Savitzky–Golay de ordem 2; lacunas permanecem vazias. Janela principal: 7 quadros. Derivadas usam o relógio original. A tabela mostra a sensibilidade dos picos, sem tratá-los como valores precisos.</p>
<table><tr><th>Janela</th><th>Pico velocidade (km/h)</th><th>Pico aceleração (m/s²)</th><th>Pico desaceleração (m/s²)</th></tr>{"".join(sensitivity)}</table>
<p>Nos vídeos, a velocidade é arredondada e omitida quando a amplitude entre filtros excede 25% da média ou 0,7 m/s (o maior). Mudanças de direção usam um limiar operacional de 25°, não um teste de habilidade.</p>
<h2>Qualidade e limites da reconstrução</h2><ul>{"".join(checks)}<li>Bola binocular aceita em {analysis["quality"]["ball_accepted_pairs"]}/{n} pares, sem impor altura zero. Gate: reprojeção ≤10 px nas duas câmeras, ângulo entre raios ≥1° e altura entre −0,1 e 4 m. Pontos rejeitados permanecem no CSV bruto.</li></ul>
<p class='note'>Os DLTs fornecidos foram transferidos para as imagens originais por registro robusto e conferência dos pontos por fluxo óptico de ida e volta. A escala e a geometria vertical continuam herdadas dessas calibrações; reprojeção pequena não prova exatidão métrica. A altura da bola apresentou valores próximos ou abaixo do plano, incompatíveis com uma leitura precisa da altura do centro da bola. Sua série vertical é um diagnóstico da reconstrução, não uma altura validada. A posição de cada jogador é a média dos tornozelos estimados pelo SAM 3D Body, projetada no plano do campo da câmera principal. Em oclusões e quedas isso pode desviar da posição real. As caixas/identidades vieram dos rastros SAM existentes; o SAM 3D Body foi reexecutado nos originais, sem máscaras. Não houve nova segmentação SAM 3.</p>
{qa_images}<p>Quadros RIFE fornecem apenas referências visuais para localizar sementes. As novas medidas de pixels usam os originais. O vídeo editorial repete quadros observados para câmera lenta, sem gerar observações intermediárias.</p>
<h2>Leitura científica</h2><p>O estudo de um contra um relaciona velocidade de drible em trajetórias curvas ao desempenho de ataque. No estudo de jogos reduzidos, testes de drible e passe se associaram ao saldo de gols em jogos 3 × 3 de jovens atletas. Essas relações de grupo contextualizam o lance; não validam sozinhas a reconstrução nem permitem atribuir causalidade a esta jogada.</p>
{"<p>As mudanças de direção seguem a leitura de Monteiro et al. (2025): a mudança de direção como velocidade angular, cruzada com a velocidade de corrida. Marques et al. (2025) estudaram o corte de 90° com sensores vestíveis; a reportagem da Agência FAPESP resume esse estudo.</p>" if analysis.get("turns") else ""}
<p>A coluna da USP apresenta o um contra um como oportunidade para estudar movimentos. A reportagem da CBN contextualiza a comparação entre dribles e fuga de presas. Aqui, a analogia descreve caminhos e distâncias; não mede intenção, reação neural ou força muscular.</p><ol>{references}</ol>
<h2>Auditoria e reprodução</h2><p><a href='manifest.json'>Manifesto</a> · <a href='inputs_sha256.json'>Entradas e hashes</a> · <a href='data/dribble_analysis.json'>Análise JSON</a> · <a href='data/raw_tracks.csv'>Posições brutas</a> · <a href='data/ball_reprojection.csv'>Bola e reprojeção</a> · <a href='data/ground_observations.csv'>Observações por câmera</a> · séries CSV para janelas 5, 7 e 9 em <code>data/</code>.</p><p>Comandos e critérios em <a href='README_reproducao.md'>README_reproducao.md</a>. Imagens dos vídeos fornecidas pelo usuário. Poses: SAM 3D Body / Meta. Ferramentas: FreeKiki / vailá.</p></main></html>"""
    path = out / "relatorio_drible.html"
    path.write_text(markup, encoding="utf-8")
    pdf = print_pdf(path, out / "relatorio_drible.pdf")
    return {"html": str(path), "pdf": str(pdf) if pdf else None}


def build_manifest_report(manifest_path, out, *, animation=False, prepare=False, sam=False):
    manifest = load_manifest(manifest_path)
    out = Path(out).resolve()
    # All derived products belong to a new results folder, never the received inputs.
    inputs = fingerprint_inputs(manifest)
    if any(
        Path(r["path"]).parent == out
        for r in inputs
        if r["path"] != str(Path(manifest_path).resolve())
    ):
        raise ValueError("output must be a separate results directory")
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "manifest.json",
        {k: v for k, v in manifest.items() if k not in ("manifest_path", "n_frames")},
    )
    write_json(out / "inputs_sha256.json", inputs)
    for cam in manifest["cameras"]:
        folder = out / "cameras" / cam["id"]
        if prepare:
            prepare_camera(cam, manifest["participants"], folder)
        if sam:
            rerun_body(cam, folder)
    analysis, poses = analyze_originals(manifest, out)
    renderer = DribbleRenderer(manifest, analysis, poses, out)
    for label, f in (
        ("inicio", 0),
        ("disputa", manifest["n_frames"] // 2),
        ("fim", manifest["n_frames"] - 1),
    ):
        Image.fromarray(renderer.render(f)).save(out / f"conferencia_{label}.jpg", quality=92)
    write_script(out, build_scenes(analysis))
    videos = (
        write_videos(renderer, out)
        if animation
        else [
            p.name
            for p in out.glob("*.mp4")
            if p.name in ("lance_sam3body.mp4", "explicacao_tv.mp4")
        ]
    )
    paths = build_html(manifest, analysis, out, videos)
    command = [
        sys.executable,
        "-m",
        "vaila.freekiki",
        "openplay-report",
        "--manifest",
        str(out / "manifest.json"),
        "--output",
        str(out),
        "--kind",
        "dribble",
        "--lang",
        "pt",
        "--animation",
        "--prepare-originals",
        "--rerun-sam",
    ]
    readme = (
        "# Drible para TV — reprodução\n\n"
        + shlex.join(command)
        + "\n\nPara reaproveitar as poses e calibrações derivadas, omita --prepare-originals e --rerun-sam. Os arquivos recebidos são somente lidos. Cada nova análise completa deve usar outra pasta de saída.\n\n"
        + json.dumps(analysis["quality"], ensure_ascii=False, indent=2)
    )
    (out / "README_reproducao.md").write_text(readme, encoding="utf-8")
    write_json(
        out / "report_summary.json",
        {"analysis": analysis["quality"], "report": paths, "videos": videos, "commands": [command]},
    )
    return {"dir": str(out), "kind": "dribble", "languages": {"pt": paths}}
