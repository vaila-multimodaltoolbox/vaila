"""
================================================================================
Script: freekiki_pass_tv.py - TV delivery package for a FreeKiki pass (+ the shot after it)
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.14
Created: 02 November 2026
Update Date: 02 November 2026

Description:
    Builds a TV folder for a pass analysed by :mod:`freekiki_pass` (``data/pass_analysis.json``),
    with the same deliverables as the dribble TV package (:mod:`freekiki_openplay_video`):

    * ``explicacao_tv.mp4`` — 60 s, 30 fps, 1920x1080, chapters and burnt-in captions, no voice;
    * ``lance_sam3body.mp4`` — the pass at its original rate with the SAM 3D Body skeletons of
      the passer and the receiver, the ball, a live top view and live speeds / distances;
    * ``relatorio_passe.html/.pdf`` (pt) and ``relatorio_passe_en.html/.pdf``, ``figuras/``;
    * ``roteiro_tv.md``, ``roteiro_legendas.csv``, ``explicacao_tv.srt``;
    * ``conferencia_*.jpg`` (visual checks), ``data/`` (analysis, series, tracks, C3D, DLT,
      shot files), ``inputs_sha256.json``, ``outputs_sha256.json``, ``validacao_entrega.json``,
      ``report_summary.json`` and ``README_reproducao.md``.

    The slow-motion chapters repeat observed frames; no frame is interpolated. Every number on
    screen comes from ``pass_analysis.json`` (speeds of the smoothed tracks, t = 0 at release).
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
from datetime import date
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    from .freekiki_openplay_kinematics import load_ball_xy, smooth_track
    from .freekiki_openplay_video import _srt_time, _text, encode_frames, frame_at_time
    from .freekiki_report import print_pdf
    from .freekiki_report_i18n import fmt_num
except ImportError:
    from freekiki_openplay_kinematics import (  # ty: ignore[unresolved-import]
        load_ball_xy,
        smooth_track,
    )
    from freekiki_openplay_video import (  # ty: ignore[unresolved-import]
        _srt_time,
        _text,
        encode_frames,
        frame_at_time,
    )
    from freekiki_report import print_pdf  # ty: ignore[unresolved-import]
    from freekiki_report_i18n import fmt_num  # ty: ignore[unresolved-import]

W, H = 1920, 1080
MAIN = (1280, 720)
SIDE = (500, 281)
BG = "#0b1728"
PASSER_C = "#7fc8ff"
RECV_C = "#5ef08a"
BALL_C = "#ff5a5f"
STAR_C = "#ffb347"
YELLOW = "#ffe25c"
MUTED = "#8da9c2"
TV_SECONDS = 60
TV_RATE = 30

REFERENCES = [
    (
        "Wilson et al. (2026; online em 2025). Dribbling and passing performances predict "
        "individual success in small-sided soccer games.",
        "https://journals.sagepub.com/doi/10.1177/17479541251346902",
    ),
    (
        "Wilson et al. (2019; online em 2018). Dribbling speed along curved paths predicts "
        "attacking performance in match-realistic one vs. one soccer games.",
        "https://doi.org/10.1080/02640414.2018.1544110",
    ),
    (
        "Santiago / Jornal da USP (2022). O futebol de um contra um é fonte para análises "
        "científicas.",
        "https://jornal.usp.br/radio-usp/o-futebol-de-um-para-um-e-fonte-para-analises-cientificas/",
    ),
]


# ---------------------------------------------------------------------------------- helpers


def _n(x, nd: int = 1) -> str:
    return "—" if x is None else fmt_num(float(x), nd, "pt")


def _kmh(mps) -> str:
    return "—" if mps is None else fmt_num(float(mps) * 3.6, 0, "pt")


def _rgb(hex_color: str) -> tuple[int, int, int]:
    return (int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16))


def _bgr(hex_color: str) -> tuple[int, int, int]:
    r, g, b = _rgb(hex_color)
    return (b, g, r)


def _readme_video(folder: Path) -> Path:
    readme = folder / "README.txt"
    for line in readme.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("video:"):
            return Path(line.split(":", 1)[1].strip())
    raise FileNotFoundError(f"no 'video:' line in {readme}")


def _video_rate(video: Path) -> Fraction:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return Fraction(out.stdout.strip().split(",")[0])


def _read_frames(video: Path, size: tuple[int, int]) -> tuple[list[np.ndarray], tuple[int, int]]:
    cap = cv2.VideoCapture(str(video))
    frames, dims = [], (0, 0)
    try:
        while True:
            ok, im = cap.read()
            if not ok:
                break
            dims = (im.shape[1], im.shape[0])
            frames.append(im)
    finally:
        cap.release()
    if not frames:
        raise FileNotFoundError(f"cannot read frames from {video}")
    return frames, dims


def _keypoints_file(run_dir: Path, stem: str) -> Path | None:
    """SAM 3D Body 2D keypoints of a FreeKiki run (``work/<stem>/sam3dinov3/...``)."""
    roots = [run_dir.parent / "work" / stem, run_dir.parent / "work"]
    for root in roots:
        hits = sorted(root.glob(f"**/{stem}_sam3dinov3_keypoints2d.csv"))
        if hits:
            return hits[-1]
    return None


def _load_poses(path: Path | None, ids: set[int]) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    out: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    if path is None or not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                pid = int(r["person_id"])
                if pid not in ids:
                    continue
                p = np.array([float(r["x_px"]), float(r["y_px"])])
            except (KeyError, ValueError):
                continue
            if np.isfinite(p).all():
                out.setdefault((int(r["frame"]), pid), {})[r["kpt_name"]] = p
    return out


def _ball_pixels(path: Path) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                if r.get("score") not in (None, "") and float(r["score"]) < 0.5:
                    continue
                p = np.array([float(r["cx"]), float(r["cy"])])
            except (KeyError, ValueError):
                continue
            if np.isfinite(p).all():
                out[int(r["frame"])] = p
    return out


def _kicker_sam_id(shot_folder: Path) -> int | None:
    path = shot_folder / "player_map.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("role") == "kicker":
                try:
                    return int(r["sam_id"])
                except (KeyError, ValueError):
                    return None
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _probe(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(out.stdout)
    stream = (data.get("streams") or [{}])[0]
    return {**stream, "duration": (data.get("format") or {}).get("duration")}


# ---------------------------------------------------------------------------------- 2nd camera

SYNC_OK_M = 0.3  # median ball residual of an accepted ball-clock sync
ID_MATCH_M = 1.0  # median ground distance of an accepted player match
ID_MARGIN = 0.75  # the best match must be at most this fraction of the runner-up


def sync_by_ball(ref_xy: np.ndarray, other_xy: np.ndarray) -> dict:
    """Map each frame of a second camera to a fractional frame of the reference, by the ball.

    Both tracks are the ball on the pitch plane (metres). The ball's position along its path is
    the clock: each ``other`` frame gets the reference time at which the reference ball was at
    the same along-track position (monotone interpolation), which handles a replay whose speed
    ramps. ``residual_m`` is the cross-check: reference ball at that time minus the other ball.
    Frames outside the reference path stay NaN.
    """
    ref = np.asarray(ref_xy, dtype=float)
    oth = np.asarray(other_xy, dtype=float)
    fr = np.flatnonzero(np.isfinite(ref).all(axis=1))
    if fr.size < 3:
        raise ValueError("the reference ball has fewer than three frames")
    direction = ref[fr[-1]] - ref[fr[0]]
    norm = float(np.linalg.norm(direction))
    if norm < 1.0:
        raise ValueError("the reference ball moves less than 1 m: no ball clock")
    u, origin = direction / norm, ref[fr[0]]
    s_ref = np.maximum.accumulate((ref[fr] - origin) @ u)
    s_oth = (oth - origin) @ u
    t_ref = np.full(len(oth), np.nan)
    residual = np.full(len(oth), np.nan)
    for j in np.flatnonzero(np.isfinite(oth).all(axis=1)):
        if s_ref[0] <= s_oth[j] <= s_ref[-1]:
            t = float(np.interp(s_oth[j], s_ref, fr))
            t_ref[j] = t
            p = np.array([np.interp(t, fr, ref[fr, 0]), np.interp(t, fr, ref[fr, 1])])
            residual[j] = float(np.linalg.norm(p - oth[j]))
    ok = np.isfinite(residual)
    steps = np.diff(t_ref)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    return {
        "t_ref": t_ref,
        "residual_m": residual,
        "mapped": int(ok.sum()),
        "frames": len(oth),
        "median_m": float(np.median(residual[ok])) if ok.any() else None,
        "max_m": float(np.max(residual[ok])) if ok.any() else None,
        "slowmo_factor": float(np.median(1.0 / steps)) if steps.size else None,
        "slowmo_range": [float(np.min(1.0 / steps)), float(np.max(1.0 / steps))]
        if steps.size
        else None,
    }


def match_ids(
    ref_tracks: dict[int, np.ndarray],
    other_tracks: dict[int, dict[int, tuple[float, float]]],
    t_ref: np.ndarray,
    *,
    max_m: float = ID_MATCH_M,
    margin: float = ID_MARGIN,
    min_frames: int = 8,
) -> dict[int, dict]:
    """For each reference id, the other camera's track closest on the pitch at the synced times.

    Accepted when its median distance is below ``max_m`` and at most ``margin`` times the
    runner-up's (two players side by side are not told apart by position alone).
    """
    out: dict[int, dict] = {}
    for pid, xy in ref_tracks.items():
        fr = np.flatnonzero(np.isfinite(xy).all(axis=1))
        cands = []
        for qid, track in other_tracks.items():
            d = []
            for j, (x, y) in track.items():
                if 0 <= j < len(t_ref) and np.isfinite(t_ref[j]) and fr[0] <= t_ref[j] <= fr[-1]:
                    p = np.array(
                        [np.interp(t_ref[j], fr, xy[fr, 0]), np.interp(t_ref[j], fr, xy[fr, 1])]
                    )
                    d.append(float(np.hypot(p[0] - x, p[1] - y)))
            if len(d) >= min_frames:
                cands.append((float(np.median(d)), int(qid), len(d)))
        cands.sort()
        if not cands:
            out[pid] = {"other_id": None, "accepted": False, "reason": "no overlapping track"}
            continue
        best = cands[0]
        second = cands[1][0] if len(cands) > 1 else None
        accepted = best[0] < max_m and (second is None or best[0] <= margin * second)
        out[pid] = {
            "other_id": best[1],
            "median_m": best[0],
            "frames": best[2],
            "runner_up_id": cands[1][1] if len(cands) > 1 else None,
            "runner_up_m": second,
            "accepted": bool(accepted),
            "reason": "ok" if accepted else "too far or too close to the runner-up",
        }
    return out


def _ground_tracks(folder: Path) -> dict[int, dict[int, tuple[float, float]]]:
    out: dict[int, dict[int, tuple[float, float]]] = {}
    path = folder / "ground_tracks.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                out.setdefault(int(r["person_id"]), {})[int(r["frame"])] = (
                    float(r["x_m"]),
                    float(r["y_m"]),
                )
            except (KeyError, ValueError):
                continue
    return out


def _crop_box(
    pts: np.ndarray, dims: tuple[int, int], *, margin: int, min_w: int
) -> tuple[int, int, int, int]:
    """16:9 box ``(x0, y0, x1, y1)`` around ``pts`` (pixels), inside the image."""
    w_img, h_img = dims
    if pts.size == 0:
        return 0, 0, w_img, h_img
    lo = pts.min(axis=0) - margin
    hi = pts.max(axis=0) + margin
    w = max(float(hi[0] - lo[0]), (hi[1] - lo[1]) * 16 / 9, float(min_w))
    w = min(w, float(w_img), h_img * 16 / 9)
    h = w * 9 / 16
    cx, cy = (lo + hi) / 2
    x0 = int(np.clip(cx - w / 2, 0, w_img - w))
    y0 = int(np.clip(cy - h / 2, 0, h_img - h))
    return x0, y0, x0 + int(w), y0 + int(h)


# ---------------------------------------------------------------------------------- scenes


def build_scenes(a: dict, *, second: bool = False) -> list[tuple[int, int, str, str, str, str]]:
    """``(start_s, end_s, chapter, title, caption, mode)``; captions use the analysis numbers."""
    t = a["timing"]
    run = t.get("receiver_run") or {}
    ball = t.get("ball") or {}
    shot = a.get("shot") or {}
    pid, rid = a["passer_id"], a["receiver_id"]
    if run.get("lead_lower_bound_s") is not None:
        run_text = (
            f"Quando a bola saiu, o {rid} já corria a {_kmh(run.get('v_first_mps'))} km/h: "
            f"a corrida começou pelo menos {_n(run['lead_lower_bound_s'])} s antes (estimativa)."
        )
    elif run.get("onset_t_s") is not None:
        when = "antes" if run["onset_t_s"] < 0 else "depois"
        run_text = (
            f"O {rid} começou a correr {_n(abs(run['onset_t_s']), 2)} s {when} da saída da bola."
        )
    else:
        run_text = (
            f"O deslocamento do {rid} durante o passe foi de {_n(run.get('displacement_m'))} m."
        )
    scenes = [
        (
            0,
            5,
            "O LANCE",
            f"O passe do {pid} para o {rid}",
            f"Um passe de {_n(t.get('passer_to_arrival_m'))} m e, em seguida, a finalização. "
            + (
                "Câmera principal em velocidade real; ao lado, o replay lento da câmera 16 mm."
                if second
                else "Velocidade real."
            ),
            "natural",
        ),
        (
            5,
            17,
            "REPLAY LENTO",
            "A bola sai do pé do passador",
            f"A bola sai a cerca de {_kmh(ball.get('v_release_mps'))} km/h. O esqueleto é uma "
            "estimativa do SAM 3D Body.",
            "replay",
        ),
        (
            17,
            29,
            "A CORRIDA DO RECEBEDOR",
            "Ele correu antes da bola sair",
            run_text,
            "receiver",
        ),
        (
            29,
            41,
            "ESPAÇO E TEMPO",
            f"{_n(t.get('passer_to_arrival_m'))} metros em {_n(t.get('flight_s'), 2)} s",
            f"Do passador ao 1º toque: {_n(t.get('passer_to_arrival_m'))} m. A bola chega a "
            f"{_kmh(ball.get('v_arrival_mps'))} km/h; média de {_kmh(ball.get('v_mean_mps'))} km/h.",
            "map",
        ),
    ]
    if shot.get("shot_xy"):
        scenes += [
            (
                41,
                52,
                "A FINALIZAÇÃO",
                f"O {rid} finaliza",
                f"Chute a {_n(shot.get('distance_to_goal_m'))} m da linha do gol; a bola sai do pé "
                f"a cerca de {_n(shot.get('speed_headline_kmh'), 0)} km/h (estimativa). Outra câmera.",
                "shot",
            ),
            (
                52,
                57,
                "PASSE E CORRIDA",
                "A bola encontra o recebedor no espaço",
                f"O {rid} se deslocou {_n(run.get('displacement_m'))} m enquanto a bola viajava. "
                "O tempo entre o passe e o chute não foi medido.",
                "map_full",
            ),
        ]
    else:
        scenes += [
            (
                41,
                57,
                "PASSE E CORRIDA",
                "A bola encontra o recebedor no espaço",
                f"O {rid} se deslocou {_n(run.get('displacement_m'))} m enquanto a bola viajava.",
                "map_full",
            )
        ]
    scenes.append(
        (
            57,
            60,
            "CIÊNCIA E FUTEBOL",
            "Velocidade, precisão e tempo",
            "Estudo: Wilson et al. (2026), drible e passe em jogos reduzidos. Contexto: USP.",
            "end",
        )
    )
    return scenes


# ---------------------------------------------------------------------------------- renderer


class PassTV:
    """Draws every picture of the package from the pass analysis and the run folders."""

    def __init__(
        self,
        analysis: dict,
        pass_folder: Path,
        shot_folder: Path | None,
        second_folder: Path | None = None,
    ) -> None:
        self.a = analysis
        self.t = analysis["timing"]
        self.pid, self.rid = int(analysis["passer_id"]), int(analysis["receiver_id"])
        self.fps = float(analysis["fps"])
        self.n = int(analysis["n_frames"])
        self.pass_folder = pass_folder
        self.video = _readme_video(pass_folder)
        self.rate = _video_rate(self.video)
        self.frames, self.dims = _read_frames(self.video, MAIN)
        self.kp_file = _keypoints_file(pass_folder, self.video.stem)
        self.poses = _load_poses(self.kp_file, {self.pid, self.rid})
        self.ball_px = _ball_pixels(pass_folder / "ball.csv")
        self.paths = {
            "ball": self._xy(analysis["ball"].get("xy_m")),
            self.pid: self._xy(analysis["passer"].get("xy_m")),
            self.rid: self._xy(analysis["receiver"].get("xy_m")),
        }
        self.series = self.t["series"]
        self.arrival = self.t.get("arrival_frame")
        self.shot = analysis.get("shot") or {}
        self.shot_frames: list[np.ndarray] = []
        self.shot_poses: dict = {}
        self.shot_ball: dict[int, np.ndarray] = {}
        self.shot_track = np.zeros((0, 2))
        self.kicker = None
        if shot_folder is not None and self.shot.get("shot_xy"):
            shot_video = _readme_video(shot_folder)
            self.shot_video = shot_video
            self.shot_frames, self.shot_dims = _read_frames(shot_video, MAIN)
            self.kicker = _kicker_sam_id(shot_folder)
            self.shot_kp_file = _keypoints_file(shot_folder, shot_video.stem)
            if self.kicker is not None:
                self.shot_poses = _load_poses(self.shot_kp_file, {self.kicker})
            self.shot_ball = _ball_pixels(shot_folder / "ball.csv")
            self.shot_track = self._ball_field(shot_folder / "ball_3d_on_field.csv")
        self.second: dict | None = None
        if second_folder is not None:
            self.second = self._load_second(Path(second_folder))
        try:
            from .sam3dinov3 import SKELETON_EDGE_NAMES
        except ImportError:
            from sam3dinov3 import SKELETON_EDGE_NAMES  # ty: ignore[unresolved-import]

        self.edges = SKELETON_EDGE_NAMES
        pts = [p[np.isfinite(p).all(axis=1)] for p in self.paths.values()]
        if len(self.shot_track):
            pts.append(self.shot_track[np.isfinite(self.shot_track).all(axis=1)])
        allp = np.vstack(pts)
        self.lo, self.hi = allp.min(axis=0) - 2.5, allp.max(axis=0) + 2.5
        goal = self.shot.get("goal_x_m")
        self.goal_x: float = (
            float(goal) if goal is not None else (-52.5 if float(np.mean(allp[:, 0])) < 0 else 52.5)
        )

    def _load_second(self, folder: Path) -> dict:
        """Second camera: ball clock against this camera, matched ids, frames, poses, ball."""
        ref_ball = smooth_track(load_ball_xy(self.pass_folder, source="ground"), self.fps)
        other_ball = smooth_track(load_ball_xy(folder, source="ground"), self.fps)
        sync = sync_by_ball(ref_ball, other_ball)
        ids = match_ids(
            {self.pid: self.paths[self.pid], self.rid: self.paths[self.rid]},
            _ground_tracks(folder),
            sync["t_ref"],
        )
        video = _readme_video(folder)
        frames, dims = _read_frames(video, MAIN)
        kp_file = _keypoints_file(folder, video.stem)
        wanted = {m["other_id"] for m in ids.values() if m.get("accepted")}
        poses = _load_poses(kp_file, {int(w) for w in wanted if w is not None})
        ball_px = _ball_pixels(folder / "ball.csv")
        # one fixed crop for the whole clip: the passer, the receiver and the ball, 16:9
        mapped = {int(j) for j in np.flatnonzero(np.isfinite(sync["t_ref"]))}
        pts = [p for (j, _), kp in poses.items() if j in mapped for p in kp.values()]
        pts += [p for j, p in ball_px.items() if j in mapped]
        crop = _crop_box(np.asarray(pts, dtype=float), dims, margin=90, min_w=720)
        return {
            "folder": folder,
            "video": video,
            "frames": frames,
            "dims": dims,
            "kp_file": kp_file,
            "poses": poses,
            "ball_px": ball_px,
            "crop": crop,
            "sync": sync,
            "ids": ids,
            "sync_ok": sync["median_m"] is not None and sync["median_m"] <= SYNC_OK_M,
        }

    def second_frame(self, tc: float) -> int | None:
        """The second-camera frame whose synced reference time is nearest to ``tc``."""
        if not self.second or not self.second["sync_ok"]:
            return None
        t = self.second["sync"]["t_ref"]
        ok = np.flatnonzero(np.isfinite(t))
        if not ok.size:
            return None
        return int(ok[np.argmin(np.abs(t[ok] - tc))])

    def second_view(self, tc: float, *, tight: bool = False) -> Image.Image | None:
        """Second camera at reference time ``tc``, cropped to the play (``tight``: receiver)."""
        j = self.second_frame(tc)
        if j is None or self.second is None:
            return None
        sec = self.second
        image = sec["frames"][min(j, len(sec["frames"]) - 1)].copy()
        rm = sec["ids"].get(self.rid) or {}
        for pid, color in ((self.pid, PASSER_C), (self.rid, RECV_C)):
            m = sec["ids"].get(pid) or {}
            if not m.get("accepted"):
                continue
            kp = sec["poses"].get((j, int(m["other_id"])))
            if kp:
                self._skeleton(image, kp, color, str(pid), 1.0, 1.0)
        if j in sec["ball_px"]:
            c = tuple(np.rint(sec["ball_px"][j]).astype(int))
            cv2.circle(image, c, 12, _bgr(BALL_C), 2, cv2.LINE_AA)
        x0, y0, x1, y1 = sec["crop"]
        if tight:
            pts = []
            kp = sec["poses"].get((j, int(rm["other_id"]))) if rm.get("accepted") else None
            if kp:
                pts += list(kp.values())
            if j in sec["ball_px"]:
                pts.append(sec["ball_px"][j])
            if pts:
                x0, y0, x1, y1 = _crop_box(np.asarray(pts), sec["dims"], margin=60, min_w=400)
        crop = cv2.resize(image[y0:y1, x0:x1], SIDE)
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _xy(rows) -> np.ndarray:
        if not rows:
            return np.zeros((0, 2))
        return np.asarray(
            [[np.nan, np.nan] if r is None or r[0] is None else [r[0], r[1]] for r in rows],
            dtype=float,
        )

    @staticmethod
    def _ball_field(path: Path) -> np.ndarray:
        rows = []
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    try:
                        rows.append([float(r["x_m"]), float(r["y_m"])])
                    except (KeyError, ValueError):
                        rows.append([np.nan, np.nan])
        return np.asarray(rows, dtype=float).reshape(-1, 2)

    def t_rel(self, f: int) -> float:
        return float(self.t["t_rel_s"][min(f, self.n - 1)])

    def value(self, key: str, f: int) -> float | None:
        s = self.series[key]
        for k in (f, f + 1):
            if 0 <= k < len(s) and s[k] is not None:
                return float(s[k])
        return None

    # --- camera pictures ---
    def _skeleton(self, image, kp: dict, color: str, label: str, sx: float, sy: float) -> None:
        bgr = _bgr(color)
        for a, b in self.edges:
            if a in kp and b in kp:
                p, q = (tuple(np.rint(kp[k] * [sx, sy]).astype(int)) for k in (a, b))
                cv2.line(image, p, q, bgr, 2, cv2.LINE_AA)
        anchor = kp.get("nose", next(iter(kp.values()))) * [sx, sy]
        x, y = np.rint(anchor + [-20, -22]).astype(int)
        x, y = max(0, min(MAIN[0] - 160, x)), max(22, min(MAIN[1] - 4, y))
        cv2.putText(
            image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 20, 35), 4, cv2.LINE_AA
        )
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 1, cv2.LINE_AA)

    def view(self, f: int, overlay: bool = True) -> Image.Image:
        f = max(0, min(f, len(self.frames) - 1))
        image = cv2.resize(self.frames[f], MAIN)
        sx, sy = MAIN[0] / self.dims[0], MAIN[1] / self.dims[1]
        if overlay:
            for pid, color, name in (
                (self.pid, PASSER_C, f"{self.pid} passador"),
                (self.rid, RECV_C, f"{self.rid} recebedor"),
            ):
                kp = self.poses.get((f, pid))
                if kp:
                    self._skeleton(image, kp, color, name, sx, sy)
            trail = [self.ball_px[k] for k in range(max(0, f - 10), f + 1) if k in self.ball_px]
            for p, q in zip(trail[:-1], trail[1:], strict=False):
                cv2.line(
                    image,
                    tuple(np.rint(p * [sx, sy]).astype(int)),
                    tuple(np.rint(q * [sx, sy]).astype(int)),
                    (100, 235, 255),
                    2,
                    cv2.LINE_AA,
                )
        if f in self.ball_px:
            c = tuple(np.rint(self.ball_px[f] * [sx, sy]).astype(int))
            cv2.circle(image, c, 10, _bgr(BALL_C), 2, cv2.LINE_AA)
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def zoom(self, f: int) -> Image.Image:
        """The original frame cropped around the receiver and the ball (no resampling of time)."""
        f = max(0, min(f, len(self.frames) - 1))
        image = self.frames[f].copy()
        pts = []
        kp = self.poses.get((f, self.rid))
        if kp:
            pts += list(kp.values())
        if f in self.ball_px:
            pts.append(self.ball_px[f])
        for pid, color in ((self.rid, RECV_C), (self.pid, PASSER_C)):
            k = self.poses.get((f, pid))
            if k:
                self._skeleton(image, k, color, str(pid), 1.0, 1.0)
        if f in self.ball_px:
            cv2.circle(
                image, tuple(np.rint(self.ball_px[f]).astype(int)), 12, _bgr(BALL_C), 2, cv2.LINE_AA
            )
        centre = np.mean(pts, axis=0) if pts else np.array(self.dims) / 2
        cw, ch = 560, 315
        x0 = int(np.clip(centre[0] - cw / 2, 0, self.dims[0] - cw))
        y0 = int(np.clip(centre[1] - ch / 2, 0, self.dims[1] - ch))
        crop = cv2.resize(image[y0 : y0 + ch, x0 : x0 + cw], SIDE)
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    def shot_view(self, k: int) -> Image.Image:
        k = max(0, min(k, len(self.shot_frames) - 1))
        image = cv2.resize(self.shot_frames[k], MAIN)
        sx, sy = MAIN[0] / self.shot_dims[0], MAIN[1] / self.shot_dims[1]
        if self.kicker is not None:
            kp = self.shot_poses.get((k, self.kicker))
            if kp:
                self._skeleton(image, kp, RECV_C, f"{self.rid} finaliza", sx, sy)
        trail = [self.shot_ball[j] for j in range(0, k + 1) if j in self.shot_ball]
        for p, q in zip(trail[:-1], trail[1:], strict=False):
            cv2.line(
                image,
                tuple(np.rint(p * [sx, sy]).astype(int)),
                tuple(np.rint(q * [sx, sy]).astype(int)),
                (100, 235, 255),
                2,
                cv2.LINE_AA,
            )
        if k in self.shot_ball:
            c = tuple(np.rint(self.shot_ball[k] * [sx, sy]).astype(int))
            cv2.circle(image, c, 10, _bgr(BALL_C), 2, cv2.LINE_AA)
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # --- top view ---
    def map(
        self, size: tuple[int, int], f: int, *, title: str, big: bool = False, shot_k: int = -1
    ) -> Image.Image:
        image = Image.new("RGB", size, "#142e35")
        draw = ImageDraw.Draw(image)
        w, h = size
        top = 60 if big else 46
        span = self.hi - self.lo
        scale = min((w - 40) / span[0], (h - top - 50) / span[1])
        ox = (w - span[0] * scale) / 2
        oy = top + (h - top - 50 - span[1] * scale) / 2

        def px(p) -> tuple[float, float]:
            # pitch y up on screen
            return (ox + (p[0] - self.lo[0]) * scale, oy + (self.hi[1] - p[1]) * scale)

        # goal line, penalty and goal areas of the attacked goal
        s = 1.0 if self.goal_x > 0 else -1.0
        line = "#3d6a70"
        draw.line([px([self.goal_x, -34]), px([self.goal_x, 34])], fill=line, width=2)
        for depth, half in ((16.5, 20.16), (5.5, 9.16)):
            x2 = self.goal_x - s * depth
            draw.line(
                [
                    px([self.goal_x, -half]),
                    px([x2, -half]),
                    px([x2, half]),
                    px([self.goal_x, half]),
                ],
                fill=line,
                width=2,
            )
        _text(draw, title, (20, 14), 24 if big else 17, bold=True)
        lw = 5 if big else 3
        dot = 11 if big else 7
        for key, color in (("ball", BALL_C), (self.pid, PASSER_C), (self.rid, RECV_C)):
            path = self.paths[key]
            last = min(f, len(path) - 1)
            for p, q in zip(path[:last], path[1 : last + 1], strict=True):
                if np.isfinite([p, q]).all():
                    draw.line([px(p), px(q)], fill=color, width=lw)
            if last >= 0 and np.isfinite(path[last]).all():
                x, y = px(path[last])
                r = dot if key != "ball" else dot - 2
                draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
                if big and key != "ball":
                    _text(draw, str(key), (x + 14, y - 16), 24, color, bold=True)
        p0, arr = self.t.get("passer_at_release_xy"), self.t.get("arrival_xy")
        if p0 and arr and self.arrival is not None and f >= self.arrival:
            a, b = px(p0), px(arr)
            n_dash = 24
            for i in range(0, n_dash, 2):
                u, v = i / n_dash, (i + 1) / n_dash
                draw.line(
                    [
                        (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u),
                        (a[0] + (b[0] - a[0]) * v, a[1] + (b[1] - a[1]) * v),
                    ],
                    fill="#f0f5fb",
                    width=2,
                )
            x, y = b
            draw.regular_polygon((x, y, 13 if big else 9), 5, fill=STAR_C)
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            _text(
                draw,
                f"{_n(self.t.get('passer_to_arrival_m'))} m",
                (mid[0] + 12, mid[1] - 14),
                28 if big else 19,
                YELLOW,
                bold=True,
            )
        if shot_k >= 0 and len(self.shot_track):
            sp = self.shot_track
            last = min(shot_k, len(sp) - 1)
            for p, q in zip(sp[:last], sp[1 : last + 1], strict=True):
                if np.isfinite([p, q]).all():
                    draw.line([px(p), px(q)], fill=BALL_C, width=lw)
            if np.isfinite(sp[0]).all():
                x, y = px(sp[0])
                draw.regular_polygon((x, y, 11 if big else 8), 3, fill=BALL_C)
        bar = 5.0 * scale
        draw.line([(20, h - 34), (20 + bar, h - 34)], fill="#dfebef", width=3)
        _text(draw, "5 m · plano do campo", (20, h - 26), 15)
        return image

    # --- frame composer ---
    def render(
        self,
        f: int,
        scene: tuple | None = None,
        elapsed: float = 0.0,
        shot_k: int = -1,
        tc: float | None = None,
    ) -> np.ndarray:
        canvas = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(canvas)
        mode = scene[5] if scene else "natural_full"
        chapter = scene[2] if scene else "SAM 3D BODY · VELOCIDADE NATURAL"
        title = scene[3] if scene else f"O passe do {self.pid} para o {self.rid}"
        _text(draw, "LANCE DECIFRADO  /  FREEKIKI + VAILÁ", (55, 29), 20, MUTED, bold=True)
        _text(draw, title, (55, 70), 43, bold=True)
        _text(draw, chapter, (1375, 45), 20, YELLOW, bold=True, width=485)
        if mode == "shot":
            t_shot = shot_k / float(self.shot.get("fps") or self.fps)
            _text(draw, f"Depois do contato no chute: {_n(t_shot, 2)} s", (1375, 108), 22)
        else:
            _text(draw, f"Desde a saída da bola: {_n(self.t_rel(f), 2)} s", (1375, 108), 22)

        if mode in ("map", "map_full"):
            canvas.paste(
                self.map(
                    MAIN,
                    f,
                    title="VISTA DE CIMA · ESTE LANCE",
                    big=True,
                    shot_k=shot_k if mode == "map_full" else -1,
                ),
                (55, 170),
            )
            canvas.paste(self.view(f).resize(SIDE), (1370, 170))
            side_label = "CÂMERA PRINCIPAL"
        elif mode == "shot":
            canvas.paste(self.shot_view(shot_k), (55, 170))
            canvas.paste(self.view(self.n - 1).resize(SIDE), (1370, 170))
            side_label = "FIM DO PASSE · 1º TOQUE"
        else:
            canvas.paste(self.view(f, overlay=mode != "natural"), (55, 170))
            second = self.second_view(float(f) if tc is None else tc)
            if second is not None:
                canvas.paste(second, (1370, 170))
                side_label = "CÂMERA 16 mm · REPLAY LENTO SINCRONIZADO"
            else:
                canvas.paste(self.zoom(f), (1370, 170))
                side_label = f"ZOOM NO RECEBEDOR ({self.rid}) E NA BOLA"
        draw = ImageDraw.Draw(canvas)
        _text(draw, side_label, (1380, 463), 17, MUTED, bold=True, width=480)
        if mode in ("map", "map_full"):
            _text(draw, "Trajetórias medidas", (1370, 510), 26, bold=True)
            _text(
                draw,
                "Posições no plano do campo; ponto no chão sob cada jogador. "
                "Estrela: 1º toque do recebedor.",
                (1370, 556),
                23,
                width=490,
            )
        else:
            canvas.paste(
                self.map(
                    (500, 300),
                    f,
                    title="TRAJETÓRIAS NO CAMPO",
                    shot_k=shot_k if mode == "shot" else -1,
                ),
                (1370, 507),
            )
        draw = ImageDraw.Draw(canvas)
        if mode == "shot":
            _text(
                draw,
                f"Chute: cerca de {_n(self.shot.get('speed_headline_kmh'), 0)} km/h (estimativa)",
                (1370, 825),
                23,
                YELLOW,
                bold=True,
            )
            _text(
                draw,
                f"{_n(self.shot.get('distance_to_goal_m'))} m da linha do gol",
                (1370, 862),
                21,
            )
        else:
            vb = self.value("ball_speed_mps", f)
            _text(draw, f"Bola: {_kmh(vb)} km/h", (1370, 815), 25, YELLOW, bold=True)
            vp, vr = self.value("passer_speed_mps", f), self.value("receiver_speed_mps", f)
            _text(
                draw,
                f"{self.pid}: {_kmh(vp)} km/h  ·  {self.rid}: {_kmh(vr)} km/h",
                (1370, 852),
                21,
            )
            dp = self.series["distance_passer_ball_m"][f]
            dr = self.series["distance_receiver_ball_m"][f]
            _text(
                draw,
                f"{self.pid}→bola {_n(dp)} m  ·  {self.rid}→bola {_n(dr)} m",
                (1370, 882),
                21,
            )
        draw.rectangle((55, 914, 1870, 1031), fill="#152b43")
        caption = (
            scene[4]
            if scene
            else "SAM 3D Body: poses estimadas no vídeo original. Velocidades do plano do campo."
        )
        _text(draw, caption, (78, 933), 29, width=1755)
        for i, (label, color) in enumerate(
            (
                (f"Passador ({self.pid})", PASSER_C),
                (f"Recebedor ({self.rid})", RECV_C),
                ("Bola", BALL_C),
            )
        ):
            _text(draw, label, (60 + i * 310, 1044), 19, color, bold=True)
        label = "30 fps · SEM NARRAÇÃO" if scene else f"{float(self.rate):.3f} fps · ORIGINAL"
        _text(draw, label, (1440, 1044), 18, MUTED)
        progress = elapsed / TV_SECONDS if scene else (f + 1) / self.n
        draw.rectangle((0, H - 6, int(W * progress), H - 1), fill=YELLOW)
        return np.asarray(canvas)


# ---------------------------------------------------------------------------------- outputs


def write_videos(tv: PassTV, out: Path, scenes: list) -> list[str]:
    encode_frames(out / "lance_sam3body.mp4", tv.rate, (tv.render(f) for f in range(tv.n)))
    n_shot = len(tv.shot_frames)

    def tv_frames():
        for k in range(TV_SECONDS * TV_RATE):
            seconds = k / TV_RATE
            scene = next(s for s in scenes if s[0] <= seconds < s[1])
            start, end, mode = scene[0], scene[1], scene[5]
            f = frame_at_time(seconds, tv.rate, tv.n, start, end, natural=mode == "natural")
            # fractional reference time: the slow replay of the 2nd camera moves between frames
            if mode == "natural":
                tc = (seconds - start) * float(tv.rate)
            else:
                tc = (seconds - start) / (end - start) * tv.n
            tc = min(max(tc, 0.0), tv.n - 1.0)
            shot_k = -1
            if mode == "shot" and n_shot:
                shot_k = frame_at_time(seconds, tv.rate, n_shot, start, end)
            elif mode == "map_full":
                f, shot_k = tv.n - 1, n_shot - 1
            elif mode == "end":
                f = tv.n - 1
            if k % 300 == 0:
                print(f">> TV: {k}/{TV_SECONDS * TV_RATE} quadros", flush=True)
            yield tv.render(f, scene, seconds, shot_k, tc=tc)

    encode_frames(out / "explicacao_tv.mp4", Fraction(TV_RATE), tv_frames())
    return ["lance_sam3body.mp4", "explicacao_tv.mp4"]


def write_script(out: Path, scenes: list) -> None:
    rows, srt = [], []
    for i, (start, end, chapter, title, caption, _mode) in enumerate(scenes, 1):
        rows.append(
            {"start_s": start, "end_s": end, "chapter": chapter, "title": title, "caption": caption}
        )
        srt.append(f"{i}\n{_srt_time(start)} --> {_srt_time(end)}\n{title}\n{caption}\n")
    with (out / "roteiro_legendas.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "explicacao_tv.srt").write_text("\n".join(srt), encoding="utf-8")
    (out / "roteiro_tv.md").write_text(
        "# Roteiro de legendas — sem narração\n\n"
        + "\n\n".join(
            f"**{r['start_s']}–{r['end_s']} s · {r['chapter']}**\n\n{r['title']}\n\n{r['caption']}"
            for r in rows
        )
        + "\n",
        encoding="utf-8",
    )


def _copy_data(tv: PassTV, pass_folder: Path, shot_folder: Path | None, out: Path) -> list[Path]:
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []

    def cp(src: Path, name: str | None = None) -> None:
        if src.is_file():
            dst = data / (name or src.name)
            shutil.copy2(src, dst)
            copied.append(src)

    for name in ("pass_analysis.json", "pass_timeseries.csv"):
        cp(pass_folder / "data" / name)
    for name in (
        "ball.csv",
        "ball_3d.csv",
        "ball_3d_on_field.csv",
        "ball_flight.json",
        "ground_tracks.csv",
        "README.txt",
    ):
        cp(pass_folder / name, f"passe_{name}")
    for pid in (tv.pid, tv.rid):
        cp(pass_folder / f"person_{pid:03d}.c3d")
        cp(pass_folder / f"person_{pid:03d}_c3d.csv")
    dlts = sorted(pass_folder.parent.glob(f"work/**/{tv.video.stem}.dlt3d"))
    if dlts:
        cp(dlts[-1], f"passe_{dlts[-1].name}")
    if tv.kp_file is not None:
        rows = []
        with tv.kp_file.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fields = list(reader.fieldnames or [])
            rows = [r for r in reader if r.get("person_id") in (str(tv.pid), str(tv.rid))]
        with (data / f"passe_keypoints2d_ids_{tv.pid}_{tv.rid}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        copied.append(tv.kp_file)
    if shot_folder is not None:
        for name in (
            "ball.csv",
            "ball_3d_on_field.csv",
            "ball_flight.json",
            "player_map.csv",
            "README.txt",
        ):
            cp(shot_folder / name, f"chute_{name}")
        for name in ("kicker_metrics.json", "kicker_metrics.csv", "setpiece.json"):
            cp(shot_folder / "data" / name, f"chute_{name}")
    return copied


def _second_text(tv: PassTV) -> str:
    """Report paragraph on the second camera (sync quality and matched ids), or ''."""
    sec = tv.second
    if not sec:
        return ""
    sync = sec["sync"]
    rng = sync.get("slowmo_range") or [None, None]
    parts = []
    for pid, label in ((tv.pid, "passador"), (tv.rid, "recebedor")):
        m = sec["ids"].get(pid) or {}
        if m.get("accepted"):
            parts.append(
                f"{label} {pid} = id {m['other_id']} na câmera 16 mm (distância mediana "
                f"{_n(m['median_m'], 2)} m)"
            )
        else:
            parts.append(f"{label} {pid}: sem correspondência segura na câmera 16 mm")
    status = (
        "aceito"
        if sec["sync_ok"]
        else "recusado (o painel lateral volta ao zoom da câmera principal)"
    )
    return (
        "<h2>A câmera 16 mm (replay lento)</h2><p>O segundo vídeo é um replay em câmera lenta do "
        f"mesmo passe ({sync['frames']} quadros). Ele foi sincronizado pela bola: cada quadro "
        "recebe o instante da câmera principal em que a bola estava no mesmo ponto do seu "
        f"caminho. {sync['mapped']} quadros sincronizados; diferença mediana entre as duas bolas "
        f"{_n(sync['median_m'], 2)} m (máxima {_n(sync['max_m'], 2)} m); o replay é "
        f"{_n(rng[0])}× a {_n(rng[1])}× mais lento que o tempo real. Sincronismo {status}. "
        + "; ".join(parts)
        + ". Esta câmera serve só para ver o lance: todas as medidas vêm da câmera principal.</p>"
    )


def _second_touch_check(tv: PassTV, out: Path) -> None:
    """``conferencia_primeiro_toque_c2.jpg``: replay frames synced around the first touch."""
    sec = tv.second
    af = tv.t.get("arrival_frame")
    if not sec or af is None or not sec["sync_ok"]:
        return
    t = sec["sync"]["t_ref"]
    picks = [j for j in range(len(t)) if np.isfinite(t[j]) and af - 2.5 <= t[j] <= af + 1.0]
    if not picks:
        return
    tiles = []
    for j in picks[:6]:
        im = tv.second_view(float(t[j]), tight=True)
        if im is None:
            continue
        im = im.copy()
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, SIDE[0], 30), fill=BG)
        _text(draw, f"16 mm quadro {j} → principal {_n(t[j], 2)}", (8, 5), 17, YELLOW, bold=True)
        tiles.append(im)
    if not tiles:
        return
    sheet = Image.new("RGB", (SIDE[0] * min(3, len(tiles)), SIDE[1] * ((len(tiles) + 2) // 3)), BG)
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % 3) * SIDE[0], (i // 3) * SIDE[1]))
    sheet.save(out / "conferencia_primeiro_toque_c2.jpg", quality=92)


def _second_data(tv: PassTV, out: Path) -> list[Path]:
    """``data/sync_c2.csv`` and the second camera's inputs, as ``data/c2_*``."""
    sec = tv.second
    if not sec:
        return []
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    sync = sec["sync"]
    with (data / "sync_c2.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["c2_frame", "main_frame", "main_t_rel_release_s", "ball_residual_m"])
        for j, (t, r) in enumerate(zip(sync["t_ref"], sync["residual_m"], strict=True)):
            rel = "" if not np.isfinite(t) else f"{t / tv.fps - tv.t['release_t_s']:.4f}"
            w.writerow(
                [
                    j,
                    "" if not np.isfinite(t) else f"{t:.3f}",
                    rel,
                    "" if not np.isfinite(r) else f"{r:.4f}",
                ]
            )
    (data / "ids_c2.json").write_text(
        json.dumps({str(k): v for k, v in sec["ids"].items()}, indent=2), encoding="utf-8"
    )
    folder: Path = sec["folder"]
    copied: list[Path] = []
    for name in ("ball.csv", "ball_3d.csv", "ground_tracks.csv", "README.txt"):
        if (folder / name).is_file():
            shutil.copy2(folder / name, data / f"c2_{name}")
            copied.append(folder / name)
    dlts = sorted(folder.parent.glob(f"work/**/{sec['video'].stem}.dlt3d"))
    if dlts:
        shutil.copy2(dlts[-1], data / f"c2_{dlts[-1].name}")
        copied.append(dlts[-1])
    ids = {
        str(m["other_id"]) for m in sec["ids"].values() if m.get("accepted") and m.get("other_id")
    }
    if sec["kp_file"] is not None and ids:
        with sec["kp_file"].open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fields = list(reader.fieldnames or [])
            rows = [r for r in reader if r.get("person_id") in ids]
        with (data / f"c2_keypoints2d_ids_{'_'.join(sorted(ids))}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            w2 = csv.DictWriter(fh, fieldnames=fields)
            w2.writeheader()
            w2.writerows(rows)
        copied.append(sec["kp_file"])
    return copied


def _package_html(
    report_html: Path,
    out: Path,
    videos: list[str],
    extra_videos: list[str],
    second_text: str = "",
) -> str:
    """The pt TV report plus a videos section and an audit section (for the package)."""
    text = report_html.read_text(encoding="utf-8")
    vids = "".join(
        f"<figure><video controls preload='metadata' src='{v}' style='width:100%'></video>"
        f"<figcaption><a href='{v}'>{v}</a></figcaption></figure>"
        for v in videos + extra_videos
    )
    block = (
        "<style>@media print{video{display:none}}</style>"
        "<h2>Os vídeos</h2>" + vids + "<p>Legendas já desenhadas no vídeo de 60 s; também em "
        "<a href='explicacao_tv.srt'>SRT</a>, <a href='roteiro_tv.md'>roteiro</a> e "
        "<a href='roteiro_legendas.csv'>CSV</a>. Sem voz. A câmera lenta repete quadros "
        "observados; nenhum quadro é interpolado.</p>" + second_text
    )
    text = re.sub(
        r"(<p class=\"lead\">.*?</p>)", lambda m: m.group(1) + block, text, count=1, flags=re.S
    )
    checks = "".join(
        f"<figure><img src='{p.name}' style='width:100%'><figcaption>"
        + (
            "1º toque visto no replay lento da câmera 16 mm, com o quadro correspondente da "
            "câmera principal."
            if p.name.endswith("_c2.jpg")
            else "Conferência visual: identidades, esqueletos e bola no vídeo original."
        )
        + "</figcaption></figure>"
        for p in sorted(out.glob("conferencia_*.jpg"))
    )
    refs = "".join(f"<li><a href='{u}'>{t}</a></li>" for t, u in REFERENCES)
    audit = (
        "<h2>Conferência e reprodução</h2>" + checks + "<p><a href='data/pass_analysis.json'>"
        "Análise JSON</a> · <a href='data/pass_timeseries.csv'>Série quadro a quadro</a> · "
        "<a href='inputs_sha256.json'>Entradas e hashes</a> · "
        "<a href='validacao_entrega.json'>Validação</a> · "
        "<a href='README_reproducao.md'>Comando de reprodução</a>.</p>"
        "<h2>Referências e contexto</h2><ol class='refs'>" + refs + "</ol>"
    )
    return text.replace("<footer>", audit + "<footer>", 1)


def build_pass_tv_package(
    pass_folder: Path | str,
    out: Path | str,
    *,
    shot_folder: Path | str | None = None,
    second_camera: Path | str | None = None,
    animation: bool = True,
    command: list[str] | None = None,
) -> dict:
    """Write the TV package of a pass already analysed (``data/pass_analysis.json``).

    ``second_camera`` is the 3D folder of another FreeKiki run of the same pass (e.g. a slow
    replay). It is synchronized by the ball (:func:`sync_by_ball`) and shown beside the main
    camera; its players are matched to the passer and receiver (:func:`match_ids`). It is for
    display only: every number comes from the main camera.
    """
    pass_folder = Path(pass_folder).resolve()
    out = Path(out).resolve()
    shot = Path(shot_folder).resolve() if shot_folder else None
    analysis_path = pass_folder / "data" / "pass_analysis.json"
    if not analysis_path.is_file():
        raise FileNotFoundError(f"run the pass analysis first: {analysis_path}")
    if out == pass_folder:
        raise ValueError("the package must go to a separate folder, not the run folder")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if "timing" not in analysis:
        raise ValueError("pass_analysis.json has no timing block: re-run openplay-report")
    out.mkdir(parents=True, exist_ok=True)
    second = Path(second_camera).resolve() if second_camera else None
    tv = PassTV(analysis, pass_folder, shot, second)
    scenes = build_scenes(analysis, second=bool(tv.second and tv.second["sync_ok"]))

    # visual checks: release, mid-pass, first touch (+ the shot contact)
    rf = int(analysis["timing"].get("release_frame") or 0)
    af = analysis["timing"].get("arrival_frame")
    checks = [("saida", rf)]
    if af is not None:
        checks += [("meio", (rf + af) // 2), ("primeiro_toque", int(af))]
    for label, f in checks:
        Image.fromarray(tv.render(f)).save(out / f"conferencia_{label}.jpg", quality=92)
    if tv.shot_frames:
        shot_scene = next(s for s in scenes if s[5] == "shot")
        Image.fromarray(tv.render(tv.n - 1, shot_scene, shot_scene[0], 0)).save(
            out / "conferencia_chute.jpg", quality=92
        )
    if tv.second is not None:
        _second_touch_check(tv, out)
    write_script(out, scenes)
    videos = (
        write_videos(tv, out, scenes)
        if animation
        else [v for v in ("lance_sam3body.mp4", "explicacao_tv.mp4") if (out / v).is_file()]
    )
    extra = []
    if shot is not None:
        anim = shot / "report_tv" / "pt" / "animacao_lance.mp4"
        if anim.is_file():
            shutil.copy2(anim, out / "chute_animacao_bola.mp4")
            extra.append("chute_animacao_bola.mp4")

    # reports: figures + pt package HTML/PDF + en copy
    figs_src = pass_folder / "report_tv" / "pt" / "figuras_openplay"
    if figs_src.is_dir():
        shutil.copytree(figs_src, out / "figuras", dirs_exist_ok=True)
    reports: dict[str, str | None] = {}
    pt_html = pass_folder / "report_tv" / "pt" / "relatorio_openplay.html"
    if pt_html.is_file():
        path = out / "relatorio_passe.html"
        path.write_text(
            _package_html(pt_html, out, videos, extra, _second_text(tv)), encoding="utf-8"
        )
        pdf = print_pdf(path, out / "relatorio_passe.pdf")
        reports.update({"html": str(path), "pdf": str(pdf) if pdf else None})
    en_dir = pass_folder / "report_tv" / "en"
    for src, name in (
        (en_dir / "openplay_report.html", "relatorio_passe_en.html"),
        (en_dir / "openplay_report.pdf", "relatorio_passe_en.pdf"),
    ):
        if src.is_file():
            shutil.copy2(src, out / name)
            reports[name] = str(out / name)

    copied = _copy_data(tv, pass_folder, shot, out) + _second_data(tv, out)
    inputs = [tv.video, analysis_path, *copied]
    if tv.shot_frames:
        inputs.append(tv.shot_video)
    if tv.second is not None:
        inputs.append(tv.second["video"])
    seen: set[Path] = set()
    rows = []
    for p in inputs:
        p = Path(p)
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        rows.append({"path": str(p), "bytes": p.stat().st_size, "sha256": _sha256(p)})
    (out / "inputs_sha256.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    cmd = command or [
        sys.executable,
        "-m",
        "vaila.freekiki",
        "openplay-report",
        "--folder",
        str(pass_folder),
        "--kind",
        "pass",
    ]
    t = analysis["timing"]
    readme = (
        "# Passe para TV — reprodução\n\n"
        + shlex.join(cmd)
        + "\n\nO comando refaz a análise (`data/pass_analysis.json` na pasta do passe), o "
        "relatório e este pacote. Os arquivos de entrada são somente lidos; cada nova entrega "
        "deve usar outra pasta de saída.\n\n"
        "## Decisões desta análise\n\n"
        f"- Passador id {analysis['passer_id']}, recebedor id {analysis['receiver_id']} (IDs SAM "
        "da câmera principal).\n"
        f"- Taxa de quadros: {analysis['fps']:.6f} ({analysis.get('fps_source')}).\n"
        f"- Bola: `{analysis.get('ball_source')}` (trajetória no chão, raio da câmera na altura da "
        "bola).\n"
        f"- Saída da bola: {t['release_t_s']:+.3f} s em relação ao 1º quadro "
        f"({t['release_source']}).\n"
        f"- 1º toque: quadro {t.get('arrival_frame')} ({t.get('arrival_source')}).\n"
        "- O chute é outro trecho, de outra câmera: o tempo entre o passe e o chute não foi "
        "medido.\n"
    )
    if tv.second is not None:
        sync = tv.second["sync"]
        readme += (
            f"- Câmera 16 mm: `{tv.second['folder']}`, sincronizada pela bola "
            f"({sync['mapped']}/{sync['frames']} quadros, diferença mediana "
            f"{sync['median_m']:.2f} m). Ids: "
            + ", ".join(
                f"{pid} → {m.get('other_id')} ({'aceito' if m.get('accepted') else 'recusado'})"
                for pid, m in tv.second["ids"].items()
            )
            + ". Só para visualização; `data/sync_c2.csv` e `data/ids_c2.json`.\n"
        )
    (out / "README_reproducao.md").write_text(readme, encoding="utf-8")

    validation: dict = {
        "created": date.today().isoformat(),
        "videos": {},
        "pass_frames": tv.n,
        "pass_rate": str(tv.rate),
        "shot_frames": len(tv.shot_frames),
        "poses_frames": {
            str(pid): sum(1 for (f, p) in tv.poses if p == pid) for pid in (tv.pid, tv.rid)
        },
        "ball_pixel_frames": len(tv.ball_px),
    }
    if tv.second is not None:
        sync = tv.second["sync"]
        validation["second_camera"] = {
            "folder": str(tv.second["folder"]),
            "frames": sync["frames"],
            "mapped": sync["mapped"],
            "sync_median_m": sync["median_m"],
            "sync_max_m": sync["max_m"],
            "slowmo_factor": sync["slowmo_factor"],
            "slowmo_range": sync["slowmo_range"],
            "ids": {str(k): v for k, v in tv.second["ids"].items()},
        }
    for v in videos + extra:
        if (out / v).is_file():
            validation["videos"][v] = _probe(out / v)
    ok_tv = validation["videos"].get("explicacao_tv.mp4", {})
    ok_nat = validation["videos"].get("lance_sam3body.mp4", {})
    validation["checks"] = {
        "tv_60s_1800_frames": ok_tv.get("nb_read_frames") == str(TV_SECONDS * TV_RATE),
        "natural_all_frames": ok_nat.get("nb_read_frames") == str(tv.n),
        "csv_rows_match_frames": sum(
            1 for _ in (out / "data" / "pass_timeseries.csv").open(encoding="utf-8")
        )
        - 1
        == tv.n
        if (out / "data" / "pass_timeseries.csv").is_file()
        else False,
    }
    if tv.second is not None:
        validation["checks"]["second_sync_ok"] = bool(tv.second["sync_ok"])
        validation["checks"]["second_ids_ok"] = all(
            m.get("accepted") for m in tv.second["ids"].values()
        )
    (out / "validacao_entrega.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "report_summary.json").write_text(
        json.dumps(
            {
                "analysis": {
                    "fps": analysis["fps"],
                    "fps_source": analysis.get("fps_source"),
                    "ball_source": analysis.get("ball_source"),
                    "release_source": t["release_source"],
                    "arrival_source": t.get("arrival_source"),
                    "passer_to_arrival_m": t.get("passer_to_arrival_m"),
                    "flight_s": t.get("flight_s"),
                },
                "report": reports,
                "videos": videos + extra,
                "commands": [cmd],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    outputs = [
        {"path": str(p.relative_to(out)), "sha256": _sha256(p)}
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "outputs_sha256.json"
    ]
    (out / "outputs_sha256.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    return {
        "dir": str(out),
        "videos": videos + extra,
        "report": reports,
        "checks": validation["checks"],
    }


__all__ = ["PassTV", "build_pass_tv_package", "build_scenes"]
