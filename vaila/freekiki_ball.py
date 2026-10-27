"""
================================================================================
Script: freekiki_ball.py - ball track with SAM 3 (text prompt "sports ball")
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.11
Created: 25 October 2026
Update Date: 26 October 2026

Description:
    One ball per frame is the most confident ``sports ball`` the SAM 3 text
    prompt finds. The ball is small and often blurred, so the result always
    carries its ``score``: a low score means a doubtful ball, and the CSV keeps
    it for filtering rather than hiding it.

    * :func:`best_ball` - pick the most confident detection of a frame (pure);
    * :func:`write_ball_csv` - ``frame, cx, cy, x1, y1, x2, y2, score`` (pure I/O);
    * :func:`detect_ball_video` - runs SAM 3 over a video (needs CUDA and the
      ``sam3.pt`` weights; not exercised by the unit tests).

    Ultralytics is imported only inside :func:`detect_ball_video`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

Detections = list[tuple[float, np.ndarray]]  # one frame: (score, xyxy), best first
Candidates = list[Detections]  # one video: a Detections per frame

BALL_PROMPT = "sports ball"
BALL_COLUMNS = ["frame", "cx", "cy", "x1", "y1", "x2", "y2", "score"]


def best_ball(detections: list[tuple[float, np.ndarray]]) -> tuple[float, np.ndarray] | None:
    """The ``(score, xyxy)`` with the highest score, ignoring non-finite boxes."""
    best: tuple[float, np.ndarray] | None = None
    for score, box in detections:
        box = np.asarray(box, dtype=np.float64).reshape(4)
        if not np.isfinite(box).all() or not np.isfinite(score):
            continue
        if best is None or score > best[0]:
            best = (float(score), box)
    return best


def ball_row(frame: int, best: tuple[float, np.ndarray] | None) -> dict:
    """One CSV row; a frame without a ball keeps empty coordinates and score."""
    if best is None:
        return {
            "frame": frame,
            "cx": "",
            "cy": "",
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
            "score": "",
        }
    score, (x1, y1, x2, y2) = best
    return {
        "frame": frame,
        "cx": round(0.5 * (x1 + x2), 2),
        "cy": round(0.5 * (y1 + y2), 2),
        "x1": round(x1, 2),
        "y1": round(y1, 2),
        "x2": round(x2, 2),
        "y2": round(y2, 2),
        "score": round(score, 4),
    }


def track_ball(
    frames: list[list[tuple[float, float, float]]],
    *,
    max_speed_px: float = 150.0,
    slack_px: float = 40.0,
    min_score: float = 0.2,
    jump_cost: float = 0.6,
) -> list[int | None]:
    """Pick at most one candidate per frame so that the picks form one smooth track.

    ``frames[t]`` holds the ``(score, cx, cy)`` candidates of frame ``t`` (already on the
    pitch). A pick earns ``score - min_score``, so a weak box is kept only when it
    continues the track. Two picks in frames ``t0 < t1`` link for free when their
    distance is within ``max_speed_px * (t1 - t0) + slack_px``; otherwise the link costs
    ``jump_cost``. The best chain is found by dynamic programming over the candidates in
    frame order. Returns the chosen candidate index per frame, or ``None``.
    """
    nodes = [
        (t, k, score, cx, cy)
        for t, cands in enumerate(frames)
        for k, (score, cx, cy) in enumerate(cands)
    ]
    best = [0.0] * len(nodes)
    back: list[int | None] = [None] * len(nodes)
    for i, (t1, _, score, x1, y1) in enumerate(nodes):
        gain = score - min_score
        start = 0.0  # a new chain, with no link to earlier picks
        for j in range(i):
            t0, _, _, x0, y0 = nodes[j]
            if t0 >= t1:
                continue
            dist = float(np.hypot(x1 - x0, y1 - y0))
            link = 0.0 if dist <= max_speed_px * (t1 - t0) + slack_px else jump_cost
            if best[j] - link > start:
                start = best[j] - link
                back[i] = j
        best[i] = gain + start
    picks: list[int | None] = [None] * len(frames)
    if not nodes:
        return picks
    i = max(range(len(nodes)), key=lambda n: best[n])
    if best[i] <= 0:
        return picks
    while i is not None:
        t, k, *_ = nodes[i]
        picks[t] = k
        i = back[i]
    return picks


def write_ball_csv(path: Path | str, rows: list[dict]) -> Path:
    """Write the per-frame ball track with the fixed :data:`BALL_COLUMNS` header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=BALL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def finite_candidates(detections: list[tuple[float, np.ndarray]]) -> Detections:
    """The finite ``(score, xyxy)`` boxes of one frame, highest score first."""
    keep: Detections = []
    for score, box in detections:
        box = np.asarray(box, dtype=np.float64).reshape(4)
        if np.isfinite(box).all() and np.isfinite(score):
            keep.append((float(score), box))
    return sorted(keep, key=lambda c: c[0], reverse=True)


def ball_candidates_video(
    video: Path | str,
    weights: Path | str,
    *,
    conf: float = 0.1,
    prompt: str = BALL_PROMPT,
) -> Candidates:
    """Every SAM 3 ``sports ball`` box of every frame, best score first.

    The pitch gate needs all of them: the most confident box of a frame can be a ball
    off the pitch (the stands, a spare ball) while the ball in play is the second one.
    Uses the ``sam3`` package (the one ``vaila_sam`` uses), not the Ultralytics SAM 3
    predictor: that one passes its text through a CLIP tokenizer that is not callable
    with the ``openai-clip`` package installed here. Needs ``weights`` (``sam3.pt``)
    and CUDA when available. Raises ``RuntimeError`` when ``sam3`` is missing.
    """
    import cv2
    import torch

    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as exc:
        raise RuntimeError("The sam3 package is not installed; install the 'sam' extra.") from exc
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"SAM 3 weights not found: {weights}")

    try:
        from .vaila_sam import _resolve_bpe_path
    except ImportError:
        from vaila_sam import _resolve_bpe_path  # ty: ignore[unresolved-import]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(
        bpe_path=str(_resolve_bpe_path()),
        checkpoint_path=str(weights),
        load_from_HF=False,
        enable_inst_interactivity=False,
        device=device,
    )
    processor = Sam3Processor(model, device=device, confidence_threshold=conf)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video}")
    out: Candidates = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            # Sam3Processor reads the size from shape[-2:], so it needs CHW, not HWC.
            # An HWC array gives width 3 and wrong boxes.
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            state = processor.set_image(torch.from_numpy(rgb).permute(2, 0, 1))
            state = processor.set_text_prompt(prompt, state)
            boxes = state["boxes"].detach().cpu().numpy()
            scores = state["scores"].detach().cpu().numpy()
            out.append(
                finite_candidates([(float(s), b) for s, b in zip(scores, boxes, strict=True)])
            )
    finally:
        cap.release()
    return out


def detect_ball_video(
    video: Path | str,
    weights: Path | str,
    *,
    conf: float = 0.1,
    prompt: str = BALL_PROMPT,
) -> list[dict]:
    """The most confident SAM 3 ball of every frame, one row per frame (no pitch gate).

    The 3D chain uses :func:`ball_candidates_video` with the pitch gate instead.
    """
    candidates = ball_candidates_video(video, weights, conf=conf, prompt=prompt)
    return [ball_row(frame, best_ball(c)) for frame, c in enumerate(candidates)]


def ball_candidates_net(
    video: Path | str,
    weights: Path | str,
    *,
    conf: float = 0.25,
    imgsz: int = 1280,
) -> Candidates:
    """Every box of the dedicated ball detector (one YOLO class ``ball``), per frame.

    The ``score`` is the detector confidence. Ultralytics is imported here only.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required for the ball detector") from exc
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Ball detector weights not found: {weights}")

    model = YOLO(str(weights))
    out: Candidates = []
    results = model.predict(
        source=str(video), conf=conf, imgsz=imgsz, stream=True, verbose=False, save=False
    )
    for result in results:
        detections: list[tuple[float, np.ndarray]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy):
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            detections.extend((float(s), b) for s, b in zip(scores, xyxy, strict=True))
        out.append(finite_candidates(detections))
    return out


def detect_ball_net_video(
    video: Path | str,
    weights: Path | str,
    *,
    conf: float = 0.25,
    imgsz: int = 1280,
) -> list[dict]:
    """The most confident ball-net box of every frame, one row per frame (no pitch gate)."""
    candidates = ball_candidates_net(video, weights, conf=conf, imgsz=imgsz)
    return [ball_row(frame, best_ball(c)) for frame, c in enumerate(candidates)]
