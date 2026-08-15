"""
===============================================================================
Project: vailá Multimodal Toolbox
Script: vaila_stroboscopic.py
===============================================================================
Author: Paulo R. P. Santiago & Antigravity (Google Deepmind)
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 21 April 2026
Update Date: 15 August 2026
Version: 0.3.105

Description:
------------
Generates a stroboscopic (chronophotography) image and video from sports &
biomechanical recordings.
Primary mode: "stromotion" (Dartfish style) extracts the moving athlete and
composites frozen snapshots over time onto an estimated clean background.
Supports AI Selfie Segmentation, Adaptive Background Subtraction (auto-fallback),
MOG2 background subtraction, 2D Pose Skeleton overlays, and Multishot Stacking.

Usage:
------
uv run python vaila/vaila_stroboscopic.py -v video.mp4 --mode stromotion -i 10
or run without arguments to open the interactive GUI settings dialog.
===============================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import tkinter as tk
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

try:
    from .cli_highlight import print_gui_cli_mirror
except ImportError:
    from cli_highlight import print_gui_cli_mirror  # ty: ignore[unresolved-import]

_SELFIE_SEGMENTER_MODELS: dict[int, tuple[str, list[str]]] = {
    0: (
        "image_segmenter_selfie_square.tflite",
        [
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite",
        ],
    ),
    1: (
        "image_segmenter_selfie_landscape.tflite",
        [
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite",
        ],
    ),
}


def _help_path() -> Path:
    return Path(__file__).resolve().parent / "help" / "vaila_stroboscopic.html"


def _is_valid_model_file(path: Path) -> bool:
    """True if path exists, is binary, >10KB and not an unpulled Git LFS pointer text file."""
    if not path.is_file():
        return False
    try:
        size = path.stat().st_size
        if size < 10_000:
            return False
        with path.open("rb") as fh:
            header = fh.read(128)
            if b"git-lfs" in header or b"version https://" in header:
                return False
        return True
    except OSError:
        return False


def _get_selfie_image_segmenter_model_path(model_selection: int) -> str:
    """Download/cache MediaPipe Tasks selfie segmenter (.tflite) next to other vaila models."""
    if model_selection not in (0, 1):
        model_selection = 1
    filename, urls = _SELFIE_SEGMENTER_MODELS[model_selection]

    candidate_dirs = [
        Path(__file__).resolve().parent / "models",
        Path(__file__).resolve().parents[1] / "models",
        Path(__file__).resolve().parent / "models" / "crop_face",
    ]

    for cdir in candidate_dirs:
        candidate = cdir / filename
        if _is_valid_model_file(candidate):
            return str(candidate.resolve())

    target_dir = Path(__file__).resolve().parent / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename

    if target_path.exists() and not _is_valid_model_file(target_path):
        with contextlib.suppress(OSError):
            target_path.unlink()

    last_err: Exception | None = None
    for url in urls:
        try:
            print(f">> vaila/vaila_stroboscopic: Downloading model ({filename})...\n  {url}")
            urllib.request.urlretrieve(url, str(target_path))
            if _is_valid_model_file(target_path):
                print(
                    f">> vaila/vaila_stroboscopic: Download completed ({target_path.stat().st_size} bytes)."
                )
                return str(target_path.resolve())
            with contextlib.suppress(OSError):
                target_path.unlink()
        except Exception as e:
            last_err = e
            if target_path.exists():
                with contextlib.suppress(OSError):
                    target_path.unlink()

    raise RuntimeError(
        f"Failed to download MediaPipe selfie segmenter model ({filename}). "
        f"Please check your internet connection or download manually into {target_dir}."
    ) from last_err


def _open_video_or_raise(video_path: Path) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    return cap


def _read_video_info(cap: cv2.VideoCapture) -> tuple[int, int, float, int]:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return width, height, fps, nframes


def _estimate_background(
    cap: cv2.VideoCapture, n_samples: int = 15, start_frame: int = 0, end_frame: int | None = None
) -> np.ndarray | None:
    if end_frame is None:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    span = max(1, end_frame - start_frame)
    step = max(1, span // max(1, n_samples))
    frames = []

    for fi in range(start_frame, end_frame, step):
        if len(frames) >= min(n_samples, 40):
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if ok and fr is not None:
            frames.append(fr)

    if not frames:
        return None
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def _segment_person_mediapipe(
    frame_bgr: np.ndarray,
    segmenter: mp.tasks.vision.ImageSegmenter,
    threshold: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Segment person using MediaPipe Tasks ImageSegmenter. Returns mask, alpha, confidence."""
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = segmenter.segment(mp_image)

    alpha = None
    max_conf = 0.0
    if result.confidence_masks:
        masks = result.confidence_masks
        raw = np.asarray(
            masks[1].numpy_view() if len(masks) >= 2 else masks[0].numpy_view(), dtype=np.float32
        )
        if raw.ndim == 3:
            raw = np.squeeze(raw, axis=-1)
        max_conf = float(raw.max()) if raw.size > 0 else 0.0
        alpha = raw
    elif result.category_mask is not None:
        raw = np.asarray(result.category_mask.numpy_view())
        if raw.ndim == 3:
            raw = np.squeeze(raw, axis=-1)
        max_conf = float((raw == 1).any())
        alpha = (raw == 1).astype(np.float32)

    if alpha is None or max_conf < threshold:
        return (
            np.zeros((h, w), dtype=np.uint8),
            np.zeros((h, w), dtype=np.float32),
            max_conf,
        )

    if alpha.shape[:2] != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = (alpha > threshold).astype(np.uint8) * 255
    return mask, alpha, max_conf


def _segment_by_background_diff(
    frame_bgr: np.ndarray,
    background_bgr: np.ndarray,
    threshold: int = 22,
    open_size: int = 5,
    dilate_size: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract moving foreground by comparing frame against estimated background."""
    h, w = frame_bgr.shape[:2]
    diff = cv2.absdiff(frame_bgr, background_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    if open_size > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    if dilate_size > 1:
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k_dil)

    alpha = mask.astype(np.float32) / 255.0
    return mask, alpha


def generate_stromotion(
    video_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    frame_interval: int = 10,
    bg_mode: str = "median",
    bg_samples: int = 15,
    seg_method: str = "auto",
    seg_threshold: float = 0.35,
    diff_threshold: int = 22,
    feather_px: int = 5,
    min_subject_area: int = 150,
    outline: bool = False,
    outline_color: tuple[int, int, int] = (255, 255, 255),
    outline_thickness: int = 2,
    start_sec: float | None = None,
    end_sec: float | None = None,
    save_individual_frames: bool = True,
    save_video: bool = True,
    model_selection: int = 1,
) -> bool:
    """Dartfish-style Stromotion effect with adaptive AI/background segmentation and live timeline composite."""
    video_path = Path(video_path)
    cap = _open_video_or_raise(video_path)
    if cap is None:
        print(f"Error opening video: {video_path}")
        return False

    width, height, fps, nframes = _read_video_info(cap)
    if width <= 0 or height <= 0 or nframes <= 0:
        print("Error: Could not read video properties.")
        cap.release()
        return False

    if output_dir is None:
        output_dir = video_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    start_frame = max(0, int(start_sec * fps)) if start_sec is not None else 0
    end_frame = min(nframes, int(end_sec * fps)) if end_sec is not None else nframes

    if end_frame <= start_frame:
        print("Error: invalid duration range.")
        cap.release()
        return False

    print(f"1/3 Estimating background ({bg_mode}, {bg_samples} samples across {end_frame - start_frame} frames)...")
    if bg_mode == "first":
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, base_frame = cap.read()
        if not ret or base_frame is None:
            base_frame = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        base_frame = _estimate_background(
            cap, n_samples=bg_samples, start_frame=start_frame, end_frame=end_frame
        )
        if base_frame is None:
            base_frame = np.zeros((height, width, 3), dtype=np.uint8)

    canvas = base_frame.copy()

    out_video_path = output_dir / f"{video_path.stem}_stromotion.mp4"
    out_img_path = output_dir / f"{video_path.stem}_stromotion.png"
    frames_dir = output_dir / f"{video_path.stem}_stromotion_frames"

    if save_individual_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    vwriter = None
    if save_video:
        fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc
        fourcc = fourcc_fn(*"mp4v")
        vwriter = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
        if not vwriter.isOpened():
            fourcc = fourcc_fn(*"avc1")
            vwriter = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
        if not vwriter.isOpened():
            print(
                ">> vaila/vaila_stroboscopic: [WARNING] Could not open video writer. Video output disabled."
            )
            vwriter = None

    # Setup MediaPipe segmenter if requested or in auto mode
    segmenter: mp.tasks.vision.ImageSegmenter | None = None
    if seg_method in ("auto", "ai"):
        try:
            model_path = _get_selfie_image_segmenter_model_path(model_selection)
            BaseOptions = mp.tasks.BaseOptions  # noqa: N806
            ImageSegmenter = mp.tasks.vision.ImageSegmenter  # noqa: N806
            ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions  # noqa: N806
            VisionRunningMode = mp.tasks.vision.RunningMode  # noqa: N806

            segmenter_options = ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE,
                output_confidence_masks=True,
                output_category_mask=True,
            )
            segmenter = ImageSegmenter.create_from_options(segmenter_options)
        except Exception as exc:
            print(f">> vaila/vaila_stroboscopic: [NOTE] AI segmenter load skipped ({exc}); using background diff.")
            segmenter = None

    # Setup MOG2 background subtractor if requested
    mog2: cv2.BackgroundSubtractorMOG2 | None = None
    if seg_method == "mog2":
        mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    print(f"2/3 Processing frames and compositing (mode={seg_method}, interval={frame_interval})...")
    sampled_instances_count = 0

    try:
        # Pre-feed MOG2 if used
        if mog2 is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(min(50, end_frame - start_frame)):
                ok, fr = cap.read()
                if ok and fr is not None:
                    mog2.apply(fr)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for fi in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # 1. Segment foreground subject for current frame
            mask = np.zeros((height, width), dtype=np.uint8)
            alpha = np.zeros((height, width), dtype=np.float32)

            if segmenter is not None:
                ai_mask, ai_alpha, conf = _segment_person_mediapipe(frame, segmenter, threshold=seg_threshold)
                ai_area = int((ai_mask > 0).sum())
                if conf >= seg_threshold and ai_area >= min_subject_area:
                    mask, alpha = ai_mask, ai_alpha
                elif seg_method == "auto":
                    # Auto fallback to background difference when AI misses full-body/distant subjects
                    mask, alpha = _segment_by_background_diff(frame, base_frame, threshold=diff_threshold)
            elif mog2 is not None:
                fg_mask = mog2.apply(frame)
                _, mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
                k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
                alpha = mask.astype(np.float32) / 255.0
            else:
                # Default: robust background difference
                mask, alpha = _segment_by_background_diff(frame, base_frame, threshold=diff_threshold)

            # Apply feathering
            if feather_px > 0 and alpha.max() > 0:
                ksize = feather_px if feather_px % 2 == 1 else feather_px + 1
                alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)

            alpha_3d = np.stack([alpha] * 3, axis=-1)
            subject_area = int((mask > 0).sum())
            is_sample_frame = ((fi - start_frame) % frame_interval == 0) and (subject_area >= min_subject_area)

            # 2. If sample frame: stamp a persistent frozen clone onto the background canvas
            if is_sample_frame:
                if outline:
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(canvas, contours, -1, outline_color, outline_thickness)

                canvas = (canvas * (1 - alpha_3d) + frame * alpha_3d).astype(np.uint8)
                sampled_instances_count += 1

                if save_individual_frames:
                    rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                    rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
                    cv2.imwrite(str(frames_dir / f"frame_{fi:06d}.png"), rgba)

            # 3. For video output: composite canvas (history of frozen snapshots) + live moving frame
            if vwriter is not None:
                live_composite = (canvas * (1 - alpha_3d) + frame * alpha_3d).astype(np.uint8)
                vwriter.write(live_composite)

            if fi % 30 == 0:
                print(
                    f"  Processed {fi - start_frame}/{end_frame - start_frame} frames ({sampled_instances_count} snapshots captured)...",
                    end="\r",
                    flush=True,
                )
    finally:
        if segmenter is not None:
            segmenter.close()

    print(f"\n3/3 Saving outputs ({sampled_instances_count} snapshots captured) to {output_dir}...")
    cv2.imwrite(str(out_img_path), canvas)
    print(f">> Saved Image: {out_img_path}")

    if vwriter:
        vwriter.release()
        print(f">> Saved Video: {out_video_path}")

    cap.release()
    return True


# =============================================================================
# BACKWARD COMPATIBILITY MODES
# =============================================================================
def generate_stack_multishot(
    video_path: str | Path,
    output_path: str | Path | None = None,
    frame_interval: int = 10,
    stack_op: str = "max",
) -> bool:
    print("Running stack mode...")
    video_path = Path(video_path)
    cap = _open_video_or_raise(video_path)
    if cap is None:
        return False
    _, _, _, nframes = _read_video_info(cap)

    ret, result = cap.read()
    if not ret or result is None:
        cap.release()
        return False

    acc = result.astype(np.float32) if stack_op == "add" else None
    count = 1

    for fi in range(1, nframes, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if stack_op == "add":
            assert acc is not None
            acc += frame.astype(np.float32)
            count += 1
        else:
            result = np.maximum(result, frame)

    cap.release()
    out_path = (
        Path(output_path)
        if output_path
        else video_path.parent / f"{video_path.stem}_multishot.png"
    )
    if stack_op == "add" and acc is not None:
        blended = (acc / count).astype(np.float32)
        cv2.normalize(blended, blended, 0, 255, cv2.NORM_MINMAX)
        out = blended.astype(np.uint8)
    else:
        out = result
    cv2.imwrite(str(out_path), out)
    print(f">> Saved Stack Image: {out_path}")
    return True


def generate_motion_stroboscopic(
    video_path: str | Path,
    output_path: str | Path | None = None,
    frame_interval: int = 1,
    threshold: int = 50,
    blur_size: int = 5,
    **kwargs: Any,
) -> bool:
    print("Running motion mode...")
    video_path = Path(video_path)
    cap = _open_video_or_raise(video_path)
    if cap is None:
        return False
    _, _, _, nframes = _read_video_info(cap)

    ret, base_frame = cap.read()
    if not ret or base_frame is None:
        cap.release()
        return False
    prev_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
    acc = np.zeros_like(base_frame, dtype=np.float32)
    change = np.zeros_like(base_frame, dtype=np.float32)

    for fi in range(1, nframes, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask, (blur_size | 1, blur_size | 1), 0)
        acc += cv2.bitwise_and(frame, frame, mask=mask).astype(np.float32)
        change += (mask[:, :, np.newaxis] > 0).astype(np.float32)
        prev_gray = gray

    cap.release()
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.divide(acc, change)
        avg[~np.isfinite(avg)] = base_frame[~np.isfinite(avg)]

    avg_f = avg.astype(np.float32)
    cv2.normalize(avg_f, avg_f, 0, 255, cv2.NORM_MINMAX)
    out = avg_f.astype(np.uint8)
    out_path = (
        Path(output_path)
        if output_path
        else video_path.parent / f"{video_path.stem}_motion_strobe.png"
    )
    cv2.imwrite(str(out_path), out)
    print(f">> Saved Motion Image: {out_path}")
    return True


def generate_stroboscopic_image(
    video_path: str | Path,
    csv_path: str | Path | None = None,
    output_path: str | Path | None = None,
    strobe_interval: int = 10,
    **kwargs: Any,
) -> bool:
    print("Running pose mode... Needs valid CSV.")
    video_path = Path(video_path)
    if csv_path is None:
        candidates = list(video_path.parent.glob(f"*{video_path.stem}*.csv"))
        if not candidates:
            candidates = list(video_path.parent.glob("*.csv"))
        if candidates:
            csv_path = candidates[0]
            print(f">> Auto-detected CSV: {csv_path}")
        else:
            print("No CSV file found.")
            return False

    if csv_path is not None:
        csv_path = Path(csv_path)
        if csv_path.exists():
            import pandas as pd

            df = pd.read_csv(csv_path)
        else:
            print(f"CSV path does not exist: {csv_path}")
            return False
    else:
        print("CSV file is required.")
        return False

    cap = _open_video_or_raise(video_path)
    if cap is None:
        print("Error opening video.")
        return False

    width, height, fps, nframes = _read_video_info(cap)
    ret, base_frame = cap.read()
    if not ret or base_frame is None:
        cap.release()
        return False

    canvas = base_frame.copy()

    for fi in range(0, nframes, strobe_interval):
        row_df = df[df["frame"] == fi]
        if row_df.empty:
            row_df = df[df["frame"].astype(float).round() == float(fi)]
        if row_df.empty:
            continue

        row = row_df.iloc[0]
        pts = {}
        is_normalized = False
        for p in range(1, 34):
            x_val = row.get(f"p{p}_x")
            y_val = row.get(f"p{p}_y")
            if (
                x_val is not None
                and y_val is not None
                and not pd.isna(x_val)
                and not pd.isna(y_val)
            ):
                if 0.0 <= float(x_val) <= 1.0 and 0.0 <= float(y_val) <= 1.0:
                    is_normalized = True
                pts[p] = (float(x_val), float(y_val))

        scaled_pts = {}
        for p, (x, y) in pts.items():
            if is_normalized:
                px = int(x * width)
                py = int(y * height)
            else:
                px = int(x)
                py = int(y)
            scaled_pts[p] = (px, py)
            cv2.circle(canvas, (px, py), 4, (0, 255, 0), -1)

        connections = [
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
            (11, 23),
            (12, 24),
            (23, 24),
            (23, 25),
            (25, 27),
            (24, 26),
            (26, 28),
            (27, 29),
            (29, 31),
            (27, 31),
            (28, 30),
            (30, 32),
            (28, 32),
        ]
        for p1, p2 in connections:
            if p1 in scaled_pts and p2 in scaled_pts:
                cv2.line(canvas, scaled_pts[p1], scaled_pts[p2], (255, 0, 0), 2)

    cap.release()
    out_path = (
        Path(output_path)
        if output_path
        else video_path.parent / f"{video_path.stem}_pose_strobe.png"
    )
    cv2.imwrite(str(out_path), canvas)
    print(f">> Saved Pose Strobe Image: {out_path}")
    return True


# =============================================================================
# GUI SETTINGS DIALOG
# =============================================================================
@dataclass(frozen=True)
class StroboscopicSettings:
    video_path: Path
    output_dir: Path | None
    csv_path: Path | None
    mode: str
    interval: int
    bg_mode: str
    bg_samples: int
    seg_method: str
    seg_threshold: float
    diff_threshold: int
    feather_px: int
    outline: bool
    outline_color: tuple[int, int, int]
    outline_thickness: int
    save_video: bool
    save_frames: bool
    stack_op: str
    model_selection: int


def _format_gui_cli(settings: StroboscopicSettings) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "vaila/vaila_stroboscopic.py",
        "-v",
        str(settings.video_path),
        "--mode",
        settings.mode,
        "-i",
        str(settings.interval),
    ]
    if settings.output_dir is not None:
        cmd += ["-o", str(settings.output_dir)]
    if settings.mode == "stromotion":
        cmd += [
            "--bg-mode",
            settings.bg_mode,
            "--bg-samples",
            str(settings.bg_samples),
            "--seg-method",
            settings.seg_method,
            "--seg-threshold",
            str(settings.seg_threshold),
            "--diff-threshold",
            str(settings.diff_threshold),
            "--feather-px",
            str(settings.feather_px),
            "--model-selection",
            str(settings.model_selection),
        ]
        if settings.outline:
            color_str = f"{settings.outline_color[0]},{settings.outline_color[1]},{settings.outline_color[2]}"
            cmd += [
                "--outline",
                "--outline-color",
                color_str,
                "--outline-thickness",
                str(settings.outline_thickness),
            ]
        if not settings.save_video:
            cmd.append("--no-video")
        if not settings.save_frames:
            cmd.append("--no-frames")
    elif settings.mode == "stack":
        cmd += ["--stack-op", settings.stack_op]
    elif settings.mode == "pose" and settings.csv_path:
        cmd += ["-c", str(settings.csv_path)]
    return cmd


class StroboscopicDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        super().__init__(parent)
        self.title("vailá - Stroboscopic / Stromotion Generator")
        self.result: StroboscopicSettings | None = None
        self.resizable(True, True)

        self.video_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.csv_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="stromotion")
        self.interval_var = tk.StringVar(value="10")
        self.bg_mode_var = tk.StringVar(value="median")
        self.bg_samples_var = tk.StringVar(value="15")
        self.seg_method_var = tk.StringVar(value="auto")
        self.seg_thresh_var = tk.StringVar(value="0.35")
        self.diff_thresh_var = tk.StringVar(value="22")
        self.feather_var = tk.StringVar(value="5")
        self.outline_var = tk.BooleanVar(value=False)
        self.outline_color = (255, 255, 255)
        self.outline_thick_var = tk.StringVar(value="2")
        self.save_video_var = tk.BooleanVar(value=True)
        self.save_frames_var = tk.BooleanVar(value=True)
        self.stack_op_var = tk.StringVar(value="max")
        self.model_sel_var = tk.StringVar(value="1")

        self._build_ui()

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="Input & Output", padding=8)
        file_frame.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(file_frame, text="Video:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.video_var, width=50).grid(
            row=0, column=1, sticky=tk.EW, padx=4
        )
        ttk.Button(file_frame, text="Browse…", command=self._browse_video).grid(row=0, column=2)

        ttk.Label(file_frame, text="Output Dir:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.output_var, width=50).grid(
            row=1, column=1, sticky=tk.EW, padx=4
        )
        ttk.Button(file_frame, text="Browse…", command=self._browse_output).grid(row=1, column=2)

        file_frame.columnconfigure(1, weight=1)

        # Mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Mode & Basic Settings", padding=8)
        mode_frame.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(mode_frame, text="Mode:").grid(row=0, column=0, sticky=tk.W)
        mode_cb = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            values=["stromotion", "pose", "motion", "stack"],
            state="readonly",
            width=18,
        )
        mode_cb.grid(row=0, column=1, sticky=tk.W, padx=4)
        mode_cb.bind("<<ComboboxSelected>>", self._on_mode_change)

        ttk.Label(mode_frame, text="Frame Interval:").grid(row=0, column=2, sticky=tk.W, padx=8)
        ttk.Entry(mode_frame, textvariable=self.interval_var, width=8).grid(
            row=0, column=3, sticky=tk.W
        )

        # Pose CSV row (hidden if not pose)
        self.csv_label = ttk.Label(mode_frame, text="Pose CSV:")
        self.csv_label.grid(row=1, column=0, sticky=tk.W, pady=4)
        self.csv_entry = ttk.Entry(mode_frame, textvariable=self.csv_var, width=35)
        self.csv_entry.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=4, pady=4)
        self.csv_btn = ttk.Button(mode_frame, text="Browse CSV…", command=self._browse_csv)
        self.csv_btn.grid(row=1, column=3, sticky=tk.W, pady=4)

        # Advanced options for Stromotion
        self.ai_frame = ttk.LabelFrame(
            main_frame, text="Stromotion (Dartfish & AI Compositing) Options", padding=8
        )
        self.ai_frame.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(self.ai_frame, text="Segmentation:").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(
            self.ai_frame,
            textvariable=self.seg_method_var,
            values=["auto", "bg_diff", "ai", "mog2"],
            state="readonly",
            width=10,
        ).grid(row=0, column=1, sticky=tk.W, padx=4)
        ttk.Label(self.ai_frame, text="(auto = AI + background diff fallback)").grid(
            row=0, column=2, columnspan=2, sticky=tk.W
        )

        ttk.Label(self.ai_frame, text="Background:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(
            self.ai_frame,
            textvariable=self.bg_mode_var,
            values=["median", "first"],
            state="readonly",
            width=10,
        ).grid(row=1, column=1, sticky=tk.W, padx=4, pady=2)

        ttk.Label(self.ai_frame, text="Median Samples:").grid(
            row=1, column=2, sticky=tk.W, padx=4, pady=2
        )
        ttk.Entry(self.ai_frame, textvariable=self.bg_samples_var, width=6).grid(
            row=1, column=3, sticky=tk.W, pady=2
        )

        ttk.Label(self.ai_frame, text="Diff Threshold:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.ai_frame, textvariable=self.diff_thresh_var, width=6).grid(
            row=2, column=1, sticky=tk.W, padx=4, pady=2
        )

        ttk.Label(self.ai_frame, text="Feather (px):").grid(
            row=2, column=2, sticky=tk.W, padx=4, pady=2
        )
        ttk.Entry(self.ai_frame, textvariable=self.feather_var, width=6).grid(
            row=2, column=3, sticky=tk.W, pady=2
        )

        ttk.Checkbutton(self.ai_frame, text="Draw Outline", variable=self.outline_var).grid(
            row=3, column=0, sticky=tk.W, pady=2
        )
        ttk.Button(self.ai_frame, text="Color…", command=self._pick_color).grid(
            row=3, column=1, sticky=tk.W, padx=4, pady=2
        )
        ttk.Label(self.ai_frame, text="Thickness:").grid(
            row=3, column=2, sticky=tk.W, padx=4, pady=2
        )
        ttk.Entry(self.ai_frame, textvariable=self.outline_thick_var, width=6).grid(
            row=3, column=3, sticky=tk.W, pady=2
        )

        ttk.Checkbutton(
            self.ai_frame, text="Save Composite Video (.mp4)", variable=self.save_video_var
        ).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(
            self.ai_frame, text="Save Extracted Frames (PNG)", variable=self.save_frames_var
        ).grid(row=4, column=2, columnspan=2, sticky=tk.W, pady=2)

        # Button Bar
        btn_bar = ttk.Frame(main_frame)
        btn_bar.pack(fill=tk.X, pady=8)

        ttk.Button(btn_bar, text="❓ Help", command=self._open_help).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_bar, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_bar, text="▶ Run", style="Accent.TButton", command=self._on_run).pack(
            side=tk.RIGHT, padx=4
        )

        self._on_mode_change(None)

    def _on_mode_change(self, _event: Any) -> None:
        mode = self.mode_var.get()
        if mode == "pose":
            self.csv_label.grid()
            self.csv_entry.grid()
            self.csv_btn.grid()
        else:
            self.csv_label.grid_remove()
            self.csv_entry.grid_remove()
            self.csv_btn.grid_remove()

        if mode == "stromotion":
            self.ai_frame.pack(fill=tk.X, padx=8, pady=4)
        else:
            self.ai_frame.pack_forget()

    def _browse_video(self) -> None:
        fn = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")],
            parent=self,
        )
        if fn:
            self.video_var.set(fn)
            if not self.output_var.get():
                self.output_var.set(str(Path(fn).parent))

    def _browse_output(self) -> None:
        dn = filedialog.askdirectory(title="Select Output Directory", parent=self)
        if dn:
            self.output_var.set(dn)

    def _browse_csv(self) -> None:
        fn = filedialog.askopenfilename(
            title="Select Pose CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            parent=self,
        )
        if fn:
            self.csv_var.set(fn)

    def _pick_color(self) -> None:
        color = colorchooser.askcolor(title="Select Outline Color", parent=self)
        if color and color[0]:
            self.outline_color = (int(color[0][0]), int(color[0][1]), int(color[0][2]))

    def _open_help(self) -> None:
        hp = _help_path()
        if hp.is_file():
            webbrowser.open_new_tab(hp.as_uri())
        else:
            webbrowser.open_new_tab("https://github.com/vaila-multimodaltoolbox/vaila")

    def _on_run(self) -> None:
        vid_str = self.video_var.get().strip()
        if not vid_str:
            messagebox.showerror("Error", "Please select an input video.", parent=self)
            return
        vid_path = Path(vid_str).expanduser().resolve()
        if not vid_path.is_file():
            messagebox.showerror("Error", f"Video file not found: {vid_path}", parent=self)
            return

        out_str = self.output_var.get().strip()
        out_path = Path(out_str).expanduser().resolve() if out_str else vid_path.parent

        csv_str = self.csv_var.get().strip()
        csv_path = Path(csv_str).expanduser().resolve() if csv_str else None

        try:
            interval = max(1, int(self.interval_var.get()))
            bg_samples = max(1, int(self.bg_samples_var.get()))
            seg_threshold = float(self.seg_thresh_var.get())
            diff_threshold = max(1, int(self.diff_thresh_var.get()))
            feather_px = max(0, int(self.feather_var.get()))
            outline_thickness = max(1, int(self.outline_thick_var.get()))
            model_selection = int(self.model_sel_var.get())
        except ValueError as exc:
            messagebox.showerror("Error", f"Invalid numeric input: {exc}", parent=self)
            return

        self.result = StroboscopicSettings(
            video_path=vid_path,
            output_dir=out_path,
            csv_path=csv_path,
            mode=self.mode_var.get(),
            interval=interval,
            bg_mode=self.bg_mode_var.get(),
            bg_samples=bg_samples,
            seg_method=self.seg_method_var.get(),
            seg_threshold=seg_threshold,
            diff_threshold=diff_threshold,
            feather_px=feather_px,
            outline=bool(self.outline_var.get()),
            outline_color=self.outline_color,
            outline_thickness=outline_thickness,
            save_video=bool(self.save_video_var.get()),
            save_frames=bool(self.save_frames_var.get()),
            stack_op=self.stack_op_var.get(),
            model_selection=model_selection,
        )
        self.destroy()


def run_stroboscopic(existing_root: Any | None = None) -> None:
    """GUI launcher called from vaila.py or standalone."""
    owns_root = False
    root = existing_root
    if root is None:
        root = tk.Tk()
        root.withdraw()
        owns_root = True

    dialog = StroboscopicDialog(root)
    root.wait_window(dialog)
    settings = dialog.result

    if owns_root:
        root.destroy()

    if settings is None:
        print("Operation canceled by user.")
        return

    cli_cmd = _format_gui_cli(settings)
    print_gui_cli_mirror("vaila/vaila_stroboscopic", cli_cmd)

    if settings.mode == "stromotion":
        generate_stromotion(
            settings.video_path,
            output_dir=settings.output_dir,
            frame_interval=settings.interval,
            bg_mode=settings.bg_mode,
            bg_samples=settings.bg_samples,
            seg_method=settings.seg_method,
            seg_threshold=settings.seg_threshold,
            diff_threshold=settings.diff_threshold,
            feather_px=settings.feather_px,
            outline=settings.outline,
            outline_color=settings.outline_color,
            outline_thickness=settings.outline_thickness,
            save_individual_frames=settings.save_frames,
            save_video=settings.save_video,
            model_selection=settings.model_selection,
        )
    elif settings.mode == "stack":
        generate_stack_multishot(
            settings.video_path,
            output_path=settings.output_dir,
            frame_interval=settings.interval,
            stack_op=settings.stack_op,
        )
    elif settings.mode == "motion":
        generate_motion_stroboscopic(
            settings.video_path,
            output_path=settings.output_dir,
            frame_interval=settings.interval,
        )
    elif settings.mode == "pose":
        generate_stroboscopic_image(
            settings.video_path,
            csv_path=settings.csv_path,
            output_path=settings.output_dir,
            strobe_interval=settings.interval,
        )


# =============================================================================
# CLI PARSER & MAIN
# =============================================================================
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stroboscopic / Stromotion Image & Video Generator (vailá)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--video", type=Path, default=None, help="Path to input video")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Path to output directory or image"
    )
    parser.add_argument("-i", "--interval", type=int, default=10, help="Frame interval")
    parser.add_argument(
        "--mode",
        choices=("stromotion", "pose", "motion", "stack"),
        default="stromotion",
        help="Effect mode",
    )
    parser.add_argument(
        "-c", "--csv", type=Path, default=None, help="Path to pose CSV (for --mode pose)"
    )
    parser.add_argument(
        "--bg-mode",
        choices=("median", "first"),
        default="median",
        help="(stromotion) Background estimation mode",
    )
    parser.add_argument(
        "--bg-samples", type=int, default=15, help="(stromotion) Samples for median BG"
    )
    parser.add_argument(
        "--seg-method",
        choices=("auto", "bg_diff", "ai", "mog2"),
        default="auto",
        help="(stromotion) Segmentation method (auto uses AI with background diff fallback)",
    )
    parser.add_argument(
        "--seg-threshold",
        type=float,
        default=0.35,
        help="(stromotion) AI confidence threshold",
    )
    parser.add_argument(
        "--diff-threshold",
        type=int,
        default=22,
        help="(stromotion) Background difference pixel threshold",
    )
    parser.add_argument(
        "--feather-px", type=int, default=5, help="(stromotion) Edge feathering blur px"
    )
    parser.add_argument(
        "--min-area", type=int, default=150, help="(stromotion) Minimum subject area in pixels"
    )
    parser.add_argument(
        "--outline", action="store_true", help="(stromotion) Draw outline around subjects"
    )
    parser.add_argument(
        "--outline-color", default="255,255,255", help="(stromotion) Outline color R,G,B"
    )
    parser.add_argument(
        "--outline-thickness", type=int, default=2, help="(stromotion) Outline thickness px"
    )
    parser.add_argument("--start-sec", type=float, default=None, help="Start time in seconds")
    parser.add_argument("--end-sec", type=float, default=None, help="End time in seconds")
    parser.add_argument("--no-video", action="store_true", help="(stromotion) Disable video output")
    parser.add_argument(
        "--no-frames", action="store_true", help="(stromotion) Disable individual frames extraction"
    )
    parser.add_argument(
        "--stack-op", choices=("max", "add"), default="max", help="(stack) Stack operator"
    )
    parser.add_argument(
        "--model-selection",
        type=int,
        choices=(0, 1),
        default=1,
        help="(stromotion) 1=Landscape, 0=Square",
    )
    parser.add_argument("--open-help", action="store_true", help="Open documentation in browser")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.open_help:
        hp = _help_path()
        if hp.is_file():
            webbrowser.open_new_tab(hp.as_uri())
        return

    if args.video is None:
        run_stroboscopic()
        return

    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        parser.error(f"Input video file not found: {video_path}")

    # Parse outline color
    outline_color = (255, 255, 255)
    if args.outline_color:
        parts = [int(p.strip()) for p in str(args.outline_color).split(",") if p.strip()]
        if len(parts) == 3:
            outline_color = (parts[0], parts[1], parts[2])

    if args.mode == "stromotion":
        generate_stromotion(
            video_path,
            output_dir=args.output,
            frame_interval=args.interval,
            bg_mode=args.bg_mode,
            bg_samples=args.bg_samples,
            seg_method=args.seg_method,
            seg_threshold=args.seg_threshold,
            diff_threshold=args.diff_threshold,
            feather_px=args.feather_px,
            min_subject_area=args.min_area,
            outline=args.outline,
            outline_color=outline_color,
            outline_thickness=args.outline_thickness,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
            save_individual_frames=not args.no_frames,
            save_video=not args.no_video,
            model_selection=args.model_selection,
        )
    elif args.mode == "stack":
        generate_stack_multishot(video_path, args.output, args.interval, args.stack_op)
    elif args.mode == "motion":
        generate_motion_stroboscopic(video_path, args.output, args.interval)
    elif args.mode == "pose":
        generate_stroboscopic_image(
            video_path, csv_path=args.csv, output_path=args.output, strobe_interval=args.interval
        )


if __name__ == "__main__":
    main()
