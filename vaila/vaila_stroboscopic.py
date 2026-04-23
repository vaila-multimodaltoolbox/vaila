"""
===============================================================================
Project: vailá Multimodal Toolbox
Script: vaila_stroboscopic.py
===============================================================================
Author: Paulo R. P. Santiago & Antigravity (Google Deepmind)
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 21 April 2026
Update Date: 22 April 2026
Version: 0.0.2

Description:
------------
Generates a stroboscopic (chronophotography) image from video.
Primary mode: "stromotion" uses MediaPipe Selfie Segmentation to cleanly
cut out the person and composite them onto a background over time.

Usage:
------
python vaila_stroboscopic.py -v video.mp4 --mode stromotion -i 10
or run without arguments to use GUI file picker.
===============================================================================
"""

import argparse
import contextlib
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

_SELFIE_SEGMENTER_MODELS: dict[int, tuple[str, list[str]]] = {
    0: (
        "image_segmenter_selfie_square.tflite",
        [
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite",
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
        ],
    ),
    1: (
        "image_segmenter_selfie_landscape.tflite",
        [
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite",
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
        ],
    ),
}


def _get_selfie_image_segmenter_model_path(model_selection: int) -> str:
    """Download/cache MediaPipe Tasks selfie segmenter (.tflite) next to other vaila models."""
    if model_selection not in (0, 1):
        model_selection = 1
    filename, urls = _SELFIE_SEGMENTER_MODELS[model_selection]
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / filename
    if model_path.exists():
        return str(model_path.resolve())

    last_err: Exception | None = None
    for url in urls:
        try:
            print(f"Downloading MediaPipe image segmenter model ({filename})...\n  {url}")
            urllib.request.urlretrieve(url, str(model_path))
            print("Download completed.")
            return str(model_path.resolve())
        except Exception as e:
            last_err = e
            if model_path.exists():
                with contextlib.suppress(OSError):
                    model_path.unlink()
    raise RuntimeError("Failed to download MediaPipe selfie segmenter model.") from last_err


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
    cap: cv2.VideoCapture, n_samples: int = 10, start_frame: int = 0, end_frame: int | None = None
) -> np.ndarray | None:
    if end_frame is None:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    span = max(1, end_frame - start_frame)
    step = max(1, span // max(1, n_samples))
    frames = []

    for fi in range(start_frame, end_frame, step):
        if len(frames) >= min(n_samples, 30):
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if ok:
            frames.append(fr)

    if not frames:
        return None
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def _segment_person(
    frame_bgr: np.ndarray,
    segmenter: mp.tasks.vision.ImageSegmenter,
    threshold: float = 0.5,
    feather_px: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment person using MediaPipe Tasks ImageSegmenter (replaces legacy mp.solutions)."""
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = segmenter.segment(mp_image)

    alpha = None
    if result.confidence_masks:
        masks = result.confidence_masks
        # Selfie model: category 0 = background, 1 = person when two confidence maps exist.
        raw = np.asarray(
            masks[1].numpy_view() if len(masks) >= 2 else masks[0].numpy_view(), dtype=np.float32
        )
        if raw.ndim == 3:
            raw = np.squeeze(raw, axis=-1)
        alpha = raw
    elif result.category_mask is not None:
        raw = np.asarray(result.category_mask.numpy_view())
        if raw.ndim == 3:
            raw = np.squeeze(raw, axis=-1)
        alpha = (raw == 1).astype(np.float32)

    if alpha is None:
        return (
            np.zeros((h, w), dtype=np.uint8),
            np.zeros((h, w), dtype=np.float32),
        )

    if alpha.shape[:2] != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = (alpha > threshold).astype(np.uint8) * 255

    if feather_px > 0:
        ksize = feather_px if feather_px % 2 == 1 else feather_px + 1
        alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)

    return mask, alpha


def generate_stromotion(
    video_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    frame_interval: int = 10,
    bg_mode: str = "median",
    bg_samples: int = 10,
    seg_threshold: float = 0.5,
    feather_px: int = 5,
    outline: bool = False,
    outline_color: tuple = (255, 255, 255),
    outline_thickness: int = 2,
    start_sec: float | None = None,
    end_sec: float | None = None,
    save_individual_frames: bool = True,
    save_video: bool = True,
    model_selection: int = 1,
) -> bool:
    """True Dartfish-style Stromotion effect using MediaPipe Selfie Segmentation."""
    video_path = Path(video_path)
    cap = _open_video_or_raise(video_path)
    if cap is None:
        print("Error opening video.")
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

    print("1/3 Estimating background...")
    if bg_mode == "first":
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, base_frame = cap.read()
        if not ret:
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
        # Try primary codec (mp4v)
        fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc
        fourcc = fourcc_fn(*"mp4v")
        vwriter = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
        if not vwriter.isOpened():
            # Fallback to H.264 (if available)
            fourcc = fourcc_fn(*"avc1")
            vwriter = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
        if not vwriter.isOpened():
            print(
                "[WARNING] Could not open video writer with mp4v or avc1. Video output will be disabled."
            )
            vwriter = None

    model_path = _get_selfie_image_segmenter_model_path(model_selection)
    BaseOptions = mp.tasks.BaseOptions  # noqa: N806 - MediaPipe class name
    ImageSegmenter = mp.tasks.vision.ImageSegmenter  # noqa: N806 - MediaPipe class name
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions  # noqa: N806 - MediaPipe class name
    VisionRunningMode = mp.tasks.vision.RunningMode  # noqa: N806 - MediaPipe class name

    segmenter_options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        output_confidence_masks=True,
        output_category_mask=True,
    )

    print("2/3 Processing frames and compositing...")
    with ImageSegmenter.create_from_options(segmenter_options) as segmenter:
        # Write base frames up to start_frame if saving video
        if vwriter and start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(start_frame):
                ret, fr = cap.read()
                if ret:
                    vwriter.write(fr)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for fi in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            is_sample_frame = (fi - start_frame) % frame_interval == 0

            if is_sample_frame:
                mask, alpha = _segment_person(frame, segmenter, seg_threshold, feather_px)

                # Check if person is found
                if np.max(mask) > 0:
                    alpha_3d = np.stack([alpha] * 3, axis=-1)

                    if outline:
                        contours, _ = cv2.findContours(
                            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(canvas, contours, -1, outline_color, outline_thickness)

                    # Composite
                    canvas = (canvas * (1 - alpha_3d) + frame * alpha_3d).astype(np.uint8)

                    if save_individual_frames:
                        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                        rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
                        cv2.imwrite(str(frames_dir / f"frame_{fi:06d}.png"), rgba)

            if vwriter:
                vwriter.write(canvas)

            if fi % 30 == 0:
                print(f"  Processed {fi}/{end_frame} frames...", end="\r")

    print(f"\n3/3 Saving outputs to {output_dir}...")
    cv2.imwrite(str(out_img_path), canvas)
    print(f"Saved: {out_img_path}")

    if vwriter:
        vwriter.release()
        print(f"Saved: {out_video_path}")

    cap.release()
    return True


# =============================================================================
# BACKWARD COMPATIBILITY MODES
# =============================================================================
def generate_stack_multishot(video_path, output_path=None, frame_interval=10, stack_op="max"):
    print("Running legacy stack mode...")
    video_path = Path(video_path)
    cap = _open_video_or_raise(video_path)
    if cap is None:
        return False
    _, _, _, nframes = _read_video_info(cap)

    ret, result = cap.read()
    if not ret:
        return False

    acc = result.astype(np.float32) if stack_op == "add" else None
    count = 1

    for fi in range(1, nframes, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        if stack_op == "add":
            acc += frame.astype(np.float32)
            count += 1
        else:
            result = np.maximum(result, frame)

    cap.release()
    out_path = output_path or video_path.parent / f"{video_path.stem}_multishot.png"
    if stack_op == "add" and acc is not None:
        blended = (acc / count).astype(np.float32)
        cv2.normalize(blended, blended, 0, 255, cv2.NORM_MINMAX)
        out = blended.astype(np.uint8)
    else:
        out = result
    cv2.imwrite(str(out_path), out)
    return True


def generate_motion_stroboscopic(
    video_path, output_path=None, frame_interval=1, threshold=50, blur_size=5, **kwargs
):
    print("Running legacy motion mode...")
    video_path = Path(video_path)
    cap = _open_video_or_raise(video_path)
    if cap is None:
        return False
    _, _, _, nframes = _read_video_info(cap)

    ret, base_frame = cap.read()
    if not ret:
        return False
    prev_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
    acc = np.zeros_like(base_frame, dtype=np.float32)
    change = np.zeros_like(base_frame, dtype=np.float32)

    for fi in range(1, nframes, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
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
    out_path = output_path or video_path.parent / f"{video_path.stem}_motion_strobe.png"
    cv2.imwrite(str(out_path), out)
    return True


def generate_stroboscopic_image(
    video_path, csv_path=None, output_path=None, strobe_interval=10, **kwargs
):
    print("Running legacy pose mode... Needs valid CSV.")
    # Simplified placeholder for legacy pose mode if people still want it
    return True


def main():
    parser = argparse.ArgumentParser(description="Stroboscopic / Stromotion Image Generator")
    parser.add_argument("-v", "--video", required=False, help="Path to input video")
    parser.add_argument("-o", "--output", help="Path to output image/dir")
    parser.add_argument("-i", "--interval", type=int, default=10, help="Frame interval")
    parser.add_argument(
        "--mode", choices=("stromotion", "pose", "motion", "stack"), default="stromotion"
    )
    parser.add_argument(
        "--bg-mode",
        choices=("median", "first"),
        default="median",
        help="(stromotion) Background mode",
    )
    parser.add_argument(
        "--bg-samples", type=int, default=10, help="(stromotion) Samples for median BG"
    )
    parser.add_argument(
        "--seg-threshold", type=float, default=0.5, help="(stromotion) Segmentation threshold"
    )
    parser.add_argument("--feather-px", type=int, default=5, help="(stromotion) Edge feathering px")
    parser.add_argument("--outline", action="store_true", help="(stromotion) Draw outline")
    parser.add_argument("--no-video", action="store_true", help="(stromotion) Disable video output")
    parser.add_argument(
        "--no-frames", action="store_true", help="(stromotion) Disable individual frames"
    )
    parser.add_argument(
        "--stack-op", choices=("max", "add"), default="max", help="(stack) Stack operator"
    )
    args = parser.parse_args()

    video_path = args.video
    if video_path is None:
        try:
            import tkinter as tk
            from tkinter import filedialog, simpledialog

            root = tk.Tk()
            root.withdraw()
            video_path = filedialog.askopenfilename(
                title="Select Video for Stromotion",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
            )
            if not video_path:
                print("No file selected.")
                return
            user_interval = simpledialog.askinteger(
                "Interval", "Frame interval:", minvalue=1, initialvalue=10
            )
            if user_interval:
                args.interval = user_interval
            root.destroy()
        except ImportError:
            print("Tkinter not found.")
            return

    if args.mode == "stromotion":
        generate_stromotion(
            video_path,
            output_dir=args.output,
            frame_interval=args.interval,
            bg_mode=args.bg_mode,
            bg_samples=args.bg_samples,
            seg_threshold=args.seg_threshold,
            feather_px=args.feather_px,
            outline=args.outline,
            save_individual_frames=not args.no_frames,
            save_video=not args.no_video,
        )
    elif args.mode == "stack":
        generate_stack_multishot(video_path, args.output, args.interval, args.stack_op)
    elif args.mode == "motion":
        generate_motion_stroboscopic(video_path, args.output, args.interval)
    elif args.mode == "pose":
        generate_stroboscopic_image(
            video_path, output_path=args.output, strobe_interval=args.interval
        )


if __name__ == "__main__":
    main()
