"""
Project: vailá
Script: rf_trackers.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 14 May 2026
Update Date: 14 May 2026
Version: 0.3.44

Description:
    Video object detection with Ultralytics YOLO26 (default weights: yolo26x.pt) and
    multi-object tracking via
    Roboflow trackers (SORT, ByteTrack, OC-SORT, BoT-SORT 2.4.x + supervision).

    Writes long-format rf_tracks.csv plus yolov26-compatible {Label}_id_XX.csv files and
    all_id_detection.csv for getpixelvideo (bbox/ID load); rec2d still uses pixel CSV after getpixelvideo.

CLI with -i: status lines plus tqdm frame progress bar on stderr; use -q / --quiet to disable.

Usage (show all options):
    uv run python -m vaila.rf_trackers --help

GUI (Tkinter — default when no CLI args):
    uv run python -m vaila.rf_trackers
    Frame B in vaila.py: Video AI tools -> Roboflow trackers (v2.4)

CLI (headless — pass input video):
    uv run python -m vaila.rf_trackers -i VIDEO.mp4 [-w WEIGHTS.pt] [--tracker botsort] ...

Example (FIFA broadcast clip on a local disk):
    uv run python -m vaila.rf_trackers \\
      -i /home/preto/data/FIFA/to_sent/BRA_KOR_234113.mp4 \\
      -w vaila/models/yolo26x.pt --tracker botsort --conf 0.25

Module help (browser / editor):
    vaila/help/rf_trackers.md
    vaila/help/rf_trackers.html

License: AGPL-3.0-or-later
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import re
import sys
import tkinter as tk
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Literal, cast

import cv2
import numpy as np

try:
    from .yolov26track import (
        VAILA_MODELS_DIR,
        _configure_ultralytics_dirs,
        create_combined_detection_csv,
        get_color_for_id,
    )
except ImportError:  # pragma: no cover
    from yolov26track import (  # ty: ignore[unresolved-import]
        VAILA_MODELS_DIR,
        _configure_ultralytics_dirs,
        create_combined_detection_csv,
        get_color_for_id,
    )

_configure_ultralytics_dirs(VAILA_MODELS_DIR)

# Default detector: YOLO26 x-large (aligned with markerless2d_mpyolo / Frame B v26 tools).
DEFAULT_RF_YOLO_WEIGHTS = VAILA_MODELS_DIR / "yolo26x.pt"

TrackerName = Literal["sort", "bytetrack", "ocsort", "botsort"]
CmcMethod = Literal["sparseOptFlow", "orb", "sift", "ecc"]


def _import_trackers() -> Any:
    return importlib.import_module("trackers")


def build_tracker(
    name: TrackerName,
    *,
    enable_cmc: bool = True,
    cmc_method: CmcMethod = "sparseOptFlow",
) -> Any:
    """Construct a Roboflow trackers instance by id (for GUI and tests)."""
    tr = _import_trackers()
    match name:
        case "sort":
            return tr.SORTTracker()
        case "bytetrack":
            return tr.ByteTrackTracker()
        case "ocsort":
            return tr.OCSORTTracker()
        case "botsort":
            return tr.BoTSORTTracker(enable_cmc=enable_cmc, cmc_method=cmc_method)
        case _:
            raise ValueError(f"Unknown tracker: {name}")


@dataclass(frozen=True)
class RfTrackersConfig:
    video_path: Path
    weights_path: Path
    tracker: TrackerName
    conf: float
    save_video: bool
    enable_cmc: bool
    cmc_method: CmcMethod
    show_progress: bool = False


def _class_name(names: dict[int, str], class_id: int | None) -> str:
    if class_id is None:
        return ""
    return str(names.get(int(class_id), str(class_id)))


def _cmc_method_from_ui(raw: str) -> CmcMethod:
    key = raw.strip().lower().replace("_", "")
    lookup: dict[str, CmcMethod] = {
        "sparseoptflow": "sparseOptFlow",
        "orb": "orb",
        "sift": "sift",
        "ecc": "ecc",
    }
    if key not in lookup:
        msg = f"Invalid CMC method: {raw!r}"
        raise ValueError(msg)
    return lookup[key]


def _sanitize_label_for_filename(label: str) -> str:
    raw = (label or "").strip()
    safe = re.sub(r"[^0-9a-zA-Z_-]+", "_", raw).strip("_")
    return safe or "object"


def _write_vaila_per_id_tracking_csvs(
    out_dir: Path,
    n_frames: int,
    rows_out: list[list[Any]],
    *,
    show_progress: bool,
) -> Path | None:
    """Write {label}_id_XX.csv + all_id_detection.csv matching yolov26track / getpixelvideo."""
    if n_frames <= 0 or not rows_out:
        return None

    # (track_id, frame) -> best row by confidence
    cell: dict[tuple[int, int], tuple[float, float, float, float, float, str]] = {}
    names_by_tid: dict[int, list[str]] = {}

    for row in rows_out:
        if len(row) < 9:
            continue
        fr, tid, x1, y1, x2, y2, conf, _cid, cname = row[:9]
        fr_i = int(fr)
        tid_i = int(tid)
        cname_s = str(cname).strip() if cname not in (None, "") else "object"
        names_by_tid.setdefault(tid_i, []).append(cname_s)
        key = (tid_i, fr_i)
        conf_f = float(conf)
        new_tup = (float(x1), float(y1), float(x2), float(y2), conf_f, cname_s)
        prev = cell.get(key)
        if prev is None or conf_f > prev[4]:
            cell[key] = new_tup

    tid_to_label: dict[int, str] = {}
    for tid_i, names in names_by_tid.items():
        tid_to_label[tid_i] = Counter(names).most_common(1)[0][0]

    header = [
        "Frame",
        "Tracker ID",
        "Label",
        "X_min",
        "Y_min",
        "X_max",
        "Y_max",
        "Confidence",
        "Color_R",
        "Color_G",
        "Color_B",
    ]

    for tid_i in sorted(tid_to_label):
        label_raw = tid_to_label[tid_i]
        label_safe = _sanitize_label_for_filename(label_raw)
        out_csv = out_dir / f"{label_safe}_id_{int(tid_i):02d}.csv"
        color_bgr = get_color_for_id(tid_i)
        color_r, color_g, color_b = int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])

        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for frame_i in range(n_frames):
                hit = cell.get((tid_i, frame_i))
                if hit is not None:
                    x1, y1, x2, y2, conf_f, _lab = hit
                    w.writerow(
                        [
                            frame_i,
                            tid_i,
                            label_raw,
                            x1,
                            y1,
                            x2,
                            y2,
                            conf_f,
                            color_r,
                            color_g,
                            color_b,
                        ]
                    )
                else:
                    w.writerow(
                        [
                            frame_i,
                            tid_i,
                            label_raw,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            color_r,
                            color_g,
                            color_b,
                        ]
                    )

    combined_path = create_combined_detection_csv(str(out_dir))
    if show_progress:
        if combined_path:
            print(
                f"[rf_trackers] vailá tracking: all_id_detection.csv -> {combined_path}",
                file=sys.stderr,
            )
            print(
                f"[rf_trackers] vailá per-ID CSVs (yolov26-style filenames) in {out_dir}",
                file=sys.stderr,
            )
        else:
            print(
                "[rf_trackers] vailá combined CSV not created (no per-ID files?)", file=sys.stderr
            )

    return Path(combined_path) if combined_path else None


def build_arg_parser() -> argparse.ArgumentParser:
    epilog = f"""examples:
  GUI (no arguments):
    {sys.executable} -m vaila.rf_trackers

  CLI with local FIFA sample:
    uv run python -m vaila.rf_trackers \\
      -i /home/preto/data/FIFA/to_sent/BRA_KOR_234113.mp4 \\
      -w vaila/models/yolo26x.pt --tracker botsort --conf 0.25

  Human-readable help in-repo:
    vaila/help/rf_trackers.md
    vaila/help/rf_trackers.html
"""
    p = argparse.ArgumentParser(
        prog="python -m vaila.rf_trackers",
        description=(
            "YOLO (Ultralytics) + Roboflow trackers 2.4.x: SORT, ByteTrack, OC-SORT, BoT-SORT. "
            "Omit -i/--input to open the GUI."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Input video file. If omitted, starts the Tkinter GUI.",
    )
    p.add_argument(
        "-w",
        "--weights",
        type=Path,
        default=DEFAULT_RF_YOLO_WEIGHTS,
        help=(
            "YOLO detection .pt (default: yolo26x.pt under vaila/models per Ultralytics bootstrap)."
        ),
    )
    p.add_argument(
        "--tracker",
        choices=("sort", "bytetrack", "ocsort", "botsort"),
        default="botsort",
        help="Roboflow tracker backend (default: botsort)",
    )
    p.add_argument(
        "--conf", type=float, default=0.25, help="YOLO confidence threshold (default: 0.25)"
    )
    p.add_argument(
        "--no-save-video",
        action="store_true",
        help="Do not write overlay MP4 (still writes rf_tracks.csv).",
    )
    p.add_argument(
        "--no-cmc",
        action="store_true",
        help="BoT-SORT only: disable camera motion compensation.",
    )
    p.add_argument(
        "--cmc-method",
        default="sparseOptFlow",
        choices=("sparseOptFlow", "orb", "sift", "ecc"),
        help="BoT-SORT CMC backend when CMC is enabled (default: sparseOptFlow).",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="No stderr progress bar or status lines (stdout result line unchanged).",
    )
    return p


def run_tracking_pipeline(cfg: RfTrackersConfig) -> Path:
    """Process video; write CSV (+ optional overlay video) under timestamped output dir."""
    from tqdm import tqdm
    from ultralytics import YOLO

    video_path = cfg.video_path
    if not video_path.is_file():
        msg = f"Video not found: {video_path}"
        raise FileNotFoundError(msg)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = video_path.parent / f"processed_rf_trackers_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        msg = f"Could not open video: {video_path}"
        raise OSError(msg)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_hint = n_frames_meta if n_frames_meta > 0 else None

    if cfg.show_progress:
        print("[rf_trackers] ---", file=sys.stderr)
        print(f"[rf_trackers] Video: {video_path}", file=sys.stderr)
        nf_txt = str(n_frames_meta) if n_frames_meta > 0 else "unknown"
        print(
            f"[rf_trackers] Stream: {width}x{height} @ {fps:.3f} fps, frames~{nf_txt}",
            file=sys.stderr,
        )
        print(f"[rf_trackers] Output: {out_dir}", file=sys.stderr)
        print(f"[rf_trackers] Weights: {cfg.weights_path}", file=sys.stderr)
        print(
            f"[rf_trackers] Tracker={cfg.tracker} conf={cfg.conf} save_video={cfg.save_video}",
            file=sys.stderr,
        )
        if cfg.tracker == "botsort":
            print(
                f"[rf_trackers] BoT-SORT CMC enabled={cfg.enable_cmc} method={cfg.cmc_method}",
                file=sys.stderr,
            )

    writer: cv2.VideoWriter | None = None
    if cfg.save_video:
        out_mp4 = out_dir / f"{video_path.stem}_rf_tracked.mp4"
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")  # ty: ignore[unresolved-attribute]
        writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (width, height))

    if cfg.show_progress:
        print("[rf_trackers] Loading YOLO model...", file=sys.stderr)
    model = YOLO(str(cfg.weights_path))
    tracker = build_tracker(
        cfg.tracker,
        enable_cmc=cfg.enable_cmc,
        cmc_method=cfg.cmc_method,
    )

    import supervision as sv

    box_annotator = sv.BoxAnnotator()

    csv_path = out_dir / "rf_tracks.csv"
    rows_out: list[list[Any]] = []

    frame_idx = 0
    pbar = tqdm(
        total=total_hint,
        unit="fr",
        desc="rf_trackers",
        disable=not cfg.show_progress,
        file=sys.stderr,
        dynamic_ncols=True,
    )
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_u8 = np.ascontiguousarray(frame, dtype=np.uint8)

            results = model.predict(frame_u8, conf=cfg.conf, verbose=False)
            raw = results[0]
            detections = sv.Detections.from_ultralytics(raw)
            names: dict[int, str] = raw.names if isinstance(raw.names, dict) else dict(raw.names)

            tracked = tracker.update(detections, frame_u8)

            if len(tracked) and tracked.tracker_id is not None:
                ann = frame_u8.copy()
                ann = box_annotator.annotate(ann, tracked)
                for j in range(len(tracked)):
                    x1, y1, _, _ = tracked.xyxy[j].astype(int)
                    tid = int(tracked.tracker_id[j])
                    cid = int(tracked.class_id[j]) if tracked.class_id is not None else None
                    cn = _class_name(names, cid)
                    lab = f"#{tid} {cn}".strip()
                    y_txt = int(max(0, y1 - 4))
                    cv2.putText(
                        ann,
                        lab,
                        (int(x1), y_txt),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
            else:
                ann = frame_u8

            if writer is not None:
                writer.write(np.ascontiguousarray(ann, dtype=np.uint8))

            if len(tracked) and tracked.tracker_id is not None:
                for j in range(len(tracked)):
                    x1, y1, x2, y2 = tracked.xyxy[j].tolist()
                    tid = int(tracked.tracker_id[j])
                    cid = int(tracked.class_id[j]) if tracked.class_id is not None else -1
                    conf_f = (
                        float(tracked.confidence[j])
                        if tracked.confidence is not None
                        else float("nan")
                    )
                    rows_out.append(
                        [
                            frame_idx,
                            tid,
                            x1,
                            y1,
                            x2,
                            y2,
                            conf_f,
                            cid,
                            _class_name(names, cid),
                        ]
                    )

            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        if writer is not None:
            writer.release()

    headers = [
        "frame",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence",
        "class_id",
        "class_name",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows_out)

    _write_vaila_per_id_tracking_csvs(
        out_dir,
        frame_idx,
        rows_out,
        show_progress=cfg.show_progress,
    )
    if cfg.show_progress:
        print(f"[rf_trackers] Frames processed: {frame_idx}", file=sys.stderr)
        print(
            f"[rf_trackers] CSV rows (detections): {len(rows_out)} -> {csv_path}", file=sys.stderr
        )
        if cfg.save_video:
            mp4_path = out_dir / f"{video_path.stem}_rf_tracked.mp4"
            print(f"[rf_trackers] Overlay video: {mp4_path}", file=sys.stderr)
        print("[rf_trackers] ---", file=sys.stderr)

    return out_dir


def run_rf_trackers_gui() -> None:
    root = tk.Tk()
    root.title("Roboflow trackers + YOLO")

    video_var = tk.StringVar()
    weights_var = tk.StringVar(value=str(DEFAULT_RF_YOLO_WEIGHTS))
    tracker_var = tk.StringVar(value="botsort")
    conf_var = tk.StringVar(value="0.25")
    save_vid_var = tk.BooleanVar(value=True)
    cmc_var = tk.BooleanVar(value=True)
    cmc_method_var = tk.StringVar(value="sparseOptFlow")

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)

    def browse_video() -> None:
        p = filedialog.askopenfilename(
            parent=root,
            title="Select video",
            filetypes=[
                ("Video", "*.mp4 *.avi *.mov *.mkv"),
                ("All", "*.*"),
            ],
        )
        if p:
            video_var.set(p)

    def browse_weights() -> None:
        p = filedialog.askopenfilename(
            parent=root,
            title="Select YOLO weights (.pt)",
            filetypes=[("Weights", "*.pt"), ("All", "*.*")],
        )
        if p:
            weights_var.set(p)

    r0 = ttk.Frame(frm)
    r0.pack(fill="x", pady=2)
    ttk.Label(r0, text="Video:").pack(side="left")
    ttk.Entry(r0, textvariable=video_var, width=48).pack(side="left", padx=4)
    ttk.Button(r0, text="Browse", command=browse_video).pack(side="left")

    r1 = ttk.Frame(frm)
    r1.pack(fill="x", pady=2)
    ttk.Label(r1, text="YOLO weights:").pack(side="left")
    ttk.Entry(r1, textvariable=weights_var, width=48).pack(side="left", padx=4)
    ttk.Button(r1, text="Browse", command=browse_weights).pack(side="left")

    r2 = ttk.Frame(frm)
    r2.pack(fill="x", pady=2)
    ttk.Label(r2, text="Tracker:").pack(side="left")
    ttk.Combobox(
        r2,
        textvariable=tracker_var,
        values=("sort", "bytetrack", "ocsort", "botsort"),
        state="readonly",
        width=14,
    ).pack(side="left", padx=4)

    ttk.Label(r2, text="conf:").pack(side="left", padx=(12, 0))
    ttk.Entry(r2, textvariable=conf_var, width=8).pack(side="left", padx=4)

    r3 = ttk.Frame(frm)
    r3.pack(fill="x", pady=2)
    ttk.Checkbutton(r3, text="Save overlay video", variable=save_vid_var).pack(side="left")
    ttk.Checkbutton(
        r3,
        text="BoT-SORT CMC (moving camera)",
        variable=cmc_var,
    ).pack(side="left", padx=12)

    r4 = ttk.Frame(frm)
    r4.pack(fill="x", pady=2)
    ttk.Label(r4, text="CMC method:").pack(side="left")
    ttk.Combobox(
        r4,
        textvariable=cmc_method_var,
        values=("sparseOptFlow", "orb", "sift", "ecc"),
        state="readonly",
        width=14,
    ).pack(side="left", padx=4)

    ttk.Label(
        frm,
        text="Uses Roboflow trackers 2.4.x. Passes frame to tracker.update for CMC.",
        wraplength=420,
    ).pack(pady=6)

    def on_go() -> None:
        try:
            _import_trackers()
        except ImportError as e:
            messagebox.showerror(
                "Missing dependency",
                "Install Roboflow trackers:\n  uv add trackers==2.4.0\n\n" + str(e),
                parent=root,
            )
            return

        vp = Path(video_var.get().strip())
        wp = Path(weights_var.get().strip())
        if not vp.is_file():
            messagebox.showerror("Error", f"Video not found:\n{vp}", parent=root)
            return
        if not wp.is_file():
            messagebox.showerror("Error", f"Weights not found:\n{wp}", parent=root)
            return

        try:
            conf = float(conf_var.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Confidence must be a number.", parent=root)
            return

        tname = tracker_var.get().strip().lower()
        if tname not in ("sort", "bytetrack", "ocsort", "botsort"):
            messagebox.showerror("Error", f"Invalid tracker: {tname}", parent=root)
            return

        cm_raw = cmc_method_var.get().strip()
        try:
            cmc_norm = _cmc_method_from_ui(cm_raw)
        except ValueError:
            messagebox.showerror("Error", f"Invalid CMC method: {cm_raw}", parent=root)
            return

        cfg = RfTrackersConfig(
            video_path=vp,
            weights_path=wp,
            tracker=cast(TrackerName, tname),
            conf=conf,
            save_video=bool(save_vid_var.get()),
            enable_cmc=bool(cmc_var.get()) if tname == "botsort" else False,
            cmc_method=cmc_norm,
        )

        root.withdraw()
        try:
            out = run_tracking_pipeline(cfg)
            msg = f"Done.\nOutput:\n{out}"
            messagebox.showinfo("Roboflow trackers", msg, parent=root)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", str(e), parent=root)
        finally:
            root.destroy()

    ttk.Button(frm, text="Run", command=on_go).pack(pady=8)
    ttk.Button(frm, text="Quit", command=root.destroy).pack()

    root.mainloop()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.input is None:
        run_rf_trackers_gui()
        return

    try:
        _import_trackers()
    except ImportError as e:
        print(
            "Missing dependency: install Roboflow trackers (see `uv add trackers==2.4.0`).",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        sys.exit(1)

    vp = args.input.expanduser().resolve()
    wp = args.weights.expanduser().resolve()
    if not vp.is_file():
        parser.error(f"Video not found: {vp}")
    if not wp.is_file():
        parser.error(f"Weights not found: {wp}")

    tname = cast(TrackerName, args.tracker)
    cfg = RfTrackersConfig(
        video_path=vp,
        weights_path=wp,
        tracker=tname,
        conf=args.conf,
        save_video=not args.no_save_video,
        enable_cmc=(not args.no_cmc) if tname == "botsort" else False,
        cmc_method=cast(CmcMethod, args.cmc_method),
        show_progress=not args.quiet,
    )
    out_dir = run_tracking_pipeline(cfg)
    print(f"[rf_trackers] Done. Output directory:\n{out_dir}")


if __name__ == "__main__":
    main()
