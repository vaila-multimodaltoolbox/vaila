"""AI soccer-field keypoints (pixels) for vailá.

Goal
----
Provide an **automatic first pass** of soccer-field keypoints in pixel space to seed:
- manual corrections in :mod:`vaila.getpixelvideo`
- homography fitting in :mod:`vaila.soccerfield_calib`
- dataset building for YOLOv26 fine-tuning (:mod:`vaila.yolotrain`)

Important note about the Hugging Face model
------------------------------------------
The Hugging Face repo ``Simon9/football-field-detection-roboflow`` is a *wrapper* around
Roboflow Inference and does **not** ship a standalone `.pt` weight file. See:
`https://huggingface.co/Simon9/football-field-detection-roboflow/tree/main`.

Therefore this module supports two backends:
1) **Roboflow Inference** (needs `inference` + `supervision` + ROBOFLOW_API_KEY)
2) **Ultralytics local weights** (user provides a local `.pt` pose model)

Outputs
-------
Writes a timestamped folder:
- ``field_keypoints_raw.csv``: rows ``kp_00..kp_N`` with x,y,conf
- ``field_keypoints_template.csv``: rows for each ``point_name`` in `models/soccerfield_ref3d.csv`
  with empty x,y (meant to be filled/renamed in getpixelvideo)
- ``frame_XXXXXX_overlay.png``: overlay visualization for review
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import os
import tkinter as tk
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import pandas as pd

try:
    from .soccerfield_calib import load_field_reference
except ImportError:
    from soccerfield_calib import load_field_reference  # ty: ignore[unresolved-import]


HF_REPO_ID = "Simon9/football-field-detection-roboflow"
ROBOFLOW_DEFAULT_MODEL_ID = "football-field-detection-f07vi/14"


@dataclass(frozen=True)
class DetectedKp:
    name: str
    x: float
    y: float
    conf: float


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n > 0:
        frame_index = int(np.clip(frame_index, 0, max(0, n - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        raise OSError(f"Could not read frame {frame_index} from {video_path}")
    return bgr


def _draw_kps(bgr: np.ndarray, kps: list[DetectedKp]) -> np.ndarray:
    out = bgr.copy()
    for kp in kps:
        if not np.isfinite(kp.x) or not np.isfinite(kp.y):
            continue
        x = int(round(kp.x))
        y = int(round(kp.y))
        c = (0, 255, 0) if kp.conf >= 0.5 else (0, 165, 255)
        cv2.circle(out, (x, y), 5, c, -1)
        cv2.putText(
            out,
            f"{kp.name}:{kp.conf:.2f}",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            c,
            1,
            cv2.LINE_AA,
        )
    return out


def _backend_roboflow(
    bgr: np.ndarray, *, api_key: str, model_id: str, conf: float
) -> list[DetectedKp]:
    """Roboflow Inference backend (remote or local provider depending on inference config)."""
    try:
        get_model = importlib.import_module("inference").get_model  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError(
            "Roboflow backend requires the `inference` package.\n\n"
            "Install with:\n"
            "  uv add inference\n"
            "  uv add supervision\n"
        ) from e
    try:
        sv = importlib.import_module("supervision")
    except Exception as e:
        raise RuntimeError(
            "Roboflow backend requires `supervision`.\n\nInstall with:\n  uv add supervision\n"
        ) from e

    try:
        model = get_model(model_id=model_id, api_key=api_key)
    except Exception as e:
        # Common failure: 401 Unauthorized (invalid key / not authorized for model)
        name = type(e).__name__
        msg = str(e)
        if "Unauthorized" in msg or "NotAuthorized" in name or "401" in msg:
            raise RuntimeError(
                "Roboflow API unauthorized (HTTP 401).\n\n"
                "Fix checklist:\n"
                "  - Make sure ROBOFLOW_API_KEY is valid (regen in Roboflow if leaked).\n"
                "  - Confirm your account has access to the model_id you requested.\n"
                "  - Re-check model_id spelling (expected e.g. football-field-detection-f07vi/14).\n\n"
                "Offline alternative (no Roboflow API): export/download a YOLO pose `.pt` "
                "and run this module with `--backend ultralytics --weights /path/model.pt`.\n\n"
                f"Details: {name}: {msg}"
            ) from e
        raise
    # inference accepts np arrays; keep BGR for simplicity (wrapper usually handles it)
    result = model.infer(bgr, confidence=float(conf))[0]
    key_points = sv.KeyPoints.from_inference(result)
    if key_points.xy is None or len(key_points.xy) == 0:
        return []
    xy = key_points.xy[0]
    cf = key_points.confidence[0] if key_points.confidence is not None else None
    out: list[DetectedKp] = []
    for i in range(xy.shape[0]):
        out.append(
            DetectedKp(
                name=f"kp_{i:02d}",
                x=float(xy[i, 0]),
                y=float(xy[i, 1]),
                conf=float(cf[i]) if cf is not None else float("nan"),
            )
        )
    return out


def _backend_ultralytics(
    bgr: np.ndarray, *, weights: Path, conf: float, imgsz: int, device: str | int | None = None
) -> list[DetectedKp]:
    """Ultralytics backend (local `.pt` weights)."""
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "Ultralytics is not installed. Install with `uv add ultralytics`."
        ) from e

    model = YOLO(str(weights))
    res = model.predict(bgr, imgsz=int(imgsz), conf=float(conf), device=device, verbose=False)
    if not res or res[0].keypoints is None:
        return []
    # Take the best detection (highest mean confidence)
    kpt = res[0].keypoints
    data_obj = getattr(kpt, "data", None)
    if data_obj is None:
        return []
    data = data_obj.detach().cpu().numpy() if hasattr(data_obj, "detach") else np.asarray(data_obj)
    if data.size == 0:
        return []
    best = max(range(data.shape[0]), key=lambda i: float(np.nanmean(data[i, :, 2])))
    pts = data[best]
    out: list[DetectedKp] = []
    for i in range(pts.shape[0]):
        out.append(
            DetectedKp(
                name=f"kp_{i:02d}", x=float(pts[i, 0]), y=float(pts[i, 1]), conf=float(pts[i, 2])
            )
        )
    return out


def _resolve_ultralytics_weights_path(p: Path) -> Path | None:
    """Resolve weights path; tolerate Ultralytics auto-suffixed run dirs (e.g. `name-3`)."""
    try:
        pr = p.expanduser().resolve()
        if pr.is_file():
            return pr
    except Exception:
        pr = p

    # If a user passed ".../<run_name>/weights/best.pt" but Ultralytics created "<run_name>-N",
    # try to find it under the repo root.
    run_name = p.parent.parent.name if p.parent.name == "weights" else p.stem
    repo_root = Path(__file__).resolve().parents[1]
    pattern = f"**/{run_name}*/weights/{p.name}"
    try:
        cands = list(repo_root.rglob(pattern))
    except Exception:
        cands = []
    if not cands:
        return None

    # Pick most recently modified candidate
    best = max(cands, key=lambda q: q.stat().st_mtime if q.exists() else 0.0)
    return best


def detect_field_keypoints(
    video_path: Path,
    *,
    frame_index: int = 0,
    backend: str = "roboflow",
    roboflow_api_key: str | None = None,
    roboflow_model_id: str = ROBOFLOW_DEFAULT_MODEL_ID,
    ultralytics_weights: Path | None = None,
    conf: float = 0.3,
    imgsz: int = 1280,
    device: str | int | None = None,
    draw_min_conf: float = 0.3,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    bgr = _read_frame(video_path, frame_index)

    if backend == "roboflow":
        api_key = (roboflow_api_key or os.environ.get("ROBOFLOW_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(
                "Roboflow backend selected but no API key was provided.\n\n"
                "Either:\n"
                "  - set ROBOFLOW_API_KEY in your environment, or\n"
                "  - enter the API key in the GUI (Roboflow backend).\n\n"
                "Alternative: use backend=ultralytics with local weights."
            )
        kps = _backend_roboflow(bgr, api_key=api_key, model_id=roboflow_model_id, conf=conf)
    elif backend == "ultralytics":
        if ultralytics_weights is None:
            raise RuntimeError("Ultralytics backend requires --weights PATH_TO_MODEL.pt")
        resolved_w = _resolve_ultralytics_weights_path(ultralytics_weights)
        if resolved_w is None or not resolved_w.is_file():
            raise FileNotFoundError(
                f"Ultralytics weights not found: {ultralytics_weights}\n"
                f"Tip: Ultralytics may suffix the run dir (e.g. '{ultralytics_weights.parent.parent.name}-3')."
            )
        kps = _backend_ultralytics(
            bgr, weights=resolved_w, conf=conf, imgsz=imgsz, device=device
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    # Raw CSV
    raw_path = out_dir / "field_keypoints_raw.csv"
    pd.DataFrame([{"name": kp.name, "x": kp.x, "y": kp.y, "conf": kp.conf} for kp in kps]).to_csv(
        raw_path, index=False
    )

    # vailá getpixelvideo format (single frame)
    gv_path = out_dir / "field_keypoints_getpixelvideo.csv"
    max_points = len(kps)
    cols = ["frame"]
    for i in range(1, max_points + 1):
        cols += [f"p{i}_x", f"p{i}_y"]
    df_gv = pd.DataFrame(np.nan, index=[int(frame_index)], columns=cols)
    df_gv["frame"] = int(frame_index)
    for i, kp in enumerate(kps, start=1):
        if float(kp.conf) >= float(draw_min_conf):
            df_gv.at[int(frame_index), f"p{i}_x"] = float(kp.x)
            df_gv.at[int(frame_index), f"p{i}_y"] = float(kp.y)
    df_gv.to_csv(gv_path, index=False, na_rep="")

    # Overlay markers CSV: same schema as getpixelvideo, but semantically "what was drawn/saved"
    markers_path = out_dir / "field_keypoints_overlay_markers.csv"
    df_gv.to_csv(markers_path, index=False, na_rep="")
    # Template for vailá point names (to be filled/renamed after manual correction)
    ref = load_field_reference()
    template_path = out_dir / "field_keypoints_template.csv"
    pd.DataFrame([{"name": kp.name, "x": np.nan, "y": np.nan} for kp in ref]).to_csv(
        template_path, index=False
    )

    overlay = _draw_kps(bgr, [kp for kp in kps if float(kp.conf) >= float(draw_min_conf)])
    overlay_path = out_dir / f"frame_{frame_index:06d}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    # Short README
    readme = out_dir / "README_field_keypoints.txt"
    readme.write_text(
        "\n".join(
            [
                "Soccer-field keypoints (AI seed)",
                f"video={video_path.resolve()}",
                f"frame_index={frame_index}",
                f"backend={backend}",
                f"roboflow_model_id={roboflow_model_id}",
                f"hf_repo={HF_REPO_ID}",
                f"raw_csv={raw_path.name}",
                f"getpixelvideo_csv={gv_path.name}",
                f"overlay_markers_csv={markers_path.name}",
                f"template_csv={template_path.name}",
                f"overlay={overlay_path.name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return out_dir


def detect_field_keypoints_video(
    video_path: Path,
    *,
    backend: str = "roboflow",
    roboflow_api_key: str | None = None,
    roboflow_model_id: str = ROBOFLOW_DEFAULT_MODEL_ID,
    ultralytics_weights: Path | None = None,
    conf: float = 0.3,
    imgsz: int = 1280,
    device: str | int | None = None,
    draw_min_conf: float = 0.3,
    out_dir: Path,
    start: int = 0,
    stride: int = 10,
    max_frames: int | None = None,
    write_overlay_video: bool = False,
) -> Path:
    """Run inference across a video and write one CSV with per-frame keypoints."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = n if n > 0 else None

    start_i = max(0, int(start))
    stride_i = max(1, int(stride))
    max_frames_i = None if max_frames is None else max(1, int(max_frames))

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_i))

    writer: cv2.VideoWriter | None = None
    overlay_path = out_dir / "field_keypoints_overlay.mp4"
    if write_overlay_video:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if w > 0 and h > 0:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(overlay_path), fourcc, float(fps / stride_i), (w, h))

    rows: list[dict[str, object]] = []
    # getpixelvideo format: one row per frame, columns p1_x/p1_y..pK_x/pK_y
    gv_cols = ["frame"] + [f"p{i}_x" for i in range(1, 33)] + [f"p{i}_y" for i in range(1, 33)]
    # Fix ordering to p1_x,p1_y,p2_x,p2_y...
    gv_cols = ["frame"] + [c for i in range(1, 33) for c in (f"p{i}_x", f"p{i}_y")]
    total_frames = n_frames if n_frames is not None else (start_i + stride_i * (max_frames_i or 0))
    df_gv = pd.DataFrame(np.nan, index=range(int(total_frames)), columns=gv_cols)
    df_gv["frame"] = df_gv.index
    processed = 0
    frame_index = start_i
    while True:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break

        if (frame_index - start_i) % stride_i == 0:
            print(
                f"[field_kps] frame={frame_index}"
                f"{'' if n_frames is None else f'/{n_frames - 1}'}"
                f" processed={processed + 1}"
                f"{'' if max_frames_i is None else f'/{max_frames_i}'}"
            )
            if backend == "roboflow":
                api_key = (roboflow_api_key or os.environ.get("ROBOFLOW_API_KEY") or "").strip()
                if not api_key:
                    cap.release()
                    raise RuntimeError(
                        "Roboflow backend selected but no API key was provided.\n\n"
                        "Either:\n"
                        "  - set ROBOFLOW_API_KEY in your environment, or\n"
                        "  - enter the API key in the GUI (Roboflow backend).\n\n"
                        "Alternative: use backend=ultralytics with local weights."
                    )
                kps = _backend_roboflow(bgr, api_key=api_key, model_id=roboflow_model_id, conf=conf)
            elif backend == "ultralytics":
                if ultralytics_weights is None:
                    cap.release()
                    raise RuntimeError("Ultralytics backend requires --weights PATH_TO_MODEL.pt")
                resolved_w = _resolve_ultralytics_weights_path(ultralytics_weights)
                if resolved_w is None or not resolved_w.is_file():
                    cap.release()
                    raise FileNotFoundError(
                        f"Ultralytics weights not found: {ultralytics_weights}\n"
                        f"Tip: Ultralytics may suffix the run dir (e.g. '{ultralytics_weights.parent.parent.name}-3')."
                    )
                kps = _backend_ultralytics(
                    bgr, weights=resolved_w, conf=conf, imgsz=imgsz, device=device
                )
            else:
                cap.release()
                raise ValueError(f"Unknown backend: {backend!r}")

            for kp in kps:
                rows.append(
                    {
                        "frame": int(frame_index),
                        "name": kp.name,
                        "x": float(kp.x),
                        "y": float(kp.y),
                        "conf": float(kp.conf),
                    }
                )

            if writer is not None:
                writer.write(_draw_kps(bgr, [kp for kp in kps if float(kp.conf) >= float(draw_min_conf)]))

            # Fill getpixelvideo row (blank for low-confidence points)
            if 0 <= int(frame_index) < len(df_gv):
                for i, kp in enumerate(kps, start=1):
                    if i > 32:
                        break
                    if float(kp.conf) >= float(draw_min_conf):
                        df_gv.at[int(frame_index), f"p{i}_x"] = float(kp.x)
                        df_gv.at[int(frame_index), f"p{i}_y"] = float(kp.y)

            processed += 1
            if max_frames_i is not None and processed >= max_frames_i:
                break

        frame_index += 1
        if n_frames is not None and frame_index >= n_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()

    out_csv = out_dir / "field_keypoints_video.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_gv = out_dir / "field_keypoints_getpixelvideo.csv"
    df_gv.to_csv(out_gv, index=False, na_rep="")

    out_markers = out_dir / "field_keypoints_overlay_markers.csv"
    df_gv.to_csv(out_markers, index=False, na_rep="")
    readme = out_dir / "README_field_keypoints_video.txt"
    readme.write_text(
        "\n".join(
            [
                "Soccer-field keypoints (AI seed) - VIDEO MODE",
                f"video={video_path.resolve()}",
                f"backend={backend}",
                f"roboflow_model_id={roboflow_model_id}",
                f"hf_repo={HF_REPO_ID}",
                f"start={start_i}",
                f"stride={stride_i}",
                f"max_frames={max_frames_i}",
                f"out_csv={out_csv.name}",
                f"getpixelvideo_csv={out_gv.name}",
                f"overlay_markers_csv={out_markers.name}",
                f"overlay_video={overlay_path.name if write_overlay_video else ''}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return out_dir


def run_gui() -> None:
    root = tk.Tk()
    root.withdraw()

    video = filedialog.askopenfilename(
        title="Select soccer video (broadcast)",
        filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")],
    )
    if not video:
        return
    out_parent = filedialog.askdirectory(title="Select output folder")
    if not out_parent:
        return

    dlg = tk.Toplevel(root)
    dlg.title("Soccer field keypoints (AI)")
    frm = ttk.Frame(dlg, padding=10)
    frm.grid(row=0, column=0, sticky="nsew")

    mode_var = tk.StringVar(value="frame")
    backend_var = tk.StringVar(value="roboflow")
    frame_var = tk.StringVar(value="0")
    start_var = tk.StringVar(value="0")
    stride_var = tk.StringVar(value="10")
    max_frames_var = tk.StringVar(value="")
    overlay_video_var = tk.BooleanVar(value=False)
    conf_var = tk.StringVar(value="0.3")
    draw_min_conf_var = tk.StringVar(value="0.3")
    imgsz_var = tk.StringVar(value="1280")
    device_var = tk.StringVar(value="")
    model_id_var = tk.StringVar(value=ROBOFLOW_DEFAULT_MODEL_ID)
    weights_var = tk.StringVar(value="")
    api_key_var = tk.StringVar(value=os.environ.get("ROBOFLOW_API_KEY", ""))

    ttk.Label(frm, text="Mode:").grid(row=0, column=0, sticky="w")
    ttk.Combobox(frm, textvariable=mode_var, values=("frame", "video"), width=18, state="readonly").grid(
        row=0, column=1, sticky="w"
    )
    ttk.Label(frm, text="Backend:").grid(row=1, column=0, sticky="w", pady=(6, 0))
    ttk.Combobox(frm, textvariable=backend_var, values=("roboflow", "ultralytics"), width=18).grid(
        row=1, column=1, sticky="w", pady=(6, 0)
    )
    ttk.Label(frm, text="Frame index (frame mode):").grid(row=2, column=0, sticky="w", pady=(6, 0))
    ttk.Entry(frm, textvariable=frame_var, width=10).grid(row=2, column=1, sticky="w", pady=(6, 0))

    ttk.Label(frm, text="Start frame (video):").grid(row=3, column=0, sticky="w", pady=(6, 0))
    ttk.Entry(frm, textvariable=start_var, width=10).grid(row=3, column=1, sticky="w", pady=(6, 0))
    ttk.Label(frm, text="Stride (video):").grid(row=4, column=0, sticky="w")
    ttk.Entry(frm, textvariable=stride_var, width=10).grid(row=4, column=1, sticky="w")
    ttk.Label(frm, text="Max frames (video, empty=all):").grid(row=5, column=0, sticky="w")
    ttk.Entry(frm, textvariable=max_frames_var, width=10).grid(row=5, column=1, sticky="w")
    ttk.Checkbutton(frm, text="Write overlay .mp4 (video)", variable=overlay_video_var).grid(
        row=6, column=0, columnspan=2, sticky="w", pady=(6, 0)
    )

    ttk.Label(frm, text="Confidence:").grid(row=7, column=0, sticky="w", pady=(6, 0))
    ttk.Entry(frm, textvariable=conf_var, width=10).grid(row=7, column=1, sticky="w", pady=(6, 0))
    ttk.Label(frm, text="Draw/save min conf:").grid(row=8, column=0, sticky="w")
    ttk.Entry(frm, textvariable=draw_min_conf_var, width=10).grid(row=8, column=1, sticky="w")
    ttk.Label(frm, text="imgsz (ultralytics):").grid(row=9, column=0, sticky="w")
    ttk.Entry(frm, textvariable=imgsz_var, width=10).grid(row=9, column=1, sticky="w")
    ttk.Label(frm, text="Device (ultralytics):").grid(row=10, column=0, sticky="w")
    ttk.Entry(frm, textvariable=device_var, width=10).grid(row=10, column=1, sticky="w")

    lbl_model = ttk.Label(frm, text="Roboflow model id:")
    ent_model = ttk.Entry(frm, textvariable=model_id_var, width=38)
    lbl_api = ttk.Label(frm, text="Roboflow API key (optional):")
    ent_api = ttk.Entry(frm, textvariable=api_key_var, width=38, show="*")
    lbl_model.grid(row=11, column=0, sticky="w", pady=(6, 0))
    ent_model.grid(row=11, column=1, columnspan=2, sticky="ew", pady=(6, 0))
    lbl_api.grid(row=12, column=0, sticky="w", pady=(6, 0))
    ent_api.grid(row=12, column=1, columnspan=2, sticky="ew", pady=(6, 0))

    ttk.Label(frm, text="Ultralytics weights (.pt):").grid(row=13, column=0, sticky="w", pady=(6, 0))
    ttk.Entry(frm, textvariable=weights_var, width=38).grid(
        row=13, column=1, columnspan=2, sticky="ew", pady=(6, 0)
    )

    def _browse_weights() -> None:
        p = filedialog.askopenfilename(
            title="Select YOLO pose weights (.pt)",
            filetypes=[("PyTorch", "*.pt"), ("All files", "*.*")],
        )
        if p:
            weights_var.set(p)

    ttk.Button(frm, text="Browse…", command=_browse_weights).grid(row=13, column=3, padx=(6, 0))

    frm.columnconfigure(2, weight=1)

    def _sync_backend_rows(*_args: object) -> None:
        b = backend_var.get().strip()
        if b == "roboflow":
            lbl_model.grid()
            ent_model.grid()
            lbl_api.grid()
            ent_api.grid()
        else:
            lbl_model.grid_remove()
            ent_model.grid_remove()
            lbl_api.grid_remove()
            ent_api.grid_remove()

    backend_var.trace_add("write", _sync_backend_rows)
    _sync_backend_rows()

    result: dict[str, object] = {"ok": False, "err": ""}

    def _run() -> None:
        try:
            out_base = Path(out_parent) / f"processed_field_kps_{_timestamp()}"
            backend = backend_var.get().strip()
            api_override = api_key_var.get().strip() or None
            device_raw = device_var.get().strip()
            device: str | int | None = None if device_raw == "" else device_raw
            if mode_var.get().strip() == "frame":
                detect_field_keypoints(
                    Path(video),
                    frame_index=int(frame_var.get().strip() or "0"),
                    backend=backend,
                    roboflow_api_key=api_override,
                    roboflow_model_id=model_id_var.get().strip() or ROBOFLOW_DEFAULT_MODEL_ID,
                    ultralytics_weights=Path(weights_var.get()).expanduser().resolve()
                    if weights_var.get().strip()
                    else None,
                    conf=float(conf_var.get().strip() or "0.3"),
                    imgsz=int(imgsz_var.get().strip() or "1280"),
                    device=device,
                    draw_min_conf=float(draw_min_conf_var.get().strip() or "0.3"),
                    out_dir=out_base,
                )
            else:
                mf_raw = (max_frames_var.get() or "").strip()
                max_frames = None if mf_raw == "" else int(mf_raw)
                detect_field_keypoints_video(
                    Path(video),
                    backend=backend,
                    roboflow_api_key=api_override,
                    roboflow_model_id=model_id_var.get().strip() or ROBOFLOW_DEFAULT_MODEL_ID,
                    ultralytics_weights=Path(weights_var.get()).expanduser().resolve()
                    if weights_var.get().strip()
                    else None,
                    conf=float(conf_var.get().strip() or "0.3"),
                    imgsz=int(imgsz_var.get().strip() or "1280"),
                    device=device,
                    draw_min_conf=float(draw_min_conf_var.get().strip() or "0.3"),
                    out_dir=out_base,
                    start=int(start_var.get().strip() or "0"),
                    stride=int(stride_var.get().strip() or "10"),
                    max_frames=max_frames,
                    write_overlay_video=bool(overlay_video_var.get()),
                )
            result["ok"] = True
            messagebox.showinfo("Done", f"Wrote outputs under:\n{out_base}", parent=dlg)
            dlg.destroy()
        except Exception as e:
            result["err"] = str(e)
            messagebox.showerror("Field keypoints failed", str(e), parent=dlg)

    def _open_help() -> None:
        help_html = Path(__file__).resolve().parent / "help" / "soccerfield_keypoints_ai.html"
        if help_html.exists():
            webbrowser.open(help_html.as_uri())
        else:
            messagebox.showinfo(
                "Help",
                "Help file not found. See docs/fifa_workflow.md or run with --help.",
                parent=dlg,
            )

    btns = ttk.Frame(frm)
    btns.grid(row=14, column=0, columnspan=4, pady=(10, 0))
    ttk.Button(btns, text="Run", command=_run).pack(side=tk.LEFT, padx=4)
    ttk.Button(btns, text="Help", command=_open_help).pack(side=tk.LEFT, padx=4)
    ttk.Button(btns, text="Close", command=dlg.destroy).pack(side=tk.LEFT, padx=4)

    dlg.transient(root)
    dlg.grab_set()
    root.wait_window(dlg)
    root.destroy()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Soccer-field keypoints (AI) → pixel CSV seed")
    parser.add_argument("-i", "--input", type=Path, help="Input video")
    parser.add_argument("-o", "--output", type=Path, help="Output directory")
    parser.add_argument("--mode", choices=["frame", "video"], default="frame")
    parser.add_argument("--frame", type=int, default=0, help="Frame index for inference")
    parser.add_argument("--start", type=int, default=0, help="Start frame for video mode")
    parser.add_argument("--stride", type=int, default=10, help="Process every Nth frame in video mode")
    parser.add_argument("--max-frames", type=int, default=None, help="Max processed frames in video mode")
    parser.add_argument("--overlay-video", action="store_true", help="Write an overlay .mp4 in video mode")
    parser.add_argument("--backend", choices=["roboflow", "ultralytics"], default="roboflow")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument(
        "--draw-min-conf",
        type=float,
        default=0.3,
        help="Only draw/save points with conf >= this value (raw CSV still contains all).",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Ultralytics imgsz")
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Ultralytics device (e.g. '0', 'cpu'). Empty = auto. Used only for backend=ultralytics.",
    )
    parser.add_argument("--roboflow-model-id", type=str, default=ROBOFLOW_DEFAULT_MODEL_ID)
    parser.add_argument("--weights", type=Path, default=None, help="Ultralytics weights (.pt)")
    args = parser.parse_args(argv)

    if args.input and args.output:
        out_base = args.output / f"processed_field_kps_{_timestamp()}"
        device: str | int | None = None if (args.device or "").strip() == "" else args.device
        if args.mode == "frame":
            detect_field_keypoints(
                args.input,
                frame_index=args.frame,
                backend=args.backend,
                roboflow_model_id=args.roboflow_model_id,
                ultralytics_weights=args.weights,
                conf=args.conf,
                draw_min_conf=args.draw_min_conf,
                imgsz=args.imgsz,
                device=device,
                out_dir=out_base,
            )
        else:
            detect_field_keypoints_video(
                args.input,
                backend=args.backend,
                roboflow_model_id=args.roboflow_model_id,
                ultralytics_weights=args.weights,
                conf=args.conf,
                draw_min_conf=args.draw_min_conf,
                imgsz=args.imgsz,
                device=device,
                out_dir=out_base,
                start=args.start,
                stride=args.stride,
                max_frames=args.max_frames,
                write_overlay_video=bool(args.overlay_video),
            )
        print(f"Done. Output: {out_base}")
        return

    run_gui()


if __name__ == "__main__":
    main()
