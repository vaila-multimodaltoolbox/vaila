"""
markerless_3d_gui.py — GUI configurator + TOML + batch runner for monocular 3D (meters by default)

What it does
- Launches a Tkinter GUI to collect all parameters and save a .toml config file (with sensible defaults).
- Batch processes a directory of inputs:
    • If the input is a VIDEO: extracts 2D MediaPipe → lifts 2D→3D (VideoPose3D) → anchors to ground (DLT2D/JSON/Click) → calibrates vertical by leg length → optional DLT3D refine → exports CSV/C3D.
    • If the input is a CSV (MediaPipe pixels, shape [T, 33*2]): skips extraction and continues the same pipeline. Width/Height/FPS for CSVs can be provided in the GUI (CSV defaults).

Defaults set for your case
- Units = METERS ("m") and conversions handled internally.
- DLT2D default: /mnt/data/cam2_calib2D.dlt2d
- DLT3D default: /mnt/data/cam2_calib3D.dlt3d
- Input dir default: /mnt/data
- Pattern default: "*.mp4;*.mov;*.avi;*_mp_pixel.csv" (videos + existing 2D CSVs)
- Leg length default = 0.42 m (can be changed per participant in GUI)

Requires
pip install numpy opencv-python mediapipe torch ezc3d scipy
"""

from __future__ import annotations
import os, sys, json, math, glob, time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import cv2

try:
    import torch
except Exception:
    torch = None
try:
    import mediapipe as mp
except Exception:
    mp = None
try:
    from ezc3d import c3d as ezc3d
except Exception:
    ezc3d = None

from scipy.spatial import procrustes
from scipy.optimize import least_squares

try:
    import tomllib as _toml_reader  # Python 3.11+
except Exception:  # noqa: BLE001
    try:
        import tomli as _toml_reader  # type: ignore
    except Exception:  # noqa: BLE001
        _toml_reader = None  # parsed only if available

# =====================
# Small TOML helpers
# =====================


def _toml_escape(s: str) -> str:
    # escape backslashes and double quotes for TOML strings
    return s.replace("\\", "\\\\").replace('"', '\\"')


def dict_to_toml(d: Dict, prefix: str = "") -> str:
    lines: List[str] = []
    # simple (non-container) keys first
    scalars = {k: v for k, v in d.items() if not isinstance(v, (dict, list))}
    if scalars and prefix:
        lines.append(f"[{prefix}]")
        for k, v in scalars.items():
            if isinstance(v, str):
                lines.append(f'{k} = "{_toml_escape(v)}"')
            elif isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                    v = 0.0
                lines.append(f"{k} = {v}")
            elif v is None:
                lines.append(f'{k} = ""')
            else:
                lines.append(f"# {k} = unsupported scalar type")
        lines.append("")
    # nested dicts
    for k, v in d.items():
        if isinstance(v, dict):
            new_prefix = f"{prefix}.{k}" if prefix else k
            sub = dict_to_toml(v, new_prefix)
            if sub:
                lines.append(sub)
    # lists (arrays)
    for k, v in d.items():
        if isinstance(v, list):
            if v and isinstance(v[0], list):
                arr = ", ".join("[" + ", ".join(str(x) for x in row) + "]" for row in v)
                if prefix:
                    lines.append(f"[{prefix}]")
                lines.append(f"{k} = [ {arr} ]")
                lines.append("")
            else:
                arr = ", ".join(
                    f'"{_toml_escape(x)}"' if isinstance(x, str) else str(x) for x in v
                )
                if prefix:
                    lines.append(f"[{prefix}]")
                lines.append(f"{k} = [ {arr} ]")
                lines.append("")
    return "\n".join([ln for ln in lines if ln is not None])


# =====================
# Pose mappings
# =====================
MEDIAPIPE_TO_COCO17 = {
    0: 0,
    2: 1,
    5: 2,
    7: 3,
    8: 4,
    11: 5,
    12: 6,
    13: 7,
    14: 8,
    15: 9,
    16: 10,
    23: 11,
    24: 12,
    25: 13,
    26: 14,
    27: 15,
    28: 16,
}
COCO_IDX = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}
COCO17_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# =====================
# Core math helpers
# =====================


def normalize_screen_coordinates(xy: np.ndarray, w: int, h: int) -> np.ndarray:
    out = np.empty_like(xy, dtype=np.float32)
    out[..., 0] = xy[..., 0] / w * 2.0 - 1.0
    out[..., 1] = xy[..., 1] / w * 2.0 - (h / w)
    return out


def mediapipe_to_coco17(kps33_xy: np.ndarray) -> np.ndarray:
    T = kps33_xy.shape[0]
    out = np.zeros((T, 17, 2), dtype=np.float32)
    for mp_idx, coco_idx in MEDIAPIPE_TO_COCO17.items():
        out[:, coco_idx, :] = kps33_xy[:, mp_idx, :]
    return out


# =====================
# VideoPose3D
# =====================


def build_vp3d_model(
    vp3d_dir: str, njoints: int = 17, channels: int = 1024, dropout: float = 0.25
):
    if torch is None:
        raise ImportError("torch is required for VideoPose3D")
    vp3d_path = Path(vp3d_dir).resolve()

    # Avoid collision with a pip package named 'common'.
    for modname in [
        k for k in list(sys.modules) if k == "common" or k.startswith("common.")
    ]:
        del sys.modules[modname]

    # Ensure the VideoPose3D repo path has priority
    if str(vp3d_path) not in sys.path:
        sys.path.insert(0, str(vp3d_path))

    # Explicitly import VideoPose3D's common/model.py from disk
    import importlib.util

    model_py = vp3d_path / "common" / "model.py"
    if not model_py.exists():
        raise ImportError(
            f"Could not find VideoPose3D model file at: {model_py}. "
            "Ensure 'common/model.py' exists under the specified vp3d_dir."
        )
    spec = importlib.util.spec_from_file_location("vp3d_common_model", str(model_py))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {model_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    TemporalModel = getattr(mod, "TemporalModel")

    # Standard VideoPose3D expects num_joints_out = number of joints (e.g., 17)
    model = TemporalModel(
        njoints,
        in_features=2,
        num_joints_out=njoints,
        filter_widths=[3, 3, 3, 3, 3],
        causal=False,
        dropout=dropout,
        channels=channels,
    )
    return model


def load_vp3d_checkpoint(model, ckpt_path: str):
    chk = torch.load(ckpt_path, map_location="cpu")
    key = "model_pos" if "model_pos" in chk else list(chk.keys())[0]
    model.load_state_dict(chk[key], strict=False)
    model.eval()
    return model


def sliding_windows(X: np.ndarray, receptive: int, pad: bool = True) -> np.ndarray:
    T = X.shape[0]
    half = receptive // 2
    if pad:
        Xpad = np.pad(X, ((half, half), (0, 0), (0, 0)), mode="edge")
        starts = [i for i in range(T)]
        slices = [Xpad[i : i + receptive] for i in starts]
        return np.stack(slices, axis=0)
    else:
        starts = [i for i in range(T - receptive + 1)]
        return np.stack([X[i : i + receptive] for i in starts], axis=0)


def infer_vp3d(
    model, seq2d_norm: np.ndarray, batch_size: Optional[int] = None
) -> np.ndarray:
    """Run VideoPose3D with progress and bounded memory.

    The model expects input shaped [B, T, J, 2]. We build sliding windows of length
    receptive field and feed them in batches along the first dimension.
    """
    device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu")
    model.to(device)
    # Choose a conservative default batch size
    if batch_size is None:
        batch_size = 1024 if device.type == "cuda" else 64
    receptive = model.receptive_field()
    Xw = sliding_windows(seq2d_norm, receptive=receptive, pad=True)  # [T, R, J, 2]

    T = Xw.shape[0]
    print(
        f"[Vp3D] Inference device: {device}; frames={T}; receptive={receptive}; batch={batch_size}"
    )
    sys.stdout.flush()

    out_np = np.zeros((T, 17, 3), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        start = 0
        while start < T:
            end = min(T, start + batch_size)
            try:
                X = torch.from_numpy(Xw[start:end]).float().to(device)  # [B, R, J, 2]
                Y = model(X)  # [B, R', J_out, 3] or [B, J_out, 3]
                if Y.dim() == 4:
                    center = Y.shape[1] // 2
                    Y = Y[:, center, :, :]
                out_np[start:end] = Y.detach().cpu().numpy().astype(np.float32)
                start = end
                # Lightweight progress
                if (start // batch_size) % max(1, (T // batch_size) // 10 + 1) == 0:
                    print(f"[Vp3D] {start}/{T} frames")
                    sys.stdout.flush()
            except RuntimeError as e:
                msg = str(e)
                if "out of memory" in msg.lower() or "not enough memory" in msg.lower():
                    new_bs = max(8, batch_size // 2)
                    if new_bs == batch_size:
                        raise
                    print(
                        f"[Vp3D] OOM at batch_size={batch_size}. Retrying with batch_size={new_bs}..."
                    )
                    sys.stdout.flush()
                    batch_size = new_bs
                    torch.cuda.empty_cache() if device.type == "cuda" else None
                else:
                    raise

    return out_np


# =====================
# Ground + DLT
# =====================


def dlt8_to_H(L: List[float]):
    L1, L2, L3, L4, L5, L6, L7, L8 = L
    H = np.array([[L1, L2, L3], [L4, L5, L6], [L7, L8, 1.0]], dtype=np.float64)
    Hinv = np.linalg.inv(H)
    return H, Hinv


def pick_points_on_image(image: np.ndarray, n: int = 4) -> np.ndarray:
    pts = []
    img = image.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(
                img,
                f"P{len(pts)}",
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    cv2.namedWindow("Click 4 ground points")
    cv2.setMouseCallback("Click 4 ground points", on_mouse)
    while True:
        cv2.imshow("Click 4 ground points", img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        if len(pts) == n:
            break
    cv2.destroyAllWindows()
    return np.array(pts, dtype=np.float32)


def image_to_ground_xy(Hinv: np.ndarray, uv: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, np.float32).reshape(1, -1, 2)
    xy = cv2.perspectiveTransform(uv, Hinv)[0]
    return xy


def detect_foot_contacts(kp2d_px: np.ndarray, fps: float, ankle_idx: int) -> np.ndarray:
    vel = np.linalg.norm(np.diff(kp2d_px[:, ankle_idx, :], axis=0), axis=1) * fps
    thr = np.percentile(vel, 20.0)
    frames = np.where(vel <= thr)[0]
    frames = np.unique(np.clip(frames, 0, kp2d_px.shape[0] - 1))
    return frames


def fit_plane_normal(pts3d: np.ndarray) -> np.ndarray:
    C = pts3d.mean(axis=0)
    U, S, Vt = np.linalg.svd(pts3d - C)
    n = Vt[-1]
    return n / (np.linalg.norm(n) + 1e-9)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = np.array([1, 0, 0], np.float32)
        if abs(a[0]) > 0.9:
            axis = np.array([0, 1, 0], np.float32)
        v = np.cross(a, axis)
        v = v / (np.linalg.norm(v) + 1e-9)
        R = -np.eye(3, dtype=np.float32)
        R += 2 * np.outer(v, v)
        return R
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], np.float32)
    R = np.eye(3, dtype=np.float32) + K + K @ K * (1.0 / (1.0 + c + 1e-9))
    return R


def anchor_with_vertical(
    pred3d_mm: np.ndarray,
    coco2d_px: np.ndarray,
    fps: float,
    Hinv: np.ndarray,
    units: str = "m",
    prefer_side: str = "left",
    leg_len_m: Optional[float] = 0.42,
    z_weight: float = 10.0,
    use_anisotropic_z: bool = True,
) -> Tuple[np.ndarray, Dict]:
    scale_u = (
        1000.0 if units.lower() == "m" else 1.0
    )  # convert ground XY meters→mm for matching Vp3D units
    idxA = COCO_IDX["left_ankle"] if prefer_side == "left" else COCO_IDX["right_ankle"]
    idxK = COCO_IDX["left_knee"] if prefer_side == "left" else COCO_IDX["right_knee"]

    fL = detect_foot_contacts(coco2d_px, fps, COCO_IDX["left_ankle"])
    fR = detect_foot_contacts(coco2d_px, fps, COCO_IDX["right_ankle"])
    frames = np.unique(np.concatenate([fL, fR]))
    if len(frames) < 10:
        frames = np.arange(0, pred3d_mm.shape[0], max(1, pred3d_mm.shape[0] // 30))

    ankles = np.concatenate(
        [
            pred3d_mm[frames, COCO_IDX["left_ankle"], :],
            pred3d_mm[frames, COCO_IDX["right_ankle"], :],
        ],
        axis=0,
    )
    n_pred = fit_plane_normal(ankles)
    Rz = rotation_between(n_pred, np.array([0, 0, 1], np.float32))
    P = (Rz @ pred3d_mm.reshape(-1, 3).T).T.reshape(pred3d_mm.shape)

    rows, rhs = [], []
    for t in frames:
        for a in [COCO_IDX["left_ankle"], COCO_IDX["right_ankle"]]:
            p = P[t, a, :]
            g = image_to_ground_xy(Hinv, coco2d_px[t, a, :]) * scale_u
            gx, gy = float(g[0, 0]), float(g[0, 1])
            rows.append([p[0], 1, 0, 0])
            rhs.append(gx)
            rows.append([p[1], 0, 1, 0])
            rhs.append(gy)
            rows.append([z_weight * p[2], 0, 0, z_weight])
            rhs.append(0.0)
    A = np.asarray(rows, np.float64)
    b = np.asarray(rhs, np.float64)
    s_xy, tx, ty, tz = np.linalg.lstsq(A, b, rcond=None)[0]
    Txyz = np.array([tx, ty, tz], np.float32)

    Pw = np.empty_like(P, dtype=np.float32)
    for t in range(P.shape[0]):
        Pw[t] = s_xy * P[t] + Txyz

    calib = {
        "s_xy": float(s_xy),
        "Rz": Rz,
        "T": Txyz,
        "gamma_z": 1.0,
        "anisotropic": False,
    }

    if leg_len_m is not None and leg_len_m > 0:
        dists = np.linalg.norm(Pw[:, idxK, :] - Pw[:, idxA, :], axis=1)  # in mm now
        med_pred_mm = float(np.median(dists))
        target_mm = leg_len_m * 1000.0
        if med_pred_mm > 1e-3:
            if use_anisotropic_z:
                gamma = float(target_mm / med_pred_mm)
                Pw[:, :, 2] *= gamma
                zc = np.median(
                    Pw[frames, [COCO_IDX["left_ankle"], COCO_IDX["right_ankle"]], 2]
                )
                Pw[:, :, 2] -= zc
                calib.update({"gamma_z": gamma, "anisotropic": True})
            else:
                s_global = float(target_mm / med_pred_mm) * s_xy
                rows, rhs = [], []
                for t in frames:
                    for a in [COCO_IDX["left_ankle"], COCO_IDX["right_ankle"]]:
                        p = P[t, a, :] * s_global
                        g = image_to_ground_xy(Hinv, coco2d_px[t, a, :]) * scale_u
                        gx, gy = float(g[0, 0]), float(g[0, 1])
                        rows.append([1, 0, 0])
                        rhs.append(gx - p[0])
                        rows.append([0, 1, 0])
                        rhs.append(gy - p[1])
                        rows.append([0, 0, 1])
                        rhs.append(-p[2])
                A = np.asarray(rows, np.float64)
                b = np.asarray(rhs, np.float64)
                tx, ty, tz = np.linalg.lstsq(A, b, rcond=None)[0]
                Txyz2 = np.array([tx, ty, tz], np.float32)
                for t in range(P.shape[0]):
                    Pw[t] = s_global * P[t] + Txyz2
                calib.update(
                    {"s_xy": float(s_global), "T": Txyz2, "anisotropic": False}
                )

    return Pw, calib


# DLT3D
class DLT3D:
    def __init__(self, L: np.ndarray):
        self.L = L.astype(np.float64)


def load_dlt3d_from_file(path: str) -> DLT3D:
    vals = []
    with open(path, "r") as f:
        for line in f:
            parts = [p for p in line.replace(",", " ").split() if p.strip()]
            for p in parts:
                try:
                    vals.append(float(p))
                except:
                    pass
    if len(vals) < 11:
        raise ValueError("DLT3D file must contain at least 11 numeric parameters")
    return DLT3D(np.asarray(vals[:11]))


def project_dlt3d(dlt: DLT3D, XYZ: np.ndarray) -> np.ndarray:
    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    L = dlt.L
    denom = L[8] * X + L[9] * Y + L[10] * Z + 1.0
    u = (L[0] * X + L[1] * Y + L[2] * Z + L[3]) / denom
    v = (L[4] * X + L[5] * Y + L[6] * Z + L[7]) / denom
    return np.stack([u, v], axis=-1)


def euler_to_Rxyz(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def refine_with_dlt3d(
    Pw: np.ndarray,
    uv_px: np.ndarray,
    dlt: DLT3D,
    init: Optional[Dict] = None,
    use_yaw_only: bool = True,
) -> Tuple[np.ndarray, Dict]:
    T, J, _ = Pw.shape
    W = Pw.copy()
    if init is None:
        s0 = 1.0
        rz0 = 0.0
        tx0 = 0.0
        ty0 = 0.0
        tz0 = 0.0
        rx0 = 0.0
        ry0 = 0.0
    else:
        s0 = float(init.get("s", 1.0))
        R0 = init.get("R", np.eye(3))
        rz0 = math.atan2(R0[1, 0], R0[0, 0])
        tx0, ty0, tz0 = [float(x) for x in init.get("t", np.zeros(3))]
        rx0 = 0.0
        ry0 = 0.0
    x0 = (
        np.array([s0, rz0, tx0, ty0, tz0], dtype=np.float64)
        if use_yaw_only
        else np.array([s0, rx0, ry0, rz0, tx0, ty0, tz0], dtype=np.float64)
    )
    uv = uv_px.reshape(-1, 2).astype(np.float64)
    XYZ = W.reshape(-1, 3).astype(np.float64)

    def residuals(x):
        if use_yaw_only:
            s, rz, tx, ty, tz = x
            R = euler_to_Rxyz(0.0, 0.0, rz)
        else:
            s, rx, ry, rz, tx, ty, tz = x
            R = euler_to_Rxyz(rx, ry, rz)
        XYZt = (s * (XYZ @ R.T)) + np.array([tx, ty, tz])
        proj = project_dlt3d(dlt, XYZt)
        return (proj - uv).reshape(-1)

    res = least_squares(residuals, x0, method="lm", max_nfev=200)
    x = res.x
    if use_yaw_only:
        s, rz, tx, ty, tz = x
        R = euler_to_Rxyz(0.0, 0.0, rz)
    else:
        s, rx, ry, rz, tx, ty, tz = x
        R = euler_to_Rxyz(rx, ry, rz)
    t = np.array([tx, ty, tz], dtype=np.float32)
    out = np.empty_like(W, dtype=np.float32)
    for t_i in range(T):
        out[t_i] = (s * (W[t_i] @ R.T)) + t
    params = {"s": float(s), "R": R, "t": t, "cost": float(res.cost)}
    return out, params


# =====================
# MediaPipe 2D extractor
# =====================


def extract_mediapipe_csv(video_path: str, out_csv: str) -> Tuple[int, int, float, int]:
    if mp is None:
        raise ImportError("mediapipe is required for 2D extraction")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[MediaPipe] Opening {video_path}")
    print(f"[MediaPipe] Resolution: {w}x{h}  fps: {fps:.3f}  total frames: {total}")
    sys.stdout.flush()
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        smooth_landmarks=True,
    )
    rows = []
    i = 0
    t0 = time.time()
    t_last = t0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t_proc0 = time.time()
        res = pose.process(rgb)
        t_proc1 = time.time()
        if res.pose_landmarks:
            pts = []
            for lm in res.pose_landmarks.landmark:
                pts.extend([lm.x * w, lm.y * h])
        else:
            pts = [np.nan] * (33 * 2)
        rows.append(pts)
        i += 1
        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            print(
                f"[MediaPipe] {i}/{total if total>0 else '?'} frames  ({rate:.1f} fps)"
            )
            sys.stdout.flush()
        if (t_proc1 - t_proc0) > 0.25:
            print(f"[MediaPipe] slow frame {i}: {t_proc1 - t_proc0:.3f}s")
            sys.stdout.flush()
    cap.release()
    pose.close()
    arr = np.array(rows, dtype=np.float32)
    np.savetxt(out_csv, arr, delimiter=",", fmt="%.6f")
    print(f"[MediaPipe] Extraction done. Saved {arr.shape[0]} frames to {out_csv}")
    sys.stdout.flush()
    return w, h, float(fps), arr.shape[0]


# =====================
# 3D pipeline helpers
# =====================


def load_csv33(csv_path: str) -> np.ndarray:
    arr = np.loadtxt(csv_path, delimiter=",")
    if arr.ndim == 2 and arr.shape[1] == 66:
        return arr.reshape(-1, 33, 2).astype(np.float32)
    elif arr.ndim == 3 and arr.shape[1:] == (33, 2):
        return arr.astype(np.float32)
    else:
        raise ValueError(f"CSV {csv_path} must be [T,66] or [T,33,2]")


# =====================
# Config (TOML)
# =====================


def read_toml_config(path: str) -> Dict:
    if _toml_reader is None:
        raise ImportError(
            "tomllib/tomli is required to read TOML configs. Install 'tomli' for Python<3.11."
        )
    with open(path, "rb") as f:
        return _toml_reader.load(f)


def default_config() -> Dict:
    return {
        "paths": {
            "vp3d_dir": "../VideoPose3D",
            "ckpt": "../checkpoints/h36m_cpn_ft_h36m_dbb.pth",
            "dlt2d_file": "../tests/markerless_3d_analysis/cam2_calib2D.dlt2d",
            "ground_json": "",
            "dlt3d_file": "../tests/markerless_3d_analysis/cam2_calib3D.dlt3d",
        },
        "video": {
            "input_dir": "../tests/markerless_3d_analysis",
            "pattern": "*.mp4;*.mov;*.avi;*_mp_pixel.csv",
            "fps_override": 0.0,
        },
        "ground": {
            "mode": "dlt",
            "units": "m",
            "world_points_text": "0.0,0.0,1.422,0.0,1.422,1.422,0.0,1.422",
        },
        "calibration": {
            "leg_len_m": 0.42,
            "prefer_side": "left",
        },
        "csv_defaults": {
            "imgw": 2704,
            "imgh": 1520,
            "fps": 240.0,
        },
        "output": {
            "out_dir": "../tests/markerless_3d_analysis/mono3d_out",
            "save_c3d": True,
        },
    }


def write_toml_config(path: str, cfg: Dict) -> None:
    toml_str = dict_to_toml(cfg)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(toml_str if toml_str.endswith("\n") else toml_str + "\n")


# VIDEO path → full pipeline


def run_single_video(
    cfg: Dict, video_path: str, out_dir: str, base_dir: Optional[Path] = None
):
    def _resolve(p: str, expect_exists: bool = False) -> str:
        if not p:
            return ""
        candidate_paths: List[Path] = []
        pp = Path(p)
        if pp.is_absolute():
            candidate_paths.append(pp)
        else:
            if base_dir is not None:
                candidate_paths.append((base_dir / pp))
            candidate_paths.append(pp)
            if base_dir is not None:
                candidate_paths.append(base_dir / pp.name)
        if expect_exists:
            for c in candidate_paths:
                if c.exists():
                    return str(c.resolve())
        # fallback
        return str(candidate_paths[0].resolve())

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    csv2d = os.path.join(out_dir, f"{stem}_mp_pixel.csv")
    if not os.path.exists(csv2d):
        print(f"Extracting MediaPipe from {video_path}")
        w, h, fps, frames = extract_mediapipe_csv(video_path, csv2d)
    else:
        print(f"Using existing CSV: {csv2d}")
        # Load existing CSV to get dimensions
        arr = load_csv33(csv2d)
        frames = arr.shape[0]
        csv_def = cfg.get("csv_defaults", {})
        w = int(csv_def.get("imgw", 1920))
        h = int(csv_def.get("imgh", 1080))
        fps = float(csv_def.get("fps", 30.0))
        print(f"[CSV] Using csv_defaults: w={w} h={h} fps={fps}")
        sys.stdout.flush()

    print(f"Processing {frames} frames...")

    # Load 2D keypoints
    kps33_px = load_csv33(csv2d)  # [T,33,2]
    coco2d_px = mediapipe_to_coco17(kps33_px)  # [T,17,2]
    coco2d_norm = normalize_screen_coordinates(coco2d_px, w, h)  # [-1..1]

    # Build and run VideoPose3D
    paths = cfg.get("paths", {})
    vp3d_dir_resolved = _resolve(
        paths.get("vp3d_dir", "../VideoPose3D"), expect_exists=True
    )
    ckpt_resolved = _resolve(
        paths.get("ckpt", "../checkpoints/h36m_cpn_ft_h36m_dbb.pth"),
        expect_exists=False,
    )
    if not os.path.exists(ckpt_resolved):
        # Try to find a checkpoint automatically under common folders
        search_dirs = [
            Path(vp3d_dir_resolved) / "checkpoints",
            Path(vp3d_dir_resolved) / "checkpoint",
            Path(vp3d_dir_resolved) / "pretrained",
            Path(vp3d_dir_resolved) / "pretrained_models",
        ]
        candidates: List[str] = []
        for d in search_dirs:
            candidates.extend([str(p) for p in d.glob("*.pth") if p.is_file()])
            candidates.extend([str(p) for p in d.glob("*.bin") if p.is_file()])
        if candidates:
            ckpt_resolved = candidates[0]
            print(
                f"[Vp3D] Checkpoint not found in TOML; using found file: {ckpt_resolved}"
            )
        else:
            raise FileNotFoundError(
                f"Vp3D checkpoint not found. Set 'paths.ckpt' in the TOML to a valid .pth file.\n"
                f"Tried: {ckpt_resolved}\n"
                f"Also looked under: {[str(d) for d in search_dirs]}"
            )
    print(f"[Vp3D] Building model from {vp3d_dir_resolved}")
    sys.stdout.flush()
    model = build_vp3d_model(vp3d_dir_resolved)
    print(f"[Vp3D] Loading checkpoint {ckpt_resolved}")
    sys.stdout.flush()
    load_vp3d_checkpoint(model, ckpt_resolved)
    print("[Vp3D] Checkpoint loaded")
    pred3d_mm = infer_vp3d(model, coco2d_norm)  # [T,17,3] in mm (root-centered)
    print("Vp3D inference done")

    # Ground homography Hinv
    ground = cfg.get("ground", {})
    mode = str(ground.get("mode", "dlt")).lower()
    units = str(ground.get("units", "m")).lower()
    if mode == "dlt":
        dlt2d_path = _resolve(paths.get("dlt2d_file", ""), expect_exists=True)
        if not dlt2d_path or not os.path.exists(dlt2d_path):
            raise FileNotFoundError(f"DLT2D file not found: {dlt2d_path}")
        vals = []
        with open(dlt2d_path, "r", encoding="utf-8") as f:
            for line in f:
                for p in line.replace(",", " ").split():
                    try:
                        vals.append(float(p))
                    except ValueError:
                        pass
        if len(vals) < 8:
            raise ValueError("DLT2D must contain at least 8 numeric parameters")
        _, Hinv = dlt8_to_H(vals[:8])
    else:
        raise NotImplementedError(
            f"ground.mode '{mode}' not implemented yet (use 'dlt')"
        )

    # Anchor and vertical calibration
    calib = cfg.get("calibration", {})
    Pw_mm, calib_out = anchor_with_vertical(
        pred3d_mm=pred3d_mm,
        coco2d_px=coco2d_px,
        fps=fps,
        Hinv=Hinv,
        units=units,
        prefer_side=str(calib.get("prefer_side", "left")),
        leg_len_m=float(calib.get("leg_len_m", 0.42)),
    )
    print("Ground anchoring done")

    # Optional DLT3D refine
    dlt3d_file = _resolve(paths.get("dlt3d_file", ""), expect_exists=True)
    if dlt3d_file and os.path.exists(dlt3d_file):
        dlt3d = load_dlt3d_from_file(dlt3d_file)
        Pw_mm, refine_params = refine_with_dlt3d(
            Pw_mm, coco2d_px, dlt3d, init=None, use_yaw_only=True
        )
        print("DLT3D refine done")

    # Export
    out = cfg.get("output", {})
    out_base = os.path.join(out_dir, f"{stem}_mono3d")
    # CSV in meters if units=="m", else mm
    if units == "m":
        Pw = Pw_mm / 1000.0
    else:
        Pw = Pw_mm
    np.savetxt(
        out_base + ".csv", Pw.reshape(Pw.shape[0], -1), delimiter=",", fmt="%.6f"
    )
    print(f"Saved CSV to {out_base}.csv")

    # Optional C3D
    if bool(out.get("save_c3d", True)) and ezc3d is not None:
        c3d = ezc3d()
        pts = Pw.copy()  # meters or mm per units setting
        # C3D expects mm; if we are in meters, convert
        pts_mm = pts * 1000.0 if units == "m" else pts
        T, J, _ = pts_mm.shape
        data = np.ones((4, J, T), dtype=np.float32)
        data[0:3, :, :] = pts_mm.transpose(2, 1, 0)
        c3d["data"]["points"] = data
        c3d["parameters"]["POINT"]["LABELS"]["value"] = COCO17_NAMES
        c3d.write(out_base + ".c3d")
        print(f"Saved C3D to {out_base}.c3d")

    print(
        f"Saved: {out_base}.csv"
        + (" and .c3d" if bool(out.get("save_c3d", True)) and ezc3d is not None else "")
    )


# =====================
# Main
# =====================
if __name__ == "__main__":

    class App:
        def __init__(self, master: tk.Tk):
            self.master = master
            master.title("Markerless 3D Analysis")

            top = tk.Frame(master)
            top.pack(padx=10, pady=10, fill=tk.X)

            # Config file row
            row1 = tk.Frame(top)
            row1.pack(fill=tk.X)
            tk.Label(row1, text="Config TOML:").pack(side=tk.LEFT)
            self.cfg_path_var = tk.StringVar(
                value=str(
                    Path(
                        "tests/markerless_3d_analysis/markerless3d_config.toml"
                    ).resolve()
                )
            )
            tk.Entry(row1, textvariable=self.cfg_path_var, width=80).pack(
                side=tk.LEFT, padx=5, fill=tk.X, expand=True
            )
            tk.Button(row1, text="Browse", command=self.browse_cfg).pack(
                side=tk.LEFT, padx=5
            )

            # Buttons
            row2 = tk.Frame(top)
            row2.pack(fill=tk.X, pady=(8, 0))
            tk.Button(row2, text="Load TOML", command=self.load_cfg).pack(
                side=tk.LEFT, padx=5
            )
            self.btn_default = tk.Button(
                row2, text="Create Default TOML", command=self.write_default_cfg
            )
            self.btn_default.pack(side=tk.LEFT, padx=5)
            self.btn_run = tk.Button(row2, text="Run Batch", command=self.run_batch)
            self.btn_run.pack(side=tk.LEFT, padx=5)

            self.status_var = tk.StringVar(value="Ready")
            tk.Label(top, textvariable=self.status_var, fg="gray").pack(
                anchor=tk.W, pady=(8, 0)
            )

            self.cfg: Dict = {}
            self._worker_running = False

        def _set_status(self, text: str):
            self.status_var.set(text)
            self.master.update_idletasks()

        def _set_buttons_enabled(self, enabled: bool):
            state = tk.NORMAL if enabled else tk.DISABLED
            self.btn_default.config(state=state)
            self.btn_run.config(state=state)

        def browse_cfg(self):
            path = filedialog.askopenfilename(
                title="Select TOML",
                filetypes=(("TOML files", "*.toml"), ("All files", "*.*")),
            )
            if path:
                self.cfg_path_var.set(path)

        def load_cfg(self):
            try:
                self.cfg = read_toml_config(self.cfg_path_var.get())
                self.status_var.set("Config loaded")
            except Exception as e:  # noqa: BLE001
                messagebox.showerror("Error", str(e))

        def write_default_cfg(self):
            try:
                cfg = default_config()
                write_toml_config(self.cfg_path_var.get(), cfg)
                self.status_var.set("Default config written")
            except Exception as e:  # noqa: BLE001
                messagebox.showerror("Error", str(e))

        def run_batch(self):
            try:
                if self._worker_running:
                    return
                if not self.cfg:
                    self.load_cfg()
                cfg = self.cfg
                # Resolve relative paths based on TOML location
                cfg_path = Path(self.cfg_path_var.get())
                base_dir = cfg_path.parent if cfg_path.exists() else Path.cwd()

                vid_cfg = cfg.get("video", {})
                input_dir = (base_dir / str(vid_cfg.get("input_dir", "."))).resolve()
                patterns = [
                    p.strip()
                    for p in str(vid_cfg.get("pattern", "*.mp4;*_mp_pixel.csv")).split(
                        ";"
                    )
                    if p.strip()
                ]
                out_dir_cfg = cfg.get("output", {}).get(
                    "out_dir", str(input_dir / "mono3d_out")
                )
                out_dir = (
                    str((base_dir / out_dir_cfg).resolve())
                    if not os.path.isabs(out_dir_cfg)
                    else out_dir_cfg
                )
                files: List[str] = []
                for pat in patterns:
                    files.extend([str(p) for p in input_dir.glob(pat)])
                if not files:
                    messagebox.showwarning(
                        "No inputs",
                        f"No files found in\n{input_dir}\nwith patterns: {', '.join(patterns)}",
                    )
                    return
                self._set_status(f"Processing {len(files)} files...")
                self._set_buttons_enabled(False)
                self._worker_running = True

                import threading

                def _worker():
                    try:
                        for i, fp in enumerate(files, 1):
                            self.master.after(
                                0,
                                lambda i=i, fp=fp: self._set_status(
                                    f"[{i}/{len(files)}] {Path(fp).name}"
                                ),
                            )
                            run_single_video(cfg, fp, out_dir, base_dir=base_dir)
                        self.master.after(0, lambda: self._set_status("Done"))
                    except Exception as exc:  # noqa: BLE001
                        import traceback

                        tb = traceback.format_exc()

                        def _show_error(msg=str(exc), tb_str=tb):
                            messagebox.showerror("Error", f"{msg}\n\n{tb_str}")

                        self.master.after(0, _show_error)
                    finally:
                        self._worker_running = False
                        self.master.after(0, lambda: self._set_buttons_enabled(True))

                threading.Thread(target=_worker, daemon=True).start()
            except Exception as e:  # noqa: BLE001
                messagebox.showerror("Error", str(e))

    root = tk.Tk()
    app = App(root)
    root.mainloop()
