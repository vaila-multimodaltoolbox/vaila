"""FIFA Skeletal Tracking Light 2026 — data prep, SAM 3D Body preprocess, baseline NPZ, ZIP.

Upstream references:
  - Starter kit: https://github.com/FIFA-Skeletal-Light-Tracking-Challenge/FIFA-Skeletal-Tracking-Starter-Kit-2026
  - SAM 3D Body (vendored ``sam_3d_body`` + HF weights under ``vaila/models/sam-3d-dinov3/``).

CUDA is required for ``preprocess`` and ``baseline`` (SAM 3D Body and LBFGS steps use the GPU).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

try:
    from .fifa_starter_lib.camera_tracker import CameraTracker, CameraTrackerOptions
    from .fifa_starter_lib.postprocess import smoothen
except ImportError:
    from fifa_starter_lib.camera_tracker import (  # ty: ignore[unresolved-import]
        CameraTracker,
        CameraTrackerOptions,
    )
    from fifa_starter_lib.postprocess import smoothen  # ty: ignore[unresolved-import]

# SPDX-License-Identifier: MIT
# Joint subset from FIFA starter kit main.py (Body25/OpenPose → 15 challenge joints).
OPENPOSE_TO_OURS = [0, 2, 5, 3, 6, 4, 7, 9, 12, 10, 13, 11, 14, 22, 19]

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def default_sam3d_checkpoint_paths() -> tuple[Path, Path]:
    """Return (model.ckpt, mhr_model.pt) under ``vaila/models/sam-3d-dinov3/``."""
    root = _package_dir() / "models" / "sam-3d-dinov3"
    return root / "model.ckpt", root / "assets" / "mhr_model.pt"


def load_sequences(sequences_file: Path | str) -> list[str]:
    with open(sequences_file, encoding="utf-8") as f:
        lines = f.read().splitlines()
    out = [s.strip() for s in lines if s.strip() and not s.strip().startswith("#")]
    return out


def write_sequences_from_videos(video_dir: Path, out_file: Path) -> list[str]:
    names: list[str] = []
    for p in sorted(video_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            names.append(p.stem)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")
    return names


def _ffmpeg_to_mp4(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def ensure_video_mp4(src: Path, data_videos: Path) -> Path:
    """Copy or transcode to ``data_videos/<stem>.mp4``."""
    dst = data_videos / f"{src.stem}.mp4"
    data_videos.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".mp4":
        shutil.copy2(src, dst)
    else:
        _ffmpeg_to_mp4(src, dst)
    return dst


def extract_frames(video_path: Path, image_dir: Path, fps: float | None = None) -> int:
    """Write ``00001.jpg`` … OpenCV; returns frame count."""
    image_dir.mkdir(parents=True, exist_ok=True)
    for old in image_dir.glob("*.jpg"):
        old.unlink()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    idx = 0
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = 1
    if fps is not None and fps > 0 and native_fps > 0:
        step = max(1, int(round(native_fps / fps)))
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i % step == 0:
            idx += 1
            out = image_dir / f"{idx:05d}.jpg"
            cv2.imwrite(str(out), frame)
        frame_i += 1
    cap.release()
    if idx == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return idx


def fifa_prepare(
    video_source: Path,
    data_root: Path,
    sequences_out: Path | None = None,
    extract_fps: float | None = None,
) -> list[str]:
    """Populate ``data_root/videos``, ``data_root/images``, and optional ``sequences_*.txt``."""
    data_root = data_root.resolve()
    vdir = data_root / "videos"
    idir = data_root / "images"
    sequences: list[str] = []

    if video_source.is_file():
        seq = video_source.stem
        ensure_video_mp4(video_source, vdir)
        extract_frames(vdir / f"{seq}.mp4", idir / seq, fps=extract_fps)
        sequences = [seq]
    else:
        for src in sorted(video_source.iterdir()):
            if not src.is_file() or src.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            seq = src.stem
            ensure_video_mp4(src, vdir)
            extract_frames(vdir / f"{seq}.mp4", idir / seq, fps=extract_fps)
            sequences.append(seq)

    if sequences_out is not None:
        sequences_out.parent.mkdir(parents=True, exist_ok=True)
        sequences_out.write_text("\n".join(sequences) + "\n", encoding="utf-8")
    return sequences


# --- SAM 3D Body (starter kit preprocess.py logic; MIT) ---
class FifaSam3dBodyModel:
    """Wraps ``SAM3DBodyEstimator`` + Body25 keypoint selection (70 → 25)."""

    def __init__(self, device: str = "cuda") -> None:
        from sam_3d_body import SAM3DBodyEstimator  # ty: ignore[unresolved-import]
        from sam_3d_body.build_models import load_sam_3d_body  # ty: ignore[unresolved-import]

        ckpt = Path(os.environ.get("SAM3D_CHECKPOINT", "")).expanduser()
        mhr = Path(os.environ.get("SAM3D_MHR", "")).expanduser()
        if not ckpt.is_file():
            ckpt, mhr_default = default_sam3d_checkpoint_paths()
            if not mhr.is_file():
                mhr = mhr_default
        if not ckpt.is_file():
            raise FileNotFoundError(
                "SAM 3D Body checkpoint not found. Set SAM3D_CHECKPOINT or download:\n"
                "  uv run hf download facebook/sam-3d-body-dinov3 "
                "--local-dir vaila/models/sam-3d-dinov3"
            )
        if not mhr.is_file():
            raise FileNotFoundError(f"MHR weights missing: {mhr}")

        model, model_cfg = load_sam_3d_body(
            checkpoint_path=str(ckpt),
            device=device,
            mhr_path=str(mhr),
        )
        self.estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg)

    @staticmethod
    def sam3d_to_body25(kpt: np.ndarray) -> np.ndarray:
        indices_70_to_body25 = [
            0,
            69,
            6,
            8,
            41,
            5,
            7,
            62,
            -1,
            10,
            12,
            14,
            9,
            11,
            13,
            2,
            1,
            4,
            3,
            15,
            16,
            17,
            18,
            19,
            20,
        ]
        kp25 = kpt[..., indices_70_to_body25, :]
        kp25[..., 8, :] = (kpt[..., 9, :] + kpt[..., 10, :]) / 2
        return kp25

    def infer_frame(
        self,
        img_path: Path,
        boxes: np.ndarray,
        cam_int: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if boxes.size == 0 or np.isnan(boxes).all():
            return (
                np.full((0, 25, 2), np.nan),
                np.full((0, 25, 3), np.nan),
            )
        valid = ~np.isnan(boxes).any(axis=-1)
        boxes_v = boxes[valid].reshape(-1, 4)
        if len(boxes_v) == 0:
            return (
                np.full((0, 25, 2), np.nan),
                np.full((0, 25, 3), np.nan),
            )
        cam_arr = np.asarray(cam_int, dtype=np.float32).reshape(1, 3, 3)
        batch = self.estimator.process_one_image(
            str(img_path),
            bboxes=boxes_v,
            cam_int=cam_arr,
            inference_type="body",
        )
        n = len(boxes_v)
        kpt_2d = np.zeros((n, 70, 2))
        kpt_3d = np.zeros((n, 70, 3))
        for person_id, pitem in enumerate(batch):
            kpt_2d[person_id] = pitem["pred_keypoints_2d"]
            kpt_3d[person_id] = pitem["pred_keypoints_3d"]
        kpt_2d = self.sam3d_to_body25(kpt_2d)
        kpt_3d = self.sam3d_to_body25(kpt_3d)

        full_2d = np.full((boxes.shape[0], 25, 2), np.nan, dtype=np.float32)
        full_3d = np.full((boxes.shape[0], 25, 3), np.nan, dtype=np.float32)
        full_2d[valid] = kpt_2d
        full_3d[valid] = kpt_3d
        return full_2d, full_3d


def run_preprocess_sequence(
    model: FifaSam3dBodyModel,
    image_dir: Path,
    boxes: np.ndarray,
    cam_int: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_frames, num_persons, _ = boxes.shape
    skels_2d = np.zeros((num_frames, num_persons, 25, 2), dtype=np.float32)
    skels_3d = np.zeros((num_frames, num_persons, 25, 3), dtype=np.float32)
    skels_2d.fill(np.nan)
    skels_3d.fill(np.nan)
    image_files = sorted(image_dir.glob("*.jpg"))
    if len(image_files) != num_frames:
        raise ValueError(
            f"Frame count mismatch: {len(image_files)} images vs boxes {num_frames} in {image_dir}"
        )
    for frame in tqdm(range(num_frames), desc=image_dir.name):
        skels_2d[frame], skels_3d[frame] = model.infer_frame(
            image_files[frame],
            boxes[frame],
            cam_int[frame],
        )
    return skels_2d, skels_3d


def fifa_preprocess(
    data_root: Path,
    sequences_file: Path,
    *,
    skip_existing: bool = True,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("fifa preprocess requires CUDA (SAM 3D Body).")
    data_root = data_root.resolve()
    model = FifaSam3dBodyModel(device="cuda")
    sequences = load_sequences(sequences_file)
    for seq in sequences:
        skel_2d_path = data_root / "skel_2d" / f"{seq}.npy"
        skel_3d_path = data_root / "skel_3d" / f"{seq}.npy"
        if skip_existing and skel_2d_path.is_file() and skel_3d_path.is_file():
            continue
        cam = dict(np.load(data_root / "cameras" / f"{seq}.npz"))
        boxes = np.load(data_root / "boxes" / f"{seq}.npy")
        image_dir = data_root / "images" / seq
        skel_2d, skel_3d = run_preprocess_sequence(model, image_dir, boxes, cam["K"])
        skel_2d_path.parent.mkdir(parents=True, exist_ok=True)
        skel_3d_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(skel_2d_path, skel_2d)
        np.save(skel_3d_path, skel_3d)


# --- Baseline main.py (MIT) ---
def intersection_over_plane(o: np.ndarray, d: np.ndarray) -> np.ndarray:
    t = -o[2] / d[2]
    return o + t * d


def ray_from_xy(
    xy: np.ndarray,
    k: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.array([xy[0], xy[1], 1.0])
    p_norm = np.linalg.inv(k) @ p
    x_d, y_d = p_norm[0], p_norm[1]
    r2 = x_d**2 + y_d**2
    factor = 1 + k1 * r2 + k2 * (r2**2)
    x_undist = x_d / factor
    y_undist = y_d / factor
    d_cam = np.array([x_undist, y_undist, 1.0])
    direction = r.T @ d_cam
    direction = direction / np.linalg.norm(direction)
    origin = -r.T @ t
    return origin, direction


def project_points_th(obj_pts, r, c, k_mat, k_dist):
    pts_c = (r @ ((obj_pts - c).unsqueeze(-1))).squeeze(-1)
    img_pts = pts_c[:, :2] / pts_c[:, 2:]
    r2 = (img_pts**2).sum(dim=-1, keepdim=True)
    r2 = torch.clamp(r2, 0, 0.5 / min(max(torch.abs(k_dist).max().item(), 1.0), 1.0))
    p = torch.arange(1, k_dist.shape[-1] + 1, device=k_dist.device)
    img_pts = img_pts * (torch.ones_like(r2) + (k_dist * r2.pow(p)).sum(-1, keepdim=True))
    img_pts_h = torch.cat([img_pts, torch.ones_like(img_pts[:, :1])], dim=-1)
    img_pts = (k_mat @ img_pts_h.unsqueeze(-1)).squeeze(-1)[:, :2]
    return img_pts


def minimize_reprojection_error(pts_3d, pts_2d, r, c, k_mat, k_dist, iterations: int = 10):
    t_param = torch.nn.Parameter(torch.zeros_like(pts_3d).clone().detach().requires_grad_(True))
    offset = torch.tensor([3, 3, 0.2], dtype=pts_3d.dtype, device=pts_3d.device)
    lower_bounds = t_param - offset
    upper_bounds = t_param + offset
    if torch.isnan(pts_3d).any() or torch.isnan(pts_2d).any():
        raise ValueError("NaN in reprojection tuning inputs")

    def closure():
        optimizer.zero_grad()
        projected_pts = project_points_th(pts_3d + t_param, r, c, k_mat, k_dist)
        loss = torch.nn.functional.mse_loss(projected_pts, pts_2d)
        loss.backward()
        return loss

    optimizer = optim.LBFGS([t_param], line_search_fn="strong_wolfe")
    for _ in range(iterations):
        optimizer.step(closure)
        with torch.no_grad():
            t_param.copy_(torch.clamp(t_param, lower_bounds, upper_bounds))
    return t_param.detach()


def fine_tune_translation(predictions, skels_2d, cameras, rt_hist, boxes, device: torch.device):
    num_persons = predictions.shape[0]
    mid_hip_3d = predictions[..., [7, 8], :].mean(axis=-2, keepdims=False)
    mid_hip_2d = skels_2d[..., [7, 8], :].mean(axis=-2, keepdims=False).transpose(1, 0, 2)

    r_arr = np.array([k[0] for k in rt_hist])
    t_arr = np.array([k[1] for k in rt_hist])
    c_centers = (-t_arr[:, None] @ r_arr).squeeze(1)

    camera_params = {
        "K": cameras["K"][None].repeat(num_persons, axis=0),
        "R": r_arr[None].repeat(num_persons, axis=0),
        "C": c_centers[None].repeat(num_persons, axis=0),
        "k": cameras["k"][None, ..., :2].repeat(num_persons, axis=0),
    }
    valid = ~np.isnan(boxes).any(axis=-1).transpose(1, 0)
    traj_3d = minimize_reprojection_error(
        pts_3d=torch.tensor(mid_hip_3d[valid], dtype=torch.float32, device=device),
        pts_2d=torch.tensor(mid_hip_2d[valid], dtype=torch.float32, device=device),
        r=torch.tensor(camera_params["R"][valid], dtype=torch.float32, device=device),
        c=torch.tensor(camera_params["C"][valid], dtype=torch.float32, device=device),
        k_mat=torch.tensor(camera_params["K"][valid], dtype=torch.float32, device=device),
        k_dist=torch.tensor(camera_params["k"][valid], dtype=torch.float32, device=device),
    )
    return traj_3d, valid


def process_sequence(
    boxes: np.ndarray,
    cameras: dict,
    skels_3d: np.ndarray,
    skels_2d: np.ndarray,
    video_path: Path | str,
    pitch_points: np.ndarray,
    tracker_options: CameraTrackerOptions,
    device: torch.device,
) -> np.ndarray:
    num_frames, num_persons, _ = boxes.shape
    predictions = np.zeros((num_persons, num_frames, 15, 3), dtype=np.float32)
    predictions.fill(np.nan)

    video = cv2.VideoCapture(str(video_path))
    camera_tracker = CameraTracker(
        pitch_points=pitch_points,
        fps=50.0,
        options=tracker_options,
    )
    camera_tracker.initialize(
        frame_idx=0,
        K=cameras["K"][0],
        k=cameras["k"][0],
        R=cameras["R"][0],
        t=cameras["t"][0],
    )

    rt: list[tuple[np.ndarray, np.ndarray]] = []
    for frame_idx in tqdm(range(num_frames), desc=Path(video_path).stem):
        success, img = video.read()
        if not success:
            break

        state = camera_tracker.track(
            frame_idx=frame_idx,
            frame=img,
            K=cameras["K"][frame_idx],
            dist_coeffs=cameras["k"][frame_idx],
        )
        assert state is not None
        rt.append((state.R.copy(), state.t.copy()))

        r_mat, t_vec = rt[-1]
        for person in range(num_persons):
            box = boxes[frame_idx, person]
            if np.isnan(box).any():
                continue

            skel_2d = skels_2d[frame_idx, person]
            idx = int(np.argmax(skel_2d[:, 1]))
            x, y = skel_2d[idx]
            kk = cameras["K"][frame_idx]
            dist_k = cameras["k"][frame_idx]
            o, d = ray_from_xy(
                np.array([x, y], dtype=np.float64), kk, r_mat, t_vec, dist_k[0], dist_k[1]
            )
            inter = intersection_over_plane(o, d)

            skel_3d = skels_3d[frame_idx, person]
            skel_3d = skel_3d @ r_mat
            skel_3d = skel_3d - skel_3d[idx] + inter
            predictions[person, frame_idx] = skel_3d

    video.release()

    traj_3d, valid = fine_tune_translation(predictions, skels_2d, cameras, rt, boxes, device)
    predictions[valid] = predictions[valid] + traj_3d.cpu().numpy()[:, None, :]
    for person in range(num_persons):
        predictions[person] = smoothen(predictions[person])

    cameras["R"] = np.array([k[0] for k in rt], dtype=np.float32)
    cameras["t"] = np.array([k[1] for k in rt], dtype=np.float32)
    return predictions.astype(np.float32)


def fifa_baseline(
    data_root: Path,
    sequences_file: Path,
    output_npz: Path,
    *,
    max_refine_interval: int = 1,
    export_camera: bool = False,
    visualize: bool = False,
    calibration_dir: Path | None = None,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("fifa baseline requires CUDA (camera tracker + LBFGS).")
    device = torch.device("cuda")
    data_root = data_root.resolve()
    pitch_path = data_root / "pitch_points.txt"
    if not pitch_path.is_file():
        raise FileNotFoundError(f"Missing {pitch_path} (from starter kit data).")
    pitch_points = np.loadtxt(pitch_path)

    debug_stages: tuple[str, ...] = ("projection", "flow", "mask") if visualize else ()
    cal_dir = calibration_dir
    if export_camera:
        cal_dir = (cal_dir or Path("outputs/calibration")).resolve()
        cal_dir.mkdir(parents=True, exist_ok=True)

    sequences = load_sequences(sequences_file)
    solutions: dict[str, np.ndarray] = {}
    for sequence in sequences:
        camera = dict(np.load(data_root / "cameras" / f"{sequence}.npz"))
        skel2d = np.load(data_root / "skel_2d" / f"{sequence}.npy")
        skel3d = np.load(data_root / "skel_3d" / f"{sequence}.npy")
        boxes = np.load(data_root / "boxes" / f"{sequence}.npy")
        video_path = data_root / "videos" / f"{sequence}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(video_path)

        num_frames = boxes.shape[0]
        solutions[sequence] = process_sequence(
            boxes=boxes,
            cameras=camera,
            skels_3d=skel3d[:, :, OPENPOSE_TO_OURS],
            skels_2d=skel2d[:, :, OPENPOSE_TO_OURS],
            video_path=video_path,
            pitch_points=pitch_points,
            tracker_options=CameraTrackerOptions(
                refine_interval=int(np.clip(num_frames // 500, 1, max_refine_interval)),
                debug_stages=debug_stages,
            ),
            device=device,
        )
        if export_camera and cal_dir is not None:
            np.savez(cal_dir / f"{sequence}.npz", **camera)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **solutions)  # ty: ignore[invalid-argument-type]


def fifa_pack(
    submission_full: Path,
    data_root: Path,
    output_dir: Path,
    split: Literal["val", "test"],
) -> Path:
    """Build ``submission_{split}.zip`` like ``prepare_submission.py``."""
    data = np.load(submission_full)
    sequences = load_sequences(data_root / f"sequences_{split}.txt")
    submission: dict[str, np.ndarray] = {}
    for k in sequences:
        submission[k] = data[k].astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_npz = output_dir / f"submission_{split}.npz"
    zip_path = output_dir / f"submission_{split}.zip"
    np.savez_compressed(split_npz, **submission)  # ty: ignore[invalid-argument-type]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(split_npz, arcname="submission.npz")
    split_npz.unlink(missing_ok=True)
    return zip_path


# --- Bounding boxes (YOLO / SAM3) ---
def generate_boxes_yolo(
    data_root: Path,
    sequence: str,
    *,
    yolo_model: str = "yolo11n.pt",
    max_persons: int = 25,
    conf: float = 0.25,
) -> np.ndarray:
    from ultralytics import YOLO

    image_dir = data_root / "images" / sequence
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(image_dir)
    # Use vaila/models path for standard models
    models_dir = _package_dir() / "models"
    model_path = models_dir / yolo_model
    if not model_path.exists() and not os.path.isabs(yolo_model):
        # Let YOLO download to models_dir if possible, or at least load from there if it exists
        model = YOLO(str(model_path))
    else:
        model = YOLO(yolo_model)
    counts: list[int] = []
    per_frame_boxes: list[list[list[float]]] = []
    for img_path in tqdm(images, desc=f"yolo-boxes {sequence}"):
        res = model.predict(str(img_path), conf=conf, verbose=False)
        boxes_list: list[list[float]] = []
        if res and res[0].boxes is not None and len(res[0].boxes):
            xyxy = res[0].boxes.xyxy.cpu().numpy()  # ty: ignore[unresolved-attribute]
            for row in xyxy:
                boxes_list.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        per_frame_boxes.append(boxes_list)
        counts.append(len(boxes_list))
    p = min(max_persons, max(counts) if counts else 1)
    p = max(p, 1)
    f = len(images)
    out = np.full((f, p, 4), np.nan, dtype=np.float32)
    for fi, blist in enumerate(per_frame_boxes):
        for pi in range(min(len(blist), p)):
            out[fi, pi, :] = blist[pi]
    return out


def _masks_to_boxes(mask: np.ndarray) -> np.ndarray | None:
    if mask is None or mask.size == 0:
        return None
    m = (mask > 0).astype(np.uint8)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return np.array(
        [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
        dtype=np.float32,
    )


def generate_boxes_sam3_semantic(
    data_root: Path,
    sequence: str,
    *,
    sam3_checkpoint: Path | None,
    text_prompt: str = "person",
    max_persons: int = 25,
    conf: float = 0.25,
) -> np.ndarray:
    """Per-frame Ultralytics SAM3 semantic segmentation → XYXY boxes (slow; needs ``sam3.pt``)."""
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
    except ImportError as e:
        raise RuntimeError(
            "Ultralytics SAM3 semantic predictor not available. "
            "Upgrade ultralytics and place sam3.pt; see docs."
        ) from e

    image_dir = data_root / "images" / sequence
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(image_dir)
    ckpt = sam3_checkpoint
    if ckpt is None or not ckpt.is_file():
        raise FileNotFoundError("SAM3 checkpoint required for --box-source sam3 (sam3.pt).")
    overrides = {
        "conf": conf,
        "task": "segment",
        "mode": "predict",
        "model": str(ckpt),
        "half": True,
        "save": False,
        "verbose": False,
    }
    predictor = SAM3SemanticPredictor(overrides=overrides)
    f = len(images)
    out = np.full((f, max_persons, 4), np.nan, dtype=np.float32)
    for fi, img_path in enumerate(tqdm(images, desc=f"sam3-boxes {sequence}")):
        predictor.set_image(str(img_path))
        pred = predictor(text=[text_prompt])
        boxes_acc: list[np.ndarray] = []
        if pred is not None:
            for r in pred:
                if r.masks is None:
                    continue
                masks = r.masks.data.cpu().numpy()
                for mi in range(masks.shape[0]):
                    bb = _masks_to_boxes(masks[mi])
                    if bb is not None:
                        boxes_acc.append(bb)
        for pi, bb in enumerate(boxes_acc[:max_persons]):
            out[fi, pi, :] = bb
    return out


def fifa_generate_boxes(
    data_root: Path,
    sequences_file: Path,
    *,
    source: Literal["yolo", "sam3"] = "yolo",
    yolo_model: str = "yolo11n.pt",
    sam3_checkpoint: Path | None = None,
    text_prompt: str = "person",
    max_persons: int = 25,
) -> None:
    data_root = data_root.resolve()
    boxes_dir = data_root / "boxes"
    boxes_dir.mkdir(parents=True, exist_ok=True)
    for seq in load_sequences(sequences_file):
        if source == "yolo":
            arr = generate_boxes_yolo(
                data_root, seq, yolo_model=yolo_model, max_persons=max_persons
            )
        else:
            arr = generate_boxes_sam3_semantic(
                data_root,
                seq,
                sam3_checkpoint=sam3_checkpoint,
                text_prompt=text_prompt,
                max_persons=max_persons,
            )
        np.save(boxes_dir / f"{seq}.npy", arr)


def build_fifa_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FIFA Skeletal Tracking Light — vailá pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb0 = sub.add_parser(
        "bootstrap",
        help="Symlink videos + write sequences_full/val/test.txt + pitch_points.txt",
    )
    pb0.add_argument("--videos-dir", type=Path, required=True)
    pb0.add_argument("--data-root", type=Path, required=True)
    pb0.add_argument("--val-sequences", type=Path, default=None)
    pb0.add_argument("--test-sequences", type=Path, default=None)
    pb0.add_argument("--no-copy-fallback", action="store_true")
    pb0.add_argument("--overwrite-pitch-points", action="store_true")

    pr = sub.add_parser("prepare", help="Videos → data/videos + data/images (+ sequences file)")
    pr.add_argument("--video-source", type=Path, required=True)
    pr.add_argument("--data-root", type=Path, required=True)
    pr.add_argument("--sequences-out", type=Path, default=None)
    pr.add_argument("--extract-fps", type=float, default=None)

    pb = sub.add_parser("boxes", help="Generate boxes/*.npy (YOLO or SAM3)")
    pb.add_argument("--data-root", type=Path, required=True)
    pb.add_argument("--sequences", type=Path, required=True)
    pb.add_argument("--source", choices=("yolo", "sam3"), default="yolo")
    pb.add_argument("--yolo-model", type=str, default="yolo11n.pt")
    pb.add_argument("--sam3-checkpoint", type=Path, default=None)
    pb.add_argument("--text-prompt", type=str, default="person")
    pb.add_argument("--max-persons", type=int, default=25)

    pp = sub.add_parser("preprocess", help="SAM 3D Body → skel_2d / skel_3d")
    pp.add_argument("--data-root", type=Path, required=True)
    pp.add_argument("--sequences", type=Path, required=True)
    pp.add_argument("--no-skip", action="store_true")

    pl = sub.add_parser("baseline", help="Starter-kit baseline → submission NPZ")
    pl.add_argument("--data-root", type=Path, required=True)
    pl.add_argument("--sequences", type=Path, required=True)
    pl.add_argument("--output", type=Path, required=True)
    pl.add_argument("--refine-interval", type=int, default=1)
    pl.add_argument("--export-camera", action="store_true")
    pl.add_argument("--visualize", action="store_true")
    pl.add_argument("--calibration-dir", type=Path, default=None)

    pk = sub.add_parser("pack", help="Split submission_full.npz → submission_{val|test}.zip")
    pk.add_argument("--submission-full", type=Path, required=True)
    pk.add_argument("--data-root", type=Path, required=True)
    pk.add_argument("--output-dir", type=Path, required=True)
    pk.add_argument("--split", choices=("val", "test"), required=True)

    dk = sub.add_parser(
        "dlt-export",
        help="cameras/*.npz (K,R,t,k per frame) -> .dlt2d / .dlt3d for rec2d.py / rec3d.py",
    )
    dk.add_argument("--cameras-dir", type=Path, required=True)
    dk.add_argument("--output-dir", type=Path, required=True)
    dk.add_argument("--mode", choices=("2d", "3d", "both"), default="both")
    dk.add_argument("--undistort-pixels-dir", type=Path, default=None)

    return p


def main_fifa_cli(argv: list[str] | None = None) -> None:
    args = build_fifa_argparser().parse_args(argv)
    if args.cmd == "bootstrap":
        try:
            from .fifa_bootstrap import prepare_fifa_data_layout
        except ImportError:
            from fifa_bootstrap import (  # ty: ignore[unresolved-import]
                prepare_fifa_data_layout,
            )
        res = prepare_fifa_data_layout(
            videos_dir=args.videos_dir,
            data_root=args.data_root,
            val_sequences=args.val_sequences,
            test_sequences=args.test_sequences,
            copy_fallback=not args.no_copy_fallback,
            overwrite_pitch_points=args.overwrite_pitch_points,
        )
        print(res.as_summary())
    elif args.cmd == "prepare":
        fifa_prepare(
            args.video_source,
            args.data_root,
            sequences_out=args.sequences_out,
            extract_fps=args.extract_fps,
        )
    elif args.cmd == "boxes":
        fifa_generate_boxes(
            args.data_root,
            args.sequences,
            source=args.source,
            yolo_model=args.yolo_model,
            sam3_checkpoint=args.sam3_checkpoint,
            text_prompt=args.text_prompt,
            max_persons=args.max_persons,
        )
    elif args.cmd == "preprocess":
        fifa_preprocess(args.data_root, args.sequences, skip_existing=not args.no_skip)
    elif args.cmd == "baseline":
        fifa_baseline(
            args.data_root,
            args.sequences,
            args.output,
            max_refine_interval=args.refine_interval,
            export_camera=args.export_camera,
            visualize=args.visualize,
            calibration_dir=args.calibration_dir,
        )
    elif args.cmd == "pack":
        fifa_pack(args.submission_full, args.data_root, args.output_dir, args.split)
    elif args.cmd == "dlt-export":
        try:
            from .fifa_to_dlt import run_cli as _fifa_dlt_run
        except ImportError:
            from fifa_to_dlt import run_cli as _fifa_dlt_run  # ty: ignore[unresolved-import]

        raise SystemExit(
            _fifa_dlt_run(
                input_path=args.cameras_dir,
                output_dir=args.output_dir,
                mode=args.mode,
                undistort_pixels_dir=args.undistort_pixels_dir,
            )
        )
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main_fifa_cli()
