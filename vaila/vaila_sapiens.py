"""
Project: vailá
Script: vaila_sapiens.py
Authors: Paulo Santiago, Sergio Barroso, Felipe Dias, Lennin Abrão
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 06 July 2026
Update Date: 07 July 2026
Version: 0.3.75

Description:
    Sapiens2 Pose video inference for vailá (Meta 308-keypoint top-down pose).
    Install: ``uv sync --extra sapiens`` then ``bash bin/setup_sapiens2.sh``.
    Inference requires NVIDIA CUDA. Default model ``1b`` fits RTX 4090 24 GiB.

Usage (CLI — pass ``-i`` input and ``-o`` output parent; no GUI):

    # Single video → creates processed_sapiens_<timestamp>/ under -o
    uv run vaila/vaila_sapiens.py \\
        -i /path/to/video.mp4 \\
        -o /path/to/output_parent \\
        --model 1b

    # Batch: directory of videos
    uv run vaila/vaila_sapiens.py \\
        -i /path/to/videos_dir \\
        -o /path/to/output_parent \\
        --model 1b --stride 1

    # Dry-run (no GPU inference)
    uv run vaila/vaila_sapiens.py \\
        -i tests/markerless_2d_analysis/ \\
        -o /tmp/sapiens_out \\
        --model 1b --dry-run

    uv run vaila/vaila_sapiens.py --download-weights --model 1b

    # Largest model (5B) — accept HF license first; ~24 GiB checkpoint + heavy VRAM
    uv run vaila/vaila_sapiens.py --download-weights --model 5b

    # GUI (optional): omit -i/-o → Tkinter dialog; or vaila.py → YOLO + FB → Sapiens2 Pose

    # Biomechanics outputs (auto): <stem>_markers.csv, sapiens_vaila_*.csv, sapiens_points.csv
    # Feed *_markers.csv into rec2d.py / rec3d.py; long kpts in *_sapiens_vaila.csv

More detail: ``vaila/help/vaila_sapiens.md`` / ``vaila/help/vaila_sapiens.html``.

License:
    This program is licensed under the GNU Affero General Public License v3.0.
    For more details, visit: https://www.gnu.org/licenses/agpl-3.0.html
    Visit the project repository: https://github.com/vaila-multimodaltoolbox
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import gc
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tkinter as tk
import webbrowser
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

try:
    from .geometric_reid import GeometricFrameLinker
    from .sam_postprocess import VAILA_ANCHORS, _anchor_xy, _format_cell
    from .vaila_sam import _open_sam3_video_writer
except ImportError:
    from geometric_reid import GeometricFrameLinker
    from sam_postprocess import VAILA_ANCHORS, _anchor_xy, _format_cell
    from vaila_sam import _open_sam3_video_writer

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
DATASET = "shutterstock_goliath_3po"
RES_SUFFIX = "1024x768"
DEFAULT_MODEL_KEY = "1b"

SAPIENS_POSE_MODELS: dict[str, dict[str, str]] = {
    "0.4b": {
        "arch": "sapiens2_0.4b",
        "ckpt": "sapiens2_0.4b_pose.safetensors",
        "hf_repo": "facebook/sapiens2-pose-0.4b",
    },
    "0.8b": {
        "arch": "sapiens2_0.8b",
        "ckpt": "sapiens2_0.8b_pose.safetensors",
        "hf_repo": "facebook/sapiens2-pose-0.8b",
    },
    "1b": {
        "arch": "sapiens2_1b",
        "ckpt": "sapiens2_1b_pose.safetensors",
        "hf_repo": "facebook/sapiens2-pose-1b",
    },
    "5b": {
        "arch": "sapiens2_5b",
        "ckpt": "sapiens2_5b_pose.safetensors",
        "hf_repo": "facebook/sapiens2-pose-5b",
    },
}

DETECTOR_HF_REPO = "facebook/detr-resnet-101-dc5"
DETECTOR_DIRNAME = "detr-resnet-101-dc5"
# COCO body indices inside the 308-keypoint Sociopticon topology (left/right hip).
SAPIENS_MID_HIP_KPT_IDS = (11, 12)

SAPIENS_OUTPUT_FILE_GLOSSARY = """\
Biomechanics / vailá downstream CSVs (written after every run)
--------------------------------------------------------------
  <stem>_markers.csv
      getpixelvideo + REC2D/REC3D point format:
      frame,p1_x,p1_y,...,pN_x,pN_y  (bbox bottom-center anchor, stable pN slots)

  sapiens_vaila_center.csv   frame,x1,y1,...,xN,yN — bbox center (same schema as sam_vaila_*)
  sapiens_vaila_bottom.csv   bbox bottom-center (foot proxy)
  sapiens_vaila_top.csv      bbox top-center
  sapiens_vaila_left.csv     bbox left-center
  sapiens_vaila_right.csv    bbox right-center

  sapiens_points.csv
      frame,p1_x,p1_y,p1_cx,p1_cy,p1_hx,p1_hy,...  — canonical foot + bbox center + mid-hip

  sapiens_id_map.csv         pN,stable_id,n_frames,first_frame,last_frame
  sapiens_tracks.csv         frame,stable_id,x1,y1,x2,y2,mean_kpt_score
  sapiens_bbox_tracks.csv    SAM-compatible bbox tracks (frame,obj_id,x_px,y_px,w_px,h_px,score)
                             — load in getpixelvideo via Load Tracking CSV

  <stem>_id_NN_sapiens_pose.csv   getpixelvideo wide pose (frame + named x/y per keypoint)
  <stem>_getpixelvideo_pose.csv   alias when a single person is tracked (hardlink or copy)

  <stem>_sapiens_vaila.csv   long table: frame,person_id,kpt_idx,x,y,score (all 308 keypoints)
  <stem>_predictions.json    full per-frame instances (bbox + keypoints)
"""


def _sapiens_log(message: str) -> None:
    """Terminal progress line (>> prefix survives absl/mediapipe stdout filtering)."""
    print(f">> vaila/vaila_sapiens: {message}", flush=True)


def _progress_quiet() -> bool:
    flag = os.environ.get("TQDM_DISABLE", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _try_import_tqdm() -> Any | None:
    try:
        from tqdm import tqdm

        return tqdm
    except ImportError:
        return None


def _frame_progress_interval(n_frames: int) -> int:
    if n_frames <= 0:
        return 30
    return max(1, min(60, n_frames // 20))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sapiens2_root() -> Path:
    env = os.environ.get("SAPIENS_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    root = _repo_root()
    preferred = root / ".local" / "third_party" / "sapiens2"
    if preferred.is_dir():
        return preferred
    legacy = root / "sapiens2"
    if legacy.is_dir():
        return legacy
    return preferred


def _checkpoint_root() -> Path:
    env = os.environ.get("SAPIENS_CHECKPOINT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _repo_root() / "vaila" / "models" / "sapiens2"


def _normalize_model_key(model: str) -> str:
    key = model.strip().lower().replace("sapiens2_", "").replace("_pose", "")
    if key not in SAPIENS_POSE_MODELS:
        allowed = ", ".join(sorted(SAPIENS_POSE_MODELS))
        raise ValueError(f"Unknown model {model!r}; choose one of: {allowed}")
    return key


@dataclass(frozen=True)
class SapiensModelSpec:
    model_key: str
    arch: str
    config_path: Path
    checkpoint_path: Path
    detector_path: Path


def resolve_model_spec(model: str = DEFAULT_MODEL_KEY) -> SapiensModelSpec:
    """Resolve pose config + checkpoint paths for a Sapiens2 model size."""
    key = _normalize_model_key(model)
    meta = SAPIENS_POSE_MODELS[key]
    ckpt_root = _checkpoint_root()
    config_rel = (
        f"configs/keypoints308/{DATASET}/{meta['arch']}_keypoints308_{DATASET}-{RES_SUFFIX}.py"
    )
    pose_dir = _sapiens2_root() / "sapiens" / "pose"
    config_path = (pose_dir / config_rel).resolve()
    checkpoint_path = (ckpt_root / "pose" / meta["ckpt"]).resolve()
    detector_path = (ckpt_root / "detector" / DETECTOR_DIRNAME).resolve()
    return SapiensModelSpec(
        model_key=key,
        arch=meta["arch"],
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        detector_path=detector_path,
    )


def _default_pose_batch_size(model_key: str) -> int:
    if model_key == "5b":
        return 1
    if model_key in ("0.4b", "0.8b"):
        return 4
    return 2


def _is_sapiens_derived_video(path: Path) -> bool:
    """Skip overlay outputs and files inside prior ``processed_sapiens_*`` runs."""
    name = path.name.lower()
    if name.endswith("_sapiens_overlay.mp4") or name.endswith("_sapiens_overlay.avi"):
        return True
    return any(parent.name.startswith("processed_sapiens_") for parent in path.parents)


def _find_videos(path: Path) -> list[Path]:
    if path.is_file():
        return [path.resolve()]
    out: list[Path] = []
    for p in sorted(path.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if _is_sapiens_derived_video(p):
            continue
        out.append(p.resolve())
    return out


@contextlib.contextmanager
def _sapiens_pose_context() -> Iterator[Path]:
    """Temporarily use the upstream pose tree (configs + vis helpers)."""
    pose_dir = (_sapiens2_root() / "sapiens" / "pose").resolve()
    vis_dir = pose_dir / "tools" / "vis"
    old_cwd = os.getcwd()
    inserted: list[str] = []
    for entry in (pose_dir, vis_dir):
        s = str(entry)
        if s not in sys.path:
            sys.path.insert(0, s)
            inserted.append(s)
    os.chdir(pose_dir)
    try:
        yield pose_dir
    finally:
        os.chdir(old_cwd)
        for s in inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(s)


def _require_sapiens_installed() -> None:
    if importlib.util.find_spec("sapiens") is None:
        raise ImportError(
            "sapiens package not found. Run: uv sync --extra sapiens && bash bin/setup_sapiens2.sh"
        )


def _require_cuda() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Sapiens2 Pose in vailá requires an NVIDIA GPU."
            )
        dev_id = int(os.environ.get("SAPIENS_DEVICE", "0"))
        return f"cuda:{dev_id}"
    except ImportError as e:
        raise ImportError("PyTorch is required for Sapiens2 inference.") from e


class PoseInferenceSession:
    """Loads Sapiens2 pose + DETR once and runs per-frame top-down inference."""

    def __init__(
        self,
        spec: SapiensModelSpec,
        *,
        device: str,
        bbox_thr: float = 0.3,
        nms_thr: float = 0.3,
        kpt_thr: float = 0.3,
        radius: int = 4,
        thickness: int = 2,
        flip_test: bool = False,
        max_persons: int = 8,
        pose_batch_size: int = 2,
    ) -> None:
        _require_sapiens_installed()
        if not spec.config_path.is_file():
            raise FileNotFoundError(
                f"Sapiens2 pose config not found: {spec.config_path}\n"
                "Run: bash bin/setup_sapiens2.sh"
            )
        if not spec.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Sapiens2 pose checkpoint not found: {spec.checkpoint_path}\n"
                "Run: bash bin/setup_sapiens2.sh or vaila/vaila_sapiens.py --download-weights"
            )
        if not spec.detector_path.is_dir():
            raise FileNotFoundError(
                f"DETR detector not found: {spec.detector_path}\nRun: bash bin/setup_sapiens2.sh"
            )

        self.spec = spec
        self.device = device
        self.args = SimpleNamespace(
            device=device,
            det_checkpoint=str(spec.detector_path),
            bbox_thr=float(bbox_thr),
            nms_thr=float(nms_thr),
            kpt_thr=float(kpt_thr),
            radius=int(radius),
            thickness=int(thickness),
        )
        self._detector_cache: dict[str, Any] = {}
        self._visualize_fn: Any | None = None
        self.model: Any = None
        self.flip_test = bool(flip_test)
        self.max_persons = max(1, int(max_persons))
        self.pose_batch_size = max(1, int(pose_batch_size))

        _sapiens_log(
            f"Loading Sapiens2 pose ({spec.arch}) from {spec.checkpoint_path.name} on {device} …"
        )
        with _sapiens_pose_context():
            from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo
            from sapiens.pose.models import init_model

            self.model = init_model(
                str(spec.config_path),
                str(spec.checkpoint_path),
                device=device,
            )
            if int(self.model.cfg.num_keypoints) == 308:
                self.model.pose_metainfo = parse_pose_metainfo(
                    {"from_file": "configs/_base_/keypoints308.py"}
                )
            codec_type = self.model.cfg.codec.pop("type")
            if codec_type != "UDPHeatmap":
                raise RuntimeError(f"Unsupported codec type: {codec_type}")
            self.model.codec = UDPHeatmap(**self.model.cfg.codec)

            try:
                from pose_render_utils import visualize_keypoints

                self._visualize_fn = visualize_keypoints
            except ImportError:
                self._visualize_fn = None

        n_kp = int(getattr(self.model.cfg, "num_keypoints", 0) or 0)
        _sapiens_log(f"Pose model ready ({n_kp or '?'} keypoints, flip_test={self.flip_test})")
        self._warm_detector()

    def _warm_detector(self) -> None:
        _sapiens_log("Warming DETR person detector (first inference) …")
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self._detect_persons(dummy)
        _sapiens_log("DETR detector ready — BatchNorm buffer warnings from Hugging Face are normal")

    def _get_detector(self) -> tuple[Any, Any]:
        if "model" in self._detector_cache:
            return self._detector_cache["proc"], self._detector_cache["model"]
        import logging

        from transformers import DetrForObjectDetection, DetrImageProcessor

        _sapiens_log(f"Loading DETR person detector from {Path(self.args.det_checkpoint).name} …")
        tf_logger = logging.getLogger("transformers")
        prev_level = tf_logger.level
        tf_logger.setLevel(logging.ERROR)
        try:
            proc = DetrImageProcessor.from_pretrained(self.args.det_checkpoint)
            model = (
                DetrForObjectDetection.from_pretrained(self.args.det_checkpoint)
                .eval()
                .to(self.device)
            )
        finally:
            tf_logger.setLevel(prev_level)
        self._detector_cache["proc"] = proc
        self._detector_cache["model"] = model
        return proc, model

    def _detect_persons(self, image_bgr: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image
        from sapiens.pose.evaluators import nms

        proc, model = self._get_detector()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        inputs = proc(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([image_rgb.shape[:2]], device=self.device)
        results = proc.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.args.bbox_thr
        )[0]
        person_mask = results["labels"] == 1
        boxes = results["boxes"][person_mask].cpu().numpy()
        scores = results["scores"][person_mask].cpu().numpy().reshape(-1, 1)
        bboxes = np.concatenate([boxes, scores], axis=1)
        bboxes = bboxes[nms(bboxes, self.args.nms_thr), :]
        if len(bboxes) == 0:
            h, w = image_rgb.shape[:2]
            bboxes = np.array([[0, 0, w - 1, h - 1, 1.0]], dtype=np.float32)
        order = np.argsort(-bboxes[:, 4])
        bboxes = bboxes[order][: self.max_persons]
        return bboxes[:, :4]

    def process_frame(
        self, image_bgr: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Return keypoints, scores, bboxes for all detected persons."""
        import torch

        bboxes = self._detect_persons(image_bgr)
        keypoints: list[np.ndarray] = []
        keypoint_scores: list[np.ndarray] = []
        batch_size = self.pose_batch_size
        for start in range(0, len(bboxes), batch_size):
            chunk = bboxes[start : start + batch_size]
            inputs_list = []
            data_samples_list = []
            for bbox in chunk:
                data_info: dict[str, Any] = {"img": image_bgr}
                data_info["bbox"] = bbox[None]
                data_info["bbox_score"] = np.ones(1, dtype=np.float32)
                data = self.model.pipeline(data_info)
                data = self.model.data_preprocessor(data)
                inputs_list.append(data["inputs"])
                data_samples_list.append(data["data_samples"])

            inputs = torch.cat(inputs_list, dim=0)
            with torch.inference_mode():
                pred = self.model(inputs)
                if self.flip_test and self.model.pose_metainfo is not None:
                    pred_flipped = self.model(inputs.flip(-1))
                    pred_flipped = pred_flipped.flip(-1)
                    flip_indices = self.model.pose_metainfo["flip_indices"]
                    pred_flipped = pred_flipped[:, flip_indices]
                    pred = (pred + pred_flipped) / 2.0

            pred_np = pred.cpu().numpy()
            for i, data_samples in enumerate(data_samples_list):
                keypoints_i, keypoint_scores_i = self.model.codec.decode(pred_np[i])
                input_size = data_samples["meta"]["input_size"]
                bbox_center = data_samples["meta"]["bbox_center"]
                bbox_scale = data_samples["meta"]["bbox_scale"]
                keypoints_i = keypoints_i / input_size * bbox_scale + bbox_center - 0.5 * bbox_scale
                keypoints.append(keypoints_i[0])
                keypoint_scores.append(keypoint_scores_i[0])
        return keypoints, keypoint_scores, [np.asarray(b) for b in bboxes]

    def render_overlay(
        self,
        image_bgr: np.ndarray,
        keypoints: list[np.ndarray],
        keypoint_scores: list[np.ndarray],
    ) -> np.ndarray:
        if not keypoints:
            return image_bgr.copy()
        if self._visualize_fn is not None and self.model.pose_metainfo is not None:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            vis_rgb = self._visualize_fn(
                image=image_rgb,
                keypoints=keypoints,
                keypoints_visible=np.ones_like(keypoint_scores) > 0,
                keypoint_scores=keypoint_scores,
                radius=self.args.radius,
                thickness=self.args.thickness,
                kpt_thr=self.args.kpt_thr,
                skeleton=self.model.pose_metainfo["skeleton_links"],
                kpt_color=self.model.pose_metainfo["keypoint_colors"],
                link_color=self.model.pose_metainfo["skeleton_link_colors"],
            )
            return cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
        out = image_bgr.copy()
        for kpts, scores in zip(keypoints, keypoint_scores, strict=False):
            for (x, y), sc in zip(kpts, scores, strict=False):
                if float(sc) < self.args.kpt_thr:
                    continue
                cv2.circle(out, (int(x), int(y)), self.args.radius, (0, 255, 0), -1)
        return out

    def close(self) -> None:
        """Drop pose + DETR weights so the next video can reclaim VRAM."""
        self._detector_cache.clear()
        self.model = None
        self._visualize_fn = None


def _release_sapiens_gpu_memory(session: PoseInferenceSession | None = None) -> None:
    """Best-effort VRAM release between Sapiens2 batch items."""
    if session is not None:
        with contextlib.suppress(Exception):
            session.close()
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
        with contextlib.suppress(Exception):
            torch.cuda.ipc_collect()


def instances_from_frame(
    keypoints: list[np.ndarray],
    keypoint_scores: list[np.ndarray],
    bboxes: list[np.ndarray],
) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for kpts, scores, bbox in zip(keypoints, keypoint_scores, bboxes, strict=False):
        instances.append(
            {
                "bbox": [float(v) for v in np.asarray(bbox).reshape(-1)[:4]],
                "keypoints": np.asarray(kpts, dtype=float).tolist(),
                "keypoint_scores": np.asarray(scores, dtype=float).reshape(-1).tolist(),
            }
        )
    return instances


def write_vaila_pose_csv(path: Path, rows: list[tuple[int, int, int, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "person_id", "kpt_idx", "x", "y", "score"])
        writer.writerows(rows)


def flatten_instances_to_csv_rows(
    frame_idx: int, instances: list[dict[str, Any]]
) -> list[tuple[int, int, int, float, float, float]]:
    rows: list[tuple[int, int, int, float, float, float]] = []
    for inst in instances:
        person_id = int(inst.get("stable_id", 0))
        kpts = inst.get("keypoints", [])
        scores = inst.get("keypoint_scores", [])
        for kpt_idx, ((x, y), sc) in enumerate(zip(kpts, scores, strict=False)):
            rows.append((frame_idx, person_id, kpt_idx, float(x), float(y), float(sc)))
    return rows


def _nearest_pose_frame(frame_idx: int, keys: list[int]) -> int:
    if not keys:
        return 0
    return int(min(keys, key=lambda k: (abs(k - frame_idx), k)))


def _assign_stable_ids(
    frame_idx: int,
    instances: list[dict[str, Any]],
    linker: GeometricFrameLinker,
) -> list[dict[str, Any]]:
    """Attach cross-frame ``stable_id`` to each instance (Hungarian bbox linking)."""
    if not instances:
        return []
    dets = [
        {"xyxy": tuple(float(v) for v in inst["bbox"][:4])}
        for inst in instances  # type: ignore[misc]
    ]
    linked = linker.assign_frame(frame_idx, dets)
    out: list[dict[str, Any]] = []
    for inst, ld in zip(instances, linked, strict=True):
        tagged = dict(inst)
        tagged["stable_id"] = int(ld["stable_id"])
        out.append(tagged)
    return out


def _mid_hip_xy(inst: dict[str, Any], *, kpt_thr: float) -> tuple[float, float] | None:
    kpts = inst.get("keypoints") or []
    scores = inst.get("keypoint_scores") or []
    pts: list[tuple[float, float]] = []
    for idx in SAPIENS_MID_HIP_KPT_IDS:
        if idx >= len(kpts):
            continue
        sc = float(scores[idx]) if idx < len(scores) else 0.0
        if sc < kpt_thr:
            continue
        x, y = kpts[idx][:2]
        pts.append((float(x), float(y)))
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _expand_pose_timeline(
    pose_by_frame: dict[int, list[dict[str, Any]]],
    n_frames: int,
) -> dict[int, list[dict[str, Any]]]:
    """Fill every video frame using the nearest inferred pose (matches overlay logic)."""
    if n_frames <= 0:
        return {}
    keys = sorted(pose_by_frame.keys())
    if not keys:
        return dict.fromkeys(range(n_frames), [])
    return {f: list(pose_by_frame.get(_nearest_pose_frame(f, keys), [])) for f in range(n_frames)}


def _collect_stable_slots(
    timeline: dict[int, list[dict[str, Any]]],
) -> tuple[list[int], dict[int, int]]:
    """Return sorted stable IDs and stable_id -> pN slot (1-based)."""
    stable_ids = sorted(
        {
            int(inst["stable_id"])
            for insts in timeline.values()
            for inst in insts
            if "stable_id" in inst
        }
    )
    slot_by_id = {sid: pn for pn, sid in enumerate(stable_ids, start=1)}
    return stable_ids, slot_by_id


def _instance_for_stable_id(
    instances: list[dict[str, Any]], stable_id: int
) -> dict[str, Any] | None:
    for inst in instances:
        if int(inst.get("stable_id", -1)) == stable_id:
            return inst
    return None


def _anchor_from_instance(inst: dict[str, Any], anchor: str) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in inst["bbox"][:4]]
    px, py, pw, ph = x1, y1, x2 - x1, y2 - y1
    return _anchor_xy(px, py, pw, ph, anchor)


def write_sapiens_biomechanics_csvs(
    output_dir: Path,
    stem: str,
    timeline: dict[int, list[dict[str, Any]]],
    *,
    kpt_thr: float = 0.3,
) -> list[Path]:
    """Write SAM-style anchor/points/tracks CSVs plus ``<stem>_markers.csv`` for REC2D/REC3D."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stable_ids, slot_by_id = _collect_stable_slots(timeline)
    n_slots = len(stable_ids)
    n_frames = max(timeline.keys(), default=-1) + 1
    if n_frames <= 0:
        return []

    id_stats: dict[int, dict[str, int]] = {
        sid: {"n": 0, "first": -1, "last": -1} for sid in stable_ids
    }
    written: list[Path] = []

    header_parts = ["frame"]
    for i in range(1, n_slots + 1):
        header_parts.extend([f"x{i}", f"y{i}"])
    header_line = ",".join(header_parts)

    for anchor in VAILA_ANCHORS:
        rows: list[str] = []
        for frame_idx in range(n_frames):
            cells: list[str] = [str(frame_idx)]
            for sid in stable_ids:
                inst = _instance_for_stable_id(timeline.get(frame_idx, []), sid)
                if inst is None:
                    cells.extend(["", ""])
                    continue
                x, y = _anchor_from_instance(inst, anchor)
                cells.append(_format_cell(x))
                cells.append(_format_cell(y))
            rows.append(",".join(cells))
        out_path = output_dir / f"sapiens_vaila_{anchor}.csv"
        out_path.write_text(
            header_line + "\n" + "\n".join(rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
        written.append(out_path)

    pts_header = ["frame"]
    for pn in range(1, n_slots + 1):
        pts_header.extend(
            [f"p{pn}_x", f"p{pn}_y", f"p{pn}_cx", f"p{pn}_cy", f"p{pn}_hx", f"p{pn}_hy"]
        )
    pts_rows: list[str] = []
    track_rows: list[list[Any]] = [["frame", "stable_id", "x1", "y1", "x2", "y2", "mean_kpt_score"]]

    for frame_idx in range(n_frames):
        cells: list[str] = [str(frame_idx)]
        for sid in stable_ids:
            inst = _instance_for_stable_id(timeline.get(frame_idx, []), sid)
            if inst is None:
                cells.extend(["", ""] * 3)
                continue
            stats = id_stats[sid]
            stats["n"] += 1
            if stats["first"] < 0:
                stats["first"] = frame_idx
            stats["last"] = frame_idx

            foot = _anchor_from_instance(inst, "bottom")
            center = _anchor_from_instance(inst, "center")
            hip = _mid_hip_xy(inst, kpt_thr=kpt_thr)
            cells.append(_format_cell(foot[0]))
            cells.append(_format_cell(foot[1]))
            cells.append(_format_cell(center[0]))
            cells.append(_format_cell(center[1]))
            cells.append(_format_cell(hip[0] if hip else None))
            cells.append(_format_cell(hip[1] if hip else None))

            x1, y1, x2, y2 = [float(v) for v in inst["bbox"][:4]]
            scores = inst.get("keypoint_scores") or []
            mean_sc = float(np.mean(scores)) if scores else float("nan")
            track_rows.append([frame_idx, sid, x1, y1, x2, y2, mean_sc])

        pts_rows.append(",".join(cells))

    points_path = output_dir / "sapiens_points.csv"
    points_path.write_text(
        ",".join(pts_header) + "\n" + "\n".join(pts_rows) + ("\n" if pts_rows else ""),
        encoding="utf-8",
    )
    written.append(points_path)

    id_map_path = output_dir / "sapiens_id_map.csv"
    with id_map_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pN", "stable_id", "n_frames", "first_frame", "last_frame"])
        for sid in stable_ids:
            st = id_stats[sid]
            writer.writerow([slot_by_id[sid], sid, st["n"], st["first"], st["last"]])
    written.append(id_map_path)

    tracks_path = output_dir / "sapiens_tracks.csv"
    with tracks_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(track_rows)
    written.append(tracks_path)

    bbox_sam_rows: list[list[Any]] = [["frame", "obj_id", "x_px", "y_px", "w_px", "h_px", "score"]]
    for row in track_rows[1:]:
        frame_idx, sid, x1, y1, x2, y2, mean_sc = row
        bbox_sam_rows.append(
            [frame_idx, sid, x1, y1, float(x2) - float(x1), float(y2) - float(y1), mean_sc]
        )
    bbox_tracks_path = output_dir / "sapiens_bbox_tracks.csv"
    with bbox_tracks_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(bbox_sam_rows)
    written.append(bbox_tracks_path)

    markers_path = output_dir / f"{stem}_markers.csv"
    with markers_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        m_header = ["frame"]
        for pn in range(1, n_slots + 1):
            m_header.extend([f"p{pn}_x", f"p{pn}_y"])
        writer.writerow(m_header)
        for frame_idx in range(n_frames):
            row: list[Any] = [frame_idx]
            for sid in stable_ids:
                inst = _instance_for_stable_id(timeline.get(frame_idx, []), sid)
                if inst is None:
                    row.extend(["", ""])
                else:
                    x, y = _anchor_from_instance(inst, "bottom")
                    row.extend([f"{x:.4f}", f"{y:.4f}"])
            writer.writerow(row)
    written.append(markers_path)

    print(
        f"[Sapiens2] Biomechanics CSVs: {len(written)} files "
        f"({n_slots} stable person slot(s), {n_frames} frames)",
        flush=True,
    )
    return written


def _sanitize_keypoint_label(name: str, idx: int) -> str:
    """Turn Sociopticon keypoint names into safe CSV column prefixes."""
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name).strip()).strip("_")
    if not s or s[0].isdigit():
        s = f"kp{idx:03d}_{s}" if s else f"kp{idx:03d}"
    return s


def _resolve_keypoint_labels(n_kp: int, keypoint_names: list[str] | None) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for i in range(n_kp):
        raw = keypoint_names[i] if keypoint_names and i < len(keypoint_names) else f"kp{i:03d}"
        lab = _sanitize_keypoint_label(raw, i)
        base = lab
        suffix = 2
        while lab in seen:
            lab = f"{base}_{suffix}"
            suffix += 1
        seen.add(lab)
        labels.append(lab)
    return labels


def _make_getpixelvideo_pose_alias(output_dir: Path, stem: str, src: Path) -> Path | None:
    """Hardlink or copy ``<stem>_getpixelvideo_pose.csv`` next to a per-id pose CSV."""
    dst = output_dir / f"{stem}_getpixelvideo_pose.csv"
    if not src.is_file():
        return None
    with contextlib.suppress(OSError):
        if dst.exists() or dst.is_symlink():
            dst.unlink()
    try:
        os.link(str(src), str(dst))
        return dst
    except OSError:
        pass
    with contextlib.suppress(OSError):
        shutil.copy2(str(src), str(dst))
    return dst if dst.is_file() else None


def write_sapiens_getpixelvideo_pose_csvs(
    output_dir: Path,
    stem: str,
    timeline: dict[int, list[dict[str, Any]]],
    *,
    kpt_thr: float = 0.3,
    keypoint_names: list[str] | None = None,
) -> list[Path]:
    """Write wide per-person pose CSVs loadable via getpixelvideo **Load** button."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stable_ids, _slot_by_id = _collect_stable_slots(timeline)
    if not stable_ids:
        return []

    n_frames = max(timeline.keys(), default=-1) + 1
    if n_frames <= 0:
        return []

    n_kp = 0
    for instances in timeline.values():
        for inst in instances:
            kpts = inst.get("keypoints") or []
            n_kp = max(n_kp, len(kpts))
    if n_kp <= 0:
        return []

    labels = _resolve_keypoint_labels(n_kp, keypoint_names)
    header: list[str] = ["frame"]
    for lab in labels:
        header.extend([f"{lab}_x", f"{lab}_y"])

    written: list[Path] = []
    for sid in stable_ids:
        out_path = output_dir / f"{stem}_id_{int(sid):02d}_sapiens_pose.csv"
        rows: list[list[Any]] = []
        for frame_idx in range(n_frames):
            inst = _instance_for_stable_id(timeline.get(frame_idx, []), sid)
            row: list[Any] = [frame_idx]
            if inst is None:
                row.extend([""] * (n_kp * 2))
            else:
                kpts = inst.get("keypoints") or []
                scores = inst.get("keypoint_scores") or []
                for i in range(n_kp):
                    if i < len(kpts):
                        sc = float(scores[i]) if i < len(scores) else 0.0
                        if sc >= kpt_thr:
                            x, y = kpts[i][:2]
                            row.extend([f"{float(x):.4f}", f"{float(y):.4f}"])
                        else:
                            row.extend(["", ""])
                    else:
                        row.extend(["", ""])
            rows.append(row)
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)
        written.append(out_path)

    if len(written) == 1:
        alias = _make_getpixelvideo_pose_alias(output_dir, stem, written[0])
        if alias is not None:
            written.append(alias)

    _sapiens_log(
        f"getpixelvideo pose CSVs: {len(written)} file(s) "
        f"({len(stable_ids)} person(s), {n_kp} keypoints, {n_frames} frames)"
    )
    return written


def _write_failure_marker(output_dir: Path, video_path: Path, reason: str) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "FAILED_sapiens.txt").write_text(
            f"Sapiens2 Pose FAILED\n"
            f"video={video_path.resolve()}\n"
            f"timestamp={dt.datetime.now().isoformat(timespec='seconds')}\n"
            f"reason={reason}\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _format_sapiens_cli_command(
    input_path: Path | str,
    output_parent: Path | str,
    *,
    model: str = DEFAULT_MODEL_KEY,
    stride: int = 1,
    kpt_thr: float = 0.3,
    bbox_thr: float = 0.3,
    nms_thr: float = 0.3,
    device: int = 0,
    save_overlay: bool = True,
    output_base: Path | str | None = None,
    flip_test: bool = False,
) -> str:
    """Build copy-paste CLI equivalent to a GUI Sapiens2 run."""
    argv = _build_sapiens_cli_argv(
        input_path=Path(input_path),
        out_parent=Path(output_parent),
        output_base=Path(output_base) if output_base is not None else None,
        model=model,
        stride=stride,
        kpt_thr=kpt_thr,
        bbox_thr=bbox_thr,
        nms_thr=nms_thr,
        device=device,
        save_overlay=save_overlay,
        flip_test=flip_test,
        for_subprocess=False,
    )
    return shlex.join(argv)


def _build_sapiens_cli_argv(
    *,
    input_path: Path,
    out_parent: Path,
    output_base: Path | None = None,
    model: str = DEFAULT_MODEL_KEY,
    stride: int = 1,
    kpt_thr: float = 0.3,
    bbox_thr: float = 0.3,
    nms_thr: float = 0.3,
    device: int = 0,
    save_overlay: bool = True,
    flip_test: bool = False,
    for_subprocess: bool = False,
) -> list[str]:
    """Build Sapiens2 CLI argv (subprocess uses sys.executable; mirror uses uv run)."""
    if for_subprocess:
        runner: list[str] = [sys.executable, "-u", str(Path(__file__).resolve())]
    else:
        runner = ["uv", "run", "vaila/vaila_sapiens.py"]
    cmd: list[str] = [
        *runner,
        "-i",
        str(input_path.resolve()),
        "-o",
        str(out_parent.resolve()),
    ]
    if output_base is not None:
        cmd += ["--output-base", str(output_base.resolve())]
    cmd += [
        "--model",
        model,
        "--stride",
        str(stride),
        "--kpt-thr",
        str(kpt_thr),
        "--bbox-thr",
        str(bbox_thr),
        "--nms-thr",
        str(nms_thr),
        "--device",
        str(device),
    ]
    if flip_test:
        cmd.append("--flip-test")
    if not save_overlay:
        cmd.append("--no-overlay")
    return cmd


def _build_isolated_sapiens_cmd(
    *,
    video_file: Path,
    out_parent: Path,
    out_dir: Path,
    model: str,
    stride: int,
    kpt_thr: float,
    bbox_thr: float,
    nms_thr: float,
    device: int,
    save_overlay: bool,
    flip_test: bool,
    max_persons: int,
    pose_batch_size: int | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "-i",
        str(video_file.resolve()),
        "-o",
        str(out_parent.resolve()),
        "--video-output-dir",
        str(out_dir.resolve()),
        "--model",
        model,
        "--stride",
        str(stride),
        "--kpt-thr",
        str(kpt_thr),
        "--bbox-thr",
        str(bbox_thr),
        "--nms-thr",
        str(nms_thr),
        "--device",
        str(device),
        "--no-isolate-batch",
    ]
    if pose_batch_size is not None:
        cmd += ["--pose-batch-size", str(pose_batch_size)]
    cmd += ["--max-persons", str(max_persons)]
    if flip_test:
        cmd.append("--flip-test")
    if not save_overlay:
        cmd.append("--no-overlay")
    return cmd


def _run_sapiens_batch(
    videos: list[Path],
    output_base: Path,
    *,
    model: str,
    stride: int,
    save_overlay: bool,
    bbox_thr: float,
    nms_thr: float,
    kpt_thr: float,
    device: str,
    flip_test: bool = False,
    max_persons: int = 8,
    pose_batch_size: int | None = None,
) -> tuple[int, list[str]]:
    """Process videos sequentially; emit log lines parsed by the GUI subprocess poller."""
    failed: list[str] = []
    succeeded = 0
    for idx, vf in enumerate(videos, start=1):
        print(f"Processing video {idx}/{len(videos)}: {vf.name}", flush=True)
        out_dir = output_base / vf.stem
        try:
            run_sapiens_on_video(
                vf,
                out_dir,
                model=model,
                stride=stride,
                save_overlay=save_overlay,
                bbox_thr=bbox_thr,
                nms_thr=nms_thr,
                kpt_thr=kpt_thr,
                device=device,
                flip_test=flip_test,
                max_persons=max_persons,
                pose_batch_size=pose_batch_size,
            )
            succeeded += 1
            print(f"  Done: {out_dir}", flush=True)
        except Exception as e:
            failed.append(f"{vf.name}: {e}")
            _write_failure_marker(out_dir, vf, str(e))
            print(f"  ERROR on {vf.name}: {e}", flush=True)
        finally:
            _release_sapiens_gpu_memory()
    print(f"\nAll done. Output: {output_base}", flush=True)
    return succeeded, failed


def _print_sapiens_equivalent_cli(
    input_path: Path | str,
    output_parent: Path | str,
    *,
    model: str = DEFAULT_MODEL_KEY,
    stride: int = 1,
    kpt_thr: float = 0.3,
    save_overlay: bool = True,
    output_base: Path | str | None = None,
) -> None:
    """Print GUI→CLI mirror to stdout (>> prefix avoids absl eating bracketed lines)."""
    cmd = _format_sapiens_cli_command(
        input_path,
        output_parent,
        model=model,
        stride=stride,
        kpt_thr=kpt_thr,
        save_overlay=save_overlay,
        output_base=output_base,
    )
    print("\n>> vaila/vaila_sapiens: Equivalent CLI (copy/paste):", flush=True)
    print(f">>   {cmd}", flush=True)
    if output_base is None:
        print(">> (CLI creates processed_sapiens_<timestamp>/ under -o)\n", flush=True)
    else:
        print(f">> (GUI batch output: {output_base})\n", flush=True)


def _write_readme_sapiens(output_dir: Path, *, model_key: str, stride: int) -> None:
    text = (
        "vailá Sapiens2 Pose run\n"
        f"model={model_key} stride={stride}\n\n"
        "Outputs:\n"
        "  <video>_sapiens_overlay.mp4  — skeleton overlay on original timeline\n"
        "  <video>_predictions.json     — per-frame instances (308 kp Sociopticon)\n"
        "  <video>_sapiens_vaila.csv    — long CSV: frame,person_id,kpt_idx,x,y,score\n"
        "  <video>_markers.csv          — REC2D/REC3D + getpixelvideo (bbox foot anchor)\n"
        "  sapiens_vaila_*.csv          — five anchor tables (same schema as sam_vaila_*)\n"
        "  sapiens_points.csv           — foot + bbox center + mid-hip per pN\n"
        "  sapiens_id_map.csv           — stable person slot map\n"
        "  sapiens_tracks.csv           — bbox tracks with stable_id\n"
        "  sapiens_bbox_tracks.csv      — SAM-compatible bbox tracks for getpixelvideo\n"
        "  <stem>_id_NN_sapiens_pose.csv — wide pose for getpixelvideo Load\n"
        "  FAILED_sapiens.txt           — present only on failure\n\n"
        + SAPIENS_OUTPUT_FILE_GLOSSARY
    )
    with contextlib.suppress(OSError):
        (output_dir / "README_sapiens.txt").write_text(text, encoding="utf-8")


def run_sapiens_on_video(
    video_path: Path,
    output_dir: Path,
    *,
    model: str = DEFAULT_MODEL_KEY,
    stride: int = 1,
    save_overlay: bool = True,
    bbox_thr: float = 0.3,
    nms_thr: float = 0.3,
    kpt_thr: float = 0.3,
    device: str | None = None,
    flip_test: bool = False,
    max_persons: int = 8,
    pose_batch_size: int | None = None,
) -> None:
    """Run Sapiens2 pose on one video; writes overlay MP4 + JSON + CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_key = _normalize_model_key(model)
    _write_readme_sapiens(output_dir, model_key=model_key, stride=stride)
    micro_batch = (
        pose_batch_size if pose_batch_size is not None else _default_pose_batch_size(model_key)
    )

    spec = resolve_model_spec(model)
    dev = device or _require_cuda()

    cap_probe = cv2.VideoCapture(str(video_path))
    if not cap_probe.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    fps_probe = float(cap_probe.get(cv2.CAP_PROP_FPS) or 30.0)
    w_probe = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_probe = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_probe = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap_probe.release()
    stride_eff = max(1, int(stride))
    n_infer = (n_probe + stride_eff - 1) // stride_eff if n_probe > 0 else "?"
    _sapiens_log(
        f"Starting {video_path.name}: {w_probe}x{h_probe} @ {fps_probe:.2f} fps, "
        f"{n_probe} frames, stride={stride_eff} (~{n_infer} pose passes), "
        f"pose_batch={micro_batch}"
    )

    session = PoseInferenceSession(
        spec,
        device=dev,
        bbox_thr=bbox_thr,
        nms_thr=nms_thr,
        kpt_thr=kpt_thr,
        flip_test=flip_test,
        max_persons=max_persons,
        pose_batch_size=micro_batch,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    stem = video_path.stem
    overlay_path = output_dir / f"{stem}_sapiens_overlay.mp4"
    json_path = output_dir / f"{stem}_predictions.json"
    csv_path = output_dir / f"{stem}_sapiens_vaila.csv"

    writer: cv2.VideoWriter | None = None
    if save_overlay:
        writer, overlay_path = _open_sam3_video_writer(
            overlay_path,
            fps,
            (w, h),
            purpose="Sapiens2 overlay",
        )

    stride = max(1, int(stride))
    pose_by_frame: dict[int, list[dict[str, Any]]] = {}
    csv_rows: list[tuple[int, int, int, float, float, float]] = []
    frames_records: list[dict[str, Any]] = []
    image_size = [h, w]
    num_keypoints: int | None = None
    linker = GeometricFrameLinker(enabled=True)

    fi = 0
    progress_step = _frame_progress_interval(n_frames)
    tqdm_cls = None if _progress_quiet() else _try_import_tqdm()
    if tqdm_cls is not None:
        pbar = tqdm_cls(
            total=n_frames if n_frames > 0 else None,
            desc=stem[:48],
            unit="fr",
            file=sys.stderr,
            mininterval=0.25,
            dynamic_ncols=True,
        )
        _sapiens_log("Frame progress bar on stderr — model load can take 1–2 min before it moves")
    else:
        pbar = None
        _sapiens_log(
            f"Frame heartbeat every ~{progress_step} frames "
            "(GUI/quiet mode; use CLI without --quiet for tqdm bar)"
        )

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if fi % stride == 0:
                keypoints, scores, bboxes = session.process_frame(frame)
                instances = instances_from_frame(keypoints, scores, bboxes)
                instances = _assign_stable_ids(fi, instances, linker)
                pose_by_frame[fi] = instances
                csv_rows.extend(flatten_instances_to_csv_rows(fi, instances))
                if num_keypoints is None and instances:
                    num_keypoints = len(instances[0].get("keypoints", []))
                frames_records.append({"frame_index": fi, "instances": instances})
            nearest = _nearest_pose_frame(fi, sorted(pose_by_frame.keys()))
            instances_draw = pose_by_frame.get(nearest, [])
            if save_overlay and writer is not None:
                if instances_draw:
                    kpts = [np.asarray(i["keypoints"], dtype=float) for i in instances_draw]
                    scrs = [np.asarray(i["keypoint_scores"], dtype=float) for i in instances_draw]
                    vis = session.render_overlay(frame, kpts, scrs)
                else:
                    vis = frame
                writer.write(vis)
            fi += 1
            if pbar is not None:
                pbar.update(1)
            elif (
                fi == 1
                or (n_frames > 0 and fi % progress_step == 0)
                or (n_frames <= 0 and fi % progress_step == 0)
            ):
                if n_frames > 0:
                    pct = 100.0 * fi / n_frames
                    _sapiens_log(f"{stem}: frame {fi}/{n_frames} ({pct:.0f}%)")
                else:
                    _sapiens_log(f"{stem}: frame {fi}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if pbar is not None:
            pbar.close()
        _release_sapiens_gpu_memory(session)

    payload = {
        "video": stem,
        "image_size": image_size,
        "num_keypoints": num_keypoints,
        "model": spec.arch,
        "stride": stride,
        "kpt_thr_used": float(kpt_thr),
        "frames": frames_records,
    }
    _sapiens_log(f"Writing JSON + biomechanics CSVs for {stem} …")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_vaila_pose_csv(csv_path, csv_rows)
    timeline = _expand_pose_timeline(pose_by_frame, fi)
    bio_paths = write_sapiens_biomechanics_csvs(output_dir, stem, timeline, kpt_thr=float(kpt_thr))
    kp_names: list[str] | None = None
    if session.model is not None and getattr(session.model, "pose_metainfo", None):
        meta = session.model.pose_metainfo
        if isinstance(meta, dict) and meta.get("keypoint_names"):
            kp_names = [str(n) for n in meta["keypoint_names"]]
    gpv_paths = write_sapiens_getpixelvideo_pose_csvs(
        output_dir,
        stem,
        timeline,
        kpt_thr=float(kpt_thr),
        keypoint_names=kp_names,
    )
    markers_name = f"{stem}_markers.csv"
    _sapiens_log(
        f"Done {video_path.name}: overlay={overlay_path.name if save_overlay else 'skipped'}, "
        f"json={json_path.name}, csv={csv_path.name}, markers={markers_name} "
        f"({fi} frames, {len(frames_records)} inferred, "
        f"{len(bio_paths)} biomech + {len(gpv_paths)} getpixelvideo CSVs)"
    )


def build_dry_run_report(
    input_path: Path,
    output_parent: Path,
    *,
    model: str,
    stride: int,
) -> list[str]:
    lines = ["[Sapiens2] Dry-run (no inference)"]
    lines.append(f"input={input_path}")
    lines.append(f"output_parent={output_parent}")
    lines.append(f"model={model} stride={stride}")
    try:
        spec = resolve_model_spec(model)
        lines.append(f"config={spec.config_path} exists={spec.config_path.is_file()}")
        lines.append(f"checkpoint={spec.checkpoint_path} exists={spec.checkpoint_path.is_file()}")
        lines.append(f"detector={spec.detector_path} exists={spec.detector_path.is_dir()}")
    except Exception as e:
        lines.append(f"model resolution error: {e}")
    videos = _find_videos(input_path)
    lines.append(f"videos={len(videos)}")
    for vf in videos[:8]:
        lines.append(f"  - {vf.name}")
    if len(videos) > 8:
        lines.append(f"  ... and {len(videos) - 8} more")
    return lines


def download_weights(model: str = DEFAULT_MODEL_KEY) -> None:
    """Download pose checkpoint + DETR detector via Hugging Face Hub (Python API)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    key = _normalize_model_key(model)
    meta = SAPIENS_POSE_MODELS[key]
    ckpt_root = _checkpoint_root()
    pose_dir = ckpt_root / "pose"
    det_dir = ckpt_root / "detector" / DETECTOR_DIRNAME
    pose_dir.mkdir(parents=True, exist_ok=True)
    det_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Sapiens2] Downloading pose: {meta['hf_repo']}/{meta['ckpt']}")
    pose_path = hf_hub_download(
        repo_id=meta["hf_repo"],
        filename=meta["ckpt"],
        local_dir=str(pose_dir),
    )
    print(f"✓ Downloaded\n  path: {pose_path}")

    print(f"[Sapiens2] Downloading detector: {DETECTOR_HF_REPO}")
    snapshot_download(repo_id=DETECTOR_HF_REPO, local_dir=str(det_dir))
    print(f"[Sapiens2] Weights downloaded under {ckpt_root}")


class SapiensBatchProgress(tk.Toplevel):
    """Progress window for Sapiens2 GUI batch (GPU work runs in a child process)."""

    def __init__(self, parent: tk.Misc, total: int, *, output_base: Path | None = None) -> None:
        super().__init__(parent)
        self.title("Sapiens2 Pose — batch progress")
        self.geometry("720x480")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.cancelled = False
        self._total = max(1, total)
        self._output_base = output_base

        frm = ttk.Frame(self, padding=8)
        frm.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value=f"0 / {total} processed")
        ttk.Label(frm, textvariable=self.status_var, font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", pady=(0, 4)
        )
        self.progress = ttk.Progressbar(frm, maximum=self._total, value=0)
        self.progress.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(frm, text="Log:").pack(anchor="w")
        log_frame = ttk.Frame(frm)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, height=16, wrap="none", state="disabled")
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=sb.set)

        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, pady=(6, 0))
        self.cancel_btn = ttk.Button(btns, text="Request cancel", command=self._on_cancel)
        self.cancel_btn.pack(side=tk.LEFT)
        self.close_btn = ttk.Button(btns, text="Close", command=self._on_close, state="disabled")
        self.close_btn.pack(side=tk.RIGHT)

        self.transient(parent)  # ty: ignore[no-matching-overload]
        self.lift()

    def schedule_log(self, line: str) -> None:
        self.after(0, lambda s=line: self._append_log(s))

    def schedule_progress(self, done: int) -> None:
        self.after(0, lambda d=done: self._set_progress(d))

    def schedule_finish(self) -> None:
        self.after(0, self._finish)

    def _append_log(self, line: str) -> None:
        clean = line.rstrip()
        if clean:
            sys.stdout.write(clean + "\n")
            sys.stdout.flush()
        self.log_text.config(state="normal")
        self.log_text.insert("end", clean + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _set_progress(self, done: int) -> None:
        self.progress["value"] = done
        self.status_var.set(f"{done} / {self._total} processed")

    def _on_cancel(self) -> None:
        self.cancelled = True
        self.cancel_btn.config(state="disabled")
        self._append_log("[GUI] Cancel requested — waiting for current video to end...")

    def _finish(self) -> None:
        self.cancel_btn.config(state="disabled")
        self.close_btn.config(state="normal")

    def _on_close(self) -> None:
        if self.close_btn["state"] == "disabled":
            self._on_cancel()
            return
        with contextlib.suppress(tk.TclError):
            self.destroy()


def _start_sapiens_batch_subprocess(
    *,
    progress: SapiensBatchProgress,
    input_path: Path,
    out_parent: Path,
    output_base: Path,
    model: str,
    stride: int,
    kpt_thr: float,
    bbox_thr: float,
    nms_thr: float,
    device: int,
    save_overlay: bool,
    on_done: Callable[[int, int, list[str], Path], None],
) -> None:
    """Run Sapiens2 batch in an isolated child process (CUDA + Tk must not share a process)."""
    import re

    cmd = _build_sapiens_cli_argv(
        input_path=input_path,
        out_parent=out_parent,
        output_base=output_base,
        model=model,
        stride=stride,
        kpt_thr=kpt_thr,
        bbox_thr=bbox_thr,
        nms_thr=nms_thr,
        device=device,
        save_overlay=save_overlay,
        for_subprocess=True,
    )
    mirror_cmd = _build_sapiens_cli_argv(
        input_path=input_path,
        out_parent=out_parent,
        output_base=output_base,
        model=model,
        stride=stride,
        kpt_thr=kpt_thr,
        bbox_thr=bbox_thr,
        nms_thr=nms_thr,
        device=device,
        save_overlay=save_overlay,
        for_subprocess=False,
    )
    progress.schedule_log(f"[GUI] launching subprocess: {shlex.join(cmd)}")
    progress.schedule_log(f"[GUI] equivalent: {shlex.join(mirror_cmd)}")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["TQDM_DISABLE"] = "1"

    log_fd, log_path = tempfile.mkstemp(prefix="vaila_sapiens_batch_", suffix=".log", text=True)
    log_handle = os.fdopen(log_fd, "w", buffering=1)

    try:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    except Exception as e:
        log_handle.close()
        progress.schedule_log(f"[GUI] FAILED to launch subprocess: {e!r}")
        progress.schedule_finish()
        on_done(0, 0, [f"subprocess launch failed: {e!r}"], output_base)
        return

    state: dict[str, Any] = {
        "succeeded": 0,
        "failed": [],
        "actual_output": output_base,
        "total_videos": 0,
        "current_idx": 0,
        "read_pos": 0,
        "buffer": "",
        "finished": False,
    }
    re_processing = re.compile(r"^Processing video\s+(\d+)\s*/\s*(\d+)\s*:")
    re_done_line = re.compile(r"^\s*Done:\s+(.+)$")
    re_error_line = re.compile(r"^\s*ERROR on\s+(.+):\s*(.*)$")
    re_output_dir = re.compile(r"^All done\.\s*Output:\s*(.+)$")

    def _process_line(line: str) -> None:
        progress._append_log(line)
        if (m := re_processing.match(line)) is not None:
            idx = int(m.group(1))
            total = int(m.group(2))
            state["current_idx"] = idx
            state["total_videos"] = max(int(state["total_videos"]), total)
            progress._set_progress(idx - 1)
        elif re_done_line.match(line) is not None:
            state["succeeded"] = int(state["succeeded"]) + 1
            idx_done = int(state["current_idx"])
            if idx_done > 0:
                progress._set_progress(idx_done)
        elif (m := re_error_line.match(line)) is not None:
            fname = m.group(1).strip()
            err_text = m.group(2).strip()
            failed_list: list[str] = state["failed"]
            failed_list.append(f"{fname}: {err_text}")
        elif (m := re_output_dir.match(line)) is not None:
            state["actual_output"] = Path(m.group(1).strip())

    def _drain_log_file() -> None:
        try:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                fh.seek(int(state["read_pos"] or 0))
                chunk = fh.read()
                state["read_pos"] = fh.tell()
        except FileNotFoundError:
            chunk = ""
        if chunk:
            buf = str(state["buffer"] or "") + chunk
            *complete_lines, remainder = buf.split("\n")
            state["buffer"] = remainder
            for ln in complete_lines:
                if ln.strip():
                    with contextlib.suppress(tk.TclError):
                        _process_line(ln)

    def _poll() -> None:
        if progress.cancelled and proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.terminate()
        _drain_log_file()
        rc = proc.poll()
        if rc is None:
            with contextlib.suppress(tk.TclError):
                progress.after(150, _poll)
            return
        _drain_log_file()
        remainder = str(state["buffer"] or "")
        if remainder.strip():
            with contextlib.suppress(tk.TclError):
                _process_line(remainder)
        state["buffer"] = ""
        if state["finished"]:
            return
        state["finished"] = True
        with contextlib.suppress(Exception):
            log_handle.close()
        with contextlib.suppress(tk.TclError):
            progress._append_log(f"[GUI] subprocess exited with code {rc}")
        if rc != 0:
            failed_list = state["failed"]
            if isinstance(failed_list, list) and not failed_list:
                failed_list.append(f"subprocess exited with code {rc}")
            progress._append_log(f"[GUI] Full log: {log_path}")
        with contextlib.suppress(tk.TclError):
            progress._finish()
        try:
            actual_out = state["actual_output"]
            succ = int(state["succeeded"] or 0)
            total_default = len(_find_videos(input_path)) if input_path.is_dir() else 1
            total = int(state["total_videos"] or 0) or total_default
            failed_obj = state["failed"]
            failed_list = failed_obj if isinstance(failed_obj, list) else []
            on_done(succ, total, failed_list, Path(actual_out))
        except Exception as exc:
            on_done(0, 0, [f"finalize failed: {exc!r}"], output_base)

    progress.after(50, _poll)


def run_sapiens_video(existing_root: Any | None = None) -> None:
    """Tkinter GUI entry for Sapiens2 Pose video batch."""
    root = existing_root
    owns_root = False
    if root is None:
        root = tk.Tk()
        root.withdraw()
        owns_root = True
    try:
        if platform.system() in ("Windows", "Linux"):
            root.deiconify()
            root.geometry("1x1+100+100")
            root.update_idletasks()
    except Exception:
        pass

    class SapiensVideoDialog(tk.Toplevel):
        def __init__(self, master: tk.Misc) -> None:
            super().__init__(master)
            self.title("Sapiens2 Pose — video")
            self.result: tuple[Path, Path, str, int, float, bool] | None = None
            frm = ttk.Frame(self, padding=12)
            frm.pack(fill="both", expand=True)
            ttk.Label(frm, text="Input (dir or file):").grid(row=0, column=0, sticky="w")
            self.input_var = tk.StringVar()
            ttk.Entry(frm, textvariable=self.input_var, width=48).grid(
                row=1, column=0, columnspan=2
            )
            in_btns = ttk.Frame(frm)
            in_btns.grid(row=1, column=2)
            ttk.Button(in_btns, text="Dir…", command=self._browse_dir).pack(side="left", padx=1)
            ttk.Button(in_btns, text="File…", command=self._browse_file).pack(side="left", padx=1)
            ttk.Label(frm, text="Output parent folder:").grid(
                row=2, column=0, sticky="w", pady=(8, 0)
            )
            self.out_var = tk.StringVar()
            ttk.Entry(frm, textvariable=self.out_var, width=48).grid(row=3, column=0, columnspan=2)
            ttk.Button(frm, text="Browse…", command=self._browse_out).grid(row=3, column=2)
            ttk.Label(frm, text="Model (RTX 4090 default: 1b):").grid(
                row=4, column=0, sticky="w", pady=(8, 0)
            )
            self.model_var = tk.StringVar(value="1b")
            ttk.Combobox(
                frm,
                textvariable=self.model_var,
                values=["0.4b", "0.8b", "1b", "5b"],
                width=10,
                state="readonly",
            ).grid(row=5, column=0, sticky="w")
            ttk.Label(frm, text="Stride (1=every frame):").grid(
                row=6, column=0, sticky="w", pady=(8, 0)
            )
            self.stride_var = tk.StringVar(value="1")
            ttk.Entry(frm, textvariable=self.stride_var, width=8).grid(row=7, column=0, sticky="w")
            ttk.Label(frm, text="Keypoint threshold:").grid(
                row=8, column=0, sticky="w", pady=(8, 0)
            )
            self.kpt_var = tk.StringVar(value="0.3")
            ttk.Entry(frm, textvariable=self.kpt_var, width=8).grid(row=9, column=0, sticky="w")
            self.overlay_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(frm, text="Save overlay MP4", variable=self.overlay_var).grid(
                row=10, column=0, sticky="w", pady=(8, 0)
            )
            btns = ttk.Frame(frm)
            btns.grid(row=11, column=0, columnspan=3, pady=12)
            ttk.Button(btns, text="Help", command=self._open_help).pack(side="left", padx=4)
            ttk.Button(btns, text="Run", command=self._on_run).pack(side="left", padx=4)
            ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="left", padx=4)
            self.transient(master)  # ty: ignore[no-matching-overload]
            self.grab_set()
            self.lift()
            self.focus_force()

        def _browse_dir(self) -> None:
            d = filedialog.askdirectory(parent=self, title="Select folder with videos")
            if d:
                self.input_var.set(d)

        def _browse_file(self) -> None:
            p = filedialog.askopenfilename(
                parent=self,
                title="Select video file",
                filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")],
            )
            if p:
                self.input_var.set(p)

        def _browse_out(self) -> None:
            d = filedialog.askdirectory(title="Output parent folder")
            if d:
                self.out_var.set(d)

        def _open_help(self) -> None:
            help_path = _repo_root() / "vaila" / "help" / "vaila_sapiens.html"
            if help_path.is_file():
                webbrowser.open_new_tab(help_path.as_uri())

        def _on_run(self) -> None:
            inp = self.input_var.get().strip()
            out = self.out_var.get().strip()
            if not inp or not out:
                messagebox.showerror("Error", "Input and output are required.", parent=self)
                return
            try:
                stride = int(self.stride_var.get().strip() or "1")
                kpt_thr = float(self.kpt_var.get().strip() or "0.3")
            except ValueError:
                messagebox.showerror(
                    "Error", "Stride and kpt threshold must be numeric.", parent=self
                )
                return
            self.result = (
                Path(inp),
                Path(out),
                self.model_var.get().strip() or "1b",
                max(1, stride),
                kpt_thr,
                bool(self.overlay_var.get()),
            )
            self.destroy()

    dlg = SapiensVideoDialog(root)
    root.wait_window(dlg)
    if dlg.result is None:
        if owns_root:
            root.destroy()
        return

    input_path, out_parent, model, stride, kpt_thr, save_overlay = dlg.result

    # Validate model spec / weights existence before starting
    try:
        spec = resolve_model_spec(model)
        missing = []
        if not spec.config_path.is_file():
            missing.append(f"- Configuration file: {spec.config_path}")
        if not spec.checkpoint_path.is_file():
            missing.append(f"- Checkpoint file (pose): {spec.checkpoint_path}")
        if not spec.detector_path.is_dir():
            missing.append(f"- DETR Detector directory: {spec.detector_path}")

        if missing:
            missing_str = "\n".join(missing)
            msg = (
                f"Sapiens2 Pose weights or configurations for model size '{model}' are missing:\n\n"
                f"{missing_str}\n\n"
                f"Please download the weights and prepare the model files by running:\n"
                f"  uv run vaila/vaila_sapiens.py --download-weights --model {model}\n\n"
                f"Or run the complete setup bootstrap:\n"
                f"  bash bin/setup_sapiens2.sh\n\n"
                f"Would you like to attempt downloading the weights automatically now?"
            )
            ans = messagebox.askyesno("Sapiens2 — Weights/Config Missing", msg, parent=root)
            if ans:
                print(f"[Sapiens2] Downloading weights automatically for model {model}...")
                download_weights(model)
                # Re-verify after download
                if not spec.checkpoint_path.is_file() or not spec.detector_path.is_dir():
                    messagebox.showerror(
                        "Sapiens2 — Error",
                        "Automatic download finished, but files are still missing. "
                        "Please check your internet connection and Hugging Face access, and try downloading manually.",
                        parent=root,
                    )
                    if owns_root:
                        root.destroy()
                    return
            else:
                if owns_root:
                    root.destroy()
                return
    except Exception as err:
        messagebox.showerror(
            "Sapiens2 — Model Error", f"Failed to verify weights: {err}", parent=root
        )
        if owns_root:
            root.destroy()
        return

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = out_parent / f"processed_sapiens_{ts}"
    output_base.mkdir(parents=True, exist_ok=True)
    videos = _find_videos(input_path)
    if not videos:
        messagebox.showinfo("Sapiens2", "No videos found.", parent=root)
        if owns_root:
            root.destroy()
        return

    progress = SapiensBatchProgress(root, total=len(videos), output_base=output_base)

    def _on_done(succeeded: int, total: int, failed: list[str], out_base: Path) -> None:
        summary = f"Processed {succeeded}/{total} video(s).\nOutput: {out_base}"
        print("\n[Sapiens2] GUI batch finished")
        print(summary)
        if failed:
            summary += f"\n\nFailed ({len(failed)}):\n" + "\n".join(failed[:20])
            if len(failed) > 20:
                summary += f"\n…and {len(failed) - 20} more."
            root.after(
                0,
                lambda: messagebox.showwarning(
                    "Sapiens2 — finished with errors", summary, parent=root
                ),
            )
        else:
            root.after(0, lambda: messagebox.showinfo("Sapiens2 — done", summary, parent=root))

    _start_sapiens_batch_subprocess(
        progress=progress,
        input_path=input_path,
        out_parent=out_parent,
        output_base=output_base,
        model=model,
        stride=stride,
        kpt_thr=kpt_thr,
        bbox_thr=0.3,
        nms_thr=0.3,
        device=int(os.environ.get("SAPIENS_DEVICE", "0")),
        save_overlay=save_overlay,
        on_done=_on_done,
    )

    root.wait_window(progress)
    if owns_root:
        root.destroy()


SAPIENS_CLI_EXAMPLES = """\
Sapiens2 — copy/paste CLI recipes
=================================

# 0. Open help / setup page in your default browser
uv run vaila/vaila_sapiens.py --open-help

# 1. Print these examples again (no GPU work)
uv run vaila/vaila_sapiens.py --print-examples

# 2. Download facebook/sapiens2 weights (default 1b; use 5b for max quality)
uv run vaila/vaila_sapiens.py --download-weights --model 1b
uv run vaila/vaila_sapiens.py --download-weights --model 5b

# 3. Process video using default 1b model
uv run vaila/vaila_sapiens.py \\
  -i path/to/video.mp4 \\
  -o path/to/output_parent/ \\
  --model 1b \\
  --stride 1

# 4. Process video at a faster pace (every 2nd frame)
uv run vaila/vaila_sapiens.py \\
  -i path/to/video.mp4 \\
  -o path/to/output_parent/ \\
  --stride 2

# 5. Process a directory of videos in batch
uv run vaila/vaila_sapiens.py \\
  -i path/to/videos_dir/ \\
  -o path/to/output_parent/ \\
  --model 0.4b

Tips
----
* GUI (no args)              : uv run vaila/vaila_sapiens.py
* Requires NVIDIA CUDA and sapiens2 cloned locally via bin/setup_sapiens2.sh
* Model 5b                   : uv run vaila/vaila_sapiens.py --download-weights --model 5b
* OOM on RTX 4090            : use --model 1b (default); avoid --flip-test; pass a single .mp4 not a folder with old overlays
* REC2D/REC3D                : use <stem>_markers.csv (foot anchor, stable pN slots)
* Full reference            : vaila/help/vaila_sapiens.md  (or --open-help).
"""


def _print_sapiens_cli_examples() -> None:
    """Dump the copy-paste cheat sheet to stdout (used by ``--print-examples``)."""
    print(SAPIENS_CLI_EXAMPLES, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sapiens2 308-keypoint pose on video (vailá). Pass --print-examples for a "
            "copy-paste recipe sheet, --open-help for the full HTML reference."
        ),
        epilog=SAPIENS_CLI_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--input", type=Path, help="Input video or directory")
    parser.add_argument("-o", "--output", type=Path, help="Output parent directory")
    parser.add_argument(
        "--output-base",
        type=Path,
        default=None,
        help="Pre-created batch directory (GUI subprocess; skips new timestamp folder).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_KEY,
        choices=sorted(SAPIENS_POSE_MODELS.keys()),
        help="Sapiens2 pose model size (1b recommended for RTX 4090)",
    )
    parser.add_argument("--stride", type=int, default=1, help="Run pose every N frames")
    parser.add_argument("--kpt-thr", type=float, default=0.3, help="Keypoint threshold")
    parser.add_argument(
        "--bbox-thr", type=float, default=0.3, help="Bounding box detection threshold"
    )
    parser.add_argument("--nms-thr", type=float, default=0.3, help="NMS threshold")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--no-overlay", action="store_true", help="Skip overlay MP4 generation")
    parser.add_argument(
        "--flip-test",
        action="store_true",
        help="Enable left-right flip test (higher quality, ~2× VRAM; off by default on 24 GiB).",
    )
    parser.add_argument(
        "--max-persons",
        type=int,
        default=8,
        help="Max DETR persons per frame (top scores; default 8). Lower if VRAM is tight.",
    )
    parser.add_argument(
        "--pose-batch-size",
        type=int,
        default=None,
        help="Pose micro-batch size (default: 1 for 5b, 2 for 1b, 4 for 0.4b/0.8b).",
    )
    parser.add_argument(
        "--no-isolate-batch",
        action="store_true",
        help="Disable subprocess-per-video isolation (debug only; OOM may cascade).",
    )
    parser.add_argument(
        "--video-output-dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Perform dry run to check files and configurations"
    )
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help="Download facebook/sapiens2 model weights into vaila/models/sapiens2/.",
    )
    parser.add_argument(
        "--open-help",
        action="store_true",
        help="Open Sapiens2 setup instructions (HTML) in the default browser and exit.",
    )
    parser.add_argument(
        "--print-examples", action="store_true", help="Print copy-paste CLI recipes and exit."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal terminal output (disables tqdm frame bar and heartbeat lines).",
    )
    args = parser.parse_args()

    if args.print_examples:
        _print_sapiens_cli_examples()
        return

    if args.open_help:
        help_path = _repo_root() / "vaila" / "help" / "vaila_sapiens.html"
        if help_path.is_file():
            webbrowser.open_new_tab(help_path.as_uri())
        else:
            webbrowser.open("https://github.com/facebookresearch/sapiens2/blob/main/docs/POSE.md")
        return

    if args.download_weights:
        download_weights(args.model)
        return

    if args.input is None or args.output is None:
        run_sapiens_video()
        return

    inp = args.input.resolve()
    out_parent = args.output.resolve()
    if args.output_base is not None:
        output_base = args.output_base.resolve()
        output_base.mkdir(parents=True, exist_ok=True)
    else:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = out_parent / f"processed_sapiens_{ts}"
        output_base.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        lines = build_dry_run_report(inp, output_base, model=args.model, stride=args.stride)
        report = output_base / "SAPIENS_DRY_RUN.txt"
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for line in lines:
            print(line)
        print(f"Dry-run report: {report}")
        return

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["SAPIENS_DEVICE"] = str(args.device)
    if args.quiet:
        os.environ["TQDM_DISABLE"] = "1"
    device = f"cuda:{args.device}"
    videos = _find_videos(inp)
    if not videos:
        print(f"No videos under {inp}")
        raise SystemExit(1)

    flip_test = bool(args.flip_test)
    save_overlay = not args.no_overlay
    max_persons = max(1, int(args.max_persons))
    pose_batch_size = (
        max(1, int(args.pose_batch_size)) if args.pose_batch_size is not None else None
    )

    if args.video_output_dir is not None:
        out_dir = args.video_output_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        video = videos[0]
        try:
            run_sapiens_on_video(
                video,
                out_dir,
                model=args.model,
                stride=args.stride,
                save_overlay=save_overlay,
                bbox_thr=args.bbox_thr,
                nms_thr=args.nms_thr,
                kpt_thr=args.kpt_thr,
                device=device,
                flip_test=flip_test,
                max_persons=max_persons,
                pose_batch_size=pose_batch_size,
            )
            print(f"  Done: {out_dir}")
        except Exception as e:
            _write_failure_marker(out_dir, video, str(e))
            print(f"  ERROR on {video.name}: {e}")
            raise SystemExit(1) from e
        return

    use_isolation = not args.no_isolate_batch
    if use_isolation:
        scope = "single-video" if len(videos) == 1 else "each video"
        print(f"[Sapiens2] subprocess-per-video isolation: ENABLED ({scope} in a fresh process)")
        failed: list[str] = []
        succeeded = 0
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        if args.quiet:
            env["TQDM_DISABLE"] = "1"
        else:
            env.pop("TQDM_DISABLE", None)
        for idx, vf in enumerate(videos, 1):
            print(f"Processing video {idx}/{len(videos)}: {vf.name}", flush=True)
            if not args.quiet:
                _sapiens_log(
                    "Spawning isolated worker — expect 1–2 min of model load, "
                    "then a frame progress bar on stderr"
                )
            out_dir = output_base / vf.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = _build_isolated_sapiens_cmd(
                video_file=vf,
                out_parent=out_parent,
                out_dir=out_dir,
                model=args.model,
                stride=args.stride,
                kpt_thr=args.kpt_thr,
                bbox_thr=args.bbox_thr,
                nms_thr=args.nms_thr,
                device=args.device,
                save_overlay=save_overlay,
                flip_test=flip_test,
                max_persons=max_persons,
                pose_batch_size=pose_batch_size,
            )
            try:
                rc = subprocess.call(cmd, env=env)
            except KeyboardInterrupt:
                print(f"  INTERRUPTED on {vf.name}")
                raise
            if rc == 0:
                succeeded += 1
                print(f"  Done: {out_dir}", flush=True)
            else:
                failed.append(f"{vf.name}: subprocess exit={rc}")
                print(f"  ERROR on {vf.name}: subprocess exit={rc}", flush=True)
        print(f"\nAll done. Output: {output_base}", flush=True)
        if failed:
            raise SystemExit(1)
        if succeeded == 0:
            raise SystemExit(1)
        return

    succeeded, failed = _run_sapiens_batch(
        videos,
        output_base,
        model=args.model,
        stride=args.stride,
        save_overlay=save_overlay,
        bbox_thr=args.bbox_thr,
        nms_thr=args.nms_thr,
        kpt_thr=args.kpt_thr,
        device=device,
        flip_test=flip_test,
        max_persons=max_persons,
        pose_batch_size=pose_batch_size,
    )
    if failed:
        raise SystemExit(1)
    if succeeded == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
