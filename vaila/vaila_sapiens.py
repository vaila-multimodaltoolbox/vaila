"""
Project: vailá
Script: vaila_sapiens.py
Authors: Paulo Santiago, Sergio Barroso, Felipe Dias, Lennin Abrão
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 06 July 2026
Update Date: 16 July 2026
Version: 0.3.85

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
    from .geometric_reid import (
        GeometricFrameLinker,
        GeometricLinkerConfig,
        assignment_min_cost,
        write_reid_links_csv,
    )
    from .sam_postprocess import VAILA_ANCHORS, _anchor_xy, _format_cell
    from .vaila_sam import _open_sam3_video_writer
except ImportError:
    from geometric_reid import (  # ty: ignore[unresolved-import]
        GeometricFrameLinker,
        GeometricLinkerConfig,
        assignment_min_cost,
        write_reid_links_csv,
    )
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

DEFAULT_BBOX_THR = 0.3
DEFAULT_NMS_THR = 0.3
DEFAULT_KPT_THR = 0.3
DEFAULT_MAX_PERSONS = 8

# Geometric Re-ID defaults — same linker family as SAM3; stronger direction + mobility
# gates because DETR is per-frame (SAM3 has native video IDs before stabilize).
DEFAULT_REID_MAX_GAP = 12
DEFAULT_REID_MAX_DIST_PX = 180.0
DEFAULT_REID_MIN_IOU = 0.05
DEFAULT_REID_DIRECTION_WEIGHT = 1.0
DEFAULT_REID_STATIC_SPEED = 5.0
DEFAULT_REID_STATIC_RADIUS_PX = 65.0
DEFAULT_REID_KPT_OKS_WEIGHT = 0.4
DEFAULT_APPEARANCE_REID_THRESHOLD = 0.6

SAPIENS_CLI_FULL_INFERENCE_EXAMPLE = """\
# Full inference example — all user-facing flags (copy/paste, adjust paths)
uv run vaila/vaila_sapiens.py \\
  -i /path/to/video.mp4 \\
  -o /path/to/output_parent/ \\
  --model 1b \\
  --stride 1 \\
  --device 0 \\
  --bbox-thr 0.3 \\
  --nms-thr 0.3 \\
  --max-persons 8 \\
  --kpt-thr 0.3 \\
  --pose-batch-size 2 \\
  --flip-test \\
  --stabilize-ids \\
  --reid-max-gap 12 \\
  --reid-max-dist 180 \\
  --reid-min-iou 0.05 \\
  --reid-direction-weight 1.0 \\
  --reid-static-speed 5.0 \\
  --reid-static-radius 65 \\
  --appearance-reid \\
  --no-overlay \\
  --no-draw-id \\
  --quiet

# Re-render overlay MP4 from an existing run (loads model for skeleton viz only; no re-inference)
uv run vaila/vaila_sapiens.py \\
  --rerender-overlay \\
  -i /path/to/video.mp4 \\
  -o /path/to/processed_sapiens_<ts>/<stem>/

# Dry-run (no GPU): add --dry-run and drop inference-only flags
uv run vaila/vaila_sapiens.py -i /path/to/videos/ -o /tmp/out --model 1b --dry-run

# Download weights only:
uv run vaila/vaila_sapiens.py --download-weights --model 1b
"""

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
  sapiens_reid_links.csv     frame,raw_id,stable_id (geometric Re-ID audit, like sam_reid_links.csv)
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


def _parse_gui_inference_fields(
    *,
    stride_s: str,
    kpt_thr_s: str,
    bbox_thr_s: str,
    nms_thr_s: str,
    max_persons_s: str,
    device_s: str,
    pose_batch_s: str,
) -> tuple[int, float, float, float, int, int, int | None]:
    """Validate Tkinter inference fields; raises ValueError on bad input."""
    stride = max(1, int(stride_s.strip() or "1"))
    kpt_thr = float(kpt_thr_s.strip() or str(DEFAULT_KPT_THR))
    bbox_thr = float(bbox_thr_s.strip() or str(DEFAULT_BBOX_THR))
    nms_thr = float(nms_thr_s.strip() or str(DEFAULT_NMS_THR))
    max_persons = max(1, int(max_persons_s.strip() or str(DEFAULT_MAX_PERSONS)))
    device = max(0, int(device_s.strip() or "0"))
    pose_batch_raw = pose_batch_s.strip()
    pose_batch_size = None if not pose_batch_raw else max(1, int(pose_batch_raw))
    for name, value in (
        ("kpt-thr", kpt_thr),
        ("bbox-thr", bbox_thr),
        ("nms-thr", nms_thr),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0 (got {value})")
    return stride, kpt_thr, bbox_thr, nms_thr, max_persons, device, pose_batch_size


def _parse_gui_threshold_fields(
    *,
    stride_s: str,
    kpt_thr_s: str,
    bbox_thr_s: str,
    nms_thr_s: str,
    max_persons_s: str,
) -> tuple[int, float, float, float, int]:
    """Backward-compatible wrapper for threshold-only parsing."""
    stride, kpt_thr, bbox_thr, nms_thr, max_persons, _device, _pose_batch = (
        _parse_gui_inference_fields(
            stride_s=stride_s,
            kpt_thr_s=kpt_thr_s,
            bbox_thr_s=bbox_thr_s,
            nms_thr_s=nms_thr_s,
            max_persons_s=max_persons_s,
            device_s="0",
            pose_batch_s="",
        )
    )
    return stride, kpt_thr, bbox_thr, nms_thr, max_persons


def _progress_quiet() -> bool:
    flag = os.environ.get("TQDM_DISABLE", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}

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


def _pose_render_utils_path() -> Path:
    """Absolute path to Sapiens2 ``pose_render_utils.py`` (not a Python package)."""
    return (
        _sapiens2_root() / "sapiens" / "pose" / "tools" / "vis" / "pose_render_utils.py"
    ).resolve()


def _load_visualize_keypoints() -> Callable[..., np.ndarray] | None:
    """Load ``visualize_keypoints`` via file path (avoids bare sys.path imports)."""
    vis_mod = _pose_render_utils_path()
    if not vis_mod.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "vaila_sapiens_pose_render_utils",
        vis_mod,
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    fn = getattr(module, "visualize_keypoints", None)
    return fn if callable(fn) else None


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
            from sapiens.pose.datasets import (  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
                UDPHeatmap,
                parse_pose_metainfo,
            )
            from sapiens.pose.models import (  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
                init_model,
            )

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

            # File-path load: pose_render_utils lives outside the sapiens package.
            self._visualize_fn = _load_visualize_keypoints()

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
        from sapiens.pose.evaluators import (  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
            nms,
        )

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
            return np.zeros((0, 4), dtype=np.float32)
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
        *,
        instances: list[dict[str, Any]] | None = None,
        draw_id: bool = True,
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
            out = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
        else:
            out = image_bgr.copy()
            for kpts, scores in zip(keypoints, keypoint_scores, strict=False):
                for (x, y), sc in zip(kpts, scores, strict=False):
                    if float(sc) < self.args.kpt_thr:
                        continue
                    cv2.circle(out, (int(x), int(y)), self.args.radius, (0, 255, 0), -1)
        if draw_id and instances:
            out = _draw_person_id_labels(out, instances, draw_bbox=True)
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


def _tag_raw_ids(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-frame DETR order IDs (1..N) before cross-frame geometric Re-ID."""
    return [{**inst, "raw_id": i} for i, inst in enumerate(instances, start=1)]


def _instance_link_xy(
    inst: dict[str, Any], *, kpt_thr: float = DEFAULT_KPT_THR
) -> tuple[float, float]:
    """Foot proxy (SAM canonical anchor) or mid-hip when confident — for Re-ID linking."""
    hip = _mid_hip_xy(inst, kpt_thr=kpt_thr)
    if hip is not None:
        return hip
    x1, y1, x2, y2 = [float(v) for v in inst["bbox"][:4]]
    return (x1 + x2) * 0.5, y2


def _sapiens_reid_config(
    *,
    max_gap: int = DEFAULT_REID_MAX_GAP,
    max_dist: float = DEFAULT_REID_MAX_DIST_PX,
    min_iou: float = DEFAULT_REID_MIN_IOU,
    direction_weight: float = DEFAULT_REID_DIRECTION_WEIGHT,
    static_speed: float = DEFAULT_REID_STATIC_SPEED,
    static_radius_px: float = DEFAULT_REID_STATIC_RADIUS_PX,
    kpt_oks_weight: float = DEFAULT_REID_KPT_OKS_WEIGHT,
    kpt_thr: float = DEFAULT_KPT_THR,
) -> GeometricLinkerConfig:
    """Shared geometric linker tuning (SAM / yolov26track / reid_markers family)."""
    return GeometricLinkerConfig(
        max_gap=max(1, int(max_gap)),
        max_centroid_dist_px=float(max_dist),
        min_iou=float(min_iou),
        direction_weight=float(direction_weight),
        static_speed_threshold=float(static_speed),
        static_anchor_radius_px=float(static_radius_px),
        static_mismatch_penalty=4.0,
        mobility_warmup_frames=6,
        kpt_oks_weight=float(kpt_oks_weight),
        kpt_thr=float(kpt_thr),
    )


def _instances_to_frame_dets(
    instances: list[dict[str, Any]],
    *,
    kpt_thr: float = DEFAULT_KPT_THR,
) -> list[dict[str, Any]]:
    """Build linker detection dicts from Sapiens pose instances."""
    frame_dets: list[dict[str, Any]] = []
    for i, inst in enumerate(instances):
        raw_id = int(inst.get("raw_id", i + 1))
        frame_dets.append(
            {
                "raw_id": raw_id,
                "tracker_id": raw_id,
                "xyxy": tuple(float(v) for v in inst["bbox"][:4]),
                "link_xy": _instance_link_xy(inst, kpt_thr=kpt_thr),
                "keypoints": inst.get("keypoints"),
                "keypoint_scores": inst.get("keypoint_scores"),
            }
        )
    return frame_dets


class SapiensTemporalLinker:
    """Online temporal ID linker — assigns ``temporal_id`` during inference."""

    def __init__(self, config: GeometricLinkerConfig | None = None) -> None:
        cfg = config or _sapiens_reid_config()
        self.linker = GeometricFrameLinker(enabled=True, config=cfg, start_stable_id=0)

    def assign_instances(
        self,
        frame_idx: int,
        instances: list[dict[str, Any]],
        *,
        kpt_thr: float = DEFAULT_KPT_THR,
    ) -> list[dict[str, Any]]:
        if not instances:
            return []
        linked = self.linker.assign_frame(
            frame_idx, _instances_to_frame_dets(instances, kpt_thr=kpt_thr)
        )
        out: list[dict[str, Any]] = []
        for inst, ld in zip(instances, linked, strict=True):
            tagged = dict(inst)
            tid = int(ld["stable_id"])
            tagged["temporal_id"] = tid
            tagged["stable_id"] = tid
            out.append(tagged)
        return out


def _stabilize_sapiens_pose_timeline(
    pose_by_frame: dict[int, list[dict[str, Any]]],
    *,
    config: GeometricLinkerConfig | None = None,
    kpt_thr: float = DEFAULT_KPT_THR,
) -> tuple[dict[int, list[dict[str, Any]]], list[tuple[int, int, int]]]:
    """Single-pass forward geometric Re-ID (Hungarian + velocity + OKS)."""
    if not pose_by_frame:
        return {}, []
    cfg = config or _sapiens_reid_config(kpt_thr=kpt_thr)
    linker = GeometricFrameLinker(enabled=True, config=cfg, start_stable_id=0)
    stabilized: dict[int, list[dict[str, Any]]] = {}
    for frame_idx in sorted(pose_by_frame.keys()):
        instances = pose_by_frame[frame_idx]
        linked = linker.assign_frame(
            frame_idx, _instances_to_frame_dets(instances, kpt_thr=kpt_thr)
        )
        out_instances: list[dict[str, Any]] = []
        for inst, ld in zip(instances, linked, strict=True):
            tagged = dict(inst)
            tagged["stable_id"] = int(ld["stable_id"])
            out_instances.append(tagged)
        stabilized[frame_idx] = out_instances
    return stabilized, linker.reid_links


def _merge_bidirectional_pose_timelines(
    forward: dict[int, list[dict[str, Any]]],
    backward: dict[int, list[dict[str, Any]]],
    *,
    kpt_thr: float = DEFAULT_KPT_THR,
) -> dict[int, list[dict[str, Any]]]:
    """Prefer backward stable IDs from mid-video onward (reid_markers pattern)."""
    if not forward:
        return {}
    frames = sorted(forward.keys())
    if not frames:
        return forward
    mid = (frames[0] + frames[-1]) // 2
    merged: dict[int, list[dict[str, Any]]] = {}
    for frame_idx in frames:
        if frame_idx <= mid or frame_idx not in backward:
            merged[frame_idx] = forward[frame_idx]
            continue
        fwd_insts = forward[frame_idx]
        bwd_insts = backward[frame_idx]
        if not fwd_insts:
            merged[frame_idx] = []
            continue
        if len(fwd_insts) != len(bwd_insts):
            merged[frame_idx] = bwd_insts
            continue
        n = len(fwd_insts)
        cost = np.full((n, n), 1e6, dtype=float)
        fwd_xy = [_instance_link_xy(inst, kpt_thr=kpt_thr) for inst in fwd_insts]
        bwd_xy = [_instance_link_xy(inst, kpt_thr=kpt_thr) for inst in bwd_insts]
        for i, fxy in enumerate(fwd_xy):
            for j, bxy in enumerate(bwd_xy):
                cost[i, j] = float(np.hypot(fxy[0] - bxy[0], fxy[1] - bxy[1]))
        pairs = assignment_min_cost(cost)
        combined = [dict(fwd_insts[i]) for i in range(n)]
        for i, j in pairs:
            combined[i]["stable_id"] = int(
                bwd_insts[j].get("stable_id", combined[i].get("stable_id", 0))
            )
        merged[frame_idx] = combined
    return merged


def _stabilize_sapiens_pose_timeline_bidirectional(
    pose_by_frame: dict[int, list[dict[str, Any]]],
    *,
    config: GeometricLinkerConfig | None = None,
    kpt_thr: float = DEFAULT_KPT_THR,
) -> tuple[dict[int, list[dict[str, Any]]], list[tuple[int, int, int, int]]]:
    """Forward + backward geometric Re-ID with mid-video merge."""
    if not pose_by_frame:
        return {}, []
    cfg = config or _sapiens_reid_config(kpt_thr=kpt_thr)
    forward, fwd_links = _stabilize_sapiens_pose_timeline(
        pose_by_frame, config=cfg, kpt_thr=kpt_thr
    )
    rev_frames = sorted(pose_by_frame.keys(), reverse=True)
    rev_pose: dict[int, list[dict[str, Any]]] = {}
    for frame_idx in rev_frames:
        rev_pose[-frame_idx] = pose_by_frame[frame_idx]
    backward_rev, _bwd_links = _stabilize_sapiens_pose_timeline(
        rev_pose, config=cfg, kpt_thr=kpt_thr
    )
    backward = {(-fi): insts for fi, insts in backward_rev.items()}
    merged = _merge_bidirectional_pose_timelines(forward, backward, kpt_thr=kpt_thr)
    audit: list[tuple[int, int, int, int]] = []
    for frame_idx in sorted(merged.keys()):
        for inst in merged[frame_idx]:
            audit.append(
                (
                    frame_idx,
                    int(inst.get("raw_id", 0)),
                    int(inst.get("temporal_id", inst.get("stable_id", 0))),
                    int(inst.get("stable_id", 0)),
                )
            )
    return merged, audit


def _apply_raw_ids_as_stable(
    pose_by_frame: dict[int, list[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    """Skip geometric Re-ID: ``stable_id`` mirrors per-frame ``raw_id``."""
    out: dict[int, list[dict[str, Any]]] = {}
    for frame_idx, instances in pose_by_frame.items():
        tagged: list[dict[str, Any]] = []
        for i, inst in enumerate(instances):
            row = dict(inst)
            row["stable_id"] = int(inst.get("raw_id", i + 1))
            tagged.append(row)
        out[frame_idx] = tagged
    return out


def _run_sapiens_appearance_reid(
    video_path: Path,
    pose_by_frame: dict[int, list[dict[str, Any]]],
    *,
    threshold: float = DEFAULT_APPEARANCE_REID_THRESHOLD,
    device: str = "cuda:0",
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, int]]:
    """Optional OSNet merge of ``stable_id`` after geometric temporal linking."""
    if not pose_by_frame:
        return pose_by_frame, {}
    try:
        import torch

        try:
            from .reid_yolotrack import _get_reid_model
        except ImportError:
            from reid_yolotrack import _get_reid_model  # ty: ignore[unresolved-import]
    except ImportError as exc:
        _sapiens_log(f"boxmot/torch not available; skip appearance ReID: {exc}")
        return pose_by_frame, {}

    dev = device if torch.cuda.is_available() else "cpu"
    try:
        reid_model = _get_reid_model("osnet_x0_25_msmt17.pt", dev)
    except Exception as exc:
        _sapiens_log(f"Could not load OSNet ReID model; skip appearance ReID: {exc}")
        return pose_by_frame, {}

    id_features: dict[int, list[Any]] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        _sapiens_log(f"Could not open video for appearance ReID: {video_path}")
        return pose_by_frame, {}

    max_frame = max(pose_by_frame.keys()) if pose_by_frame else 0
    fi = 0
    try:
        while fi <= max_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if fi in pose_by_frame:
                for inst in pose_by_frame[fi]:
                    sid = int(inst.get("stable_id", -1))
                    if sid < 0:
                        continue
                    bbox = inst.get("bbox") or []
                    if len(bbox) < 4:
                        continue
                    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
                    h, w = frame.shape[:2]
                    x1 = max(0, min(w - 1, x1))
                    x2 = max(0, min(w, x2))
                    y1 = max(0, min(h - 1, y1))
                    y2 = max(0, min(h, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    feature = reid_model(crop)
                    id_features.setdefault(sid, []).append(feature)
            fi += 1
    finally:
        cap.release()

    if not id_features:
        return pose_by_frame, {}

    avg_features: dict[int, Any] = {}
    for sid, feats in id_features.items():
        if feats:
            avg_features[sid] = torch.stack(feats).mean(dim=0)

    sorted_ids = sorted(avg_features.keys())
    id_mapping: dict[int, int] = {}
    processed: set[int] = set()
    next_id = 0
    for sid in sorted_ids:
        if sid in processed:
            continue
        cluster = [sid]
        processed.add(sid)
        for other in sorted_ids:
            if other in processed or other == sid:
                continue
            sim = torch.cosine_similarity(
                avg_features[sid].unsqueeze(0),
                avg_features[other].unsqueeze(0),
            )
            if float(sim) > float(threshold):
                cluster.append(other)
                processed.add(other)
        for member in cluster:
            id_mapping[member] = next_id
        next_id += 1

    if len(id_mapping) == len(sorted_ids) and all(k == v for k, v in id_mapping.items()):
        return pose_by_frame, id_mapping

    remapped: dict[int, list[dict[str, Any]]] = {}
    for frame_idx, instances in pose_by_frame.items():
        rows: list[dict[str, Any]] = []
        for inst in instances:
            row = dict(inst)
            old = int(inst.get("stable_id", 0))
            row["stable_id"] = int(id_mapping.get(old, old))
            rows.append(row)
        remapped[frame_idx] = rows
    _sapiens_log(
        f"Appearance ReID: merged {len(sorted_ids)} geometric track(s) -> {next_id} identity/ies"
    )
    return remapped, id_mapping


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


_SAPIENS_ID_COLORS_BGR: tuple[tuple[int, int, int], ...] = (
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 128, 255),
    (255, 128, 0),
    (128, 0, 255),
    (128, 255, 0),
)


def _color_for_stable_id(stable_id: int) -> tuple[int, int, int]:
    """Distinct BGR color per ``stable_id`` (overlay bbox + label)."""
    return _SAPIENS_ID_COLORS_BGR[int(stable_id) % len(_SAPIENS_ID_COLORS_BGR)]


def _format_instance_id_label(stable_id: int) -> str:
    """Overlay tag — same ``#N`` style as SAM3 (no redundant ``pN`` slot suffix)."""
    return f"#{int(stable_id)}"


def _draw_person_id_labels(
    image_bgr: np.ndarray,
    instances: list[dict[str, Any]],
    *,
    draw_bbox: bool = True,
) -> np.ndarray:
    """Draw per-person ``#N`` tag (+ optional bbox) on an overlay frame."""
    if not instances:
        return image_bgr
    out = image_bgr
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    for inst in instances:
        sid = int(inst.get("stable_id", 0))
        if sid < 0:
            continue
        color = _color_for_stable_id(sid)
        bbox = inst.get("bbox") or []
        if len(bbox) >= 4:
            x1 = max(0, min(w - 1, int(round(float(bbox[0])))))
            y1 = max(0, min(h - 1, int(round(float(bbox[1])))))
            x2 = max(0, min(w - 1, int(round(float(bbox[2])))))
            y2 = max(0, min(h - 1, int(round(float(bbox[3])))))
            if draw_bbox and x2 > x1 and y2 > y1:
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        else:
            x1, y1 = 0, 0
        label = _format_instance_id_label(sid)
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        pad = 3
        tx = max(0, min(w - tw - pad * 2, x1))
        ty = max(th + baseline + pad * 2, y1 - 4)
        bx1 = tx
        by1 = max(0, ty - th - baseline - pad * 2)
        bx2 = min(w - 1, tx + tw + pad * 2)
        by2 = min(h - 1, ty + pad)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 0, 0), thickness=-1)
        cv2.putText(
            out,
            label,
            (tx + pad, by2 - pad - baseline),
            font,
            scale,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
    return out


def _write_sapiens_overlay_from_timeline(
    video_path: Path,
    overlay_path: Path,
    timeline: dict[int, list[dict[str, Any]]],
    session: PoseInferenceSession,
    *,
    draw_id: bool = True,
    desc: str = "overlay",
) -> Path:
    """Second video pass: skeleton + ID tags from a stabilized pose timeline."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer, overlay_path = _open_sam3_video_writer(
        overlay_path,
        fps,
        (w, h),
        purpose=f"Sapiens2 {desc}",
    )

    fi = 0
    tqdm_cls = None if _progress_quiet() else _try_import_tqdm()
    pbar = (
        tqdm_cls(total=n_frames if n_frames > 0 else None, desc=desc[:48], unit="fr")
        if tqdm_cls is not None
        else None
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            instances_draw = timeline.get(fi, [])
            if instances_draw:
                kpts = [np.asarray(i["keypoints"], dtype=float) for i in instances_draw]
                scrs = [np.asarray(i["keypoint_scores"], dtype=float) for i in instances_draw]
                vis = session.render_overlay(
                    frame,
                    kpts,
                    scrs,
                    instances=instances_draw,
                    draw_id=draw_id,
                )
            else:
                vis = frame
            writer.write(vis)
            fi += 1
            if pbar is not None:
                pbar.update(1)
    finally:
        cap.release()
        writer.release()
        if pbar is not None:
            pbar.close()
    return overlay_path


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


_SAPIENS308_KEYPOINT_NAMES_CACHE: list[str] | None = None


def _keypoint_names_from_pose_metainfo(
    meta: dict[str, Any] | None,
    *,
    n_kp: int | None = None,
) -> list[str] | None:
    """Ordered Sociopticon labels from Sapiens ``pose_metainfo`` (308 kp)."""
    if not isinstance(meta, dict):
        return None
    raw_names = meta.get("keypoint_names")
    if isinstance(raw_names, (list, tuple)) and raw_names:
        names = [str(n) for n in raw_names]
        if n_kp is not None:
            names = names[:n_kp]
        return names if names else None

    id2name = meta.get("keypoint_id2name")
    if not isinstance(id2name, dict) or not id2name:
        return None

    limit = n_kp if n_kp is not None else int(meta.get("num_keypoints") or len(id2name))
    names: list[str] = []
    for i in range(limit):
        val = id2name.get(i)
        if val is None:
            val = id2name.get(str(i))
        if val is None:
            return None
        names.append(str(val))
    return names


def _load_sapiens308_keypoint_names_cached() -> list[str] | None:
    """Load Sociopticon 308 labels from upstream ``keypoints308.py`` (cached)."""
    global _SAPIENS308_KEYPOINT_NAMES_CACHE
    if _SAPIENS308_KEYPOINT_NAMES_CACHE is not None:
        return list(_SAPIENS308_KEYPOINT_NAMES_CACHE)
    try:
        _require_sapiens_installed()
        with _sapiens_pose_context():
            from sapiens.pose.datasets import (  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
                parse_pose_metainfo,
            )

            meta = parse_pose_metainfo({"from_file": "configs/_base_/keypoints308.py"})
        names = _keypoint_names_from_pose_metainfo(meta)
        if names:
            _SAPIENS308_KEYPOINT_NAMES_CACHE = list(names)
        return names
    except Exception:
        return None


def _resolve_sapiens_keypoint_names(
    session: PoseInferenceSession | None,
    *,
    n_kp: int | None = None,
) -> list[str] | None:
    meta: dict[str, Any] | None = None
    if session is not None and getattr(session, "model", None) is not None:
        raw_meta = getattr(session.model, "pose_metainfo", None)
        if isinstance(raw_meta, dict):
            meta = raw_meta
    names = _keypoint_names_from_pose_metainfo(meta, n_kp=n_kp)
    if names and (n_kp is None or len(names) >= n_kp):
        return names[:n_kp] if n_kp is not None else names
    fallback = _load_sapiens308_keypoint_names_cached()
    if fallback and (n_kp is None or len(fallback) >= n_kp):
        return fallback[:n_kp] if n_kp is not None else fallback
    return names


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


@dataclass(frozen=True, slots=True)
class SapiensGuiSettings:
    input_path: Path
    out_parent: Path
    model: str
    stride: int
    kpt_thr: float
    bbox_thr: float
    nms_thr: float
    max_persons: int
    device: int
    flip_test: bool
    save_overlay: bool
    draw_id: bool
    stabilize_ids: bool
    reid_bidirectional: bool
    appearance_reid: bool
    appearance_reid_threshold: float
    reid_static_speed: float
    reid_static_radius: float
    pose_batch_size: int | None


def _format_sapiens_cli_command(
    input_path: Path | str,
    output_parent: Path | str,
    *,
    model: str = DEFAULT_MODEL_KEY,
    stride: int = 1,
    kpt_thr: float = DEFAULT_KPT_THR,
    bbox_thr: float = DEFAULT_BBOX_THR,
    nms_thr: float = DEFAULT_NMS_THR,
    max_persons: int = DEFAULT_MAX_PERSONS,
    device: int = 0,
    save_overlay: bool = True,
    draw_id: bool = True,
    stabilize_ids: bool = True,
    reid_max_gap: int = DEFAULT_REID_MAX_GAP,
    reid_max_dist: float = DEFAULT_REID_MAX_DIST_PX,
    reid_min_iou: float = DEFAULT_REID_MIN_IOU,
    reid_direction_weight: float = DEFAULT_REID_DIRECTION_WEIGHT,
    reid_static_speed: float = DEFAULT_REID_STATIC_SPEED,
    reid_static_radius: float = DEFAULT_REID_STATIC_RADIUS_PX,
    reid_bidirectional: bool = True,
    appearance_reid: bool = False,
    appearance_reid_threshold: float = DEFAULT_APPEARANCE_REID_THRESHOLD,
    flip_test: bool = False,
    pose_batch_size: int | None = None,
    quiet: bool = False,
) -> str:
    """Build copy-paste CLI equivalent to a GUI Sapiens2 run.

    User-facing mirror uses only ``-o`` (parent). Internal ``--output-base`` is
    for GUI/worker subprocesses and is never printed here.
    """
    argv = _build_sapiens_cli_argv(
        input_path=Path(input_path),
        out_parent=Path(output_parent),
        model=model,
        stride=stride,
        kpt_thr=kpt_thr,
        bbox_thr=bbox_thr,
        nms_thr=nms_thr,
        max_persons=max_persons,
        device=device,
        save_overlay=save_overlay,
        draw_id=draw_id,
        stabilize_ids=stabilize_ids,
        reid_max_gap=reid_max_gap,
        reid_max_dist=reid_max_dist,
        reid_min_iou=reid_min_iou,
        reid_direction_weight=reid_direction_weight,
        reid_static_speed=reid_static_speed,
        reid_static_radius=reid_static_radius,
        reid_bidirectional=reid_bidirectional,
        appearance_reid=appearance_reid,
        appearance_reid_threshold=appearance_reid_threshold,
        flip_test=flip_test,
        pose_batch_size=pose_batch_size,
        quiet=quiet,
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
    kpt_thr: float = DEFAULT_KPT_THR,
    bbox_thr: float = DEFAULT_BBOX_THR,
    nms_thr: float = DEFAULT_NMS_THR,
    max_persons: int = DEFAULT_MAX_PERSONS,
    device: int = 0,
    save_overlay: bool = True,
    draw_id: bool = True,
    stabilize_ids: bool = True,
    reid_max_gap: int = DEFAULT_REID_MAX_GAP,
    reid_max_dist: float = DEFAULT_REID_MAX_DIST_PX,
    reid_min_iou: float = DEFAULT_REID_MIN_IOU,
    reid_direction_weight: float = DEFAULT_REID_DIRECTION_WEIGHT,
    reid_static_speed: float = DEFAULT_REID_STATIC_SPEED,
    reid_static_radius: float = DEFAULT_REID_STATIC_RADIUS_PX,
    reid_bidirectional: bool = True,
    appearance_reid: bool = False,
    appearance_reid_threshold: float = DEFAULT_APPEARANCE_REID_THRESHOLD,
    flip_test: bool = False,
    pose_batch_size: int | None = None,
    quiet: bool = False,
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
        "--max-persons",
        str(max(1, int(max_persons))),
        "--device",
        str(device),
    ]
    if pose_batch_size is not None:
        cmd += ["--pose-batch-size", str(max(1, int(pose_batch_size)))]
    if flip_test:
        cmd.append("--flip-test")
    if stabilize_ids:
        cmd.append("--stabilize-ids")
    else:
        cmd.append("--no-stabilize-ids")
    if (
        reid_max_gap != DEFAULT_REID_MAX_GAP
        or reid_max_dist != DEFAULT_REID_MAX_DIST_PX
        or reid_min_iou != DEFAULT_REID_MIN_IOU
        or reid_direction_weight != DEFAULT_REID_DIRECTION_WEIGHT
        or reid_static_speed != DEFAULT_REID_STATIC_SPEED
        or reid_static_radius != DEFAULT_REID_STATIC_RADIUS_PX
    ):
        cmd += [
            "--reid-max-gap",
            str(reid_max_gap),
            "--reid-max-dist",
            str(reid_max_dist),
            "--reid-min-iou",
            str(reid_min_iou),
            "--reid-direction-weight",
            str(reid_direction_weight),
            "--reid-static-speed",
            str(reid_static_speed),
            "--reid-static-radius",
            str(reid_static_radius),
        ]
    if not reid_bidirectional:
        cmd.append("--no-reid-bidirectional")
    if appearance_reid:
        cmd.append("--appearance-reid")
        if appearance_reid_threshold != DEFAULT_APPEARANCE_REID_THRESHOLD:
            cmd += ["--appearance-reid-threshold", str(appearance_reid_threshold)]
    if not save_overlay:
        cmd.append("--no-overlay")
    if not draw_id:
        cmd.append("--no-draw-id")
    if quiet:
        cmd.append("--quiet")
    return cmd


def _build_isolated_sapiens_cmd(
    *,
    video_file: Path,
    out_parent: Path,
    output_base: Path,
    out_dir: Path,
    model: str,
    stride: int,
    kpt_thr: float,
    bbox_thr: float,
    nms_thr: float,
    device: int,
    save_overlay: bool,
    draw_id: bool,
    stabilize_ids: bool,
    flip_test: bool,
    max_persons: int,
    pose_batch_size: int | None,
    reid_max_gap: int = DEFAULT_REID_MAX_GAP,
    reid_max_dist: float = DEFAULT_REID_MAX_DIST_PX,
    reid_min_iou: float = DEFAULT_REID_MIN_IOU,
    reid_direction_weight: float = DEFAULT_REID_DIRECTION_WEIGHT,
    reid_static_speed: float = DEFAULT_REID_STATIC_SPEED,
    reid_static_radius: float = DEFAULT_REID_STATIC_RADIUS_PX,
    reid_bidirectional: bool = True,
    appearance_reid: bool = False,
    appearance_reid_threshold: float = DEFAULT_APPEARANCE_REID_THRESHOLD,
) -> list[str]:
    cmd = _build_sapiens_cli_argv(
        input_path=video_file,
        out_parent=out_parent,
        output_base=output_base,
        model=model,
        stride=stride,
        kpt_thr=kpt_thr,
        bbox_thr=bbox_thr,
        nms_thr=nms_thr,
        max_persons=max_persons,
        device=device,
        save_overlay=save_overlay,
        draw_id=draw_id,
        stabilize_ids=stabilize_ids,
        reid_max_gap=reid_max_gap,
        reid_max_dist=reid_max_dist,
        reid_min_iou=reid_min_iou,
        reid_direction_weight=reid_direction_weight,
        reid_static_speed=reid_static_speed,
        reid_static_radius=reid_static_radius,
        reid_bidirectional=reid_bidirectional,
        appearance_reid=appearance_reid,
        appearance_reid_threshold=appearance_reid_threshold,
        flip_test=flip_test,
        pose_batch_size=pose_batch_size,
        for_subprocess=True,
    )
    o_idx = cmd.index(str(out_parent.resolve()))
    insert_at = o_idx + 1
    cmd[insert_at:insert_at] = [
        "--video-output-dir",
        str(out_dir.resolve()),
        "--no-isolate-batch",
    ]
    return cmd


def _run_sapiens_batch(
    videos: list[Path],
    output_base: Path,
    *,
    model: str,
    stride: int,
    save_overlay: bool,
    draw_id: bool = True,
    stabilize_ids: bool = True,
    reid_max_gap: int = DEFAULT_REID_MAX_GAP,
    reid_max_dist: float = DEFAULT_REID_MAX_DIST_PX,
    reid_min_iou: float = DEFAULT_REID_MIN_IOU,
    reid_direction_weight: float = DEFAULT_REID_DIRECTION_WEIGHT,
    reid_static_speed: float = DEFAULT_REID_STATIC_SPEED,
    reid_static_radius: float = DEFAULT_REID_STATIC_RADIUS_PX,
    reid_bidirectional: bool = True,
    appearance_reid: bool = False,
    appearance_reid_threshold: float = DEFAULT_APPEARANCE_REID_THRESHOLD,
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
                draw_id=draw_id,
                stabilize_ids=stabilize_ids,
                reid_max_gap=reid_max_gap,
                reid_max_dist=reid_max_dist,
                reid_min_iou=reid_min_iou,
                reid_direction_weight=reid_direction_weight,
                reid_static_speed=reid_static_speed,
                reid_static_radius=reid_static_radius,
                reid_bidirectional=reid_bidirectional,
                appearance_reid=appearance_reid,
                appearance_reid_threshold=appearance_reid_threshold,
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
    kpt_thr: float = DEFAULT_KPT_THR,
    bbox_thr: float = DEFAULT_BBOX_THR,
    nms_thr: float = DEFAULT_NMS_THR,
    max_persons: int = DEFAULT_MAX_PERSONS,
    device: int = 0,
    save_overlay: bool = True,
    draw_id: bool = True,
    stabilize_ids: bool = True,
    reid_max_gap: int = DEFAULT_REID_MAX_GAP,
    reid_max_dist: float = DEFAULT_REID_MAX_DIST_PX,
    reid_min_iou: float = DEFAULT_REID_MIN_IOU,
    reid_direction_weight: float = DEFAULT_REID_DIRECTION_WEIGHT,
    reid_static_speed: float = DEFAULT_REID_STATIC_SPEED,
    reid_static_radius: float = DEFAULT_REID_STATIC_RADIUS_PX,
    reid_bidirectional: bool = True,
    appearance_reid: bool = False,
    appearance_reid_threshold: float = DEFAULT_APPEARANCE_REID_THRESHOLD,
    flip_test: bool = False,
    pose_batch_size: int | None = None,
    quiet: bool = False,
) -> None:
    """Print GUI→CLI mirror to stdout (>> prefix avoids absl eating bracketed lines)."""
    cmd = _format_sapiens_cli_command(
        input_path,
        output_parent,
        model=model,
        stride=stride,
        kpt_thr=kpt_thr,
        bbox_thr=bbox_thr,
        nms_thr=nms_thr,
        max_persons=max_persons,
        device=device,
        save_overlay=save_overlay,
        draw_id=draw_id,
        stabilize_ids=stabilize_ids,
        reid_max_gap=reid_max_gap,
        reid_max_dist=reid_max_dist,
        reid_min_iou=reid_min_iou,
        reid_direction_weight=reid_direction_weight,
        reid_static_speed=reid_static_speed,
        reid_static_radius=reid_static_radius,
        reid_bidirectional=reid_bidirectional,
        appearance_reid=appearance_reid,
        appearance_reid_threshold=appearance_reid_threshold,
        flip_test=flip_test,
        pose_batch_size=pose_batch_size,
        quiet=quiet,
    )
    print("\n>> vaila/vaila_sapiens: Equivalent CLI (copy/paste):", flush=True)
    print(f">>   {cmd}", flush=True)
    print(">> (CLI creates processed_sapiens_<timestamp>/ under -o)\n", flush=True)


def _write_readme_sapiens(output_dir: Path, *, model_key: str, stride: int) -> None:
    text = (
        "vailá Sapiens2 Pose run\n"
        f"model={model_key} stride={stride}\n\n"
        "Outputs:\n"
        "  <video>_sapiens_overlay.mp4  — skeleton overlay + id NN tags (see sapiens_id_map.csv)\n"
        "  <video>_predictions.json     — per-frame instances (308 kp Sociopticon)\n"
        "  <video>_sapiens_vaila.csv    — long CSV: frame,person_id,kpt_idx,x,y,score\n"
        "  <video>_markers.csv          — REC2D/REC3D + getpixelvideo (bbox foot anchor)\n"
        "  sapiens_vaila_*.csv          — five anchor tables (same schema as sam_vaila_*)\n"
        "  sapiens_points.csv           — foot + bbox center + mid-hip per pN\n"
        "  sapiens_id_map.csv           — stable person slot map\n"
        "  sapiens_reid_links.csv       — geometric Re-ID audit (frame,raw_id,stable_id)\n"
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
    draw_id: bool = True,
    bbox_thr: float = 0.3,
    nms_thr: float = 0.3,
    kpt_thr: float = 0.3,
    device: str | None = None,
    flip_test: bool = False,
    max_persons: int = 8,
    pose_batch_size: int | None = None,
    stabilize_ids: bool = True,
    reid_max_gap: int = DEFAULT_REID_MAX_GAP,
    reid_max_dist: float = DEFAULT_REID_MAX_DIST_PX,
    reid_min_iou: float = DEFAULT_REID_MIN_IOU,
    reid_direction_weight: float = DEFAULT_REID_DIRECTION_WEIGHT,
    reid_static_speed: float = DEFAULT_REID_STATIC_SPEED,
    reid_static_radius: float = DEFAULT_REID_STATIC_RADIUS_PX,
    reid_bidirectional: bool = True,
    appearance_reid: bool = False,
    appearance_reid_threshold: float = DEFAULT_APPEARANCE_REID_THRESHOLD,
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
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    stem = video_path.stem
    overlay_path = output_dir / f"{stem}_sapiens_overlay.mp4"
    json_path = output_dir / f"{stem}_predictions.json"
    csv_path = output_dir / f"{stem}_sapiens_vaila.csv"
    reid_links_path = output_dir / "sapiens_reid_links.csv"

    stride = max(1, int(stride))
    pose_by_frame: dict[int, list[dict[str, Any]]] = {}
    image_size = [h, w]
    num_keypoints: int | None = None
    reid_config = _sapiens_reid_config(
        max_gap=reid_max_gap,
        max_dist=reid_max_dist,
        min_iou=reid_min_iou,
        direction_weight=reid_direction_weight,
        static_speed=reid_static_speed,
        static_radius_px=reid_static_radius,
        kpt_thr=float(kpt_thr),
    )
    temporal_linker = SapiensTemporalLinker(reid_config) if stabilize_ids else None

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
                instances = _tag_raw_ids(instances_from_frame(keypoints, scores, bboxes))
                if temporal_linker is not None and instances:
                    instances = temporal_linker.assign_instances(
                        fi, instances, kpt_thr=float(kpt_thr)
                    )
                if instances:
                    pose_by_frame[fi] = instances
                if num_keypoints is None and instances:
                    num_keypoints = len(instances[0].get("keypoints", []))
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
                    _sapiens_log(f"{stem}: infer {fi}/{n_frames} ({pct:.0f}%)")
                else:
                    _sapiens_log(f"{stem}: infer frame {fi}")
    finally:
        cap.release()
        if pbar is not None:
            pbar.close()

    if stabilize_ids:
        if reid_bidirectional:
            pose_by_frame, reid_audit = _stabilize_sapiens_pose_timeline_bidirectional(
                pose_by_frame, config=reid_config, kpt_thr=float(kpt_thr)
            )
            with reid_links_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(("frame", "raw_id", "temporal_id", "stable_id"))
                writer.writerows(reid_audit)
        else:
            pose_by_frame, reid_links = _stabilize_sapiens_pose_timeline(
                pose_by_frame, config=reid_config, kpt_thr=float(kpt_thr)
            )
            write_reid_links_csv(
                str(reid_links_path),
                reid_links,
                ("frame", "raw_id", "stable_id"),
            )
        if appearance_reid:
            pose_by_frame, _app_map = _run_sapiens_appearance_reid(
                video_path,
                pose_by_frame,
                threshold=float(appearance_reid_threshold),
                device=str(dev),
            )
        n_tracks = len(
            {
                int(inst["stable_id"])
                for insts in pose_by_frame.values()
                for inst in insts
                if "stable_id" in inst
            }
        )
        _sapiens_log(
            f"Temporal Re-ID ({'bidirectional' if reid_bidirectional else 'forward'}): "
            f"{n_tracks} track(s) -> {reid_links_path.name}"
        )
    else:
        pose_by_frame = _apply_raw_ids_as_stable(pose_by_frame)
        reid_links = []

    csv_rows: list[tuple[int, int, int, float, float, float]] = []
    frames_records: list[dict[str, Any]] = []
    for frame_idx in sorted(pose_by_frame.keys()):
        instances = pose_by_frame[frame_idx]
        csv_rows.extend(flatten_instances_to_csv_rows(frame_idx, instances))
        frames_records.append({"frame_index": frame_idx, "instances": instances})

    timeline = _expand_pose_timeline(pose_by_frame, fi)

    kp_names = _resolve_sapiens_keypoint_names(session, n_kp=num_keypoints)

    if save_overlay:
        _sapiens_log(f"Writing overlay from stabilized IDs ({overlay_path.name}) …")
        _write_sapiens_overlay_from_timeline(
            video_path,
            overlay_path,
            timeline,
            session,
            draw_id=draw_id,
            desc=f"{stem} overlay",
        )

    _release_sapiens_gpu_memory(session)

    payload = {
        "video": stem,
        "image_size": image_size,
        "num_keypoints": num_keypoints,
        "keypoint_names": kp_names or [],
        "model": spec.arch,
        "stride": stride,
        "kpt_thr_used": float(kpt_thr),
        "stabilize_ids": bool(stabilize_ids),
        "reid_bidirectional": bool(reid_bidirectional),
        "appearance_reid": bool(appearance_reid),
        "reid_max_gap": int(reid_config.max_gap),
        "reid_max_dist_px": float(reid_config.max_centroid_dist_px),
        "reid_min_iou": float(reid_config.min_iou),
        "reid_direction_weight": float(reid_config.direction_weight),
        "reid_kpt_oks_weight": float(reid_config.kpt_oks_weight),
        "reid_static_speed": float(reid_config.static_speed_threshold),
        "reid_static_radius_px": float(reid_config.static_anchor_radius_px),
        "frames": frames_records,
    }
    _sapiens_log(f"Writing JSON + biomechanics CSVs for {stem} …")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_vaila_pose_csv(csv_path, csv_rows)
    timeline = _expand_pose_timeline(pose_by_frame, fi)
    bio_paths = write_sapiens_biomechanics_csvs(output_dir, stem, timeline, kpt_thr=float(kpt_thr))
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


def _load_predictions_timeline(
    json_path: Path,
) -> tuple[str, dict[int, list[dict[str, Any]]]]:
    """Load per-frame instances from ``<stem>_predictions.json``."""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    stem = str(payload.get("video", json_path.stem.replace("_predictions", "")))
    pose_by_frame: dict[int, list[dict[str, Any]]] = {}
    for rec in payload.get("frames", []):
        fi = int(rec["frame_index"])
        pose_by_frame[fi] = list(rec.get("instances", []))
    return stem, pose_by_frame


def rerender_sapiens_overlay(
    video_path: Path,
    run_dir: Path,
    *,
    model: str = DEFAULT_MODEL_KEY,
    kpt_thr: float = DEFAULT_KPT_THR,
    device: str | None = None,
    draw_id: bool = True,
) -> Path:
    """Re-write overlay MP4 from existing predictions JSON (no pose re-inference)."""
    stem = video_path.stem
    json_path = run_dir / f"{stem}_predictions.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing predictions JSON: {json_path}")

    _, pose_by_frame = _load_predictions_timeline(json_path)
    if not pose_by_frame:
        raise ValueError(f"No pose frames in {json_path}")

    cap_probe = cv2.VideoCapture(str(video_path))
    if not cap_probe.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    n_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap_probe.release()

    timeline = _expand_pose_timeline(
        pose_by_frame, n_frames if n_frames > 0 else max(pose_by_frame) + 1
    )
    _, slot_by_id = _collect_stable_slots(timeline)
    overlay_path = run_dir / f"{stem}_sapiens_overlay.mp4"
    _sapiens_log(
        f"Rerendering overlay from {json_path.name} → {overlay_path.name} "
        f"({len(slot_by_id)} person id(s), draw_id={draw_id})"
    )

    dev = device or _require_cuda()
    spec = resolve_model_spec(model)
    session = PoseInferenceSession(spec, device=dev, kpt_thr=kpt_thr)
    try:
        return _write_sapiens_overlay_from_timeline(
            video_path,
            overlay_path,
            timeline,
            session,
            draw_id=draw_id,
            desc=f"{stem[:40]} rerender",
        )
    finally:
        _release_sapiens_gpu_memory(session)


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
    max_persons: int,
    device: int,
    save_overlay: bool,
    draw_id: bool,
    stabilize_ids: bool,
    reid_max_gap: int,
    reid_max_dist: float,
    reid_min_iou: float,
    reid_direction_weight: float,
    reid_static_speed: float,
    reid_static_radius: float,
    reid_bidirectional: bool,
    appearance_reid: bool,
    appearance_reid_threshold: float,
    flip_test: bool,
    pose_batch_size: int | None,
    quiet: bool,
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
        max_persons=max_persons,
        device=device,
        save_overlay=save_overlay,
        draw_id=draw_id,
        stabilize_ids=stabilize_ids,
        reid_max_gap=reid_max_gap,
        reid_max_dist=reid_max_dist,
        reid_min_iou=reid_min_iou,
        reid_direction_weight=reid_direction_weight,
        reid_static_speed=reid_static_speed,
        reid_static_radius=reid_static_radius,
        reid_bidirectional=reid_bidirectional,
        appearance_reid=appearance_reid,
        appearance_reid_threshold=appearance_reid_threshold,
        flip_test=flip_test,
        pose_batch_size=pose_batch_size,
        quiet=quiet,
        for_subprocess=True,
    )
    progress.schedule_log(f"[GUI] launching subprocess: {shlex.join(cmd)}")

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
            self.result: SapiensGuiSettings | None = None
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
            self.model_combo = ttk.Combobox(
                frm,
                textvariable=self.model_var,
                values=["0.4b", "0.8b", "1b", "5b"],
                width=10,
                state="readonly",
            )
            self.model_combo.grid(row=5, column=0, sticky="w")
            self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)
            ttk.Label(frm, text="Stride (1 = every frame):").grid(
                row=6, column=0, sticky="w", pady=(8, 0)
            )
            self.stride_var = tk.StringVar(value="1")
            ttk.Entry(frm, textvariable=self.stride_var, width=8).grid(row=7, column=0, sticky="w")

            thr_frm = ttk.LabelFrame(frm, text="Detection & keypoint thresholds", padding=8)
            thr_frm.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(10, 0))
            ttk.Label(thr_frm, text="BBox det. (--bbox-thr):").grid(row=0, column=0, sticky="w")
            self.bbox_var = tk.StringVar(value=str(DEFAULT_BBOX_THR))
            ttk.Entry(thr_frm, textvariable=self.bbox_var, width=8).grid(
                row=0, column=1, sticky="w", padx=(4, 16)
            )
            ttk.Label(thr_frm, text="NMS (--nms-thr):").grid(row=0, column=2, sticky="w")
            self.nms_var = tk.StringVar(value=str(DEFAULT_NMS_THR))
            ttk.Entry(thr_frm, textvariable=self.nms_var, width=8).grid(
                row=0, column=3, sticky="w", padx=(4, 0)
            )
            ttk.Label(thr_frm, text="Keypoint (--kpt-thr):").grid(
                row=1, column=0, sticky="w", pady=(8, 0)
            )
            self.kpt_var = tk.StringVar(value=str(DEFAULT_KPT_THR))
            ttk.Entry(thr_frm, textvariable=self.kpt_var, width=8).grid(
                row=1, column=1, sticky="w", padx=(4, 16), pady=(8, 0)
            )
            ttk.Label(thr_frm, text="Max persons / frame:").grid(
                row=1, column=2, sticky="w", pady=(8, 0)
            )
            self.max_persons_var = tk.StringVar(value=str(DEFAULT_MAX_PERSONS))
            ttk.Entry(thr_frm, textvariable=self.max_persons_var, width=8).grid(
                row=1, column=3, sticky="w", padx=(4, 0), pady=(8, 0)
            )
            ttk.Label(
                thr_frm,
                text="bbox/nms/kpt: 0.0–1.0 confidence · max persons caps crowded scenes",
                font=("TkDefaultFont", 8),
            ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))

            adv_frm = ttk.LabelFrame(frm, text="GPU & advanced (CLI flags)", padding=8)
            adv_frm.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(10, 0))
            ttk.Label(adv_frm, text="CUDA device (--device):").grid(row=0, column=0, sticky="w")
            self.device_var = tk.StringVar(value="0")
            ttk.Entry(adv_frm, textvariable=self.device_var, width=8).grid(
                row=0, column=1, sticky="w", padx=(4, 16)
            )
            ttk.Label(adv_frm, text="Pose batch (--pose-batch-size):").grid(
                row=0, column=2, sticky="w"
            )
            self.pose_batch_var = tk.StringVar(
                value=str(_default_pose_batch_size(self.model_var.get()))
            )
            ttk.Entry(adv_frm, textvariable=self.pose_batch_var, width=8).grid(
                row=0, column=3, sticky="w", padx=(4, 0)
            )
            ttk.Label(
                adv_frm,
                text="default per model (1 for 5b, 2 for 1b, 4 for 0.4b/0.8b) · lower if OOM with many people",
                font=("TkDefaultFont", 8),
            ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 0))

            self.flip_test_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                adv_frm, text="Flip-test (--flip-test, 2× VRAM)", variable=self.flip_test_var
            ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
            self.overlay_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(adv_frm, text="Save overlay MP4", variable=self.overlay_var).grid(
                row=2, column=2, columnspan=2, sticky="w", pady=(8, 0)
            )
            self.draw_id_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                adv_frm, text="Draw person IDs on overlay", variable=self.draw_id_var
            ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(4, 0))
            self.stabilize_ids_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                adv_frm,
                text="Temporal Re-ID (online linker + OKS + bidirectional)",
                variable=self.stabilize_ids_var,
            ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))
            self.reid_bidirectional_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                adv_frm,
                text="Bidirectional refine (--reid-bidirectional, default on)",
                variable=self.reid_bidirectional_var,
            ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(2, 0))
            self.appearance_reid_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                adv_frm,
                text="Appearance ReID (--appearance-reid, needs boxmot)",
                variable=self.appearance_reid_var,
            ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(2, 0))
            ttk.Label(adv_frm, text="App. threshold:").grid(
                row=6, column=2, sticky="e", pady=(2, 0)
            )
            self.appearance_reid_thr_var = tk.StringVar(
                value=str(DEFAULT_APPEARANCE_REID_THRESHOLD)
            )
            ttk.Entry(adv_frm, textvariable=self.appearance_reid_thr_var, width=8).grid(
                row=6, column=3, sticky="w", padx=(4, 0), pady=(2, 0)
            )
            ttk.Label(adv_frm, text="Static speed (--reid-static-speed):").grid(
                row=7, column=0, sticky="w", pady=(6, 0)
            )
            self.reid_static_speed_var = tk.StringVar(value=str(DEFAULT_REID_STATIC_SPEED))
            ttk.Entry(adv_frm, textvariable=self.reid_static_speed_var, width=8).grid(
                row=7, column=1, sticky="w", padx=(4, 16), pady=(6, 0)
            )
            ttk.Label(adv_frm, text="Static radius (--reid-static-radius):").grid(
                row=7, column=2, sticky="w", pady=(6, 0)
            )
            self.reid_static_radius_var = tk.StringVar(value=str(DEFAULT_REID_STATIC_RADIUS_PX))
            ttk.Entry(adv_frm, textvariable=self.reid_static_radius_var, width=8).grid(
                row=7, column=3, sticky="w", padx=(4, 0), pady=(6, 0)
            )

            btns = ttk.Frame(frm)
            btns.grid(row=10, column=0, columnspan=3, pady=12)
            ttk.Button(btns, text="Help", command=self._open_help).pack(side="left", padx=4)
            ttk.Button(btns, text="Run", command=self._on_run).pack(side="left", padx=4)
            ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="left", padx=4)
            self.transient(master)  # ty: ignore[no-matching-overload]
            self.grab_set()
            self.lift()
            self.focus_force()

        def _on_model_change(self, _event: tk.Event | None = None) -> None:
            model = self.model_var.get().strip() or "1b"
            self.pose_batch_var.set(str(_default_pose_batch_size(model)))

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
                stride, kpt_thr, bbox_thr, nms_thr, max_persons, device, pose_batch_size = (
                    _parse_gui_inference_fields(
                        stride_s=self.stride_var.get(),
                        kpt_thr_s=self.kpt_var.get(),
                        bbox_thr_s=self.bbox_var.get(),
                        nms_thr_s=self.nms_var.get(),
                        max_persons_s=self.max_persons_var.get(),
                        device_s=self.device_var.get(),
                        pose_batch_s=self.pose_batch_var.get(),
                    )
                )
            except ValueError as exc:
                messagebox.showerror("Error", str(exc), parent=self)
                return
            try:
                appearance_thr = float(self.appearance_reid_thr_var.get().strip())
                reid_static_speed = float(self.reid_static_speed_var.get().strip())
                reid_static_radius = float(self.reid_static_radius_var.get().strip())
            except ValueError:
                messagebox.showerror(
                    "Error",
                    "Appearance threshold and static Re-ID fields must be numbers.",
                    parent=self,
                )
                return
            self.result = SapiensGuiSettings(
                input_path=Path(inp),
                out_parent=Path(out),
                model=self.model_var.get().strip() or "1b",
                stride=stride,
                kpt_thr=kpt_thr,
                bbox_thr=bbox_thr,
                nms_thr=nms_thr,
                max_persons=max_persons,
                device=device,
                flip_test=bool(self.flip_test_var.get()),
                save_overlay=bool(self.overlay_var.get()),
                draw_id=bool(self.draw_id_var.get()),
                stabilize_ids=bool(self.stabilize_ids_var.get()),
                reid_bidirectional=bool(self.reid_bidirectional_var.get()),
                appearance_reid=bool(self.appearance_reid_var.get()),
                appearance_reid_threshold=appearance_thr,
                reid_static_speed=reid_static_speed,
                reid_static_radius=reid_static_radius,
                pose_batch_size=pose_batch_size,
            )
            self.destroy()

    dlg = SapiensVideoDialog(root)
    root.wait_window(dlg)
    if dlg.result is None:
        if owns_root:
            root.destroy()
        return

    settings = dlg.result
    input_path = settings.input_path
    out_parent = settings.out_parent
    model = settings.model
    stride = settings.stride
    kpt_thr = settings.kpt_thr
    bbox_thr = settings.bbox_thr
    nms_thr = settings.nms_thr
    max_persons = settings.max_persons
    device = settings.device
    flip_test = settings.flip_test
    save_overlay = settings.save_overlay
    draw_id = settings.draw_id
    stabilize_ids = settings.stabilize_ids
    reid_bidirectional = settings.reid_bidirectional
    appearance_reid = settings.appearance_reid
    appearance_reid_threshold = settings.appearance_reid_threshold
    reid_static_speed = settings.reid_static_speed
    reid_static_radius = settings.reid_static_radius
    pose_batch_size = settings.pose_batch_size

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

    _print_sapiens_equivalent_cli(
        input_path,
        out_parent,
        model=model,
        stride=stride,
        kpt_thr=kpt_thr,
        bbox_thr=bbox_thr,
        nms_thr=nms_thr,
        max_persons=max_persons,
        device=device,
        flip_test=flip_test,
        save_overlay=save_overlay,
        draw_id=draw_id,
        stabilize_ids=stabilize_ids,
        reid_bidirectional=reid_bidirectional,
        appearance_reid=appearance_reid,
        appearance_reid_threshold=appearance_reid_threshold,
        reid_static_speed=reid_static_speed,
        reid_static_radius=reid_static_radius,
        pose_batch_size=pose_batch_size,
        quiet=True,
    )

    os.environ["SAPIENS_DEVICE"] = str(device)

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
        bbox_thr=bbox_thr,
        nms_thr=nms_thr,
        max_persons=max_persons,
        device=device,
        save_overlay=save_overlay,
        draw_id=draw_id,
        stabilize_ids=stabilize_ids,
        reid_max_gap=DEFAULT_REID_MAX_GAP,
        reid_max_dist=DEFAULT_REID_MAX_DIST_PX,
        reid_min_iou=DEFAULT_REID_MIN_IOU,
        reid_direction_weight=DEFAULT_REID_DIRECTION_WEIGHT,
        reid_static_speed=reid_static_speed,
        reid_static_radius=reid_static_radius,
        reid_bidirectional=reid_bidirectional,
        appearance_reid=appearance_reid,
        appearance_reid_threshold=appearance_reid_threshold,
        flip_test=flip_test,
        pose_batch_size=pose_batch_size,
        quiet=True,
        on_done=_on_done,
    )

    root.wait_window(progress)
    if owns_root:
        root.destroy()


SAPIENS_CLI_EXAMPLES = (
    """\
Sapiens2 — copy/paste CLI recipes
=================================

"""
    + SAPIENS_CLI_FULL_INFERENCE_EXAMPLE
    + """
# Open help / setup page in your default browser
uv run vaila/vaila_sapiens.py --open-help

# Print these examples again (no GPU work)
uv run vaila/vaila_sapiens.py --print-examples

Tips
----
* GUI (no args)              : uv run vaila/vaila_sapiens.py
* GUI Run prints >> Equivalent CLI with all flags you chose (copy/paste to reproduce)
* Requires NVIDIA CUDA and sapiens2 cloned locally via bin/setup_sapiens2.sh
* Model 5b                   : uv run vaila/vaila_sapiens.py --download-weights --model 5b
* OOM on RTX 4090            : use --model 1b (default); avoid --flip-test; pass a single .mp4 not a folder with old overlays
* REC2D/REC3D                : use <stem>_markers.csv (foot anchor, stable pN slots)
* Full reference            : vaila/help/vaila_sapiens.md  (or --open-help).
"""
)


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
    parser.add_argument(
        "--kpt-thr",
        type=float,
        default=DEFAULT_KPT_THR,
        help=("Min per-joint confidence (0.0–1.0) for overlay masking and getpixelvideo wide CSVs"),
    )
    parser.add_argument(
        "--bbox-thr",
        type=float,
        default=DEFAULT_BBOX_THR,
        help="Min DETR person-detection score (0.0–1.0) to keep a bounding box",
    )
    parser.add_argument(
        "--nms-thr",
        type=float,
        default=DEFAULT_NMS_THR,
        help="NMS IoU threshold (0.0–1.0) between overlapping person boxes",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--no-overlay", action="store_true", help="Skip overlay MP4 generation")
    parser.add_argument(
        "--no-draw-id",
        action="store_true",
        help="Do not draw person id NN / pN tags on the overlay video",
    )
    parser.add_argument(
        "--rerender-overlay",
        action="store_true",
        help=(
            "Re-write overlay MP4 from existing *_predictions.json in -o run dir "
            "(requires -i source video; loads model for skeleton viz only)"
        ),
    )
    parser.add_argument(
        "--stabilize-ids",
        action="store_true",
        default=True,
        help="Geometric Re-ID after inference (Hungarian + velocity; default on, like SAM3)",
    )
    parser.add_argument(
        "--no-stabilize-ids",
        action="store_false",
        dest="stabilize_ids",
        help="Disable geometric Re-ID (per-frame raw_id only)",
    )
    parser.add_argument(
        "--reid-max-gap",
        type=int,
        default=DEFAULT_REID_MAX_GAP,
        help="Max frame gap for geometric Re-ID linker",
    )
    parser.add_argument(
        "--reid-max-dist",
        type=float,
        default=DEFAULT_REID_MAX_DIST_PX,
        help="Max centroid distance (px) for geometric Re-ID",
    )
    parser.add_argument(
        "--reid-min-iou",
        type=float,
        default=DEFAULT_REID_MIN_IOU,
        help="Min bbox IoU gate for geometric Re-ID",
    )
    parser.add_argument(
        "--reid-direction-weight",
        type=float,
        default=DEFAULT_REID_DIRECTION_WEIGHT,
        help="Velocity-direction penalty weight for geometric Re-ID",
    )
    parser.add_argument(
        "--reid-static-speed",
        type=float,
        default=DEFAULT_REID_STATIC_SPEED,
        help="Speed threshold (px/frame) for static-track anchoring",
    )
    parser.add_argument(
        "--reid-static-radius",
        type=float,
        default=DEFAULT_REID_STATIC_RADIUS_PX,
        help="Anchor radius (px) for static-track anchoring",
    )
    parser.add_argument(
        "--reid-bidirectional",
        action="store_true",
        default=True,
        help="Forward + backward geometric Re-ID merge (default on)",
    )
    parser.add_argument(
        "--no-reid-bidirectional",
        action="store_false",
        dest="reid_bidirectional",
        help="Single forward geometric Re-ID pass only",
    )
    parser.add_argument(
        "--appearance-reid",
        action="store_true",
        help="Optional OSNet appearance merge after geometric Re-ID (requires boxmot)",
    )
    parser.add_argument(
        "--appearance-reid-threshold",
        type=float,
        default=DEFAULT_APPEARANCE_REID_THRESHOLD,
        help="Cosine similarity threshold for appearance Re-ID (default 0.6)",
    )
    parser.add_argument(
        "--flip-test",
        action="store_true",
        help="Enable left-right flip test (higher quality, ~2× VRAM; off by default on 24 GiB).",
    )
    parser.add_argument(
        "--max-persons",
        type=int,
        default=DEFAULT_MAX_PERSONS,
        help="Max DETR persons per frame (top scores; default 8). Lower if VRAM is tight.",
    )
    parser.add_argument(
        "--pose-batch-size",
        type=int,
        default=None,
        help=(
            "Person crops per GPU pose pass per frame (default: auto — 1 for 5b, 2 for 1b, 4 for 0.4b/0.8b). "
            "Lower to 1 if CUDA OOM on crowded multi-person frames."
        ),
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

    if args.rerender_overlay:
        videos = _find_videos(inp)
        if not videos:
            print(f"No videos under {inp}")
            raise SystemExit(1)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ["SAPIENS_DEVICE"] = str(args.device)
        if args.quiet:
            os.environ["TQDM_DISABLE"] = "1"
        device = f"cuda:{args.device}"
        draw_id = not args.no_draw_id
        for vf in videos:
            try:
                out_path = rerender_sapiens_overlay(
                    vf,
                    out_parent,
                    model=args.model,
                    kpt_thr=args.kpt_thr,
                    device=device,
                    draw_id=draw_id,
                )
                print(f"  Done: {out_path}")
            except Exception as e:
                print(f"  ERROR on {vf.name}: {e}")
                raise SystemExit(1) from e
        return

    if args.video_output_dir is not None:
        out_dir = args.video_output_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        videos = _find_videos(inp)
        if not videos:
            print(f"No videos under {inp}")
            raise SystemExit(1)
        video = videos[0]
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ["SAPIENS_DEVICE"] = str(args.device)
        if args.quiet:
            os.environ["TQDM_DISABLE"] = "1"
        device = f"cuda:{args.device}"
        flip_test = bool(args.flip_test)
        save_overlay = not args.no_overlay
        draw_id = not args.no_draw_id
        stabilize_ids = bool(args.stabilize_ids)
        max_persons = max(1, int(args.max_persons))
        pose_batch_size = (
            max(1, int(args.pose_batch_size)) if args.pose_batch_size is not None else None
        )
        try:
            run_sapiens_on_video(
                video,
                out_dir,
                model=args.model,
                stride=args.stride,
                save_overlay=save_overlay,
                draw_id=draw_id,
                stabilize_ids=stabilize_ids,
                reid_max_gap=args.reid_max_gap,
                reid_max_dist=args.reid_max_dist,
                reid_min_iou=args.reid_min_iou,
                reid_direction_weight=args.reid_direction_weight,
                reid_static_speed=args.reid_static_speed,
                reid_static_radius=args.reid_static_radius,
                reid_bidirectional=bool(args.reid_bidirectional),
                appearance_reid=bool(args.appearance_reid),
                appearance_reid_threshold=float(args.appearance_reid_threshold),
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
    draw_id = not args.no_draw_id
    stabilize_ids = bool(args.stabilize_ids)
    max_persons = max(1, int(args.max_persons))
    pose_batch_size = (
        max(1, int(args.pose_batch_size)) if args.pose_batch_size is not None else None
    )

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
                output_base=output_base,
                out_dir=out_dir,
                model=args.model,
                stride=args.stride,
                kpt_thr=args.kpt_thr,
                bbox_thr=args.bbox_thr,
                nms_thr=args.nms_thr,
                device=args.device,
                save_overlay=save_overlay,
                draw_id=draw_id,
                stabilize_ids=stabilize_ids,
                flip_test=flip_test,
                max_persons=max_persons,
                pose_batch_size=pose_batch_size,
                reid_max_gap=args.reid_max_gap,
                reid_max_dist=args.reid_max_dist,
                reid_min_iou=args.reid_min_iou,
                reid_direction_weight=args.reid_direction_weight,
                reid_static_speed=args.reid_static_speed,
                reid_static_radius=args.reid_static_radius,
                reid_bidirectional=bool(args.reid_bidirectional),
                appearance_reid=bool(args.appearance_reid),
                appearance_reid_threshold=float(args.appearance_reid_threshold),
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
        draw_id=draw_id,
        stabilize_ids=stabilize_ids,
        reid_max_gap=args.reid_max_gap,
        reid_max_dist=args.reid_max_dist,
        reid_min_iou=args.reid_min_iou,
        reid_direction_weight=args.reid_direction_weight,
        reid_static_speed=args.reid_static_speed,
        reid_static_radius=args.reid_static_radius,
        reid_bidirectional=bool(args.reid_bidirectional),
        appearance_reid=bool(args.appearance_reid),
        appearance_reid_threshold=float(args.appearance_reid_threshold),
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
