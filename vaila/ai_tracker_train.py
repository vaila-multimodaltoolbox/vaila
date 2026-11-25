"""
Project: vailá Multimodal Toolbox
Script: ai_tracker_train.py — Offline fine-tune for Track AI backbones / discriminator

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 18 September 2026
Update Date: 24 November 2026
Version: 0.4.19

Description:
    Train (or continue) the AI Tracker assets. Two data sources:

    * ``--data data.yaml`` — YOLO detect/pose datasets exported by getpixelvideo Save ML.
    * ``--from-samples ROOT`` — the human-placed samples that AI Track saved under
      ``ROOT/samples/*.npz`` (ROOT is the AI Track folder, ``<home>/ai_tracker``). This is what
      the "Retrain" button of the AI models window runs.

    Modes:

    * **backbone** — fine-tune ResNet/MobileNet/EfficientNet and save an embedding-compatible
      ``.pth`` (head stripped for Track AI).
    * **discriminator** — ridge discriminator (``discriminator_<profile>__<key>.npz``) built
      with the SAME features the live tracker uses (classical 808 + deep embedding of the
      backbone, via ``AITracker._discriminator_feature``), so ``AITracker.load_checkpoint``
      accepts it. A classic (808) one is written too.
    * **both** — run backbone then discriminator (the discriminator uses the chosen backbone).

    ``--from-samples`` details:

    * The backbone is a binary marker/background classifier trained on the context crops
      (2x block window). ``--unfreeze last`` (default) trains only the last stage plus a new
      2-class head (``layer4`` for ResNets, the last 3 ``features`` blocks of MobileNetV3, the
      last 2 of EfficientNet); ``--unfreeze all`` trains the whole network. Frozen BatchNorm
      layers stay in eval mode.
    * Hold-out without leakage: with samples from >= 2 videos whole videos are held out
      (about ``--val-fraction`` of the samples); with one video the last contiguous block of
      frames is. The metric is the held-out ROC AUC of a ridge discriminator trained on the
      train split with the live-tracker features, computed with the previous backbone
      ("before") and with the fine-tuned one ("after"). The new backbone is saved to
      ``backbones/<variant>_finetuned_<ts>.pth`` and becomes the active one in
      ``manifest.json`` ONLY if after >= before; otherwise the file stays for inspection and
      the previous backbone remains active (status ``kept_previous``).
    * The discriminators are then rebuilt on ALL samples for the backbone that stays active.
    * ``--device auto`` picks CUDA only if >= 3 GiB are free, else CPU.
    * Result: ``manifest["last_retrain"]`` (started, finished, status
      activated|kept_previous|error, mode, variant, device, n_samples, n_videos, backbone,
      metric, before, after, reason, log) and a log at ``ROOT/logs/retrain_<ts>.log``.
      Exit code 0 = finished (activated or not), non-zero = error.

Usage:
    uv run --no-sync python -m vaila.ai_tracker_train
    uv run --no-sync python -m vaila.ai_tracker_train \\
        --data /path/to/data.yaml --mode both --weights resnet50_imagenet.pth \\
        --epochs 10 --profile athletics
    uv run --no-sync python -m vaila.ai_tracker_train \\
        --from-samples ~/vaila_ai_models/ai_tracker --mode both --device auto
    uv run --no-sync python -m vaila.ai_tracker_train --from-samples ROOT --dry-run

License: AGPLv3
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import shlex
import sys
import tempfile
import threading
import tkinter as tk
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import IO, Any, cast

import cv2
import numpy as np

try:
    from . import ai_tracker_store as store
    from .tracking.ai_tracker import (
        _BACKBONE_FEATURE_DIM,
        _BACKBONE_HEAD_ATTR,
        AITracker,
        AITrackerParameters,
        DeepFeatureExtractor,
        _default_checkpoint_dir,
        _normalize_backbone_variant,
        ensure_backbone_weights,
    )
except ImportError:
    import ai_tracker_store as store  # ty: ignore[unresolved-import]
    from tracking.ai_tracker import (  # ty: ignore[unresolved-import]
        _BACKBONE_FEATURE_DIM,
        _BACKBONE_HEAD_ATTR,
        AITracker,
        AITrackerParameters,
        DeepFeatureExtractor,
        _default_checkpoint_dir,
        _normalize_backbone_variant,
        ensure_backbone_weights,
    )

try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    import torchvision.transforms as tv_transforms
    from torch.utils.data import DataLoader, Dataset

    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    # ``cast("Any", None)`` keeps the names usable for the type checker; every torch code path
    # is guarded by TORCH_OK.
    torch = cast("Any", None)
    nn = cast("Any", None)
    tv_models = cast("Any", None)
    tv_transforms = cast("Any", None)
    DataLoader = cast("Any", None)
    Dataset = object  # type: ignore[misc, assignment]

HELP_MD = Path(__file__).resolve().parent / "help" / "ai_tracker_train.md"
HELP_HTML = Path(__file__).resolve().parent / "help" / "ai_tracker_train.html"

# COCO-80 first names used only as GUI/doc hint (avoid importing getpixelvideo).
COCO80_HINT = ("person", "sports ball", "bicycle", "car", "dog", "cat", "bottle", "chair")

PREFIX = ">> vaila/ai_tracker_train:"
MIN_FREE_GPU_GIB = 3.0  # --device auto picks CUDA only with at least this much free memory
RIDGE_LAMBDA = 0.05  # same value as AITracker.retrain_online_model
_FLIP_SHAPES = ("point", "circle")  # a mirrored marker is still a marker; boxes/ellipses may not be
_log_fh: IO[str] | None = None


def _say(message: str) -> None:
    """Print a ``>> vaila/ai_tracker_train:`` progress line (and tee it into the run log)."""
    line = f"{PREFIX} {message}"
    print(line, flush=True)
    if _log_fh is not None:
        with contextlib.suppress(OSError, ValueError):
            _log_fh.write(line + "\n")
            _log_fh.flush()


def retrain_argv(root: str | Path, mode: str = "both", device: str = "auto") -> list[str]:
    """Command line that retrains from the samples of an AI Track folder (frozen contract with
    the models window: exit 0 = finished, non-zero = error, ``>>`` progress lines)."""
    return [
        sys.executable,
        "-m",
        "vaila.ai_tracker_train",
        "--from-samples",
        str(root),
        "--mode",
        mode,
        "--device",
        device,
    ]


@dataclass
class TrainConfig:
    data_yaml: Path
    weights: Path
    mode: str  # backbone | discriminator | both
    epochs: int = 10
    batch: int = 16
    lr: float = 1e-4
    device: str = "auto"
    profile: str = "default"
    max_samples: int = 4000
    dry_run: bool = False


def _ai_tracker_dir() -> Path:
    return _default_checkpoint_dir()


def _list_pth_files(directory: Path | None = None) -> list[Path]:
    d = directory or _ai_tracker_dir()
    if not d.is_dir():
        return []
    found: list[Path] = []
    for pat in ("*.pth", "*.pt"):
        found.extend(sorted(d.glob(pat)))
    return [p for p in found if p.is_file() and p.stat().st_size > 1_000_000]


def _infer_variant_from_path(path: Path) -> str:
    name = path.stem.lower()
    for v in ("resnet152", "mobilenet_v3_small", "efficientnet_b0", "resnet50"):
        if v in name:
            return v
    return "resnet50"


def _load_yaml_simple(path: Path) -> dict[str, Any]:
    """Minimal YAML reader for Ultralytics data.yaml (no PyYAML required)."""
    data: dict[str, Any] = {}
    text = path.read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip().strip("'\"")
        if key == "names":
            # names: ['a', 'b'] or names: {0: a}
            if val.startswith("["):
                inner = val.strip("[]")
                names = [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]
                data["names"] = names
            elif val.startswith("{"):
                data["names"] = val  # parsed lazily
            else:
                data["names"] = [val] if val else []
        elif key == "nc":
            with contextlib.suppress(ValueError):
                data["nc"] = int(val)
        elif key in {"train", "val", "test", "path"}:
            data[key] = val
        elif key == "kpt_shape":
            data["kpt_shape"] = val
    return data


def _resolve_split_images(yaml_data: dict[str, Any], dataset_dir: Path, key: str) -> Path | None:
    raw = yaml_data.get(key)
    if not raw:
        # vaila layout fallback
        cand = dataset_dir / key / "images"
        return cand if cand.is_dir() else None
    p = Path(str(raw))
    if not p.is_absolute():
        p = (dataset_dir / p).resolve()
    if p.is_dir():
        return p
    # Sometimes train: images/train
    alt = dataset_dir / str(raw)
    return alt if alt.is_dir() else None


def _label_path_for_image(image_path: Path) -> Path:
    # .../train/images/foo.jpg -> .../train/labels/foo.txt
    parts = list(image_path.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
    except ValueError:
        return image_path.with_suffix(".txt")
    return Path(*parts).with_suffix(".txt")


def _parse_yolo_boxes(
    label_path: Path,
    img_w: int,
    img_h: int,
) -> list[tuple[int, int, int, int, int]]:
    """Return list of (cls, x, y, w, h) in pixel top-left + size."""
    if not label_path.is_file():
        return []
    boxes: list[tuple[int, int, int, int, int]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        # YOLO normalized center → pixel xywh
        pw = bw * img_w
        ph = bh * img_h
        x = int(round((cx * img_w) - pw / 2.0))
        y = int(round((cy * img_h) - ph / 2.0))
        w = max(2, int(round(pw)))
        h = max(2, int(round(ph)))
        x = max(0, min(img_w - 2, x))
        y = max(0, min(img_h - 2, y))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        boxes.append((cls_id, x, y, w, h))
    return boxes


def _class_names_from_yaml(yaml_data: dict[str, Any], dataset_dir: Path) -> list[str]:
    names = yaml_data.get("names")
    if isinstance(names, list) and names:
        return [str(n) for n in names]
    classes_txt = dataset_dir / "classes.txt"
    if classes_txt.is_file():
        return [
            ln.strip() for ln in classes_txt.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
    manifest = dataset_dir / "ai_tracker_train.json"
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            cls = payload.get("classes") or []
            if cls:
                return [str(c) for c in cls]
        except Exception:
            pass
    return ["object"]


def collect_crop_samples(
    data_yaml: Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
    max_samples: int = 4000,
    seed: int = 42,
) -> tuple[list[tuple[Path, int, int, int, int, int]], list[str]]:
    """Collect (image_path, cls, x, y, w, h) crops from a YOLO dataset."""
    dataset_dir = data_yaml.parent.resolve()
    yaml_data = _load_yaml_simple(data_yaml)
    class_names = _class_names_from_yaml(yaml_data, dataset_dir)
    samples: list[tuple[Path, int, int, int, int, int]] = []
    for split in splits:
        img_dir = _resolve_split_images(yaml_data, dataset_dir, split)
        if img_dir is None:
            continue
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            h, w = im.shape[:2]
            for cls_id, x, y, bw, bh in _parse_yolo_boxes(_label_path_for_image(img_path), w, h):
                samples.append((img_path, cls_id, x, y, bw, bh))
    rng = random.Random(seed)
    if len(samples) > max_samples:
        samples = rng.sample(samples, max_samples)
    return samples, class_names


def _cuda_free_gib() -> float | None:
    """Free GPU memory in GiB, or None when it cannot be queried."""
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:
        return None
    return float(free) / 1024**3


def _resolve_device(requested: str) -> str:
    """``cpu``, ``cuda`` or ``auto``. ``auto`` picks CUDA only when ``torch.cuda`` is available
    and at least ``MIN_FREE_GPU_GIB`` GiB are free (SAM, YOLO or tests may be using the card);
    the free memory and the choice are printed."""
    if not TORCH_OK:
        return "cpu"
    req = (requested or "auto").strip().lower()
    if req == "cpu":
        return "cpu"
    if req in {"auto", "cuda", "0"}:
        if not torch.cuda.is_available():
            if req != "auto":
                _say("device: cuda requested but not available -> cpu")
            return "cpu"
        free = _cuda_free_gib()
        shown = "unknown" if free is None else f"{free:.1f} GiB"
        if req == "auto" and (free is None or free < MIN_FREE_GPU_GIB):
            _say(f"device: GPU free memory {shown} < {MIN_FREE_GPU_GIB:.0f} GiB -> cpu")
            return "cpu"
        _say(f"device: GPU free memory {shown} -> cuda")
        return "cuda"
    return req


class _CropDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int, int, int, int, int]], transform: Any) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        path, cls_id, x, y, w, h = self.samples[idx]
        im = cv2.imread(str(path))
        if im is None:
            im = np.zeros((224, 224, 3), dtype=np.uint8)
        crop = im[y : y + h, x : x + w]
        if crop.size == 0:
            crop = im
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb)
        return tensor, int(cls_id)


def _load_backbone_with_head(
    variant: str,
    weights_path: Path,
    num_classes: int,
    device: str,
) -> Any:
    variant = _normalize_backbone_variant(variant)
    model = tv_models.get_model(variant, weights=None)
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    head_attr = _BACKBONE_HEAD_ATTR[variant]
    feat_dim = _BACKBONE_FEATURE_DIM[variant]
    if head_attr == "fc":
        model.fc = nn.Linear(feat_dim, num_classes)
    else:
        # MobileNetV3 / EfficientNet: classifier is Sequential ending in Linear
        classifier = getattr(model, head_attr)
        if isinstance(classifier, nn.Sequential) and len(classifier) > 0:
            in_f = None
            for layer in reversed(list(classifier)):
                if isinstance(layer, nn.Linear):
                    in_f = layer.in_features
                    break
            if in_f is None:
                in_f = feat_dim
            new_layers = list(classifier.children())[:-1] + [nn.Linear(in_f, num_classes)]
            setattr(model, head_attr, nn.Sequential(*new_layers))
        else:
            setattr(model, head_attr, nn.Linear(feat_dim, num_classes))
    model.to(device)
    return model


def _strip_head_and_save(
    model: Any, variant: str, out_path: Path, *, announce: bool = True
) -> None:
    """Save full state_dict (including Identity-ready head) for DeepFeatureExtractor."""
    variant = _normalize_backbone_variant(variant)
    head_attr = _BACKBONE_HEAD_ATTR[variant]
    # Replace head with Identity so loaders that expect embedding weights still work;
    # DeepFeatureExtractor also forces Identity after load.
    setattr(model, head_attr, nn.Identity())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path))
    if announce:
        _say(f"saved backbone -> {out_path}")


def _train_loop(
    model: Any,
    loader: Any,
    epochs: int,
    lr: float,
    device: str,
    params: Any,
    *,
    class_weights: Any = None,
) -> list[dict[str, float]]:
    """Cross-entropy epoch loop shared by the YOLO crops and the samples paths.

    ``params`` are the parameters AdamW optimises (all of the model, or only the unfrozen
    stage). BatchNorm layers whose parameters are all frozen are kept in eval mode so the
    frozen part of the network does not drift through its running statistics. Returns one
    ``{"epoch", "loss", "acc"}`` dict per epoch and prints the progress lines.
    """
    opt = torch.optim.AdamW(params, lr=lr)
    weight = None if class_weights is None else class_weights.to(device)
    crit = nn.CrossEntropyLoss(weight=weight)
    epochs = max(1, int(epochs))
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
                p.requires_grad for p in module.parameters(recurse=False)
            ):
                module.eval()
        total_loss = 0.0
        n = 0
        correct = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
            correct += int((logits.argmax(dim=1) == yb).sum().item())
        avg = total_loss / max(1, n)
        acc = correct / max(1, n)
        history.append({"epoch": float(epoch + 1), "loss": avg, "acc": acc})
        _say(f"epoch {epoch + 1}/{epochs} loss={avg:.4f} acc={acc:.3f}")
    return history


def train_backbone(config: TrainConfig) -> Path | None:
    if not TORCH_OK:
        raise RuntimeError("PyTorch/torchvision required for backbone training.")
    samples, class_names = collect_crop_samples(config.data_yaml, max_samples=config.max_samples)
    if len(samples) < 2:
        raise RuntimeError(f"Need at least 2 labeled crops; got {len(samples)}.")
    num_classes = max(len(class_names), max(s[1] for s in samples) + 1)
    device = _resolve_device(config.device)
    variant = _infer_variant_from_path(config.weights)
    _say(
        f"backbone train variant={variant} classes={num_classes} samples={len(samples)} "
        f"device={device}"
    )
    backbones = store.layout(_ai_tracker_dir())["backbones"]
    if config.dry_run:
        out = backbones / f"{variant}_finetuned_DRYRUN.pth"
        _say(f"dry-run would write {out}")
        return out

    transform = tv_transforms.Compose(
        [
            tv_transforms.ToPILImage(),
            tv_transforms.Resize((224, 224)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = _CropDataset(samples, transform)
    loader = DataLoader(
        dataset,
        batch_size=max(1, config.batch),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    model = _load_backbone_with_head(variant, config.weights, num_classes, device)
    _train_loop(model, loader, config.epochs, config.lr, device, model.parameters())

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = backbones / f"{variant}_finetuned_{stamp}.pth"
    _strip_head_and_save(model.cpu(), variant, out_path)
    with contextlib.suppress(OSError):
        store.update_manifest_file(_ai_tracker_dir(), out_path, kind="backbone", variant=variant)
    return out_path


# ---------------------------------------------------------------- features, ridge, AUC


def make_feature_tracker(variant: str, weights: Path, device: str) -> AITracker:
    """A tracker whose ``_discriminator_feature`` is the live one (classical 808 + deep
    embedding of ``weights``). The extractor is built here, with the requested device, instead
    of ``DeepFeatureExtractor.get_shared`` (which would always use CUDA when it is there)."""
    tracker = AITracker(AITrackerParameters(use_deep_features=False, resnet_variant=variant))
    extractor = DeepFeatureExtractor(
        use_cuda=(device == "cuda"), weights_path=weights, variant=variant
    )
    if not extractor.enabled:
        raise RuntimeError(f"Could not load backbone features from {weights}")
    tracker.params.use_deep_features = True
    tracker.extractor = extractor
    return tracker


def _embed_patches(tracker: AITracker, patches: list[np.ndarray], shapes: list[str]) -> np.ndarray:
    """(N, d) float32: the live discriminator feature of every patch."""
    rows = [
        tracker._discriminator_feature(p, str(s or "point"))
        for p, s in zip(patches, shapes, strict=True)
    ]
    return np.stack(rows, axis=0).astype(np.float32)


def ridge_fit(
    X: np.ndarray, y: np.ndarray, reg_lambda: float = RIDGE_LAMBDA
) -> tuple[np.ndarray, float]:
    """Ridge exactly as ``AITracker.retrain_online_model`` (no intercept in the solve, bias =
    mean residual). Dual form (N x N) when N <= d, primal ``(X^T X + lambda I) w = X^T y``
    (d x d, bounded memory) when N is larger. Both give the same ``w``."""
    X64 = np.asarray(X, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    n, d = X64.shape
    if n <= d:
        alpha = np.linalg.solve(X64 @ X64.T + reg_lambda * np.eye(n), y64)
        w = X64.T @ alpha
    else:
        w = np.linalg.solve(X64.T @ X64 + reg_lambda * np.eye(d), X64.T @ y64)
    b = float(np.mean(y64 - X64 @ w))
    return w.astype(np.float32), b


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """ROC AUC (probability a positive outranks a negative, ties count half) with numpy only.
    None when only one class is present."""
    s = np.asarray(scores, dtype=np.float64)
    pos = np.asarray(labels) > 0
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0  # average rank of a tie group
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _heldout_auc(
    X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray
) -> float | None:
    """Held-out ROC AUC of a ridge trained on the train split of ``X`` (live-tracker features)."""
    if len(train_idx) < 2 or len(val_idx) < 2:
        return None
    w, b = ridge_fit(X[train_idx], y[train_idx])
    return roc_auc(X[val_idx] @ w + b, y[val_idx])


def _write_discriminators(
    root: Path,
    profile: str,
    tracker: AITracker,
    X: np.ndarray,
    y: np.ndarray,
    *,
    dry_run: bool = False,
) -> dict[str, Path]:
    """Ridge on ``X`` (the live features of ``tracker``'s backbone) saved with the tracker's own
    writer to ``discriminator_<profile>__<checkpoint_key>.npz``, plus the classic (808) one
    from the first 808 columns of the same features. Returns ``{key: path}``."""
    written: dict[str, Path] = {}
    classic = AITracker(AITrackerParameters(use_deep_features=False))
    for trk, feats in ((tracker, X), (classic, X[:, : classic.effective_feature_dim()])):
        key = trk.checkpoint_key()
        path = store.discriminator_path(root, profile, key)
        written[key] = path
        if dry_run:
            continue
        w, b = ridge_fit(feats, y)
        trk.discriminator_w, trk.discriminator_b = w, b
        trk._live_n_samples = float(len(y))
        if not trk.save_checkpoint(path):
            raise RuntimeError(f"could not write the discriminator {path}")
        store.update_manifest_file(
            root,
            path,
            kind="discriminator",
            n_samples=float(len(y)),
            feat_dim=int(feats.shape[1]),
            key=key,
        )
        _say(f"saved discriminator ({key}, N={len(y)}, dim={feats.shape[1]}) -> {path}")
    return written


def _random_neg_box(
    img_w: int, img_h: int, boxes: list[tuple[int, int, int, int, int]], rng: random.Random
) -> tuple[int, int, int, int] | None:
    bw = max(16, int(img_w * 0.08))
    bh = max(16, int(img_h * 0.14))
    for _ in range(40):
        x = rng.randint(0, max(0, img_w - bw))
        y = rng.randint(0, max(0, img_h - bh))
        cx, cy = x + bw / 2, y + bh / 2
        overlaps = False
        for _, bx, by, bw2, bh2 in boxes:
            if bx <= cx <= bx + bw2 and by <= cy <= by + bh2:
                overlaps = True
                break
        if not overlaps:
            return x, y, bw, bh
    return None


def train_discriminator(config: TrainConfig, weights_override: Path | None = None) -> Path | None:
    """Offline ridge discriminator from YOLO crops + random negatives.

    Features are the live tracker's own (``AITracker._discriminator_feature``: classical 808 +
    backbone embedding, crops resized to the tracker block window), and the file is saved with
    the tracker's writer under the key of the backbone, so ``AITracker.load_checkpoint`` accepts
    it. A classic (808) discriminator is written as well. Returns the deep file.
    """
    weights = weights_override or config.weights
    samples, _class_names = collect_crop_samples(config.data_yaml, max_samples=config.max_samples)
    if len(samples) < 2:
        raise RuntimeError(f"Need at least 2 labeled crops; got {len(samples)}.")
    variant = _infer_variant_from_path(weights)
    device = _resolve_device(config.device)
    tracker = make_feature_tracker(variant, weights, device)
    block_w, block_h = tracker.params.block_window

    def feature(crop: np.ndarray) -> np.ndarray:
        patch = cv2.resize(crop, (int(block_w), int(block_h)), interpolation=cv2.INTER_AREA)
        return tracker._discriminator_feature(patch, "point")

    rng = random.Random(42)
    feats: list[np.ndarray] = []
    labels: list[float] = []
    # Group samples by image for negatives
    by_image: dict[Path, list[tuple[int, int, int, int, int]]] = {}
    for path, cls_id, x, y, w, h in samples:
        by_image.setdefault(path, []).append((cls_id, x, y, w, h))

    for path, boxes in by_image.items():
        im = cv2.imread(str(path))
        if im is None:
            continue
        ih, iw = im.shape[:2]
        for _cls, x, y, w, h in boxes:
            crop = im[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            feats.append(feature(crop))
            labels.append(1.0)
            neg = _random_neg_box(iw, ih, boxes, rng)
            if neg is not None:
                nx, ny, nw, nh = neg
                ncrop = im[ny : ny + nh, nx : nx + nw]
                if ncrop.size > 0:
                    feats.append(feature(ncrop))
                    labels.append(-1.0)

    if len(feats) < 2:
        raise RuntimeError("Not enough feature vectors for discriminator training.")

    X = np.stack(feats, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.float32)
    _say(
        f"discriminator ridge N={X.shape[0]} dim={X.shape[1]} "
        f"pos={int((y > 0).sum())} neg={int((y < 0).sum())} key={tracker.checkpoint_key()}"
    )
    written = _write_discriminators(
        _ai_tracker_dir(), config.profile, tracker, X, y, dry_run=config.dry_run
    )
    out_path = written[tracker.checkpoint_key()]
    if config.dry_run:
        _say(f"dry-run would write {out_path}")
    return out_path


def run_training(config: TrainConfig) -> dict[str, Path | None]:
    results: dict[str, Path | None] = {"backbone": None, "discriminator": None}
    mode = config.mode.lower().strip()
    if mode not in {"backbone", "discriminator", "both"}:
        raise ValueError(f"Unknown mode: {config.mode}")
    backbone_path: Path | None = None
    if mode in {"backbone", "both"}:
        backbone_path = train_backbone(config)
        results["backbone"] = backbone_path
    if mode in {"discriminator", "both"}:
        results["discriminator"] = train_discriminator(
            config, weights_override=backbone_path if mode == "both" else None
        )
    return results


# ---------------------------------------------------------------- --from-samples


@dataclass
class SamplesConfig:
    root: Path
    mode: str = "both"  # backbone | discriminator | both
    variant: str | None = None  # None = variant of the active backbone, else resnet50
    weights: Path | None = None  # None = active backbone, else the official <variant>_imagenet.pth
    epochs: int = 5
    batch: int = 32
    lr: float = 1e-4
    unfreeze: str = "last"  # last | all
    device: str = "auto"
    profile: str = "default"
    min_samples: int = 40
    val_fraction: float = 0.2
    seed: int = 42
    dry_run: bool = False


@dataclass
class SampleBank:
    """All human-placed samples of ``root/samples/*.npz``, one entry per crop."""

    patches: list[np.ndarray]  # BGR uint8, tracker block size
    contexts: list[np.ndarray]  # BGR uint8, twice the block window
    labels: np.ndarray  # +1 marker / -1 background (int8)
    frames: np.ndarray
    markers: np.ndarray
    videos: list[str]
    shapes: list[str]
    files: list[Path]

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def n_videos(self) -> int:
        return len(set(self.videos))


def load_sample_bank(root: Path) -> SampleBank:
    """Reads every ``root/samples/*.npz`` (unreadable files are skipped with a message)."""
    patches: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    markers: list[np.ndarray] = []
    videos: list[str] = []
    shapes: list[str] = []
    files: list[Path] = []
    folder = Path(root) / "samples"
    for path in sorted(folder.glob("*.npz")) if folder.is_dir() else []:
        try:
            data = store.load_samples(path)
            n = len(data["labels"])
            names = [str(v) for v in data["videos"]]
            shape_names = [str(v) for v in data["shapes"]]
            if not (len(data["patches"]) == len(data["contexts"]) == n == len(names)):
                raise ValueError("inconsistent array lengths")
            patches.extend(list(data["patches"]))
            contexts.extend(list(data["contexts"]))
            labels.append(np.asarray(data["labels"], dtype=np.int8))
            frames.append(np.asarray(data["frames"], dtype=np.int64))
            markers.append(np.asarray(data["markers"], dtype=np.int64))
            videos.extend(name or f"<{path.stem}>" for name in names)
            shapes.extend(shape_names)
            files.append(path)
        except Exception as err:
            _say(f"skipping unreadable sample file {path.name}: {err}")
    cat = np.concatenate
    return SampleBank(
        patches=patches,
        contexts=contexts,
        labels=cat(labels) if labels else np.zeros(0, dtype=np.int8),
        frames=cat(frames) if frames else np.zeros(0, dtype=np.int64),
        markers=cat(markers) if markers else np.zeros(0, dtype=np.int64),
        videos=videos,
        shapes=shapes,
        files=files,
    )


def split_holdout(
    videos: list[str],
    frames: np.ndarray,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Train/validation split with no leakage. Returns ``(train_idx, val_idx, kind)``.

    * >= 2 videos (``kind="videos"``): whole videos are held out, picked (seeded shuffle) until
      the held-out share of the samples is as close to ``val_fraction`` as it gets; at least
      one video on each side.
    * one video (``kind="frames"``): the last ``val_fraction`` of its unique frames, as ONE
      contiguous block, so a click and its negatives (same frame) never sit on both sides.

    Raises ``ValueError`` when no split is possible (one video with a single frame).
    """
    n = len(videos)
    idx = np.arange(n)
    names = sorted(set(videos))
    if len(names) >= 2:
        rng = random.Random(seed)
        order = names[:]
        rng.shuffle(order)
        counts = {v: videos.count(v) for v in names}
        target = val_fraction * n
        chosen: list[str] = []
        held = 0
        for v in order:
            if len(chosen) == len(names) - 1:
                break
            if chosen and abs(held + counts[v] - target) >= abs(held - target):
                break
            chosen.append(v)
            held += counts[v]
        val_mask = np.array([v in set(chosen) for v in videos], dtype=bool)
        return idx[~val_mask], idx[val_mask], "videos"
    unique = np.unique(np.asarray(frames))
    if len(unique) < 2:
        raise ValueError("cannot hold out data: the samples come from one video and a single frame")
    k = min(len(unique) - 1, max(1, round(val_fraction * len(unique))))
    cut = unique[len(unique) - k]
    val_mask = np.asarray(frames) >= cut
    return idx[~val_mask], idx[val_mask], "frames"


class _SampleDataset(Dataset):
    """Context crops of the samples as (tensor, class 0 background / 1 marker). With
    ``augment``: small colour jitter and, for point/circle samples only, a horizontal flip."""

    def __init__(
        self,
        contexts: list[np.ndarray],
        labels: np.ndarray,
        shapes: list[str],
        *,
        augment: bool,
        seed: int,
    ) -> None:
        self.contexts = contexts
        self.classes = (np.asarray(labels) > 0).astype(np.int64)
        self.shapes = shapes
        self.augment = augment
        self.rng = random.Random(seed)
        steps: list[Any] = [tv_transforms.ToPILImage()]
        if augment:
            steps.append(
                tv_transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)
            )
        steps += [
            tv_transforms.Resize((224, 224)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform = tv_transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.classes)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        rgb = cv2.cvtColor(np.ascontiguousarray(self.contexts[idx]), cv2.COLOR_BGR2RGB)
        if self.augment and self.shapes[idx] in _FLIP_SHAPES and self.rng.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
        return self.transform(rgb), int(self.classes[idx])


def _set_trainable(model: Any, variant: str, unfreeze: str) -> list[Any]:
    """Freeze the network except the last stage and the (new) head, or nothing with ``all``.
    ResNets: ``layer4`` + ``fc``; MobileNetV3-small: last 3 ``features`` blocks + ``classifier``;
    EfficientNet-B0: last 2 ``features`` blocks + ``classifier``. Returns the trainable params."""
    if unfreeze == "all":
        return list(model.parameters())
    for prm in model.parameters():
        prm.requires_grad = False
    head_attr = _BACKBONE_HEAD_ATTR[variant]
    if head_attr == "fc":
        stages = [model.layer4, model.fc]
    else:
        n_blocks = 3 if variant == "mobilenet_v3_small" else 2
        stages = [*list(model.features)[-n_blocks:], getattr(model, head_attr)]
    for stage in stages:
        for prm in stage.parameters():
            prm.requires_grad = True
    return [prm for prm in model.parameters() if prm.requires_grad]


def _save_backbone(model: Any, variant: str, root: Path) -> Path:
    """Strip the head and save ``root/backbones/<variant>_finetuned_<ts>.pth`` atomically."""
    backbones = store.layout(root)["backbones"]
    backbones.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = backbones / f"{variant}_finetuned_{stamp}.pth"
    with tempfile.TemporaryDirectory(prefix=".tmp_", dir=root) as tmp:
        staged = Path(tmp) / target.name
        _strip_head_and_save(model.cpu(), variant, staged, announce=False)
        store.atomic_copy(staged, target)
    _say(f"saved backbone -> {target}")
    return target


def _finetune_backbone(
    cfg: SamplesConfig,
    bank: SampleBank,
    train_idx: np.ndarray,
    variant: str,
    weights: Path,
    device: str,
) -> Any:
    """Binary marker/background fine-tune on the context crops of the train split."""
    torch.manual_seed(cfg.seed)
    train_idx = np.asarray(train_idx)
    labels = bank.labels[train_idx]
    dataset = _SampleDataset(
        [bank.contexts[i] for i in train_idx],
        labels,
        [bank.shapes[i] for i in train_idx],
        augment=True,
        seed=cfg.seed,
    )
    loader = DataLoader(
        dataset, batch_size=max(1, cfg.batch), shuffle=True, num_workers=0, drop_last=False
    )
    counts = np.bincount(dataset.classes, minlength=2).astype(np.float64)
    class_weights = torch.tensor(
        counts.sum() / (2.0 * np.maximum(counts, 1.0)), dtype=torch.float32
    )
    model = _load_backbone_with_head(variant, weights, 2, device)
    params = _set_trainable(model, variant, cfg.unfreeze)
    n_train = sum(int(prm.numel()) for prm in params)
    n_all = sum(int(prm.numel()) for prm in model.parameters())
    _say(
        f"fine-tune {variant} unfreeze={cfg.unfreeze} trainable={n_train}/{n_all} params, "
        f"{len(dataset)} crops (pos={int(counts[1])} neg={int(counts[0])}), "
        f"epochs={cfg.epochs} batch={cfg.batch} lr={cfg.lr} device={device}"
    )
    _train_loop(model, loader, cfg.epochs, cfg.lr, device, params, class_weights=class_weights)
    return model


def _pick_variant(cfg: SamplesConfig) -> str:
    if cfg.variant:
        return _normalize_backbone_variant(cfg.variant)
    if cfg.weights is not None:
        return _infer_variant_from_path(cfg.weights)
    active = store.read_manifest(cfg.root).get("active_backbones") or {}
    if active:
        names = sorted(k for k in active if k in _BACKBONE_FEATURE_DIM)
        if names:
            return "resnet50" if "resnet50" in names else names[0]
    return "resnet50"


def _pick_weights(cfg: SamplesConfig, variant: str) -> Path | None:
    if cfg.weights is not None:
        return cfg.weights if cfg.weights.is_file() else None
    active = store.active_backbone(cfg.root, variant)
    if active is not None:
        return active
    return ensure_backbone_weights(variant)  # official weights; never prompts


def _relpath(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _format_samples_command(cfg: SamplesConfig) -> str:
    parts = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "vaila.ai_tracker_train",
        "--from-samples",
        str(cfg.root),
        "--mode",
        cfg.mode,
        "--epochs",
        str(cfg.epochs),
        "--batch",
        str(cfg.batch),
        "--lr",
        str(cfg.lr),
        "--unfreeze",
        cfg.unfreeze,
        "--device",
        cfg.device,
        "--profile",
        cfg.profile,
        "--min-samples",
        str(cfg.min_samples),
        "--val-fraction",
        str(cfg.val_fraction),
        "--seed",
        str(cfg.seed),
    ]
    if cfg.variant:
        parts += ["--variant", cfg.variant]
    if cfg.weights is not None:
        parts += ["--weights", str(cfg.weights)]
    if cfg.dry_run:
        parts.append("--dry-run")
    return " ".join(shlex.quote(p) for p in parts)


def _record_last_retrain(root: Path, record: dict[str, Any]) -> None:
    manifest = store.read_manifest(root)
    manifest["last_retrain"] = record
    store.write_manifest(root, manifest)


def _retrain(cfg: SamplesConfig, record: dict[str, Any]) -> tuple[str, str]:
    """The retrain proper. Fills ``record``; returns ``(status, reason)``."""
    root = cfg.root
    if not TORCH_OK:
        return "error", "PyTorch/torchvision required"
    if cfg.mode not in {"backbone", "discriminator", "both"}:
        return "error", f"unknown mode {cfg.mode!r}"
    bank = load_sample_bank(root)
    record["n_samples"] = len(bank)
    record["n_videos"] = bank.n_videos
    if len(bank) == 0:
        return "error", f"no samples under {root / 'samples'}"
    n_pos = int((bank.labels > 0).sum())
    _say(
        f"samples: {len(bank)} crops (pos={n_pos} neg={len(bank) - n_pos}) from "
        f"{bank.n_videos} video(s) in {len(bank.files)} file(s)"
    )
    try:
        train_idx, val_idx, kind = split_holdout(
            bank.videos, bank.frames, cfg.val_fraction, cfg.seed
        )
    except ValueError as err:
        return "error", str(err)
    y = np.where(bank.labels > 0, 1.0, -1.0).astype(np.float32)
    _say(f"hold-out by {kind}: train={len(train_idx)} validation={len(val_idx)}")
    if len(train_idx) < cfg.min_samples:
        return "error", (
            f"only {len(train_idx)} training samples after the hold-out "
            f"(minimum {cfg.min_samples}); mark more points"
        )
    if len(np.unique(y[train_idx])) < 2:
        return "error", "the training split has a single class (needs markers and background)"

    variant = _pick_variant(cfg)
    record["variant"] = variant
    previous = _pick_weights(cfg, variant)
    if previous is None:
        return "error", f"no {variant} weights found (pass --weights)"
    _say(f"variant={variant} previous backbone={previous}")
    if cfg.dry_run:
        _say(
            f"dry-run: would train mode={cfg.mode} on {len(train_idx)} crops and write nothing now"
        )
        return "dry_run", "dry run"

    device = _resolve_device(cfg.device)
    record["device"] = device
    shapes = bank.shapes

    def embed(weights: Path) -> np.ndarray:
        tracker = make_feature_tracker(variant, weights, device)
        _say(f"features {variant} ({weights.name}) for {len(bank)} samples")
        return _embed_patches(tracker, bank.patches, shapes)

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    chosen = previous
    X = embed(previous)
    first_auc = _heldout_auc(X, y, train_idx, val_idx)
    status = "activated"
    reason = "discriminators rebuilt on the current backbone"

    if cfg.mode == "discriminator":
        record["before"], record["after"] = None, first_auc
        _say(f"held-out AUC of the discriminator: {fmt(first_auc)}")
    else:
        before = first_auc
        record["before"] = before
        _say(f"held-out AUC before (previous backbone): {fmt(before)}")
        model = _finetune_backbone(cfg, bank, train_idx, variant, previous, device)
        new_path = _save_backbone(model, variant, root)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        record["backbone"] = _relpath(root, new_path)
        store.update_manifest_file(root, new_path, kind="backbone", variant=variant)
        new_X = embed(new_path)
        after = _heldout_auc(new_X, y, train_idx, val_idx)
        record["after"] = after
        _say(f"held-out AUC after (fine-tuned backbone): {fmt(after)}")
        if before is not None and after is not None and after >= before:
            store.set_active_backbone(root, variant, new_path)
            chosen, X = new_path, new_X
            reason = f"held-out AUC {before:.4f} -> {after:.4f}; new backbone is active"
        else:
            status = "kept_previous"
            if before is None or after is None:
                why = "held-out AUC unavailable (one class in the validation split)"
            else:
                why = f"held-out AUC did not hold up ({before:.4f} -> {after:.4f})"
            reason = f"{why}; kept the previous backbone, new one saved as {new_path.name}"
        _say(reason)

    if cfg.mode in {"discriminator", "both"}:
        tracker = make_feature_tracker(variant, chosen, device)
        _write_discriminators(root, cfg.profile, tracker, X, y)
    return status, reason


def run_from_samples(cfg: SamplesConfig) -> int:
    """Retrain from ``cfg.root/samples``; always records ``manifest["last_retrain"]`` (except in a
    dry run, which writes nothing). Returns 0 when the run finished, 2 on error."""
    global _log_fh
    root = Path(cfg.root).expanduser()
    cfg.root = root
    started = datetime.now()
    record: dict[str, Any] = {
        "started": started.isoformat(timespec="seconds"),
        "finished": "",
        "status": "error",
        "mode": cfg.mode,
        "variant": cfg.variant or "",
        "device": "",
        "n_samples": 0,
        "n_videos": 0,
        "backbone": None,
        "metric": "heldout_auc",
        "before": None,
        "after": None,
        "reason": "",
        "log": None,
    }
    if not root.is_dir():
        print(f"ERROR: AI Track folder not found: {root}", file=sys.stderr)
        return 2
    log_path: Path | None = None
    if not cfg.dry_run:
        log_dir = store.layout(root)["logs"]
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"retrain_{started.strftime('%Y%m%d_%H%M%S')}.log"
        _log_fh = log_path.open("w", encoding="utf-8")
        record["log"] = _relpath(root, log_path)
    try:
        _say(f"Equivalent CLI\n{_format_samples_command(cfg)}")
        try:
            status, reason = _retrain(cfg, record)
        except Exception as err:
            status, reason = "error", f"{type(err).__name__}: {err}"
            if _log_fh is not None:
                _log_fh.write(traceback.format_exc())
        if cfg.dry_run:  # a dry run records and writes nothing
            if status == "error":
                print(f"ERROR: {reason}", file=sys.stderr)
            return 2 if status == "error" else 0
        record.update(
            status=status, reason=reason, finished=datetime.now().isoformat(timespec="seconds")
        )
        _record_last_retrain(root, record)
        if status == "error":
            _say(f"ERROR: {reason}")
            print(f"ERROR: {reason}", file=sys.stderr)
            return 2
        _say(f"finished: {status} ({reason})")
        return 0
    finally:
        if _log_fh is not None:
            _log_fh.close()
            _log_fh = None


def _format_cli_command(config: TrainConfig) -> str:
    parts = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "vaila.ai_tracker_train",
        "--data",
        str(config.data_yaml),
        "--weights",
        str(config.weights),
        "--mode",
        config.mode,
        "--epochs",
        str(config.epochs),
        "--batch",
        str(config.batch),
        "--lr",
        str(config.lr),
        "--device",
        config.device,
        "--profile",
        config.profile,
        "--max-samples",
        str(config.max_samples),
    ]
    if config.dry_run:
        parts.append("--dry-run")
    return " ".join(shlex.quote(p) for p in parts)


def run_ai_tracker_train_gui() -> None:
    """Tkinter GUI entry (Markerless 2D → Train AI Tracker)."""
    root = tk.Tk()
    root.title("vailá — Train AI Tracker")
    root.geometry("720x480")

    data_var = tk.StringVar()
    weights_var = tk.StringVar()
    mode_var = tk.StringVar(value="both")
    epochs_var = tk.StringVar(value="10")
    batch_var = tk.StringVar(value="16")
    lr_var = tk.StringVar(value="0.0001")
    device_var = tk.StringVar(value="auto")
    profile_var = tk.StringVar(value="default")
    status_var = tk.StringVar(value="Select data.yaml and a .pth under ai_tracker/")

    pth_list = _list_pth_files()
    if pth_list:
        weights_var.set(str(pth_list[0]))

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)

    def browse_yaml() -> None:
        path = filedialog.askopenfilename(
            title="Select data.yaml",
            filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
        )
        if path:
            data_var.set(path)

    def browse_weights() -> None:
        initial = str(_ai_tracker_dir())
        path = filedialog.askopenfilename(
            title="Select backbone .pth",
            initialdir=initial,
            filetypes=[("PyTorch", "*.pth *.pt"), ("All", "*.*")],
        )
        if path:
            weights_var.set(path)

    row = 0
    ttk.Label(frm, text="data.yaml").grid(row=row, column=0, sticky="w")
    ttk.Entry(frm, textvariable=data_var, width=64).grid(row=row, column=1, sticky="ew")
    ttk.Button(frm, text="Browse", command=browse_yaml).grid(row=row, column=2, padx=4)
    row += 1
    ttk.Label(frm, text="Base .pth").grid(row=row, column=0, sticky="w")
    ttk.Entry(frm, textvariable=weights_var, width=64).grid(row=row, column=1, sticky="ew")
    ttk.Button(frm, text="Browse", command=browse_weights).grid(row=row, column=2, padx=4)
    row += 1
    if pth_list:
        ttk.Label(frm, text="Quick pick").grid(row=row, column=0, sticky="w")
        combo = ttk.Combobox(frm, values=[str(p) for p in pth_list], width=62)
        combo.grid(row=row, column=1, sticky="ew")
        combo.bind("<<ComboboxSelected>>", lambda _e: weights_var.set(combo.get()))
        row += 1
    ttk.Label(frm, text="Mode").grid(row=row, column=0, sticky="w")
    ttk.Combobox(
        frm, textvariable=mode_var, values=["backbone", "discriminator", "both"], width=20
    ).grid(row=row, column=1, sticky="w")
    row += 1
    ttk.Label(frm, text="Epochs").grid(row=row, column=0, sticky="w")
    ttk.Entry(frm, textvariable=epochs_var, width=12).grid(row=row, column=1, sticky="w")
    row += 1
    ttk.Label(frm, text="Batch").grid(row=row, column=0, sticky="w")
    ttk.Entry(frm, textvariable=batch_var, width=12).grid(row=row, column=1, sticky="w")
    row += 1
    ttk.Label(frm, text="LR").grid(row=row, column=0, sticky="w")
    ttk.Entry(frm, textvariable=lr_var, width=12).grid(row=row, column=1, sticky="w")
    row += 1
    ttk.Label(frm, text="Device").grid(row=row, column=0, sticky="w")
    ttk.Combobox(frm, textvariable=device_var, values=["auto", "cuda", "cpu"], width=12).grid(
        row=row, column=1, sticky="w"
    )
    row += 1
    ttk.Label(frm, text="Disc. profile").grid(row=row, column=0, sticky="w")
    ttk.Entry(frm, textvariable=profile_var, width=24).grid(row=row, column=1, sticky="w")
    row += 1

    log = tk.Text(frm, height=12, wrap="word")
    log.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=8)
    frm.rowconfigure(row, weight=1)
    frm.columnconfigure(1, weight=1)
    row += 1
    ttk.Label(frm, textvariable=status_var).grid(row=row, column=0, columnspan=3, sticky="w")

    def append_log(msg: str) -> None:
        log.insert("end", msg + "\n")
        log.see("end")

    def build_config() -> TrainConfig:
        return TrainConfig(
            data_yaml=Path(data_var.get()).expanduser().resolve(),
            weights=Path(weights_var.get()).expanduser().resolve(),
            mode=mode_var.get().strip() or "both",
            epochs=max(1, int(epochs_var.get() or "10")),
            batch=max(1, int(float(batch_var.get() or "16"))),
            lr=float(lr_var.get() or "0.0001"),
            device=device_var.get().strip() or "auto",
            profile=(profile_var.get().strip() or "default"),
        )

    def on_run() -> None:
        try:
            cfg = build_config()
        except Exception as exc:
            messagebox.showerror("Train AI Tracker", str(exc))
            return
        if not cfg.data_yaml.is_file():
            messagebox.showerror("Train AI Tracker", f"Missing data.yaml: {cfg.data_yaml}")
            return
        if not cfg.weights.is_file():
            messagebox.showerror("Train AI Tracker", f"Missing weights: {cfg.weights}")
            return
        cli = _format_cli_command(cfg)
        print(f">> vaila/ai_tracker_train: Equivalent CLI\n{cli}")
        append_log(f"Equivalent CLI:\n{cli}")
        status_var.set("Training…")

        def worker() -> None:
            try:
                results = run_training(cfg)
                msg = (
                    f"Done. backbone={results.get('backbone')} "
                    f"discriminator={results.get('discriminator')}"
                )
                root.after(0, lambda: (append_log(msg), status_var.set("Finished")))
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                root.after(
                    0,
                    lambda: (
                        append_log(err),
                        status_var.set("Failed"),
                        messagebox.showerror("Train AI Tracker", err),
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    def on_help() -> None:
        if HELP_HTML.is_file():
            import webbrowser

            webbrowser.open_new_tab(HELP_HTML.as_uri())
        elif HELP_MD.is_file():
            append_log(HELP_MD.read_text(encoding="utf-8")[:4000])
        else:
            append_log(
                "Train AI Tracker: fine-tune .pth backbone and/or discriminator.npz "
                "from getpixelvideo Save ML datasets (pose or detect)."
            )

    btn_row = ttk.Frame(frm)
    btn_row.grid(row=row + 1, column=0, columnspan=3, pady=6)
    ttk.Button(btn_row, text="Run", command=on_run).pack(side="left", padx=4)
    ttk.Button(btn_row, text="Help", command=on_help).pack(side="left", padx=4)
    ttk.Button(btn_row, text="Close", command=root.destroy).pack(side="left", padx=4)

    root.mainloop()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train AI Tracker backbone / discriminator")
    p.add_argument("--gui", action="store_true", help="Open Tkinter GUI.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--data", "-d", help="Path to data.yaml from getpixelvideo Save ML.")
    src.add_argument(
        "--from-samples",
        metavar="ROOT",
        help="AI Track folder (<home>/ai_tracker): retrain from its samples/*.npz.",
    )
    p.add_argument(
        "--weights",
        "-w",
        help="Base .pth (absolute path, or under the AI Track folder). With --from-samples the "
        "default is the active backbone, else the official <variant>_imagenet.pth.",
    )
    p.add_argument(
        "--mode",
        choices=["backbone", "discriminator", "both"],
        default="both",
        help="What to train.",
    )
    p.add_argument("--variant", choices=sorted(_BACKBONE_FEATURE_DIM), help="--from-samples only.")
    p.add_argument("--epochs", type=int, help="default 10 (--data) / 5 (--from-samples)")
    p.add_argument("--batch", type=int, help="default 16 (--data) / 32 (--from-samples)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--unfreeze",
        choices=["last", "all"],
        default="last",
        help="--from-samples: train only the last stage + a new 2-class head (default) or all.",
    )
    p.add_argument("--device", default="auto", help="auto (CUDA only with >= 3 GiB free)|cpu|cuda")
    p.add_argument("--profile", default="default", help="discriminator_<profile>__<key>.npz")
    p.add_argument("--max-samples", type=int, default=4000, help="--data only.")
    p.add_argument("--min-samples", type=int, default=40, help="--from-samples: minimum train.")
    p.add_argument("--val-fraction", type=float, default=0.2, help="--from-samples hold-out.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ai-models",
        metavar="DIR",
        help="AI Track home folder (same option as getpixelvideo --ai-models).",
    )
    p.add_argument("--dry-run", action="store_true", help="Plan only; write nothing.")
    return p


def _samples_config(args: argparse.Namespace) -> SamplesConfig:
    weights = Path(args.weights).expanduser() if args.weights else None
    root = Path(args.from_samples).expanduser()
    if weights is not None and not weights.is_file():
        for alt in (root / args.weights, root / "backbones" / args.weights):
            if alt.is_file():
                weights = alt
                break
    return SamplesConfig(
        root=root,
        mode=args.mode,
        variant=args.variant,
        weights=weights.resolve() if weights is not None and weights.is_file() else weights,
        epochs=args.epochs if args.epochs is not None else 5,
        batch=args.batch if args.batch is not None else 32,
        lr=args.lr,
        unfreeze=args.unfreeze,
        device=args.device,
        profile=args.profile,
        min_samples=args.min_samples,
        val_fraction=args.val_fraction,
        seed=args.seed,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.ai_models:
        store.set_session_home(args.ai_models)
    if args.from_samples:
        cfg_s = _samples_config(args)
        if cfg_s.weights is not None and not cfg_s.weights.is_file():
            print(f"ERROR: weights not found: {args.weights}", file=sys.stderr)
            return 2
        return run_from_samples(cfg_s)
    if args.gui or not args.data:
        run_ai_tracker_train_gui()
        return 0
    weights = Path(args.weights).expanduser() if args.weights else None
    if weights is None:
        listed = _list_pth_files()
        if not listed:
            print("ERROR: no .pth under vaila/models/ai_tracker/; pass --weights", file=sys.stderr)
            return 2
        weights = listed[0]
    elif not weights.is_file():
        alt = _ai_tracker_dir() / args.weights
        if alt.is_file():
            weights = alt
        else:
            print(f"ERROR: weights not found: {args.weights}", file=sys.stderr)
            return 2
    cfg = TrainConfig(
        data_yaml=Path(args.data).expanduser().resolve(),
        weights=weights.resolve(),
        mode=args.mode,
        epochs=args.epochs if args.epochs is not None else 10,
        batch=args.batch if args.batch is not None else 16,
        lr=args.lr,
        device=args.device,
        profile=args.profile,
        max_samples=args.max_samples,
        dry_run=args.dry_run,
    )
    print(f"{PREFIX} Equivalent CLI\n{_format_cli_command(cfg)}")
    results = run_training(cfg)
    print(f"{PREFIX} results {results}")
    return 0


def run_ai_tracker_train() -> None:
    """GUI entry used by vaila.py chooser."""
    run_ai_tracker_train_gui()


if __name__ == "__main__":
    raise SystemExit(main())
