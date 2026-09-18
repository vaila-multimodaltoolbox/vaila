"""
Project: vailá Multimodal Toolbox
Script: ai_tracker_train.py — Offline fine-tune for Track AI backbones / discriminator

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 18 September 2026
Update Date: 18 September 2026
Version: 0.4.4

Description:
    Train (or continue) AI Tracker assets under ``vaila/models/ai_tracker/`` using
    YOLO detect/pose datasets exported by getpixelvideo Save ML:

    * **backbone** — fine-tune ResNet/MobileNet/EfficientNet with CE on crops,
      then save an embedding-compatible ``.pth`` (head stripped for Track AI).
    * **discriminator** — offline ridge regression on deep features (pos/neg crops),
      saved as ``discriminator_<profile>.npz``.
    * **both** — run backbone then discriminator (discriminator uses the new .pth).

Usage:
    uv run --no-sync python -m vaila.ai_tracker_train
    uv run --no-sync python -m vaila.ai_tracker_train \\
        --data /path/to/data.yaml --mode both --weights resnet50_imagenet.pth \\
        --epochs 10 --profile athletics

License: AGPLv3
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import shlex
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np

try:
    from .tracking.ai_tracker import (
        _BACKBONE_FEATURE_DIM,
        _BACKBONE_HEAD_ATTR,
        DeepFeatureExtractor,
        _default_checkpoint_dir,
        _normalize_backbone_variant,
        default_checkpoint_path,
    )
except ImportError:
    from tracking.ai_tracker import (  # ty: ignore[unresolved-import]
        _BACKBONE_FEATURE_DIM,
        _BACKBONE_HEAD_ATTR,
        DeepFeatureExtractor,
        _default_checkpoint_dir,
        _normalize_backbone_variant,
        default_checkpoint_path,
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
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    tv_models = None  # type: ignore[assignment]
    tv_transforms = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc, assignment]

HELP_MD = Path(__file__).resolve().parent / "help" / "ai_tracker_train.md"
HELP_HTML = Path(__file__).resolve().parent / "help" / "ai_tracker_train.html"

# COCO-80 first names used only as GUI/doc hint (avoid importing getpixelvideo).
COCO80_HINT = ("person", "sports ball", "bicycle", "car", "dog", "cat", "bottle", "chair")


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


def _resolve_device(requested: str) -> str:
    if not TORCH_OK:
        return "cpu"
    req = (requested or "auto").strip().lower()
    if req == "cpu":
        return "cpu"
    if req in {"auto", "cuda", "0"}:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
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


def _strip_head_and_save(model: Any, variant: str, out_path: Path) -> None:
    """Save full state_dict (including Identity-ready head) for DeepFeatureExtractor."""
    variant = _normalize_backbone_variant(variant)
    head_attr = _BACKBONE_HEAD_ATTR[variant]
    # Replace head with Identity so loaders that expect embedding weights still work;
    # DeepFeatureExtractor also forces Identity after load.
    setattr(model, head_attr, nn.Identity())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path))
    print(f">> vaila/ai_tracker_train: saved backbone -> {out_path}")


def train_backbone(config: TrainConfig) -> Path | None:
    if not TORCH_OK:
        raise RuntimeError("PyTorch/torchvision required for backbone training.")
    samples, class_names = collect_crop_samples(config.data_yaml, max_samples=config.max_samples)
    if len(samples) < 2:
        raise RuntimeError(f"Need at least 2 labeled crops; got {len(samples)}.")
    num_classes = max(len(class_names), max(s[1] for s in samples) + 1)
    device = _resolve_device(config.device)
    variant = _infer_variant_from_path(config.weights)
    print(
        f">> vaila/ai_tracker_train: backbone train variant={variant} "
        f"classes={num_classes} samples={len(samples)} device={device}"
    )
    if config.dry_run:
        out = _ai_tracker_dir() / f"{variant}_finetuned_DRYRUN.pth"
        print(f">> vaila/ai_tracker_train: dry-run would write {out}")
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
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(max(1, config.epochs)):
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
        print(
            f">> vaila/ai_tracker_train: epoch {epoch + 1}/{config.epochs} "
            f"loss={avg:.4f} acc={acc:.3f}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _ai_tracker_dir() / f"{variant}_finetuned_{stamp}.pth"
    _strip_head_and_save(model.cpu(), variant, out_path)
    return out_path


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
    """Offline ridge discriminator from YOLO crops + random negatives."""
    weights = weights_override or config.weights
    samples, _class_names = collect_crop_samples(config.data_yaml, max_samples=config.max_samples)
    if len(samples) < 2:
        raise RuntimeError(f"Need at least 2 labeled crops; got {len(samples)}.")
    variant = _infer_variant_from_path(weights)
    extractor = DeepFeatureExtractor(
        use_cuda=(_resolve_device(config.device) == "cuda"), weights_path=weights, variant=variant
    )
    if not extractor.enabled:
        raise RuntimeError(f"Could not load backbone features from {weights}")

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
            emb = extractor.extract_embedding(crop)
            if emb is None:
                continue
            feats.append(emb.astype(np.float32))
            labels.append(1.0)
            neg = _random_neg_box(iw, ih, boxes, rng)
            if neg is not None:
                nx, ny, nw, nh = neg
                ncrop = im[ny : ny + nh, nx : nx + nw]
                nemb = extractor.extract_embedding(ncrop)
                if nemb is not None:
                    feats.append(nemb.astype(np.float32))
                    labels.append(-1.0)

    if len(feats) < 2:
        raise RuntimeError("Not enough feature vectors for discriminator training.")

    X = np.stack(feats, axis=0)
    y = np.asarray(labels, dtype=np.float32)
    n = X.shape[0]
    reg_lambda = 0.05
    print(
        f">> vaila/ai_tracker_train: discriminator ridge N={n} dim={X.shape[1]} "
        f"pos={int((y > 0).sum())} neg={int((y < 0).sum())}"
    )
    if config.dry_run:
        out = default_checkpoint_path(config.profile)
        print(f">> vaila/ai_tracker_train: dry-run would write {out}")
        return out

    K = X @ X.T + reg_lambda * np.eye(n, dtype=np.float32)
    alpha = np.linalg.solve(K, y)
    w = X.T @ alpha
    b = float(np.mean(y - X @ w))
    out_path = default_checkpoint_path(config.profile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        w=w.astype(np.float32),
        b=np.float32(b),
        n_samples=np.float32(n),
        feat_dim=np.int32(w.shape[0]),
    )
    print(f">> vaila/ai_tracker_train: saved discriminator -> {out_path}")
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
    p.add_argument("--data", "-d", help="Path to data.yaml from getpixelvideo Save ML.")
    p.add_argument(
        "--weights",
        "-w",
        help="Base .pth under vaila/models/ai_tracker/ (or absolute path).",
    )
    p.add_argument(
        "--mode",
        choices=["backbone", "discriminator", "both"],
        default="both",
        help="What to train.",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--profile", default="default", help="discriminator_<profile>.npz")
    p.add_argument("--max-samples", type=int, default=4000)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
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
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        device=args.device,
        profile=args.profile,
        max_samples=args.max_samples,
        dry_run=args.dry_run,
    )
    print(f">> vaila/ai_tracker_train: Equivalent CLI\n{_format_cli_command(cfg)}")
    results = run_training(cfg)
    print(f">> vaila/ai_tracker_train: results {results}")
    return 0


def run_ai_tracker_train() -> None:
    """GUI entry used by vaila.py chooser."""
    run_ai_tracker_train_gui()


if __name__ == "__main__":
    raise SystemExit(main())
