"""
================================================================================
Script: freekiki_heatmap.py - FreeKiki heatmap backend (ResNet + deconv head)
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.6
Created: 28 September 2026
Update Date: 28 September 2026

Description:
    Second network family for ``freekiki.py`` so the 49 soccer-field keypoints
    do not depend on one detector framework. Pure PyTorch + torchvision (BSD):

      * model: torchvision ResNet (default ``resnet50``, ImageNet weights read
        only from the local torch hub cache, never downloaded here) + three
        4x4 stride-2 deconvolutions (256 ch) + 1x1 conv -> 49 heatmaps at
        stride 4 (SimpleBaseline, Xiao et al. 2018). The pitch is one instance
        per frame, so no detection head is needed;
      * input: the frame letterboxed into a fixed 16:9 canvas (default
        1024x576, both sides multiples of 32), one affine for letterbox and
        augmentation (scale, rotation, shift, horizontal flip with the kiki
        ``flip_idx`` identity swap, brightness/contrast);
      * target: Gaussian (sigma 2 heatmap cells) at every labelled point;
        points with visibility 0 get an all-zero map, as YOLO-pose trains
        keypoint objectness on the same labels (v=0 -> "not present");
      * decode: arg-max + sub-cell quadratic fit on the log heatmap (exact for
        a Gaussian peak), then the inverse affine back to frame pixels. The
        keypoint confidence is the peak value (0..1); the frame "box"
        confidence is the highest peak;
      * checkpoints: ``weights/last.pt`` every epoch (model + optimizer +
        scheduler + epoch, so ``resume`` continues it) and ``weights/best.pt``
        (best val PCK10_all). Both carry ``backend: freekiki_heatmap_v1`` so
        :func:`model_backend` tells them apart from Ultralytics weights.
        ``results.csv`` / ``args.yaml`` follow the Ultralytics run layout that
        ``freekiki.py status/resume`` already read.

    Dataset: the same YOLO-pose labels of ``datasets/kiki49`` (read only).

Usage:
    uv run --no-sync vaila/freekiki.py train -w WS --backend heatmap --epochs 1 --fraction 0.01

License:
    GNU Affero General Public License v3.0
"""

from __future__ import annotations

import csv
import json
import math
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torch import nn

BACKEND = "freekiki_heatmap_v1"
STRIDE = 4
SIGMA = 2.0
PAD_VALUE = (114, 114, 114)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULTS = {"epochs": 60, "imgsz": 1024, "batch": 16, "lr": 1e-3, "workers": 8}
RESULTS_FIELDS = [
    "epoch",
    "time",
    "lr",
    "train/loss",
    "val/recall",
    "val/precision",
    "val/err_median",
    "val/pck10_all",
]
FITNESS_COL = "val/pck10_all"


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def input_size(imgsz: int) -> tuple[int, int]:
    """16:9 network canvas ``(W, H)``: W = ``imgsz`` rounded up to 32, H the 16:9 height up to 32."""
    width = max(64, math.ceil(int(imgsz) / 32) * 32)
    return width, max(64, math.ceil(width * 9 / 16 / 32) * 32)


def letterbox_matrix(w: int, h: int, width: int, height: int) -> np.ndarray:
    """3x3 affine: frame pixels -> centered, aspect-kept canvas pixels."""
    s = min(width / w, height / h)
    return np.array(
        [[s, 0.0, (width - w * s) / 2], [0.0, s, (height - h * s) / 2], [0.0, 0.0, 1.0]]
    )


def augment_matrix(
    rng, width: int, height: int, *, scale=0.2, degrees=5.0, translate=0.05, flip=False
) -> np.ndarray:
    """Random scale/rotation/shift about the canvas center, then an optional mirror."""
    c = np.array([[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1.0]])
    a = math.radians(rng.uniform(-degrees, degrees))
    s = rng.uniform(1 - scale, 1 + scale)
    r = np.array(
        [[s * math.cos(a), -s * math.sin(a), 0], [s * math.sin(a), s * math.cos(a), 0], [0, 0, 1]]
    )
    t = np.array(
        [
            [1, 0, width / 2 + rng.uniform(-translate, translate) * width],
            [0, 1, height / 2 + rng.uniform(-translate, translate) * height],
            [0, 0, 1.0],
        ]
    )
    m = t @ r @ c
    if flip:
        m = np.array([[-1, 0, width], [0, 1, 0], [0, 0, 1.0]]) @ m
    return m


def apply_affine(m: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return np.asarray(xy, dtype=float) @ m[:2, :2].T + m[:2, 2]


def to_cells(xy: np.ndarray, stride: int = STRIDE) -> np.ndarray:
    """Canvas pixels -> heatmap cell coordinates (cell j covers pixels j*stride..(j+1)*stride)."""
    return (np.asarray(xy, dtype=float) + 0.5) / stride - 0.5


def from_cells(uv: np.ndarray, stride: int = STRIDE) -> np.ndarray:
    return (np.asarray(uv, dtype=float) + 0.5) * stride - 0.5


def render_heatmaps(uv, vis, shape: tuple[int, int], sigma: float = SIGMA):
    """Gaussian targets ``(B, K, h, w)`` from cell coordinates ``uv (B, K, 2)`` and ``vis (B, K)``."""
    h, w = shape
    ys = torch.arange(h, device=uv.device, dtype=torch.float32).view(1, 1, h, 1)
    xs = torch.arange(w, device=uv.device, dtype=torch.float32).view(1, 1, 1, w)
    u = uv[..., 0].float().unsqueeze(-1).unsqueeze(-1)
    v = uv[..., 1].float().unsqueeze(-1).unsqueeze(-1)
    g = torch.exp(-((xs - u) ** 2 + (ys - v) ** 2) / (2 * sigma**2))
    return g * (vis > 0).float().unsqueeze(-1).unsqueeze(-1)


def decode_heatmaps(hm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(K, h, w)`` heatmaps -> cell coordinates ``(K, 2)`` and peak values ``(K,)`` in 0..1.

    Sub-cell offset: vertex of the parabola through the log values of the peak
    and its two neighbours on each axis (exact for a Gaussian peak).
    """
    k, h, w = hm.shape
    flat = hm.reshape(k, -1)
    idx = flat.argmax(axis=1)
    peak = flat[np.arange(k), idx]
    py, px = np.divmod(idx, w)
    uv = np.stack([px, py], axis=1).astype(float)
    log = np.log(np.clip(hm, 1e-10, None))
    for i in range(k):
        for axis, (lo, hi) in enumerate(((0, w - 1), (0, h - 1))):
            c = (px[i], py[i])[axis]
            if not lo < c < hi:
                continue
            if axis == 0:
                a, b, d = log[i, py[i], c - 1], log[i, py[i], c], log[i, py[i], c + 1]
            else:
                a, b, d = log[i, c - 1, px[i]], log[i, c, px[i]], log[i, c + 1, px[i]]
            den = a - 2 * b + d
            if den < 0:
                uv[i, axis] += float(np.clip(0.5 * (a - d) / den, -0.5, 0.5))
    return uv, np.clip(peak, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def label_path(img_path: Path) -> Path:
    """Ultralytics rule: the last ``images`` folder of the path becomes ``labels``."""
    parts = list(Path(img_path).parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_label(path: Path, nkp: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised ``xy (K, 2)`` and visibility ``(K,)`` of the first instance (zeros if none)."""
    if Path(path).is_file():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) == 5 + nkp * 3:
                kps = np.asarray(values[5:], dtype=float).reshape(nkp, 3)
                return kps[:, :2], kps[:, 2]
    return np.zeros((nkp, 2)), np.zeros(nkp)


def list_images(ds_root: Path, entry: str) -> list[Path]:
    """Images of a data.yaml split entry: a folder, or a .txt list (manifests repeat lines)."""
    path = Path(entry) if Path(entry).is_absolute() else Path(ds_root) / entry
    if path.suffix == ".txt":
        lines = path.read_text(encoding="utf-8").splitlines()
        return [Path(x.strip()) for x in lines if x.strip()]
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in path.iterdir() if p.suffix.lower() in exts)


def to_tensor(canvas_bgr: np.ndarray) -> torch.Tensor:
    rgb = canvas_bgr[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy())


class KikiHeatDataset(torch.utils.data.Dataset):
    """YOLO-pose images/labels -> ``(image tensor, cell xy (K, 2), visibility (K,))``.

    Augmentation is seeded by ``(seed, epoch, index)``; set ``epoch`` before
    each epoch (workers are re-created per epoch, so they see it).
    """

    def __init__(
        self, images, flip_idx, input_wh, *, train: bool, seed: int = 0, stride: int = STRIDE
    ):
        self.images = [Path(p) for p in images]
        self.flip_idx = np.asarray(flip_idx, dtype=int)
        self.nkp = len(self.flip_idx)
        self.width, self.height = input_wh
        self.train = train
        self.seed = seed
        self.stride = stride
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        img_path = self.images[index]
        frame = cv2.imread(str(img_path))
        if frame is None:
            raise OSError(f"Cannot read image {img_path}")
        h, w = frame.shape[:2]
        xy, vis = read_label(label_path(img_path), self.nkp)
        xy = xy * (w, h)
        m = letterbox_matrix(w, h, self.width, self.height)
        if self.train:
            rng = np.random.default_rng([self.seed, self.epoch, index])
            flip = bool(rng.random() < 0.5)
            m = augment_matrix(rng, self.width, self.height, flip=flip) @ m
            if flip:  # mirrored image: point k now shows the landmark of flip_idx[k]
                xy, vis = xy[self.flip_idx], vis[self.flip_idx]
        canvas = cv2.warpAffine(
            frame, m[:2], (self.width, self.height), flags=cv2.INTER_LINEAR, borderValue=PAD_VALUE
        )
        if self.train:
            alpha, beta = rng.uniform(0.75, 1.25), rng.uniform(-25, 25)
            canvas = cv2.convertScaleAbs(canvas, alpha=alpha, beta=beta)
        pts = apply_affine(m, xy)
        inside = (
            (pts[:, 0] >= 0)
            & (pts[:, 0] < self.width)
            & (pts[:, 1] >= 0)
            & (pts[:, 1] < self.height)
        )
        vis = np.where((vis > 0) & inside, 1.0, 0.0)
        return (
            to_tensor(canvas),
            torch.from_numpy(to_cells(pts, self.stride).astype(np.float32)),
            torch.from_numpy(vis.astype(np.float32)),
        )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def cached_imagenet_state(backbone: str) -> dict:
    """ImageNet weights of a torchvision model from the local hub cache (no download)."""
    import torchvision

    weights = torchvision.models.get_model_weights(backbone).DEFAULT  # ty: ignore[unresolved-attribute]
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not path.is_file():
        raise FileNotFoundError(
            f"ImageNet weights of {backbone} are not in the local cache ({path}). "
            f"Downloading them needs your approval ({weights.url}); or train with --no-pretrained."
        )
    return torch.load(path, map_location="cpu", weights_only=True)


class KikiHeatNet(nn.Module):
    """torchvision ResNet trunk (stride 32) + 3 deconvolutions (stride 4) + 1x1 heatmap head."""

    def __init__(self, backbone: str = "resnet50", nkp: int = 49, *, imagenet: dict | None = None):
        import torchvision

        super().__init__()
        net = getattr(torchvision.models, backbone)(weights=None)
        if imagenet is not None:
            net.load_state_dict(imagenet)
        self.backbone = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        layers: list[nn.Module] = []
        channels = net.fc.in_features
        for _ in range(3):
            layers += [
                nn.ConvTranspose2d(channels, 256, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ]
            channels = 256
        self.deconv = nn.Sequential(*layers)
        self.head = nn.Conv2d(256, nkp, 1)
        for mod in [*self.deconv.modules(), self.head]:
            if isinstance(mod, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(mod.weight, std=0.001)
            if isinstance(mod, nn.Conv2d) and mod.bias is not None:
                nn.init.zeros_(mod.bias)

    def forward(self, x):
        return self.head(self.deconv(self.backbone(x)))


def torch_device(device: str | None) -> torch.device:
    """Ultralytics-style device strings: None/'' -> cuda:0 if available, '0' -> cuda:0, 'cpu'."""
    if device in (None, ""):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = str(device)
    return torch.device(f"cuda:{device}" if device.isdigit() else device)


def model_backend(path) -> str:
    """``heatmap`` for a checkpoint written by this module, else ``yolo``."""
    path = Path(path)
    if not path.is_file() or not zipfile.is_zipfile(path):
        return "yolo"
    with zipfile.ZipFile(path) as z:
        pkl = next((n for n in z.namelist() if n.endswith("/data.pkl")), None)
        return "heatmap" if pkl and BACKEND.encode() in z.read(pkl) else "yolo"


class HeatmapPredictor:
    """``predict(frame BGR) -> (frame conf, xy (K, 2) frame px, conf (K,))`` for a heatmap model."""

    backend = "heatmap"

    def __init__(self, net: KikiHeatNet, input_wh, device=None):
        self.device = torch_device(device)
        self.net = net.to(self.device).eval().to(memory_format=torch.channels_last)  # ty: ignore[no-matching-overload]
        self.width, self.height = input_wh
        self.imgsz = self.width

    @classmethod
    def load(cls, path, device=None) -> HeatmapPredictor:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("backend") != BACKEND:
            raise ValueError(f"{path} is not a FreeKiki heatmap checkpoint")
        net = KikiHeatNet(ckpt["backbone"], int(ckpt["nkp"]))
        net.load_state_dict(ckpt["model"])
        return cls(net, tuple(ckpt["input_wh"]), device)

    @torch.inference_mode()
    def predict(self, frame) -> tuple[float, np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        m = letterbox_matrix(w, h, self.width, self.height)
        canvas = cv2.warpAffine(
            frame, m[:2], (self.width, self.height), flags=cv2.INTER_LINEAR, borderValue=PAD_VALUE
        )
        x = to_tensor(canvas)[None].to(self.device).to(memory_format=torch.channels_last)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self._amp):
            hm = self.net(x)
        uv, conf = decode_heatmaps(hm[0].float().cpu().numpy())
        xy = apply_affine(np.linalg.inv(m), from_cells(uv))
        return float(conf.max()), xy, conf

    @property
    def _amp(self) -> bool:
        return self.device.type == "cuda"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _lr_lambda(total_steps: int, warmup: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    return f


def _save(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _append_result(path: Path, row: dict) -> None:
    new = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def best_fitness(results_csv: Path) -> tuple[float, int | None]:
    """Highest ``val/pck10_all`` (first maximum) and its epoch in a heatmap ``results.csv``."""
    best, epoch = float("-inf"), None
    if Path(results_csv).is_file():
        with Path(results_csv).open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    value = float(row.get(FITNESS_COL) or "nan")
                except ValueError:
                    continue
                if value == value and value > best:
                    best, epoch = value, int(row["epoch"])
    return best, epoch


def train_heatmap(
    run_dir,
    *,
    train_images,
    flip_idx,
    val_fn=None,
    epochs: int = DEFAULTS["epochs"],
    imgsz: int = DEFAULTS["imgsz"],
    batch: int | None = DEFAULTS["batch"],
    workers: int | None = DEFAULTS["workers"],
    lr: float = DEFAULTS["lr"],
    backbone: str = "resnet50",
    pretrained: bool = True,
    init: str | None = None,
    device: str | None = None,
    seed: int = 0,
    resume: bool = False,
    extra_args: dict | None = None,
    log=print,
) -> Path:
    """Train (or ``resume``) a heatmap model in ``run_dir``; returns ``weights/best.pt``.

    ``val_fn(predictor) -> dict`` scores the model on the val images (FreeKiki
    ``score_predictions`` overall: recall, precision, err_median, pck10_all)
    after every epoch; ``best.pt`` follows ``val/pck10_all``. Without it the
    last epoch is also the best. ``init``: a heatmap checkpoint whose weights
    start this run (continued fine-tune) instead of ImageNet. ``extra_args``
    (e.g. fraction, manifest) is stored in ``args.yaml`` and the checkpoint so
    ``resume`` can rebuild the same train list.
    """
    run_dir = Path(run_dir)
    wdir = run_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    last_pt, best_pt, results = wdir / "last.pt", wdir / "best.pt", run_dir / "results.csv"
    dev = torch_device(device)
    ckpt: dict[str, Any] = {}
    if resume:
        ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
        if ckpt.get("backend") != BACKEND or ckpt.get("optimizer") is None:
            raise ValueError(f"{last_pt} is not a resumable heatmap checkpoint")
        a = ckpt["args"]
        epochs, imgsz, batch, lr = a["epochs"], a["imgsz"], batch or a["batch"], a["lr"]
        backbone, seed, workers = a["backbone"], a["seed"], workers or a["workers"]
    batch = int(batch or DEFAULTS["batch"])
    workers = int(DEFAULTS["workers"] if workers is None else workers)
    torch.manual_seed(seed)
    input_wh = input_size(imgsz)
    nkp = len(flip_idx)
    imagenet = None
    if not resume and init is None and pretrained:
        imagenet = cached_imagenet_state(backbone)
    net = KikiHeatNet(backbone, nkp, imagenet=imagenet)
    if resume:
        net.load_state_dict(ckpt["model"])
    elif init is not None:
        start = torch.load(init, map_location="cpu", weights_only=False)
        if start.get("backend") != BACKEND or start.get("backbone") != backbone:
            raise ValueError(f"--base {init} is not a {backbone} heatmap checkpoint")
        net.load_state_dict(start["model"])
    net = net.to(dev).to(memory_format=torch.channels_last)  # ty: ignore[no-matching-overload]

    ds = KikiHeatDataset(train_images, flip_idx, input_wh, train=True, seed=seed)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=int(batch),
        shuffle=True,
        num_workers=int(workers),
        pin_memory=dev.type == "cuda",
        drop_last=len(ds) > int(batch),
        generator=torch.Generator().manual_seed(seed),
    )
    steps = max(1, len(loader)) * epochs
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(steps, min(1000, steps // 10 + 1)))
    first = 1
    args = {
        "backend": "heatmap",
        "backbone": backbone,
        "model": f"{backbone}-{'imagenet' if pretrained else 'scratch'}"
        if init is None
        else str(init),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "input_wh": list(input_wh),
        "batch": int(batch),
        "workers": int(workers),
        "lr": float(lr),
        "seed": int(seed),
        "device": str(dev),
        "stride": STRIDE,
        "sigma": SIGMA,
        "train_images": len(ds),
    } | dict(extra_args or {})
    if resume:
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        first = int(ckpt["epoch"]) + 1
        args = ckpt["args"] | {"batch": int(batch), "workers": int(workers)}
        log(f"heatmap resume: epoch {first}/{epochs} from {last_pt}")
    else:
        (run_dir / "args.yaml").write_text(yaml.safe_dump(args, sort_keys=False), "utf-8")
    best, _ = best_fitness(results)
    t0 = time.perf_counter() - float(ckpt.get("train_time", 0.0))
    meta = {"backend": BACKEND, "backbone": backbone, "nkp": nkp, "input_wh": list(input_wh)}
    meta |= {"flip_idx": [int(i) for i in flip_idx], "args": args}
    amp = dev.type == "cuda"
    for epoch in range(first, epochs + 1):
        ds.epoch = epoch
        net.train()
        total, n = 0.0, 0
        for step, (x, uv, vis) in enumerate(loader, 1):
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            uv, vis = uv.to(dev, non_blocking=True), vis.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                out = net(x)
            loss = nn.functional.mse_loss(out.float(), render_heatmaps(uv, vis, out.shape[-2:]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total, n = total + loss.item(), n + 1
            if step % 200 == 0:
                log(f"  epoch {epoch}/{epochs} batch {step}/{len(loader)} loss {total / n:.3e}")
        row: dict[str, Any] = {
            "epoch": epoch,
            "lr": f"{sched.get_last_lr()[0]:.3e}",
            "train/loss": f"{total / max(1, n):.4e}",
        }
        if val_fn is not None:
            m = val_fn(HeatmapPredictor(net, input_wh, dev))
            net.to(memory_format=torch.channels_last)
            row |= {f"val/{k}": m.get(k) for k in ("recall", "precision", "err_median")}
            row[FITNESS_COL] = m.get("pck10_all")
        elapsed = time.perf_counter() - t0
        row["time"] = round(elapsed, 1)
        _append_result(results, row)
        state = {k: v.detach().float().cpu() for k, v in net.state_dict().items()}
        payload = meta | {"model": state, "epoch": epoch, "train_time": elapsed}
        payload["date"] = datetime.now().isoformat(timespec="seconds")
        fitness = row.get(FITNESS_COL)
        score = float(fitness) if fitness is not None else float(epoch)
        if score > best or not best_pt.is_file():
            best = score
            _save(best_pt, payload | {"optimizer": None, "scheduler": None, "best_epoch": epoch})
        done = epoch == epochs
        _save(
            last_pt,
            payload
            | {
                "optimizer": None if done else opt.state_dict(),
                "scheduler": None if done else sched.state_dict(),
            },
        )
        log(
            f"heatmap epoch {epoch}/{epochs} loss {row['train/loss']} "
            f"val PCK10_all {fitness} recall {row.get('val/recall')} "
            f"median {row.get('val/err_median')} px@1920 | {elapsed / 3600:.2f} h"
        )
    (run_dir / "heatmap_summary.json").write_text(
        json.dumps({"best_fitness": best, "fitness": FITNESS_COL, "args": args}, indent=2),
        encoding="utf-8",
    )
    return best_pt
