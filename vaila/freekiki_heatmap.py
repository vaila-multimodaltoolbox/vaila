"""
================================================================================
Script: freekiki_heatmap.py - FreeKiki heatmap backend (HRNet-W32 / ResNet + distillation)
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.9
Created: 28 September 2026
Update Date: 18 October 2026

Description:
    Second network family for ``freekiki.py`` so the 49 soccer-field keypoints
    do not depend on one detector framework. Pure PyTorch + torchvision (BSD):

      * model:
          - ``hrnet_w32`` (Sun et al., CVPR 2019): parallel multi-resolution
            branches at strides 4/8/16/32 (32/64/128/256 ch) keeping the
            stride-4 high-resolution stream alive from stem to head, with
            multi-scale bilinear fusion -> 49 heatmaps at stride 4;
          - ``resnet18/34/50/101/152``: torchvision ResNet (ImageNet weights
            read only from the local torch hub cache) + three 4x4 stride-2
            deconvolutions (256 ch) + 1x1 conv -> 49 heatmaps at stride 4;
      * knowledge distillation (``L = L_gt + lambda * L_distill``, default
        ``lambda = 0.5``): YOLO -> HRNet cannot copy weights directly; instead
        the promoted teacher (``active``, e.g. ``freekiki_l.pt``) evaluates the
        same augmented canvas in ``eval()`` / ``no_grad`` mode and provides soft
        Gaussian targets (including unlabelled visible points where GT vis=0
        and teacher conf >= 0.2);
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
    uv run --no-sync vaila/freekiki.py train -w WS --backend heatmap --backbone hrnet_w32 --teacher active --distill-weight 0.5

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
DISTILL_WEIGHT = 0.5
TEACHER_MIN_CONF = 0.2
PAD_VALUE = (114, 114, 114)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULTS = {
    "epochs": 60,
    "imgsz": 1024,
    "batch": 16,  # ResNet heatmap (GT only)
    "batch_hrnet": 8,  # HRNet-W32 + YOLO teacher at 1024 (~2x VRAM); start 4 on 24 GiB
    "lr": 1e-3,
    "workers": 8,
    "distill_weight": DISTILL_WEIGHT,
}
RESULTS_FIELDS = [
    "epoch",
    "time",
    "lr",
    "train/loss",
    "train/loss_gt",
    "train/loss_distill",
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


def render_soft_heatmaps(
    uv: torch.Tensor,
    conf: torch.Tensor,
    shape: tuple[int, int],
    *,
    sigma: float = SIGMA,
    min_conf: float = TEACHER_MIN_CONF,
) -> torch.Tensor:
    """Soft Gaussian teacher heatmaps ``(B, K, h, w)`` scaled by teacher confidence ``conf (B, K)``.

    Only teacher predictions inside ``shape`` with ``conf >= min_conf`` emit a
    Gaussian peak (amplitude = ``conf``); others produce a zero map.
    """
    h, w = shape
    uv_f = torch.nan_to_num(uv.float(), nan=-1e6)
    conf_f = torch.nan_to_num(conf.float(), nan=0.0).clamp(0.0, 1.0)
    inside = (
        (uv_f[..., 0] >= -0.5)
        & (uv_f[..., 0] <= w - 0.5)
        & (uv_f[..., 1] >= -0.5)
        & (uv_f[..., 1] <= h - 0.5)
    )
    weight = torch.where((conf_f >= min_conf) & inside, conf_f, torch.zeros_like(conf_f))
    ys = torch.arange(h, device=uv.device, dtype=torch.float32).view(1, 1, h, 1)
    xs = torch.arange(w, device=uv.device, dtype=torch.float32).view(1, 1, 1, w)
    u = uv_f[..., 0].unsqueeze(-1).unsqueeze(-1)
    v = uv_f[..., 1].unsqueeze(-1).unsqueeze(-1)
    g = torch.exp(-((xs - u) ** 2 + (ys - v) ** 2) / (2 * sigma**2))
    return g * weight.unsqueeze(-1).unsqueeze(-1)


def compute_distill_loss(
    pred: torch.Tensor,
    gt_uv: torch.Tensor,
    gt_vis: torch.Tensor,
    teacher_uv: torch.Tensor | None = None,
    teacher_conf: torch.Tensor | None = None,
    *,
    distill_weight: float = DISTILL_WEIGHT,
    min_teacher_conf: float = TEACHER_MIN_CONF,
    sigma: float = SIGMA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined heatmap loss ``L = L_gt + distill_weight * L_distill``.

    * ``L_gt``: MSE vs ground-truth heatmaps (``vis=0`` -> zero map).
    * ``L_distill``: MSE vs teacher soft heatmaps on the same augmented canvas.
      Where ``GT vis=0`` and ``teacher_conf >= min_teacher_conf``, the student
      inherits the teacher's soft peak; where ``GT vis>0``, GT remains dominant
      while the distillation term stays active at ``distill_weight``.
    """
    shape = (int(pred.shape[-2]), int(pred.shape[-1]))
    pred_f = pred.float()
    gt_hm = render_heatmaps(gt_uv, gt_vis, shape, sigma=sigma)
    loss_gt = nn.functional.mse_loss(pred_f, gt_hm)
    if teacher_uv is None or teacher_conf is None or float(distill_weight) <= 0.0:
        zero = torch.zeros((), device=pred.device, dtype=torch.float32)
        return loss_gt, loss_gt.detach(), zero
    soft_hm = render_soft_heatmaps(
        teacher_uv, teacher_conf, shape, sigma=sigma, min_conf=min_teacher_conf
    )
    loss_distill = nn.functional.mse_loss(pred_f, soft_hm)
    total = loss_gt + float(distill_weight) * loss_distill
    return total, loss_gt.detach(), loss_distill.detach()


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
HRNET_BACKBONES = ("hrnet_w32",)
HRNET_W32_CHANNELS = (32, 64, 128, 256)


class _HRBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):  # ty: ignore[invalid-mutable-override]
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _HRBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):  # ty: ignore[invalid-mutable-override]
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


def _conv_bn(cin: int, cout: int, k: int, stride: int, relu: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(cin, cout, k, stride, k // 2, bias=False),
        nn.BatchNorm2d(cout),
    ]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class _HRModule(nn.Module):
    """One HRNet exchange unit: 4 BasicBlocks per branch, then all-to-all fusion."""

    def __init__(self, channels: tuple[int, ...], blocks: int = 4):
        super().__init__()
        n = len(channels)
        self.branches = nn.ModuleList(
            nn.Sequential(*[_HRBasicBlock(c, c) for _ in range(blocks)]) for c in channels
        )
        fuse = nn.ModuleList()
        for i in range(n):
            row = nn.ModuleList()
            for j in range(n):
                if j > i:  # lower resolution -> 1x1 conv, upsampled in forward
                    row.append(_conv_bn(channels[j], channels[i], 1, 1, relu=False))
                elif j == i:
                    row.append(nn.Identity())
                else:  # higher resolution -> (i - j) stride-2 3x3 convs
                    steps = [
                        _conv_bn(channels[j], channels[j], 3, 2, relu=True)
                        for _ in range(i - j - 1)
                    ]
                    steps.append(_conv_bn(channels[j], channels[i], 3, 2, relu=False))
                    row.append(nn.Sequential(*steps))
            fuse.append(row)
        self.fuse_layers = fuse
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:  # ty: ignore[invalid-mutable-override]
        xs = [branch(x) for branch, x in zip(self.branches, xs, strict=True)]
        out = []
        for i, row in enumerate(self.fuse_layers):
            assert isinstance(row, nn.ModuleList)
            size = xs[i].shape[-2:]
            y = xs[i]
            for j, layer in enumerate(row):
                if j == i:
                    continue
                z = layer(xs[j])
                if j > i:
                    z = nn.functional.interpolate(
                        z, size=size, mode="bilinear", align_corners=False
                    )
                y = y + z
            out.append(self.relu(y))
        return out


class HRNetW32(nn.Module):
    """HRNet-W32 trunk (Sun et al. 2019) -> concatenated multi-scale features at stride 4.

    Stem (2x stride-2 3x3, 64 ch) -> stage 1 (4 Bottlenecks, 256 ch) -> stages
    2/3/4 with 1/4/3 exchange modules over 2/3/4 parallel branches
    (32/64/128/256 ch at strides 4/8/16/32). The output upsamples every branch
    bilinearly to stride 4 and concatenates them (``480`` channels). Module
    names follow the official implementation, so a locally cached
    HRNetV2-W32 ImageNet checkpoint loads with ``strict=False``.
    """

    out_channels = sum(HRNET_W32_CHANNELS)

    def __init__(self):
        super().__init__()
        c = HRNET_W32_CHANNELS
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        down = nn.Sequential(nn.Conv2d(64, 256, 1, bias=False), nn.BatchNorm2d(256))
        self.layer1 = nn.Sequential(
            _HRBottleneck(64, 64, downsample=down), *[_HRBottleneck(256, 64) for _ in range(3)]
        )
        self.transition1 = nn.ModuleList(
            [
                _conv_bn(256, c[0], 3, 1, relu=True),
                nn.Sequential(_conv_bn(256, c[1], 3, 2, relu=True)),
            ]
        )
        self.stage2 = nn.Sequential(_HRModule(c[:2]))
        self.transition2 = self._transition(c[:2], c[:3])
        self.stage3 = nn.Sequential(*[_HRModule(c[:3]) for _ in range(4)])
        self.transition3 = self._transition(c[:3], c[:4])
        self.stage4 = nn.Sequential(*[_HRModule(c[:4]) for _ in range(3)])
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(mod, nn.BatchNorm2d):
                nn.init.ones_(mod.weight)
                nn.init.zeros_(mod.bias)

    @staticmethod
    def _transition(prev: tuple[int, ...], cur: tuple[int, ...]) -> nn.ModuleList:
        layers = nn.ModuleList(nn.Identity() for _ in prev)
        layers.append(nn.Sequential(_conv_bn(prev[-1], cur[-1], 3, 2, relu=True)))
        return layers

    def forward(self, x):  # ty: ignore[invalid-mutable-override]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)
        xs = [t(x) for t in self.transition1]
        for stage in self.stage2:
            xs = stage(xs)
        xs = [t(xs[min(i, len(xs) - 1)]) for i, t in enumerate(self.transition2)]
        for stage in self.stage3:
            xs = stage(xs)
        xs = [t(xs[min(i, len(xs) - 1)]) for i, t in enumerate(self.transition3)]
        for stage in self.stage4:
            xs = stage(xs)
        size = xs[0].shape[-2:]
        ups = [xs[0]] + [
            nn.functional.interpolate(t, size=size, mode="bilinear", align_corners=False)
            for t in xs[1:]
        ]
        return torch.cat(ups, dim=1)


def _cached_hrnet_state(backbone: str) -> dict | None:
    """Local HRNet ImageNet checkpoint (``*hrnet*w32*.pth`` in the hub cache) or None."""
    ckdir = Path(torch.hub.get_dir()) / "checkpoints"
    tag = backbone.split("_")[-1]  # "w32"
    files = sorted(ckdir.glob(f"*hrnet*{tag}*.pth")) if ckdir.is_dir() else []
    if not files:
        return None
    state = torch.load(files[0], map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    return {k.removeprefix("module."): v for k, v in state.items()}


def cached_imagenet_state(backbone: str) -> dict | None:
    """ImageNet weights of a torchvision model from the local hub cache (no download).

    ``hrnet_w32`` is not in torchvision: a local ``*hrnet*w32*.pth`` is used when
    present, otherwise ``None`` (Kaiming init; the teacher distillation carries
    the transfer) instead of raising.
    """
    if backbone in HRNET_BACKBONES:
        return _cached_hrnet_state(backbone)
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
    """Trunk + stride-4 neck + 1x1 heatmap head.

    * ``resnet*``: torchvision ResNet (stride 32) + 3 deconvolutions (stride 4).
    * ``hrnet_w32``: :class:`HRNetW32` (480 ch at stride 4) + 1x1 conv 480 -> 256.
    """

    def __init__(self, backbone: str = "resnet50", nkp: int = 49, *, imagenet: dict | None = None):
        super().__init__()
        if backbone in HRNET_BACKBONES:
            trunk = HRNetW32()
            if imagenet is not None:
                own = trunk.state_dict()
                match = {k: v for k, v in imagenet.items() if k in own and own[k].shape == v.shape}
                trunk.load_state_dict(match, strict=False)
            self.backbone = trunk
            self.deconv = nn.Sequential(
                nn.Conv2d(HRNetW32.out_channels, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Conv2d(256, nkp, 1)
            for mod in [*self.deconv.modules(), self.head]:
                if isinstance(mod, nn.Conv2d):
                    nn.init.normal_(mod.weight, std=0.001)
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
            return
        self._init_resnet(backbone, nkp, imagenet)

    def _init_resnet(self, backbone: str, nkp: int, imagenet: dict | None) -> None:
        import torchvision

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

    def forward(self, x):  # ty: ignore[invalid-mutable-override]
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
# Teacher (knowledge distillation)
# --------------------------------------------------------------------------- #
def teacher_label(teacher: Any) -> str | None:
    """Printable/serialisable name of a teacher (path string, class name or None)."""
    if teacher is None:
        return None
    if isinstance(teacher, (str, Path)):
        return str(teacher)
    return type(teacher).__name__


def _yolo_teacher_fn(path, input_wh, dev: torch.device, nkp: int):
    from ultralytics import YOLO

    model = YOLO(str(path))
    width, height = input_wh
    device = "cpu" if dev.type == "cpu" else str(dev.index or 0)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    def fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x01 = (x.float() * std + mean).clamp(0.0, 1.0).contiguous()
        results = model.predict(x01, imgsz=(height, width), conf=0.01, device=device, verbose=False)
        xy = torch.zeros((x.shape[0], nkp, 2), device=dev)
        conf = torch.zeros((x.shape[0], nkp), device=dev)
        for b, r in enumerate(results):
            kp, boxes = getattr(r, "keypoints", None), getattr(r, "boxes", None)
            if kp is None or boxes is None or len(boxes) == 0 or kp.xy.shape[-2] != nkp:
                continue
            i = int(boxes.conf.argmax())
            xy[b] = kp.xy[i].to(dev).float()
            if kp.conf is not None:
                conf[b] = kp.conf[i].to(dev).float()
            else:
                conf[b] = float(boxes.conf[i])
        return xy, conf

    return fn


def _heatmap_teacher_fn(pred: HeatmapPredictor, dev: torch.device):
    net = pred.net.to(dev).eval()

    def fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            hm = net(x.to(memory_format=torch.channels_last)).float().cpu().numpy()
        out = [decode_heatmaps(h) for h in hm]
        xy = np.stack([from_cells(uv) for uv, _ in out])
        conf = np.stack([c for _, c in out])
        return (
            torch.as_tensor(xy, dtype=torch.float32, device=dev),
            torch.as_tensor(conf, dtype=torch.float32, device=dev),
        )

    return fn


def build_teacher_fn(teacher: Any, input_wh, dev: torch.device, nkp: int, stride: int = STRIDE):
    """``fn(x normalised (B,3,H,W)) -> (uv cells (B,K,2), conf (B,K))`` for the student grid.

    ``teacher``: an Ultralytics YOLO-pose ``.pt`` path (e.g. the promoted
    ``freekiki_l.pt``), a FreeKiki heatmap checkpoint path, a
    :class:`HeatmapPredictor`, or any callable ``teacher(x) -> (xy canvas px,
    conf)``. The teacher runs frozen (``no_grad``) on the exact augmented
    canvas the student sees, so its keypoint ``k`` is already in the flipped
    label convention of that canvas (YOLO-pose trains with the same
    ``flip_idx`` swap). Returns ``None`` when ``teacher`` is None.
    """
    if teacher is None:
        return None
    if isinstance(teacher, HeatmapPredictor):
        raw = _heatmap_teacher_fn(teacher, dev)
    elif isinstance(teacher, (str, Path)):
        path = Path(teacher)
        if not path.is_file():
            raise FileNotFoundError(f"teacher model not found: {path}")
        if model_backend(path) == "heatmap":
            raw = _heatmap_teacher_fn(HeatmapPredictor.load(path, dev), dev)
        else:
            raw = _yolo_teacher_fn(path, input_wh, dev, nkp)
    elif callable(teacher):
        raw = teacher
    else:
        raise TypeError(f"unsupported teacher: {teacher!r}")

    def fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            xy, conf = raw(x)
        xy = torch.as_tensor(xy, dtype=torch.float32, device=x.device).clone()
        conf = torch.as_tensor(conf, dtype=torch.float32, device=x.device).clone()
        return (xy + 0.5) / stride - 0.5, conf

    return fn


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
    """Append one epoch row; an existing file keeps its own header (older runs resume aligned)."""
    fields = RESULTS_FIELDS
    new = not path.is_file()
    if not new:
        with path.open(encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        fields = header or RESULTS_FIELDS
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
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
    teacher: Any = None,
    distill_weight: float = DISTILL_WEIGHT,
    teacher_min_conf: float = TEACHER_MIN_CONF,
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

    ``teacher`` (see :func:`build_teacher_fn`) enables knowledge distillation:
    ``L = L_gt + distill_weight * L_distill`` with the frozen teacher evaluated
    on the same augmented batch. ``distill_weight <= 0`` or no teacher -> GT only.
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
        if teacher is None and a.get("teacher") and Path(a["teacher"]).is_file():
            teacher = a["teacher"]
        distill_weight = float(a.get("distill_weight", distill_weight if teacher else 0.0))
        teacher_min_conf = float(a.get("teacher_min_conf", teacher_min_conf))
    default_batch = DEFAULTS["batch_hrnet"] if backbone in HRNET_BACKBONES else DEFAULTS["batch"]
    batch = int(batch or default_batch)
    workers = int(DEFAULTS["workers"] if workers is None else workers)
    torch.manual_seed(seed)
    input_wh = input_size(imgsz)
    nkp = len(flip_idx)
    imagenet = None
    if not resume and init is None and pretrained:
        imagenet = cached_imagenet_state(backbone)
        if imagenet is None:
            log(
                f"{backbone}: no local ImageNet checkpoint in the torch hub cache -> Kaiming "
                "init (transfer comes from teacher distillation)"
            )
    net = KikiHeatNet(backbone, nkp, imagenet=imagenet)
    if resume:
        net.load_state_dict(ckpt["model"])
    elif init is not None:
        start = torch.load(init, map_location="cpu", weights_only=False)
        if start.get("backend") != BACKEND or start.get("backbone") != backbone:
            raise ValueError(f"--base {init} is not a {backbone} heatmap checkpoint")
        net.load_state_dict(start["model"])
    net = net.to(dev).to(memory_format=torch.channels_last)  # ty: ignore[no-matching-overload]
    distill_weight = float(distill_weight)
    teacher_fn = build_teacher_fn(teacher, input_wh, dev, nkp) if distill_weight > 0 else None
    if teacher_fn is not None:
        log(
            f"distillation: teacher {teacher_label(teacher)} | lambda {distill_weight:g} | "
            f"min conf {teacher_min_conf:g}"
        )

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
        "model": f"{backbone}-{'imagenet' if imagenet is not None else 'scratch'}"
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
        "teacher": teacher_label(teacher) if teacher_fn is not None else None,
        "distill_weight": distill_weight if teacher_fn is not None else 0.0,
        "teacher_min_conf": float(teacher_min_conf),
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
        total, total_gt, total_kd, n = 0.0, 0.0, 0.0, 0
        for step, (x, uv, vis) in enumerate(loader, 1):
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            uv, vis = uv.to(dev, non_blocking=True), vis.to(dev, non_blocking=True)
            t_uv = t_conf = None
            if teacher_fn is not None:
                t_uv, t_conf = teacher_fn(x)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                out = net(x)
            loss, loss_gt, loss_kd = compute_distill_loss(
                out,
                uv,
                vis,
                t_uv,
                t_conf,
                distill_weight=distill_weight,
                min_teacher_conf=teacher_min_conf,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total, n = total + loss.item(), n + 1
            total_gt, total_kd = total_gt + loss_gt.item(), total_kd + loss_kd.item()
            if step % 200 == 0:
                log(
                    f"  epoch {epoch}/{epochs} batch {step}/{len(loader)} loss {total / n:.3e} "
                    f"(gt {total_gt / n:.3e} kd {total_kd / n:.3e})"
                )
        row: dict[str, Any] = {
            "epoch": epoch,
            "lr": f"{sched.get_last_lr()[0]:.3e}",
            "train/loss": f"{total / max(1, n):.4e}",
            "train/loss_gt": f"{total_gt / max(1, n):.4e}",
            "train/loss_distill": f"{total_kd / max(1, n):.4e}",
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
