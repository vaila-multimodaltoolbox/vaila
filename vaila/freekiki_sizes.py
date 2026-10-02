"""
================================================================================
Script: freekiki_sizes.py - FreeKiki model size ladder (n/s/m/l/x) for YOLO26-pose
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.7
Created: 28 September 2026
Update Date: 02 October 2026

Description:
    Helpers behind ``freekiki.py sizes`` / ``freekiki.py grow`` for the
    Ultralytics backend slots ``models/freekiki_{n,s,m,l,x}.pt``:

      * audit: for every YOLO26-pose scale, the fraction of its parameters that
        match a trained source checkpoint by name AND shape. A shape match is
        not a semantic match: between scales of different width the matching
        tensors (capped-width layers, the ``kpt_shape`` = 49*3 head convs) are
        fed by differently shaped features. So a trained m does not seed n/s/x;
        those start from the official ``yolo26{n,s,x}-pose.pt``.
      * grow: m -> l only. Both scales have the same width multiplier; l is
        deeper. Every tensor that exists in m is copied. Where l concatenates
        more C3k2 outputs, the extra input channels of the fusion conv are
        zero-initialised, and every new sequential residual block (PSABlock in
        C2PSA) gets the last BatchNorm of its attention and FFN branches zeroed,
        so it starts as the identity. l then computes exactly m's function at
        step 0 (Net2Net-style deepening); training switches the new blocks on.
        The maximum output difference on a random image is measured and stored
        in the checkpoint (``freekiki.function_preserving_maxdiff``).

    The grown checkpoint is only an initialisation: it must be trained
    (``freekiki.py train --base models/freekiki_l_init.pt``) and pass the
    promotion gate before it fills the ``l`` slot.

Usage:
    uv run --no-sync vaila/freekiki.py sizes -w WORKSPACE [--src m]
    uv run --no-sync vaila/freekiki.py grow  -w WORKSPACE --src m --to l \
        --out models/freekiki_l_init.pt

License:
    GNU Affero General Public License v3.0
"""

from __future__ import annotations

import copy
import hashlib
import os
from datetime import datetime
from pathlib import Path

SCALES = ("n", "s", "m", "l", "x")
GROWABLE = {("m", "l")}  # same width multiplier, deeper target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source(path):
    """``(checkpoint, float model on CPU)`` of an Ultralytics pose checkpoint."""
    import torch

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    model = ckpt.get("ema") or ckpt.get("model")
    if model is None or not hasattr(model, "yaml"):
        raise ValueError(f"{path} is not an Ultralytics pose checkpoint.")
    return ckpt, model.float().eval()


def source_scale(model) -> str:
    scale = str(model.yaml.get("scale", ""))
    if scale not in SCALES:
        raise ValueError(f"Unknown YOLO26 scale {scale!r} in the source checkpoint.")
    return scale


def build(scale: str, like):
    """Untrained YOLO26-pose of ``scale`` with the keypoints/classes of ``like``."""
    import ultralytics
    import yaml
    from ultralytics.nn.tasks import PoseModel

    cfg_path = Path(ultralytics.__file__).parent / "cfg" / "models" / "26" / "yolo26-pose.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["scale"] = scale
    cfg["kpt_shape"] = list(like.yaml.get("kpt_shape", [49, 3]))
    nc = int(getattr(like, "nc", 0) or like.yaml.get("nc", 1))
    model = PoseModel(cfg, nc=nc, verbose=False)
    model.names = getattr(like, "names", {0: "field"})  # ty: ignore[invalid-assignment]
    return model


def audit(src) -> list[dict]:
    """Name+shape overlap of ``src`` with every scale (read only)."""
    _, model = load_source(src)
    sd = model.state_dict()
    rows = []
    for scale in SCALES:
        dst = build(scale, model).state_dict()
        floats = {k: v for k, v in dst.items() if v.dtype.is_floating_point}
        total = sum(v.numel() for v in floats.values())
        hit = sum(v.numel() for k, v in floats.items() if k in sd and sd[k].shape == v.shape)
        rows.append({"scale": scale, "params_m": total / 1e6, "match": hit / total})
    return rows


def _max_diff(a, b, imgsz=(320, 576)) -> float:
    import torch

    x = torch.randn(1, 3, *imgsz, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ya, yb = a(x), b(x)
    ya = ya[0] if isinstance(ya, tuple | list) else ya
    yb = yb[0] if isinstance(yb, tuple | list) else yb
    return float((ya - yb).abs().max())


def grow(src, out, *, to: str = "l") -> dict:
    """Deepen a trained ``m`` into a function-preserving ``to`` initialisation at ``out``."""
    import torch
    import ultralytics
    from ultralytics.nn.modules.block import PSABlock

    src, out = Path(src), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; choose another --out (it is never overwritten).")
    ckpt, model = load_source(src)
    scale = source_scale(model)
    if (scale, to) not in GROWABLE:
        raise ValueError(
            f"{scale} -> {to}: widths differ, weights cannot be carried over. "
            f"Train {to} from yolo26{to}-pose.pt instead."
        )
    dst = build(to, model)
    sd, dd = model.state_dict(), dst.state_dict()
    stats = {"copied": 0, "padded": 0, "fresh": 0, "identity_blocks": 0}
    for key, value in dd.items():
        old = sd.get(key)
        if old is None:
            stats["fresh"] += 1
        elif old.shape == value.shape:
            dd[key] = old.clone()
            stats["copied"] += 1
        elif (
            old.ndim == value.ndim == 4
            and old.shape[0] == value.shape[0]
            and old.shape[2:] == value.shape[2:]
            and old.shape[1] < value.shape[1]
        ):
            grown = torch.zeros_like(value)  # the concat grew: old channels come first
            grown[:, : old.shape[1]] = old
            dd[key] = grown
            stats["padded"] += 1
        else:
            stats["fresh"] += 1
    dst.load_state_dict(dd)
    old_modules = {name for name, _ in model.named_modules()}
    for name, module in dst.named_modules():
        if isinstance(module, PSABlock) and name not in old_modules:
            for bn in (module.attn.proj.bn, module.ffn[-1].bn):
                torch.nn.init.zeros_(bn.weight)
                torch.nn.init.zeros_(bn.bias)
            stats["identity_blocks"] += 1
    dst.eval()
    stats["max_diff"] = _max_diff(model, dst)
    stats["params_m"] = sum(p.numel() for p in dst.parameters()) / 1e6
    lineage = {
        "grown_from": str(src),
        "grown_from_sha256": _sha256(src),
        "grown_from_freekiki": ckpt.get("freekiki"),
        "scale": to,
        "function_preserving_maxdiff": stats["max_diff"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(
        {
            "epoch": -1,
            "best_fitness": None,
            "model": copy.deepcopy(dst).half(),
            "ema": None,
            "updates": None,
            "optimizer": None,
            "train_args": {"task": "pose"},  # without it Ultralytics guesses "detect"
            "date": datetime.now().isoformat(),
            "version": ultralytics.__version__,
            "freekiki": lineage,
        },
        tmp,
    )
    os.replace(tmp, out)
    return stats | {"out": str(out), "from_scale": scale, "to_scale": to}
