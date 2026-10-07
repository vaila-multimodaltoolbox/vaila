"""
================================================================================
Script: freekiki.py - FreeKiki soccer-field keypoints (49-point kiki template)
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.7
Created: 25 September 2026
Update Date: 06 October 2026

Description:
    FreeKiki (Soccer Tools) trains, retrains and runs a YOLO-pose network that
    finds the 49 soccer-field keypoints of ``vaila/models/soccerfield_kiki.csv``
    (pitch lines, goal posts/nets, corner flags, center) in broadcast video.

    Everything lives in one portable *workspace* folder, so it can be copied to
    another machine and pointed at for new trainings:

        <workspace>/
          freekiki.toml          settings + [models] slots (paths relative)
          spec/                  soccerfield_kiki.csv + soccerfield_kiki49.json
          datasets/kiki49/       imported YOLO-pose dataset (data.yaml, images, labels)
          manifests/vNNN/        oversampled train lists (immutable, see ``manifest``)
          runs/<name>/           Ultralytics training runs
          models/registry.csv    every finished run with its best-epoch metrics
          models/freekiki_<slot>.pt  best model per size: n s m l x (YOLO26-pose),
                                 hm_n..hm_x (heatmap ResNet18..152)
          models/promotion_log.csv  every promotion decision and its reasons
          outputs/               evaluations, sweeps, audits, benchmarks

    Model slots (settings schema 2): one promoted model per network size,
    ``[models.<slot>]`` in freekiki.toml records file, backend, arch, base,
    run, pose mAP50-95, PCK10 and SHA-256. ``active`` is an alias of
    ``[models] default`` (``models --default l`` switches it). A schema-1
    workspace (``[active]`` + ``models/active.pt``) is migrated on first load:
    ``active.pt`` is copied (SHA-256 checked) to ``models/freekiki_m.pt`` and
    kept. A finished run competes only for its own slot; an empty slot is
    filled when the run passes ``new_slot_min_pose_map50_95``. ``grow`` deepens
    a trained m into a function-preserving l initialisation
    (:mod:`freekiki_sizes`); the other sizes start from ``yolo26*-pose.pt``.

    Retrain starts from ``active`` (the default slot). Any non-official base
    (a slot, a run's best.pt, a grown init) is a continued fine-tune that uses
    AdamW at ``lr0=1e-4`` (cosine down to ``1e-6``), one warmup epoch and
    ``mosaic=0``. The checkpoint already finished its previous run with mosaic
    off and a learning rate near ``2e-4``; ``lr0=0.001`` plus mosaic dropped
    pose mAP50-95 from 0.857 to 0.601 in one epoch. The backbone stays
    trainable: the set is large and already the same field domain. A new run
    replaces its slot model only when it passes the promotion gate
    (``[promotion]`` in freekiki.toml, mode ``gate``): candidate and slot model are
    evaluated on the same val images and thresholds; pose mAP50-95, overall
    PCK (misses count as failures), the rate of valid field homographies and
    the recall / median error of the critical keypoints must not get worse
    beyond the configured tolerances. ``runs/<name>/train_manifest.json``
    records seed, arguments, versions and dataset/weights hashes.

    A training stopped by the Stop button, a crash or a power loss keeps
    ``runs/<name>/weights/last.pt`` (saved after every epoch); ``resume``
    continues it from the last completed epoch and then registers it.

    ``manifest`` writes ``manifests/vNNN/`` (never inside the dataset): a train
    image list where images with rare keypoints (near-side corners, corner
    flag bases) repeat, by repeat-factor sampling. ``train --manifest vNNN``
    trains on it; val/test stay the dataset's, so evaluations stay comparable.
    The list is dataset-relative. At train time it is expanded to absolute
    paths under ``runs/<name>/sampling/``. Ultralytics may rebuild the derived
    ``labels/train.cache``; images and labels are not modified.

    Evaluate measures a model on a labelled split (default ``val``; use
    ``test`` only for the final report): pose mAP plus, per keypoint, counts of
    reference / predicted / matched / missed points, precision and recall with
    an explicit match radius, error mean / median / P90 / P95 in px@1920, PCK
    over matches and PCK with misses as failures, identity swaps and the
    homography criteria (``freekiki_diag.py``). Raw predictions are kept, so
    ``sweep`` re-scores thresholds offline. ``audit`` checks the dataset
    read-only (points, visibility, mirrored labels, leakage).

    Detect writes, per video, a getpixelvideo-compatible wide CSV of the
    accepted points (``frame,p0_x,p0_y,...,p48_x,p48_y``; 0-based), the raw
    predictions, a per-point status CSV (detected / rejected with the reason /
    interpolated), frame issues (cuts, homography status), label-free
    ``quality.json`` (not accuracy), annotated diagnostic frames and an
    optional overlay MP4.

Usage:
    uv run vaila/freekiki.py                       # GUI
    uv run vaila/freekiki.py init   -w WORKSPACE
    uv run vaila/freekiki.py import-dataset -w WORKSPACE --src /path/kiki49_dataset
    uv run vaila/freekiki.py check  -w WORKSPACE
    uv run vaila/freekiki.py manifest -w WORKSPACE                  # manifests/vNNN (no training)
    uv run vaila/freekiki.py train  -w WORKSPACE --manifest v001 --base active
    uv run vaila/freekiki.py train  -w WORKSPACE --base yolo26m-pose.pt --epochs 150 --imgsz 1280
    uv run vaila/freekiki.py train  -w WORKSPACE --base active          # retrain / fine-tune
    uv run vaila/freekiki.py status -w WORKSPACE                       # runs + resumable ones
    uv run vaila/freekiki.py resume -w WORKSPACE [--name RUN]          # continue after stop/power loss
    uv run vaila/freekiki.py evaluate -w WORKSPACE                     # quality on val split
    uv run vaila/freekiki.py evaluate -w WORKSPACE --split test        # final report only
    uv run vaila/freekiki.py sweep  -w WORKSPACE --eval-dir outputs/processed_freekiki_eval_val_TS
    uv run vaila/freekiki.py compare -w WORKSPACE --candidate models/kiki49_RUN.pt [--promote]
    uv run vaila/freekiki.py models -w WORKSPACE [--default l]         # slots; 'active' = default
    uv run vaila/freekiki.py sizes  -w WORKSPACE [--src m]             # overlap with n/s/m/l/x
    uv run vaila/freekiki.py grow   -w WORKSPACE --src m --to l        # models/freekiki_l_init.pt
    uv run vaila/freekiki.py audit  -w WORKSPACE                       # read-only dataset audit
    uv run vaila/freekiki.py bench  -w WORKSPACE --batches 2,8,16      # speed / peak VRAM
    uv run vaila/freekiki.py detect -w WORKSPACE --video match.mp4 [--output-dir DIR]
    uv run vaila/freekiki.py detect -w WORKSPACE --video FOLDER_OF_VIDEOS [--fill-gaps 3]
    uv run vaila/freekiki.py queue  -w WORKSPACE --batch DETECT_BATCH   # frames to label
    uv run vaila/freekiki.py ingest -w WORKSPACE --src incoming/SESSION --match-id M [--split hard] [--commit]
    uv run vaila/freekiki.py evaluate -w WORKSPACE --split hard        # labelled holdout, report only

    Labeling for a retrain: ``queue`` drafts the frames worth labelling
    (not calibratable first, then half-seen rare points); getpixelvideo shows
    AI and homography ghosts; only complete frames (every visible point
    labelled or hidden) are exported/ingested, because an empty point is a
    "not visible" label. ``--split hard`` is a labelled holdout never trained on.

    On NVIDIA/CUDA machines use ``uv run --no-sync`` (see CLAUDE.md).

License:
    GNU Affero General Public License v3.0 (AGPLv3).
================================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unicodedata
import webbrowser
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import toml
import yaml

try:
    from . import freekiki_diag as diag
    from .cli_highlight import print_gui_cli_mirror
except ImportError:
    import freekiki_diag as diag  # ty: ignore[unresolved-import]
    from cli_highlight import print_gui_cli_mirror  # ty: ignore[unresolved-import]

VAILA_DIR = Path(__file__).resolve().parent
SCHEMA_CSV = VAILA_DIR / "models" / "soccerfield_kiki.csv"
SKELETON_JSON = VAILA_DIR / "skeletons" / "soccerfield_kiki49.json"
HELP_HTML = VAILA_DIR / "help" / "freekiki.html"
USER_SETTINGS = Path.home() / ".vaila" / "freekiki.toml"

NKP = 49
CONFIG_NAME = "freekiki.toml"
DATASET_NAME = "kiki49"
LEGACY_ACTIVE_MODEL = "models/active.pt"  # settings schema 1 (kept on disk, never deleted)
# Model slots (settings schema 2): one promoted model per size, per backend.
# ``active`` is an alias of ``[models] default``.
MODEL_SCALES = ("n", "s", "m", "l", "x")
HEATMAP_SLOTS = {
    "resnet18": "hm_n",
    "resnet34": "hm_s",
    "resnet50": "hm_m",
    "resnet101": "hm_l",
    "resnet152": "hm_x",
}
SLOTS = MODEL_SCALES + tuple(HEATMAP_SLOTS.values())
DEFAULT_SLOT = "m"
OFFICIAL_BASE_RE = re.compile(r"^yolo26([nsmlx])-pose\.pt$")
REGISTRY_CSV = "models/registry.csv"
REGISTRY_FIELDS = [
    "date",
    "run",
    "base",
    "dataset",
    "epochs",
    "imgsz",
    "best_epoch",
    "pose_map50",
    "pose_map50_95",
    "model",
    "promoted",
    "fitness",
    "backend",
    "slot",
    "arch",
]
# Parts of a kiki49 build needed for training; preview/, frames/ and
# tracking_eval/ are build by-products and stay behind.
DATASET_PARTS = (
    "data.yaml",
    "images",
    "labels",
    "manifest.csv",
    "keypoints_kiki49.csv",
    "cameras.jsonl",
    "reports",
)
BASE_MODELS = (
    "yolo26n-pose.pt",
    "yolo26s-pose.pt",
    "yolo26m-pose.pt",
    "yolo26l-pose.pt",
    "yolo26x-pose.pt",
    "active",
    *(f"freekiki_{s}" for s in MODEL_SCALES),
)
DEFAULT_SETTINGS = {
    "workspace": {"schema": "soccerfield_kiki49", "dataset": f"datasets/{DATASET_NAME}"},
    "models": {"default": DEFAULT_SLOT},
    "train": {
        "base": "yolo26m-pose.pt",
        "epochs": 150,
        "imgsz": 1280,
        "batch": -1,
        "patience": 30,
    },
    "detect": {"conf": 0.25, "kp_conf": 0.5, "imgsz": 1280},
    # Field geometry (freekiki_geom.GEOM_DEFAULTS has every parameter): detect
    # fills / fixes keypoints with the camera fitted to the accepted ones.
    "geometry": {"mode": "fill"},
    # new_slot_min_pose_map50_95: an empty size slot (e.g. the first ``l``)
    # takes a YOLO run only above this best-epoch pose mAP50-95.
    "promotion": dict(diag.PROMOTION_DEFAULTS) | {"new_slot_min_pose_map50_95": 0.5},
}
# Continued fine-tune from a converged/custom checkpoint (a slot model, a
# run's best.pt or a grown ``freekiki_sizes`` init; any base that is not an
# official yolo26*-pose.pt). The 150-epoch m run ended with
# mosaic off and param-group lrs near 1.7e-4..5e-4 (pose mAP50-95 0.857,
# pose precision 0.954). AdamW lr0=0.001 with mosaic=1.0 then scored 0.601 /
# 0.871 after one epoch. Stay under that final lr and keep mosaic off; the
# rare-point signal comes from the repeat-factor list. Backbone stays
# trainable (large set, same domain).
CONTINUE_FINETUNE_ARGS = {
    "optimizer": "AdamW",
    "lr0": 1e-4,
    "lrf": 0.01,
    "warmup_epochs": 1.0,
    "cos_lr": True,
    "mosaic": 0.0,
}
ACTIVE_FINETUNE_ARGS = CONTINUE_FINETUNE_ARGS  # schema-1 name
MAP50_COL = "metrics/mAP50(P)"
MAP50_95_COL = "metrics/mAP50-95(P)"
BOX_MAP50_95_COL = "metrics/mAP50-95(B)"
HEATMAP_FITNESS_COL = "val/pck10_all"  # = freekiki_heatmap.FITNESS_COL
HEATMAP_VAL_IMAGES = 500  # val images scored after every heatmap epoch (evenly spaced)
PROMOTION_LOG = "models/promotion_log.csv"
# evaluate: raw predictions are kept down to this box confidence so the
# det_conf / kp_conf thresholds can be applied (and swept) offline.
RAW_CONF = 0.01
# Review ghosts: AI points kept below kp_conf down to this confidence, and the
# distance (px@1920) within which an AI ghost and a homography ghost agree.
SUGGEST_MIN_CONF = 0.1
SUGGEST_AGREE_PX = 15.0
# Label queue: the rare points whose low-confidence frames are asked first.
RARE_KPS = (5, 29, 39, 47)
# Splits ingest can write: train, and hard = labelled holdout never trained on.
INGEST_SPLITS = ("train", "hard")
PRED_KEYS = (
    "images", "groups", "sources", "width", "height", "box_conf",
    "pred_xy", "pred_kc", "gt_xy", "gt_vis",
)  # fmt: skip
SWEEP_COLS = (
    "det_conf", "kp_conf", "images_with_detection", "n_ref", "n_pred", "n_tp",
    "n_mislocalized", "n_fn", "n_fp", "precision", "recall", "err_median",
    "pck5_all", "pck10_all", "pck25_all",
)  # fmt: skip


def _log(message: str) -> None:
    print(f"[freekiki] {message}", flush=True)


# --------------------------------------------------------------------------- #
# Keypoint schema
# --------------------------------------------------------------------------- #
def load_schema(csv_path: Path = SCHEMA_CSV) -> tuple[list[str], list[int]]:
    """Return ``(point names, flip_idx)`` from the kiki field CSV (0-based order)."""
    with Path(csv_path).open(encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["point_number"]))
    numbers = [int(r["point_number"]) for r in rows]
    names = [r["point_name"] for r in rows]
    flips = [int(r["flip_idx"]) for r in rows]
    if numbers != list(range(NKP)) or len(set(names)) != NKP or any(not n for n in names):
        raise ValueError("Invalid Kiki49 point indices or names")
    if sorted(flips) != list(range(NKP)) or any(flips[flips[i]] != i for i in range(NKP)):
        raise ValueError("Invalid Kiki49 flip_idx")
    return names, flips


def load_bones(json_path: Path = SKELETON_JSON) -> list[tuple[int, int]]:
    """Bones of the kiki49 skeleton as 0-based index pairs."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return [(int(a[1:]), int(b[1:])) for a, b in data["connections"]]


# --------------------------------------------------------------------------- #
# Workspace
# --------------------------------------------------------------------------- #
def load_settings(ws: Path) -> dict:
    path = Path(ws) / CONFIG_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Not a FreeKiki workspace (no {CONFIG_NAME}): {ws}")
    settings = toml.load(path)
    if _migrate_settings(Path(ws), settings):
        save_settings(ws, settings)
    for section, values in DEFAULT_SETTINGS.items():
        merged = dict(values)
        merged.update(settings.get(section, {}))
        settings[section] = merged
    return settings


def save_settings(ws: Path, settings: dict) -> None:
    (Path(ws) / CONFIG_NAME).write_text(toml.dumps(settings), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Model slots (settings schema 2)
# --------------------------------------------------------------------------- #
def slot_file(slot: str) -> str:
    return f"models/freekiki_{slot}.pt"


def _slot_name(name: str) -> str | None:
    """``l`` / ``freekiki_l`` / ``hm_m`` -> slot name; anything else -> None."""
    name = str(name).removeprefix("freekiki_")
    return name if name in SLOTS else None


def slot_path(ws, settings: dict, name: str = "active") -> Path:
    """Workspace file of a slot; ``active`` is the ``[models] default`` slot."""
    models = settings["models"]
    slot = models["default"] if name == "active" else _slot_name(name)
    if slot is None:
        raise ValueError(f"Unknown model slot {name!r} (active, {', '.join(SLOTS)})")
    return Path(ws) / models.get(slot, {}).get("file", slot_file(slot))


def _load_checkpoint(path) -> dict | None:
    """A torch checkpoint dict, or None for a missing/foreign/unreadable file."""
    import torch

    if not Path(path).is_file():
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return ckpt if isinstance(ckpt, dict) else None


def checkpoint_slot(path) -> tuple[str, str] | None:
    """``(slot, arch)`` from a checkpoint's own metadata; None when it cannot be read."""
    ckpt = _load_checkpoint(path)
    if ckpt is None:
        return None
    if str(ckpt.get("backend", "")).startswith("freekiki_heatmap"):
        backbone = str(ckpt.get("backbone", ""))
        slot = HEATMAP_SLOTS.get(backbone)
        return (slot, f"heatmap-{backbone}") if slot else None
    net = ckpt.get("ema") or ckpt.get("model")
    scale = (getattr(net, "yaml", None) or {}).get("scale")
    return (scale, f"yolo26{scale}-pose") if scale in MODEL_SCALES else None


def base_slot(base: str, default: str = DEFAULT_SLOT) -> tuple[str, str]:
    """``(slot, arch)`` implied by a train ``--base`` name (``active`` -> ``default``)."""
    official = OFFICIAL_BASE_RE.match(Path(str(base)).name)
    slot = official[1] if official else default if base == "active" else _slot_name(base)
    slot = slot or default
    return slot, f"yolo26{slot}-pose" if slot in MODEL_SCALES else ""


def run_slot(run_dir, base: str, default: str = DEFAULT_SLOT) -> tuple[str, str]:
    """Slot a finished run competes in: its best.pt metadata, else args.yaml, else ``base``."""
    run_dir = Path(run_dir)
    found = checkpoint_slot(run_dir / "weights" / "best.pt")
    if found:
        return found
    args_yaml = run_dir / "args.yaml"
    args = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) if args_yaml.is_file() else {}
    if (args or {}).get("backend") == "heatmap":
        backbone = str(args.get("backbone", "resnet50"))
        return HEATMAP_SLOTS.get(backbone, "hm_m"), f"heatmap-{backbone}"
    return base_slot(base, default)


def _copy_verified(src: Path, dst: Path) -> str:
    """Copy through a temporary file; the SHA-256 of the result must match ``src``."""
    sha = file_sha256(src)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    if file_sha256(tmp) != sha:
        tmp.unlink(missing_ok=True)
        raise OSError(f"Copy of {src} to {dst} is corrupt (SHA-256 mismatch)")
    tmp.replace(dst)
    return sha


def _migrate_settings(ws: Path, raw: dict) -> bool:
    """Schema 1 ``[active]`` -> schema 2 ``[models]`` slots. Idempotent; copies, never moves.

    ``models/active.pt`` is copied (SHA-256 checked) to ``models/freekiki_<slot>.pt``,
    the slot read from the checkpoint (``m`` for the kiki49 YOLO26m). The old file
    and the old ``[active]`` table (as ``[legacy_active]``) stay; the old file is
    only read, so a training that uses it keeps running.
    """
    if "models" in raw:
        return False
    old = dict(raw.get("active") or {})
    src = ws / str(old.get("model", LEGACY_ACTIVE_MODEL))
    models: dict = {"default": DEFAULT_SLOT}
    if src.is_file():
        slot, arch = checkpoint_slot(src) or (DEFAULT_SLOT, "")
        dst = ws / slot_file(slot)
        if dst.is_file():
            sha = file_sha256(dst)
            if sha != file_sha256(src):
                raise FileExistsError(
                    f"{dst} exists and differs from {src}; move one of them, then reopen."
                )
        else:
            sha = _copy_verified(src, dst)
        models = {
            "default": slot,
            slot: {
                "file": slot_file(slot),
                "backend": "heatmap" if slot.startswith("hm_") else "yolo",
                "arch": arch,
                "base": "",
                "run": old.get("run", ""),
                "pose_map50_95": old.get("pose_map50_95", 0.0),
                "pck10_all": "",
                "sha256": sha,
            },
        }
        _log(f"settings schema 2: {src.name} copied to {slot_file(slot)} (slot {slot}; kept)")
    raw["models"] = models
    if raw.pop("active", None) is not None:
        raw["legacy_active"] = old
    return True


def _install_slot(ws: Path, settings: dict, slot: str, src: Path, info: dict) -> Path:
    """Copy ``src`` into a slot and record it; an empty default slot moves to ``slot``."""
    dst = ws / slot_file(slot)
    default_empty = not slot_path(ws, settings).is_file()
    sha = _copy_verified(src, dst)
    backend = "heatmap" if slot.startswith("hm_") else "yolo"
    entry: dict = {"file": slot_file(slot), "backend": backend, "arch": "", "base": "", "run": ""}
    entry |= {"pose_map50_95": 0.0, "pck10_all": ""} | info | {"sha256": sha}
    settings["models"][slot] = entry
    if default_empty:
        settings["models"]["default"] = slot
    settings.pop("active", None)
    save_settings(ws, settings)
    return dst


def list_models(ws, *, default: str | None = None) -> list[dict]:
    """Print the model slots; ``default`` sets which slot ``active`` means."""
    ws = Path(ws).expanduser().resolve()
    settings = load_settings(ws)
    models = settings["models"]
    if default:
        slot = _slot_name(default)
        if slot is None or not slot_path(ws, settings, slot).is_file():
            raise ValueError(f"Slot {default!r} has no model; it cannot be the default.")
        models["default"] = slot
        save_settings(ws, settings)
    rows = []
    for slot in SLOTS:
        path = slot_path(ws, settings, slot)
        entry = models.get(slot) or {}
        if not path.is_file():
            continue
        rows.append({"slot": slot, "default": slot == models["default"], **entry})
        score = entry.get("pck10_all") or entry.get("pose_map50_95")
        _log(
            f"{'*' if slot == models['default'] else ' '} {slot:<5} {entry.get('arch', ''):<18} "
            f"run {entry.get('run', '')}  score {score}  {path.relative_to(ws)}"
        )
    if not rows:
        _log("no promoted models yet")
    return rows


def _sizes():
    """``freekiki_sizes`` (lazy: it imports torch/Ultralytics)."""
    try:
        from . import freekiki_sizes as fs
    except ImportError:
        import freekiki_sizes as fs  # ty: ignore[unresolved-import]
    return fs


def model_sizes(ws, src: str = "active") -> list[dict]:
    """Print how much of every YOLO26 scale a trained model can seed (read only)."""
    path = resolve_model(Path(ws).expanduser().resolve(), src)
    rows = _sizes().audit(path)
    for row in rows:
        _log(
            f"{row['scale']}: {row['params_m']:6.2f} M params | name+shape match {row['match']:.1%}"
        )
    _log("only m -> l keeps the function (grow); n/s/x train from yolo26{n,s,x}-pose.pt")
    return rows


def grow_model(ws, *, src: str = "m", to: str = "l", out: str = "models/freekiki_l_init.pt"):
    """Function-preserving m -> l initialisation (see :mod:`freekiki_sizes`); no training."""
    ws = Path(ws).expanduser().resolve()
    source = resolve_model(ws, src)
    target = Path(out).expanduser()
    target = target if target.is_absolute() else ws / target
    stats = _sizes().grow(source, target, to=to)
    _log(
        f"grow {stats['from_scale']} -> {stats['to_scale']}: copied {stats['copied']}, "
        f"padded {stats['padded']}, fresh {stats['fresh']}, identity blocks "
        f"{stats['identity_blocks']}, {stats['params_m']:.2f} M params; "
        f"max |m(x) - l(x)| = {stats['max_diff']:.2e} -> {target}"
    )
    _log(
        f"next: uv run --no-sync vaila/freekiki.py train -w {ws} --base {target} "
        "--manifest v001 --epochs 60 --imgsz 1280"
    )
    return stats


def init_workspace(ws) -> Path:
    """Create (or complete) the portable workspace layout. Never deletes anything."""
    ws = Path(ws).expanduser().resolve()
    for sub in ("spec", "datasets", "runs", "models", "outputs"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    for src in (SCHEMA_CSV, SKELETON_JSON):
        shutil.copy2(src, ws / "spec" / src.name)
    if not (ws / CONFIG_NAME).is_file():
        save_settings(ws, json.loads(json.dumps(DEFAULT_SETTINGS)))
    _log(f"workspace ready: {ws}")
    return ws


def dataset_dir(ws) -> Path:
    return Path(ws) / load_settings(ws)["workspace"]["dataset"]


def refresh_yaml_path(yaml_path: Path) -> Path:
    """Point data.yaml ``path:`` at its own folder (Ultralytics needs it absolute).

    Called on import and before every train, so the workspace keeps working
    after being copied or moved to another disk/machine.
    """
    yaml_path = Path(yaml_path)
    text = yaml_path.read_text(encoding="utf-8")
    line = f'path: "{yaml_path.parent.resolve().as_posix()}"'
    new_text, count = re.subn(r"(?m)^path:.*$", line, text, count=1)
    if count == 0:
        new_text = line + "\n" + text
    if new_text != text:
        yaml_path.write_text(new_text, encoding="utf-8")
    return yaml_path


def _copy_tree_resumable(src: Path, dst: Path, counter: list[int]) -> None:
    """Copy files that are missing or differ in size (safe to re-run after a stop)."""
    for root, _dirs, files in os.walk(src):
        target_root = dst / Path(root).relative_to(src)
        target_root.mkdir(parents=True, exist_ok=True)
        for fname in files:
            if fname.endswith(".cache"):
                continue  # Ultralytics label caches are rebuilt per machine
            s, d = Path(root) / fname, target_root / fname
            if not d.exists() or d.stat().st_size != s.stat().st_size:
                shutil.copy2(s, d)
            counter[0] += 1
            if counter[0] % 1000 == 0:
                _log(f"  {counter[0]} files ...")


def import_dataset(ws, src) -> Path:
    """Copy a kiki49 YOLO-pose build into ``<ws>/datasets/kiki49`` (source is read-only)."""
    ws = Path(ws).expanduser().resolve()
    src = Path(src).expanduser().resolve()
    if not (src / "data.yaml").is_file():
        raise FileNotFoundError(f"No data.yaml in dataset source: {src}")
    dst = dataset_dir(ws)
    if dst.resolve() == src:
        raise ValueError("Source and destination dataset are the same folder.")
    dst.mkdir(parents=True, exist_ok=True)
    _log(f"importing {src} -> {dst}")
    counter = [0]
    for part in DATASET_PARTS:
        s = src / part
        if s.is_dir():
            _log(f"copying {part}/")
            _copy_tree_resumable(s, dst / part, counter)
        elif s.is_file():
            shutil.copy2(s, dst / part)
            counter[0] += 1
    refresh_yaml_path(dst / "data.yaml")
    _log(f"import done: {counter[0]} files in {dst}")
    return dst


def _validate_ingest_label(path: Path) -> int:
    lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1 or len(lines[0]) != 5 + 3 * NKP or lines[0][0] != "0":
        raise ValueError(f"{path}: expected one class-0 label with 152 fields")
    values = [float(v) for v in lines[0][1:]]
    if any(not math.isfinite(v) for v in values):
        raise ValueError(f"{path}: non-finite label value")
    cx, cy, bw, bh = values[:4]
    eps = 1e-7  # labels keep 8 decimals: a box touching the border can overshoot by 5e-9
    if not (
        0 < bw <= 1
        and 0 < bh <= 1
        and bw / 2 - eps <= cx <= 1 - bw / 2 + eps
        and bh / 2 - eps <= cy <= 1 - bh / 2 + eps
    ):
        raise ValueError(f"{path}: invalid bbox")
    visible = 0
    for i in range(NKP):
        x, y, v = values[4 + 3 * i : 7 + 3 * i]
        if v == 0:
            if x != 0 or y != 0:
                raise ValueError(f"{path}: invisible point {i} must be 0 0 0")
        elif v == 2 and 0 <= x <= 1 and 0 <= y <= 1:
            if not (
                cx - bw / 2 - 1e-7 <= x <= cx + bw / 2 + 1e-7
                and cy - bh / 2 - 1e-7 <= y <= cy + bh / 2 + 1e-7
            ):
                raise ValueError(f"{path}: visible point {i} lies outside bbox")
            visible += 1
        else:
            raise ValueError(f"{path}: invalid point {i}")
    if not visible:
        raise ValueError(f"{path}: no visible points")
    return visible


def montage_rows(src) -> dict[int, dict] | None:
    """``{montage_frame: row}`` of a montage session's ``montage_frames.csv``, or None."""
    path = Path(src) / MONTAGE_CSV
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        rows = {int(r["montage_frame"]): r for r in csv.DictReader(f)}
    for k, r in rows.items():
        group = (r.get("match") or "").strip()
        if not group or any(c in group for c in "\r\n,/"):
            raise ValueError(f"{path}: invalid match {group!r} for montage frame {k}")
        r["match"] = group
    return rows


def ingest_reviewed(
    ws,
    src,
    match_id: str | None,
    *,
    split: str = "train",
    commit: bool = False,
    dup_bits: int = 10,
) -> dict:
    """Validate an entire reviewed session, then append its frames to one split.

    ``split`` is ``train`` or ``hard`` (a labelled holdout of difficult footage
    that is never trained on). A match lives in one split only: train refuses
    matches of val/test/hard and near-duplicates of their images; hard refuses
    matches of train/val/test and near-duplicates of their images. Every frame
    must be complete (:func:`completeness_problem`). A montage session
    (``montage_frames.csv``) needs no ``match_id``: each frame takes the match
    of its source video from that file (``match_id`` overrides it).
    """
    import cv2
    import numpy as np

    ws = Path(ws).expanduser().resolve()
    src = Path(src).expanduser().resolve()
    if split not in INGEST_SPLITS:
        raise ValueError(f"--split must be one of {INGEST_SPLITS}")
    montage = montage_rows(src)
    if match_id is None and montage is None:
        raise ValueError("--match-id is required (only a montage session has a match per frame)")
    if match_id is not None and (not match_id.strip() or any(c in match_id for c in "\r\n,/")):
        raise ValueError("--match-id must be a nonempty match/sequence identifier")

    def group_of(frame: int) -> str:
        if match_id is not None:
            return match_id
        assert montage is not None
        if frame not in montage:
            raise ValueError(f"Montage frame {frame} is missing from {MONTAGE_CSV}")
        return montage[frame]["match"]

    session = load_review_session(src / "session.json")
    if session["session_id"] != src.name:
        raise ValueError("Session directory identity mismatch")
    ds = dataset_dir(ws)
    if not (ds / "data.yaml").is_file() or not (ds / "manifest.csv").is_file():
        raise ValueError("Workspace needs an imported Kiki49 dataset")
    yaml_data = yaml.safe_load((ds / "data.yaml").read_text(encoding="utf-8"))
    names, flips = load_schema()
    if (
        yaml_data.get("kpt_shape") != [NKP, 3]
        or list(yaml_data.get("flip_idx", [])) != flips
        or (_yaml_kpt_names(yaml_data) not in (None, names))
    ):
        raise ValueError("Dataset schema differs from Kiki49")
    with (src / "reviewed_frames.csv").open(encoding="utf-8") as f:
        incoming = list(csv.DictReader(f))
    if not incoming:
        raise ValueError("No reviewed frames")
    with (ds / "manifest.csv").open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        existing = list(reader)
    for field in ("split", "image", "label", "source", "group"):
        if field not in fields:
            raise ValueError(f"Manifest lacks {field}")
    other_splits = tuple(s for s in ("train", "val", "test", "hard") if s != split)
    reserved_groups = {
        diag.match_key(r["source"], r["group"]) for r in existing if r["split"] in other_splits
    } | {r["group"] for r in existing if r["split"] in other_splits}
    for group in sorted({group_of(int(r["frame"])) for r in incoming}):
        if group in reserved_groups:
            raise ValueError(f"Match {group} already belongs to another split ({other_splits})")
    if session.get("export_mode") == "full":
        _log(
            f"ingest: full dataset mode ({len(incoming)} frames, including uncorrected AI predictions)"
        )
    else:
        problems = [
            p
            for row in incoming
            if (
                p := completeness_problem(
                    session,
                    int(row["frame"]),
                    session["frames"].get(str(row["frame"]), {"points": [None] * NKP}),
                )
            )
        ]
        if problems:
            for p in problems[:10]:
                _log(f"incomplete: {p}")
            raise ValueError(
                f"{len(problems)} incomplete frame(s): open the session in getpixelvideo, "
                "label or hide (Del) the listed points, export again"
            )
    existing_stems = {Path(r["image"]).stem for r in existing}
    identities = {(r.get("video", ""), r.get("frame", "")) for r in existing}
    content_pairs = {
        (file_sha256(ds / r["image"]), file_sha256(ds / r["label"]))
        for r in existing
        if (ds / r["image"]).is_file() and (ds / r["label"]).is_file()
    }
    image_hashes = {pair[0] for pair in content_pairs}
    # train vs val/test/hard; a hard holdout must not repeat a trained frame either.
    reserved_rows = [r for r in existing if r["split"] in other_splits]
    _log(f"hashing {len(reserved_rows)} reserved images ({', '.join(other_splits)})")
    reserved = [(r, diag.dhash_image(ds / r["image"])[0]) for r in reserved_rows]
    if any(h is None for _, h in reserved):
        raise ValueError("Unreadable reserved image; run check/audit first")
    seen_stems, seen_identity, new_rows, copies = set(), set(), [], []
    extra = [
        "video",
        "frame",
        "timestamp",
        "session",
        "reviewed_at",
        "model_sha256",
        "n_corrected",
        "n_ai",
    ]
    if montage is not None:
        extra += ["source_video", "source_frame"]
    for row in incoming:
        frame = int(row["frame"])
        state = session["frames"].get(str(frame), {}).get("state")
        if state != "EXPORTED" or row["video"] != session["video"]:
            raise ValueError(f"Frame {frame} is not exported from this review session")
        img_rel, lbl_rel = Path(row["image"]), Path(row["label"])
        if (
            img_rel.parent != Path("images")
            or lbl_rel.parent != Path("labels")
            or img_rel.suffix.lower() != ".png"
            or lbl_rel.suffix.lower() != ".txt"
            or img_rel.stem != lbl_rel.stem
        ):
            raise ValueError(f"Invalid pair paths for frame {frame}")
        img, lbl = src / img_rel, src / lbl_rel
        if not img.is_file() or not lbl.is_file():
            raise ValueError(f"Missing PNG/TXT for frame {frame}")
        if row.get("image_sha256") != file_sha256(img) or row.get("label_sha256") != file_sha256(
            lbl
        ):
            raise ValueError(f"Exported PNG/TXT changed after review: frame {frame}")
        reviewed = session["frames"][str(frame)]
        if lbl.read_text(encoding="utf-8") != pose_label_line(
            reviewed["points"], session["width"], session["height"], bbox=reviewed.get("bbox")
        ):
            raise ValueError(f"Label no longer matches human review: frame {frame}")
        n_vis = _validate_ingest_label(lbl)
        if n_vis != int(row["n_visible"]):
            raise ValueError(f"Visibility count differs for frame {frame}")
        decoded = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
        if decoded is None or (decoded.shape[1], decoded.shape[0]) != (
            session["width"],
            session["height"],
        ):
            raise ValueError(f"Unreadable PNG or wrong dimensions: {img}")
        ih, _ = diag.dhash_image(img)
        for reserved_row, rh in reserved:
            if int(np.bitwise_count(ih ^ rh).sum()) <= dup_bits:
                raise ValueError(f"Near duplicate of reserved {reserved_row['image']}: {img.name}")
        identity = (row["video"], str(frame))
        previous = next(
            (
                r
                for r in existing
                if r.get("session") == session["session_id"]
                and r.get("video") == row["video"]
                and r.get("frame") == str(frame)
            ),
            None,
        )
        if previous is not None and (previous["group"], previous["split"]) != (
            group_of(frame),
            split,
        ):
            raise ValueError(
                f"Frame {frame} was already ingested as {previous['split']}/{previous['group']}"
            )
        if (
            previous is not None
            and previous["split"] == split
            and (
                file_sha256(img) == file_sha256(ds / previous["image"])
                and file_sha256(lbl) == file_sha256(ds / previous["label"])
            )
        ):
            continue  # completed earlier; safe re-run
        if (
            img_rel.stem in existing_stems
            or img_rel.stem in seen_stems
            or identity in identities
            or identity in seen_identity
        ):
            raise ValueError(f"Name or video/frame collision: {img_rel.stem}")
        content = (file_sha256(img), file_sha256(lbl))
        if content[0] in image_hashes or content in content_pairs:
            raise ValueError(f"Identical image already registered: {img_rel.stem}")
        seen_stems.add(img_rel.stem)
        seen_identity.add(identity)
        content_pairs.add(content)
        image_hashes.add(content[0])
        new_rows.append(
            {
                "split": split,
                "image": f"images/{split}/{img.name}",
                "label": f"labels/{split}/{lbl.name}",
                "source": "freekiki_review",
                "group": group_of(frame),
                "origin": "human_reviewed",
                "n_visible": str(n_vis),
                **{k: row.get(k, "") for k in extra},
            }
        )
        if montage is not None:
            new_rows[-1] |= {
                "source_video": montage[frame]["video"],
                "source_frame": montage[frame]["frame"],
            }
        copies.extend(
            (
                (img, ds / "images" / split / img.name, content[0]),
                (lbl, ds / "labels" / split / lbl.name, content[1]),
            )
        )
    expected_stems = {Path(r["image"]).stem for r in incoming}
    if {p.stem for p in (src / "images").glob("*.png")} != expected_stems or {
        p.stem for p in (src / "labels").glob("*.txt")
    } != expected_stems:
        raise ValueError("Unlisted PNG/TXT files in session")
    report = {
        "frames": len(new_rows),
        "skipped_existing": len(incoming) - len(new_rows),
        "visible_points": sum(int(r["n_visible"]) for r in new_rows),
        "ai_points": sum(int(r["n_ai"] or 0) for r in new_rows),
        "corrected_points": sum(int(r["n_corrected"] or 0) for r in new_rows),
        "sources": {"freekiki_review": len(new_rows)},
        "match_id": match_id if match_id is not None else "per montage frame",
        "groups": dict(sorted(Counter(r["group"] for r in new_rows).items())),
        "split": split,
        "committed": False,
    }
    _log(f"ingest preview: {report}")
    if commit:
        _publish_rows(
            ds, fields + [k for k in extra if k not in fields], existing, new_rows, copies
        )
        if split == "hard":
            _ensure_yaml_split(ds / "data.yaml", "hard")
        report["committed"] = True
        _log(f"ingested and committed: {report}")
    _log(f"check: uv run --no-sync vaila/freekiki.py check -w {ws}")
    _log(f"audit: uv run --no-sync vaila/freekiki.py audit -w {ws}")
    if split == "hard":
        _log(f"measure: uv run --no-sync vaila/freekiki.py evaluate -w {ws} --split hard --model l")
    else:
        _log(f"train: uv run --no-sync vaila/freekiki.py train -w {ws} --base active")
    return report


def _publish_rows(ds: Path, fields: list[str], existing: list[dict], new_rows, copies) -> None:
    """Copy ``(source, target, sha256)`` files without ever overwriting, then
    atomically rewrite ``manifest.csv`` as ``existing + new_rows``.

    Safe to re-run after an interruption: a target already holding the
    expected bytes is kept; any other existing target aborts.
    """
    for source, target, expected_sha in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_sha256(target) != expected_sha:
                raise ValueError(f"Interrupted copy conflicts with {target}")
            continue
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".ingest-", delete=False) as f:
            tmp = Path(f.name)
        try:
            shutil.copy2(source, tmp)
            if file_sha256(tmp) != expected_sha:
                raise ValueError(f"Source changed during ingest: {source}")
            os.link(tmp, target)  # fails if target appeared; never overwrite it
        finally:
            tmp.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=ds, prefix=".manifest-", delete=False
    ) as f:
        tmp = Path(f.name)
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing + list(new_rows))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, ds / "manifest.csv")


# --------------------------------------------------------------------------- #
# External datasets (super-dataset): stage, then commit to train
# --------------------------------------------------------------------------- #
EXTERNAL_SOURCES = ("roboflow", "gsr", "soccernet-rescue")
EXTERNAL_ROWS = "rows.csv"
EXTERNAL_FIELDS = (
    "split",
    "image",
    "label",
    "source",
    "group",
    "origin",
    "n_visible",
    "aux3d",
    "qa_score",
)
_ORIGIN_COLORS = {1: (0, 255, 0), 2: (255, 200, 0), 3: (0, 165, 255)}  # BGR


def _kiki49_build():
    """``kiki49_build`` package (lazy: scipy / cv2)."""
    try:
        from . import kiki49_build as kb
        from .kiki49_build import external
    except ImportError:
        import kiki49_build as kb  # ty: ignore[unresolved-import]
        from kiki49_build import external  # ty: ignore[unresolved-import]
    return kb, external


def roboflow_api_key() -> str:
    """``ROBOFLOW_API_KEY`` from the environment or the repository ``.env``."""
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    env = Path(__file__).resolve().parent.parent / ".env"
    if not key and env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "ROBOFLOW_API_KEY":
                key = value.strip().strip("\"'")
    if not key:
        raise ValueError("Set ROBOFLOW_API_KEY (environment or the vaila .env file)")
    return key


def _roboflow_get(path: str, key: str, **params) -> dict:
    import urllib.parse
    import urllib.request

    query = urllib.parse.urlencode({"api_key": key, **params})
    with urllib.request.urlopen(f"https://api.roboflow.com/{path}?{query}", timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def roboflow_projects(key: str, search=(), projects=()) -> list[tuple[str, int | None]]:
    """``[(workspace/project, version or None)]`` from ``--project`` and ``--search`` queries."""
    _, ext = _kiki49_build()
    out: dict[str, int | None] = {}
    for item in projects or ():
        name, _, ver = str(item).partition("@")
        out[name.strip("/")] = int(ver) if ver else None
    for query in [search] if isinstance(search, str) else list(search or ()):
        for page in range(1, 11):
            data = _roboflow_get("universe/search", key, q=query, page=page)
            results = data.get("results") or []
            for r in results:
                if ext.is_soccer_project(r):
                    out.setdefault(r["url"].split("universe.roboflow.com/")[-1].strip("/"), None)
            if len(results) < int(data.get("page_size") or 12):
                break
    return list(out.items())


def _latest_version(key: str, project: str) -> int:
    data = _roboflow_get(project, key)
    versions = [int(str(v.get("id", "")).rsplit("/", 1)[-1]) for v in data.get("versions", [])]
    if not versions:
        raise ValueError(f"{project}: no generated version")
    return max(versions)


def _draw_external_preview(img, kps, origin, path: Path) -> None:
    import cv2

    canvas = img.copy()
    r = max(3, canvas.shape[1] // 320)
    for k in range(len(kps)):
        if kps[k, 2] > 0:
            x, y = int(round(kps[k, 0])), int(round(kps[k, 1]))
            cv2.circle(canvas, (x, y), r, _ORIGIN_COLORS.get(int(origin[k]), (255, 255, 255)), -1)
            cv2.putText(
                canvas,
                str(k),
                (x + r, y - r),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4 * r / 3,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])


def stage_external(
    ws,
    source: str,
    *,
    projects=(),
    search=(),
    limit: int | None = None,
    model: str = "l",
    kp_conf: float | None = None,
    root=None,
) -> Path:
    """Convert an external dataset into kiki49 labels in a staging folder.

    Writes ``<ws>/incoming/external_<source>_<ts>/`` with ``images/``,
    ``labels/``, ``rows.csv`` (manifest rows, split train), ``report.md``,
    ``preview/`` and, for Roboflow, one ``mapping_<project>.csv`` per project.
    Nothing touches the dataset: ``extend --src <folder> --commit`` does.
    """
    import cv2
    import numpy as np

    if source not in EXTERNAL_SOURCES:
        raise ValueError(f"--source must be one of {EXTERNAL_SOURCES}")
    ws = Path(ws).expanduser().resolve()
    settings = load_settings(ws)
    kp_conf = float(settings["detect"]["kp_conf"] if kp_conf is None else kp_conf)
    kb, ext = _kiki49_build()
    opts = kb.Options()
    out = ws / "incoming" / f"external_{source.replace('-', '_')}_{datetime.now():%Y%m%d_%H%M%S}"
    for d in ("images", "labels", "preview"):
        (out / d).mkdir(parents=True, exist_ok=True)
    predictor = None

    def predict(img):
        nonlocal predictor
        if predictor is None:
            path = resolve_model(ws, model)
            _log(f"model: {describe_model(ws, model, path)}")
            predictor = load_predictor(str(path), fallback_imgsz=settings["detect"]["imgsz"])
        _, xy, kc = predictor.predict(img)
        return xy, kc

    rows: list[dict] = []
    rejects: Counter = Counter()
    per_group: Counter = Counter()
    vis_counts = np.zeros(NKP, dtype=int)
    previews = 0
    notes: list[str] = []

    def keep(sample, img) -> None:
        nonlocal previews
        if isinstance(sample, ext.Reject):
            rejects[sample.reason] += 1
            return
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{sample.source}__{sample.uid}")
        points = [(float(x), float(y)) if v > 0 else None for x, y, v in np.asarray(sample.kps)]
        try:
            line = pose_label_line(points, sample.width, sample.height)
        except ValueError:
            rejects["invalid_label"] += 1
            return
        img_path, lbl_path = out / "images" / f"{name}.jpg", out / "labels" / f"{name}.txt"
        if img is None:  # unchanged source image: keep its bytes
            shutil.copy2(sample.image, img_path)
        else:  # resized by the loader (e.g. a stretched Roboflow export)
            cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lbl_path.write_text(line + "\n", encoding="utf-8")
        n_vis = _validate_ingest_label(lbl_path)
        vis = np.asarray(sample.kps)[:, 2] > 0
        vis_counts[vis] += 1
        per_group[sample.group] += 1
        rows.append(
            {
                "split": "train",
                "image": f"images/{img_path.name}",
                "label": f"labels/{lbl_path.name}",
                "source": sample.source,
                "group": sample.group,
                "origin": "".join(str(int(o)) for o in sample.origin),
                "n_visible": str(n_vis),
                "aux3d": str(int(bool(sample.aux3d))),
                "qa_score": f"{float(sample.qa):.4f}",
            }
        )
        rare = any(vis[k] for k in ext.RARE_POINTS)
        if previews < 16 or (rare and previews < 32):
            canvas = img if img is not None else cv2.imread(str(img_path))
            if canvas is not None:
                _draw_external_preview(
                    canvas, np.asarray(sample.kps), sample.origin, out / "preview" / f"{name}.jpg"
                )
            previews += 1

    if source == "roboflow":
        key = roboflow_api_key()
        base = Path(root) if root else kb.DATASET_ROOT / "roboflow"
        try:
            from . import fifa_dataset_builder as fdb
        except ImportError:
            import fifa_dataset_builder as fdb  # ty: ignore[unresolved-import]
        for project, version in roboflow_projects(key, search, projects):
            try:
                version = version or _latest_version(key, project)
                export = base / f"{project.replace('/', '__')}__v{version}"
                _log(f"roboflow: {project} v{version} -> {export}")
                fdb._download_roboflow_universe(
                    project, export, api_key=key, version=version, fmt="coco"
                )
            except Exception as exc:  # one bad project must not stop the others
                notes.append(f"- `{project}`: download failed ({exc})")
                continue
            items, names = ext.read_coco_clicks(export)
            if not items:
                notes.append(f"- `{project}` v{version}: no keypoint annotations")
                continue
            mapping_csv = out / f"mapping_{project.replace('/', '__')}.csv"
            saved = export / ext.MAPPING_CSV
            aspect, why = ext.pick_aspect(export, items)
            if saved.is_file():
                mapping = ext.read_mapping(saved)
                shutil.copy2(saved, mapping_csv)
            else:
                sub = items[:: max(1, len(items) // 150)][:150]
                xs, ks, wd, cl = [], [], [], []
                for it in sub:
                    img, sx = ext.load_image(it, aspect)
                    if img is None:
                        continue
                    xy, kc = predict(img)
                    if xy is not None:
                        xy = np.asarray(xy, dtype=float).copy()
                        xy[:, 0] /= sx
                    xs.append(xy)
                    ks.append(kc)
                    wd.append(it.width * sx)
                    cl.append(it.clicks)
                mapping, table = ext.vote_mapping(cl, xs, ks, wd, kp_conf=kp_conf)
                mapping = ext.geometric_votes([it.clicks for it in items], mapping, table)
                ext.write_mapping(mapping_csv, table, names)
                shutil.copy2(mapping_csv, saved)  # reused (and editable) next time
            if len(mapping) < ext.MIN_MAPPED:
                notes.append(
                    f"- `{project}` v{version}: only {len(mapping)} keypoints aligned, skipped"
                )
                continue
            before = len(rows)
            run = items[:limit] if limit else items
            for sample, img in ext.roboflow_samples(project, run, mapping, aspect, opts):
                keep(sample, img)
            notes.append(
                f"- `{project}` v{version}: {len(items)} images, {len(mapping)}/{len(names)} "
                f"keypoints aligned, aspect {aspect:.3f} ({why}), {len(rows) - before} accepted"
            )
    elif source == "gsr":
        base = Path(root) if root else kb.DATASET_ROOT / "soccernet_gsr_2024"
        clips = ext.gsr_clips(base)
        if not clips:
            raise FileNotFoundError(f"No GSR clips (*/SNGS-*/Labels-GameState.json) in {base}")
        for n, clip in enumerate(clips[:limit] if limit else clips, 1):
            for sample, img in ext.gsr_samples(clip, opts):
                keep(sample, img)
            if n % 10 == 0:
                _log(f"  {n}/{len(clips)} clips, {len(rows)} frames")
        notes.append(
            "- GSR files give a game index, not the SoccerNet match name: the group is "
            "`gsr:<split>:game<id>` and near-duplicate dedupe is the leakage guard."
        )
    else:
        base = (
            Path(root)
            if root
            else kb.DATASET_ROOT / "soccernet_calibration_2023" / "calibration-2023"
        )
        tasks = ext.rescue_tasks(base, limit)
        for n, task in enumerate(tasks, 1):
            got = ext.rescue_sample(task, predict, opts, kp_conf=kp_conf)
            if got is not None:
                keep(*got)
            if n % 1000 == 0:
                _log(f"  {n}/{len(tasks)} images, {len(rows)} rescued")

    with (out / EXTERNAL_ROWS).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXTERNAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    names_kp, _ = load_schema()
    rare = {f"p{k}": int(vis_counts[k]) for k in ext.RARE_POINTS}
    lines = [
        f"# External dataset staging: {source}",
        "",
        f"Created: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Accepted frames: {len(rows)} | rejected: {sum(rejects.values())} | groups: {len(per_group)}",
        "",
        "Rare points: " + ", ".join(f"{k} {v}" for k, v in rare.items()),
        "",
        "## Notes",
        "",
        *(notes or ["- none"]),
        "",
        "## Rejected (reason: count)",
        "",
        *[f"- {r}: {c}" for r, c in rejects.most_common()],
        "",
        "## Visible keypoints",
        "",
        "| kp | name | frames |",
        "|---|---|---|",
        *[
            f"| {'**' if k in ext.RARE_POINTS else ''}p{k}{'**' if k in ext.RARE_POINTS else ''} "
            f"| {names_kp[k]} | {int(vis_counts[k])} |"
            for k in range(NKP)
        ],
        "",
        "Labels: annotated (green) points are human; the rest are projected by the fitted "
        "plane / camera (orange = plane, blue = camera) and passed the line-support gate. "
        "Licences: see each Roboflow project page (CC BY 4.0 / MIT) and the SoccerNet terms. "
        "Converters ported from mkvis3d `openbiomech/soccer_field`.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(
        f"extend {source}: {len(rows)} frames staged ({sum(rejects.values())} rejected) | "
        + " ".join(f"{k}={v}" for k, v in rare.items())
        + f" -> {out}"
    )
    _log(f"commit: uv run --no-sync vaila/freekiki.py extend -w {ws} --src {out} --commit")
    return out


def commit_external(ws, src, *, commit: bool = False, dup_bits: int = 10) -> dict:
    """Append a staging folder (``stage_external``) to **train**.

    Frames are dropped (not fatal: external data is bulk) when their group is
    held by val/test/hard, when they are near-duplicates of a val/test/hard
    image, of a train image or of an earlier staged frame, or identical to a
    registered image. Without ``commit`` this is a preview.
    """
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    ws = Path(ws).expanduser().resolve()
    src = Path(src).expanduser().resolve()
    ds = dataset_dir(ws)
    with (src / EXTERNAL_ROWS).open(encoding="utf-8") as f:
        incoming = list(csv.DictReader(f))
    with (ds / "manifest.csv").open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        existing = list(reader)
    dropped: Counter = Counter()
    reserved_groups = {r["group"] for r in existing if r["split"] != "train"}
    registered = {Path(r["image"]).name for r in existing}
    candidates = []
    for row in incoming:
        if row["split"] != "train":
            raise ValueError(f"{src}: external rows must target train")
        name = Path(row["image"]).name
        if row["group"] in reserved_groups:
            dropped["group_reserved_by_val_test_hard"] += 1
        elif name in registered:
            dropped["already_registered"] += 1
        else:
            _validate_ingest_label(src / row["label"])
            candidates.append(row)
    _log(f"hashing {len(existing)} dataset images and {len(candidates)} staged frames")
    with ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4))) as pool:
        old_h = list(pool.map(lambda r: diag.dhash_image(ds / r["image"])[0], existing))
        new_h = list(pool.map(lambda r: diag.dhash_image(src / r["image"])[0], candidates))
    if any(h is None for h in old_h):
        raise ValueError("Unreadable dataset image; run check/audit first")
    reserved = np.array([h for h, r in zip(old_h, existing, strict=True) if r["split"] != "train"])
    trained = np.array([h for h, r in zip(old_h, existing, strict=True) if r["split"] == "train"])
    ok = np.array([h is not None for h in new_h], dtype=bool)
    if not ok.all():
        dropped["unreadable_image"] += int((~ok).sum())
    new = np.array([h if h is not None else np.zeros(4, np.uint64) for h in new_h]).reshape(-1, 4)
    for label, ref in (("near_dup_val_test_hard", reserved), ("near_dup_train", trained)):
        if len(ref) and len(new):
            for i, _, _ in diag._near_duplicates(new, ref, dup_bits):
                if ok[i]:
                    ok[i] = False
                    dropped[label] += 1
    kept_idx: list[int] = []
    for start in range(0, len(new), 512):  # greedy: the first of near-duplicate staged frames wins
        for i in range(start, min(start + 512, len(new))):
            if not ok[i]:
                continue
            if kept_idx:
                d = np.bitwise_count(new[kept_idx] ^ new[i]).sum(axis=1)
                if int(d.min()) <= dup_bits:
                    ok[i] = False
                    dropped["near_dup_staged"] += 1
                    continue
            kept_idx.append(i)
    image_shas = {file_sha256(ds / r["image"]) for r in existing if r["split"] != "train"}
    new_rows, copies = [], []
    for i in kept_idx:
        row = candidates[i]
        img, lbl = src / row["image"], src / row["label"]
        isha, lsha = file_sha256(img), file_sha256(lbl)
        if isha in image_shas:
            dropped["identical_reserved_image"] += 1
            continue
        target = {"image": f"images/train/{img.name}", "label": f"labels/train/{lbl.name}"}
        new_rows.append({k: row.get(k, "") for k in EXTERNAL_FIELDS} | target)
        copies += [(img, ds / target["image"], isha), (lbl, ds / target["label"], lsha)]
    report = {
        "staged": len(incoming),
        "frames": len(new_rows),
        "dropped": dict(dropped.most_common()),
        "sources": dict(Counter(r["source"] for r in new_rows)),
        "groups": len({r["group"] for r in new_rows}),
        "committed": False,
    }
    _log(f"extend preview: {report}")
    if commit and new_rows:
        _publish_rows(
            ds, fields + [k for k in EXTERNAL_FIELDS if k not in fields], existing, new_rows, copies
        )
        report["committed"] = True
        _log(f"extend committed: {len(new_rows)} frames -> train")
        _log(f"check: uv run --no-sync vaila/freekiki.py check -w {ws}")
    return report


def _ensure_yaml_split(yaml_path: Path, split: str) -> None:
    """Add ``<split>: images/<split>`` to data.yaml (workspace copy) when missing."""
    text = yaml_path.read_text(encoding="utf-8")
    if re.search(rf"(?m)^{split}:", text):
        return
    text = text.rstrip("\n") + f"\n{split}: images/{split}\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=yaml_path.parent, prefix=".data-", delete=False
    ) as f:
        tmp = Path(f.name)
        f.write(text)
    os.replace(tmp, yaml_path)


def _yaml_kpt_names(data: dict) -> list[str] | None:
    names = data.get("kpt_names")
    if isinstance(names, dict) and len(names) == 1:
        names = next(iter(names.values()))
    return list(names) if isinstance(names, (list, tuple)) else None


def check_dataset(ws, *, sample_labels: int = 50) -> list[str]:
    """Validate the workspace dataset against the kiki49 schema. Returns issues."""
    root = dataset_dir(ws)
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        return [f"data.yaml not found: {yaml_path} (run import-dataset first)"]
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names, flip_idx = load_schema()
    issues: list[str] = []
    if list(data.get("kpt_shape") or []) != [NKP, 3]:
        issues.append(f"kpt_shape is {data.get('kpt_shape')}, expected [{NKP}, 3]")
    if list(data.get("flip_idx") or []) != flip_idx:
        issues.append("flip_idx differs from soccerfield_kiki.csv")
    yaml_names = _yaml_kpt_names(data)
    if yaml_names is not None and yaml_names != names:
        bad = [i for i, (a, b) in enumerate(zip(yaml_names, names, strict=False)) if a != b]
        issues.append(f"kpt_names differ from soccerfield_kiki.csv at {bad or 'length'}")
    ncols = 5 + NKP * 3
    for split in ("train", "val", "test", "hard"):
        entry = data.get(split)
        if not entry:
            if split in ("train", "val"):
                issues.append(f"data.yaml has no '{split}' split")
            continue
        img_dir = root / str(entry)
        lbl_dir = root / str(entry).replace("images", "labels", 1)
        n_img = sum(1 for _ in img_dir.glob("*")) if img_dir.is_dir() else 0
        labels = sorted(lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else []
        _log(f"{split:5s}: {n_img} images, {len(labels)} labels")
        if n_img == 0:
            issues.append(f"{split}: no images in {img_dir}")
        for lbl in labels[:sample_labels]:
            for line in lbl.read_text(encoding="utf-8").splitlines():
                if line.strip() and len(line.split()) != ncols:
                    issues.append(f"{lbl.name}: {len(line.split())} columns, expected {ncols}")
                    break
    for issue in issues:
        _log(f"ISSUE: {issue}")
    _log("dataset OK" if not issues else f"{len(issues)} issue(s) found")
    return issues


# --------------------------------------------------------------------------- #
# Training + model registry
# --------------------------------------------------------------------------- #
def read_best_metrics(results_csv: Path) -> dict:
    """The epoch Ultralytics saved as ``best.pt`` from its ``results.csv``.

    Ultralytics 8.4 pose fitness = pose mAP50-95 + box mAP50-95 (first
    maximum wins); older CSVs without box columns fall back to pose only.
    A heatmap run has no mAP: its fitness is val PCK10_all (with misses) and
    the pose mAP fields stay blank.
    """
    best = {"best_epoch": "", "pose_map50": 0.0, "pose_map50_95": 0.0, "fitness": 0.0}
    if not Path(results_csv).is_file():
        return best
    top = float("-inf")
    with Path(results_csv).open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if HEATMAP_FITNESS_COL in [k.strip() for k in reader.fieldnames or []]:
            best = {"best_epoch": "", "pose_map50": "", "pose_map50_95": "", "fitness": ""}
            for raw in reader:
                row = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
                try:
                    fitness = float(row.get(HEATMAP_FITNESS_COL) or "nan")
                except ValueError:
                    continue
                if fitness == fitness and fitness > top:
                    top = fitness
                    best |= {"best_epoch": row.get("epoch", ""), "fitness": round(fitness, 5)}
            return best
        for raw in reader:
            row = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
            try:
                pose = float(row.get(MAP50_95_COL, "nan"))
                box = float(row.get(BOX_MAP50_95_COL) or 0.0)
            except ValueError:
                continue
            fitness = pose + box
            if fitness == fitness and fitness > top:
                top = fitness
                best = {
                    "best_epoch": row.get("epoch", ""),
                    "pose_map50": float(row.get(MAP50_COL) or 0.0),
                    "pose_map50_95": pose,
                    "fitness": round(fitness, 5),
                }
    return best


def find_eval(ws, model_path, *, split: str, like: dict | None = None) -> Path | None:
    """Newest evaluation of the same weights (SHA-256) on ``split``.

    With ``like`` (another eval summary) the images and thresholds must match too.
    """
    sha = file_sha256(model_path)
    keys = ("n_images", "images_sha1", "det_conf", "kp_conf", "match_px")
    for d in sorted(
        (Path(ws) / "outputs").glob(f"processed_freekiki_eval_{split}_*"), reverse=True
    ):
        try:
            s = json.loads((d / "eval_summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        same = (
            s.get("eval_schema") == diag.EVAL_SCHEMA
            and s.get("model_sha256") == sha
            and all(s.get(k) == like.get(k) for k in keys if like)
        )
        if same and (d / "per_keypoint.csv").is_file():
            return d
    return None


def promotion_decision(ws, candidate: str, settings: dict, baseline: str | None = None) -> dict:
    """Gate a candidate against ``baseline`` (default: the active model) on the same split
    and thresholds."""
    gate = settings["promotion"]
    split = gate.get("split", "val")
    cand_dir = find_eval(ws, candidate, split=split) or evaluate(ws, model=candidate, split=split)
    like = json.loads((cand_dir / "eval_summary.json").read_text(encoding="utf-8"))
    active = baseline or str(slot_path(ws, settings))
    base_dir = find_eval(ws, active, split=split, like=like) or evaluate(
        ws,
        model=active,
        split=split,
        det_conf=like["det_conf"],
        kp_conf=like["kp_conf"],
        match_px=like["match_px"],
    )
    decision = diag.compare_evals(base_dir, cand_dir, gate)
    (cand_dir / "promotion_decision.json").write_text(json.dumps(decision, indent=2), "utf-8")
    return decision


def log_promotion(ws, row: dict) -> None:
    _append_csv(Path(ws) / PROMOTION_LOG, row)


def _upgrade_registry(ws) -> list[str]:
    """Add missing :data:`REGISTRY_FIELDS` columns to an older ``registry.csv``; return its header.

    Additive only: existing columns and values stay, new columns are blank,
    except ``backend``, which is read back from ``runs/<run>/args.yaml``.
    """
    registry = Path(ws) / REGISTRY_CSV
    with registry.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [c for c in REGISTRY_FIELDS if c not in header]
    if not header or not missing:
        return header or REGISTRY_FIELDS
    header += missing
    for row in rows:
        if "backend" in missing and row.get("run"):
            row["backend"] = run_backend(Path(ws) / "runs" / row["run"])
    tmp = registry.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, restval="")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(registry)
    return header


def register_run(ws, run_dir, *, base: str, epochs: int, imgsz: int) -> dict:
    """Copy a run's best.pt into ``models/``, log it, promote it into its size slot.

    The slot (``n``..``x`` for YOLO26 scales, ``hm_n``..``hm_x`` for heatmap
    ResNet18..152) comes from the checkpoint; the run competes only with the
    model already in that slot (``models/freekiki_<slot>.pt``).
    ``settings['promotion']['mode']``:
      * ``gate`` (default): candidate and slot model are evaluated on the same
        split (``val``) and compared with :func:`freekiki_diag.compare_evals`
        (pose mAP, PCK with misses, homography rate, critical keypoints);
      * ``map``: best-epoch validation pose mAP50-95 only (legacy);
      * ``never``: keep the slot model (use ``compare --promote`` later).
    The first model of a workspace is always promoted; the first model of an
    empty slot needs ``new_slot_min_pose_map50_95``. Heatmap runs are never
    promoted automatically. Every decision is appended to
    ``models/promotion_log.csv``.
    """
    ws = Path(ws)
    run_dir = Path(run_dir)
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        raise FileNotFoundError(f"Training produced no best.pt: {best_pt}")
    settings = load_settings(ws)
    metrics = read_best_metrics(run_dir / "results.csv")
    backend = run_backend(run_dir)
    slot, arch = run_slot(run_dir, base, settings["models"]["default"])
    model_rel = f"models/{DATASET_NAME}_{run_dir.name}.pt"
    shutil.copy2(best_pt, ws / model_rel)
    entry = settings["models"].get(slot) or {}
    current = slot_path(ws, settings, slot)
    gate = settings["promotion"]
    mode = str(gate.get("mode", "gate"))
    reasons: list[str] = []
    if backend == "heatmap":
        # Never replaces a slot model on its own: compare it on val first.
        mode = "heatmap"
        promoted, reasons = (
            False,
            [
                f"heatmap backend: automatic promotion off; use compare --candidate {model_rel} "
                f"(add --promote to fill slot {slot})"
            ],
        )
    elif not any(slot_path(ws, settings, s).is_file() for s in SLOTS):
        promoted, reasons = True, ["first model of the workspace"]
    elif not current.is_file():
        floor = float(gate.get("new_slot_min_pose_map50_95", 0.5))
        promoted = metrics["pose_map50_95"] >= floor
        reasons = [f"empty slot {slot}: pose mAP50-95 {metrics['pose_map50_95']} vs min {floor}"]
    elif mode == "map":
        promoted = metrics["pose_map50_95"] > float(entry.get("pose_map50_95", 0.0))
        reasons = [f"pose mAP50-95 {entry.get('pose_map50_95')} -> {metrics['pose_map50_95']}"]
    elif mode == "gate":
        decision = promotion_decision(ws, str(ws / model_rel), settings, str(current))
        promoted, reasons = bool(decision["promote"]), decision["reasons"] or ["gate passed"]
    else:
        promoted, reasons = False, [f"promotion mode '{mode}'"]
    if promoted:
        _install_slot(
            ws,
            settings,
            slot,
            best_pt,
            {
                "arch": arch,
                "base": str(base),
                "run": run_dir.name,
                "pose_map50_95": metrics["pose_map50_95"],
                "pck10_all": metrics["fitness"] if backend == "heatmap" else "",
            },
        )
    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run": run_dir.name,
        "base": base,
        "dataset": settings["workspace"]["dataset"],
        "epochs": epochs,
        "imgsz": imgsz,
        **metrics,
        "model": model_rel,
        "promoted": promoted,
        "backend": backend,
        "slot": slot,
        "arch": arch,
    }
    registry = ws / REGISTRY_CSV
    header = _upgrade_registry(ws) if registry.is_file() else REGISTRY_FIELDS
    new_file = not registry.is_file()
    with registry.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    log_promotion(
        ws,
        {
            "date": row["date"],
            "candidate": model_rel,
            "mode": mode,
            "promoted": promoted,
            "reasons": " | ".join(reasons),
        },
    )
    score = (
        f"val PCK10_all={metrics['fitness']}"
        if backend == "heatmap"
        else f"pose mAP50-95={metrics['pose_map50_95']:.4f}"
    )
    _log(
        f"run {run_dir.name} (slot {slot}): {score} ({mode}) "
        f"-> {'PROMOTED to ' + slot_file(slot) if promoted else 'kept the slot model'}"
        + ("" if promoted else " | " + "; ".join(reasons))
    )
    return row


def resolve_model(ws, model: str) -> str:
    """Model name -> file.

    ``active`` (= the ``[models] default`` slot) or a slot (``l``, ``freekiki_l``,
    ``hm_m``) -> its workspace model; else a workspace-relative/absolute path;
    else a named YOLO .pt (resolved by yolotrain).
    """
    ws = Path(ws)
    if model == "active" or _slot_name(model):
        path = slot_path(ws, load_settings(ws), model)
        if not path.is_file():
            raise FileNotFoundError(f"No {model} model yet ({path}). Train one first.")
        return str(path)
    for candidate in (Path(model).expanduser(), ws / model):
        if candidate.is_file():
            return str(candidate.resolve())
    return model  # named Ultralytics weights, resolved by yolotrain


def describe_model(ws, model: str, path: str | Path | None = None) -> str:
    """One line saying exactly which network ``model`` is (slot, run, arch, score, SHA-256).

    ``active`` and slot names are resolved through ``[models]`` in freekiki.toml;
    a file path gets its run name (``runs/<run>/weights``) and its SHA-256.
    """
    ws = Path(ws)
    path = Path(path or resolve_model(ws, model))
    settings = load_settings(ws)
    models = settings["models"]
    slot = models["default"] if model == "active" else _slot_name(str(model))
    if slot is None:  # a file: is it a slot model or a run checkpoint?
        slot = next(
            (
                name
                for name in SLOTS
                if isinstance(models.get(name), dict)
                and (ws / models[name].get("file", slot_file(name))).resolve() == path.resolve()
            ),
            None,
        )
    if slot is not None and isinstance(models.get(slot), dict):
        entry = models[slot]
        alias = "active -> " if model == "active" else ""
        default = " (default)" if slot == models["default"] else ""
        return (
            f"{alias}slot {slot}{default} = {entry.get('file', slot_file(slot))} | run "
            f"{entry.get('run') or '?'} | {entry.get('arch') or entry.get('backend', '?')} | "
            f"val pose mAP50-95 {entry.get('pose_map50_95') or '?'} | "
            f"sha256 {str(entry.get('sha256', ''))[:12] or '?'}"
        )
    run = path.parent.parent.name if path.parent.name == "weights" else "-"
    sha = file_sha256(path)[:12] if path.is_file() else "?"
    return f"file {path} | run {run} | sha256 {sha} (not a promoted slot)"


def train(
    ws,
    *,
    base: str | None = None,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | float | None = None,
    device: str | None = None,
    name: str | None = None,
    fraction: float = 1.0,
    patience: int | None = None,
    seed: int = 0,
    workers: int | None = None,
    manifest: str | None = None,
    backend: str = "yolo",
    backbone: str = "resnet50",
    pretrained: bool = True,
    lr: float | None = None,
) -> dict:
    """Train (from a base model) or retrain (``base='active'``) on the workspace dataset.

    ``backend='heatmap'`` trains the vailá-native ResNet heatmap network
    (:mod:`freekiki_heatmap`, torch/torchvision only) on the same labels instead
    of Ultralytics YOLO; ``base`` is then only used when it is a heatmap
    checkpoint (continued fine-tune). Heatmap runs are never auto-promoted.

    ``manifest`` (e.g. ``v001``) trains on the oversampled list of
    ``manifests/<manifest>/`` instead of ``images/train`` (see
    :func:`make_manifest`); val/test are unchanged.

    ``runs/<name>/train_manifest.json`` records what is needed to reproduce the
    run: seed, arguments, library versions, SHA-256 of the base weights and
    of the dataset ``manifest.csv`` / ``data.yaml`` and the sampling manifest.
    """
    try:
        from . import yolotrain
    except ImportError:
        import yolotrain  # ty: ignore[unresolved-import]

    ws = Path(ws).expanduser().resolve()
    defaults = load_settings(ws)["train"]
    name = name or f"{DATASET_NAME}_{datetime.now():%Y%m%d_%H%M%S}"
    if (ws / "runs" / name).is_dir():
        info = run_state(ws, ws / "runs" / name)
        if info["state"] in ("running", "resumable"):
            raise RunStateError(
                f"Run '{name}' is {info['state']} (epoch {info['epochs_done']}/{info['epochs']}); "
                f"a new Train would overwrite it. Use: resume --name {name}"
            )
    if backend == "heatmap":
        return _train_heatmap(
            ws,
            base=base,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            name=name,
            fraction=fraction,
            seed=seed,
            workers=workers,
            manifest=manifest,
            backbone=backbone,
            pretrained=pretrained,
            lr=lr,
        )
    if backend != "yolo":
        raise ValueError(f"Unknown backend {backend!r} (yolo or heatmap)")
    base = base or defaults["base"]
    epochs = int(epochs or defaults["epochs"])
    imgsz = int(imgsz or defaults["imgsz"])
    batch = defaults["batch"] if batch is None else batch
    patience = int(defaults["patience"] if patience is None else patience)
    yaml_path = refresh_yaml_path(dataset_dir(ws) / "data.yaml")
    sampling = None
    if manifest:
        yaml_path, sampling = materialize_manifest(ws, manifest, ws / "runs" / name)
    model_path = resolve_model(ws, base)
    _log(
        f"train: base={model_path} epochs={epochs} imgsz={imgsz} batch={batch} run={name}"
        + (f" manifest={manifest}" if manifest else "")
    )
    extra: dict = {"patience": patience, "seed": seed, "deterministic": True}
    if Path(model_path).is_file() and not OFFICIAL_BASE_RE.match(Path(model_path).name):
        extra.update(CONTINUE_FINETUNE_ARGS)
        _log(
            f"continued fine-tune from {Path(model_path).name}: AdamW lr0=1e-4 lrf=0.01 "
            "warmup_epochs=1 cos_lr mosaic=0 freeze=none"
        )
    if fraction < 1.0:
        extra["fraction"] = fraction
    if workers is not None:
        extra["workers"] = int(workers)
    with _running_marker(ws / "runs" / name):
        write_train_manifest(
            ws / "runs" / name,
            base=model_path,
            yaml_path=yaml_path,
            args={"epochs": epochs, "imgsz": imgsz, "batch": batch, "device": device} | extra,
            dataset=dataset_dir(ws),
            sampling=sampling,
        )
        yolotrain.train_yolo_dataset(
            str(yaml_path),
            task="pose",
            model=model_path,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=str(ws / "runs"),
            name=name,
            extra_train_args=extra,
        )
    return register_run(ws, ws / "runs" / name, base=base, epochs=epochs, imgsz=imgsz)


def _heatmap_data(
    ws: Path, run_dir: Path, manifest: str | None, fraction: float, seed: int
) -> tuple[Path, dict | None, list[Path], Any]:
    """Train list, val scorer and data.yaml of a heatmap run (same labels as YOLO).

    ``fraction`` < 1 keeps a seeded random subset of the train list (so a
    resume rebuilds the same one). ``val_fn`` scores at most
    ``HEATMAP_VAL_IMAGES`` evenly spaced val images with the FreeKiki tables.
    """
    import numpy as np

    fh = _heatmap()
    settings = load_settings(ws)
    ds = dataset_dir(ws)
    yaml_path = refresh_yaml_path(ds / "data.yaml")
    sampling = None
    if manifest:
        yaml_path, sampling = materialize_manifest(ws, manifest, run_dir)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    train_images = fh.list_images(ds, data["train"])
    if fraction < 1.0:
        keep = np.random.default_rng(seed).permutation(len(train_images))
        keep = np.sort(keep[: max(1, round(len(train_images) * fraction))])
        train_images = [train_images[i] for i in keep]
    val_all = fh.list_images(ds, data["val"])
    step = max(1, math.ceil(len(val_all) / HEATMAP_VAL_IMAGES))
    val_images = val_all[::step]
    lbl_dir = ds / str(data["val"]).replace("images", "labels", 1)
    index = manifest_index(ws)
    det_conf = float(settings["detect"]["conf"])
    kp_conf = float(settings["detect"]["kp_conf"])

    def val_fn(predictor) -> dict:
        pred = collect_predictions(predictor, val_images, lbl_dir, index)
        return diag.score_predictions(
            pred, det_conf=det_conf, kp_conf=kp_conf, match_px=25.0, with_calib=False
        )["overall"]

    _log(f"heatmap data: {len(train_images)} train images, {len(val_images)} val images/epoch")
    return yaml_path, sampling, train_images, val_fn


def _train_heatmap(
    ws: Path,
    *,
    base,
    epochs,
    imgsz,
    batch,
    device,
    name: str,
    fraction: float,
    seed: int,
    workers,
    manifest,
    backbone: str,
    pretrained: bool,
    lr,
) -> dict:
    """``train --backend heatmap`` (see :func:`train`)."""
    fh = _heatmap()
    run_dir = ws / "runs" / name
    init = None
    if base:
        path = resolve_model(ws, base)
        if Path(path).is_file() and fh.model_backend(path) == "heatmap":
            init = path
        else:
            _log(f"heatmap: --base {base} is not a heatmap checkpoint; ignored ({backbone} start)")
    epochs = int(epochs or fh.DEFAULTS["epochs"])
    imgsz = int(imgsz or fh.DEFAULTS["imgsz"])
    batch = int(batch) if batch and float(batch) >= 1 else fh.DEFAULTS["batch"]
    workers = fh.DEFAULTS["workers"] if workers is None else int(workers)
    lr = float(lr or fh.DEFAULTS["lr"])
    _, flip_idx = load_schema()
    with _running_marker(run_dir):
        yaml_path, sampling, train_images, val_fn = _heatmap_data(
            ws, run_dir, manifest, fraction, seed
        )
        start = init or f"{backbone}-{'imagenet' if pretrained else 'scratch'}"
        _log(
            f"train: backend=heatmap start={start} epochs={epochs} imgsz={imgsz} "
            f"batch={batch} lr={lr} run={name}" + (f" manifest={manifest}" if manifest else "")
        )
        write_train_manifest(
            run_dir,
            base=init or start,
            yaml_path=yaml_path,
            args={
                "backend": "heatmap",
                "backbone": backbone,
                "pretrained": pretrained,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "workers": workers,
                "lr": lr,
                "device": device,
                "seed": seed,
                "fraction": fraction,
            },
            dataset=dataset_dir(ws),
            sampling=sampling,
        )
        fh.train_heatmap(
            run_dir,
            train_images=train_images,
            flip_idx=flip_idx,
            val_fn=val_fn,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            lr=lr,
            backbone=backbone,
            pretrained=pretrained,
            init=init,
            device=device,
            seed=seed,
            extra_args={"fraction": fraction, "manifest": manifest},
            log=_log,
        )
    return register_run(ws, run_dir, base=init or start, epochs=epochs, imgsz=imgsz)


def write_train_manifest(
    run_dir: Path,
    *,
    base: str,
    yaml_path: Path,
    args: dict,
    dataset: Path | None = None,
    sampling: dict | None = None,
) -> Path:
    """``train_manifest.json``: seed/args, versions and input hashes of a run."""
    import platform

    versions = {"python": platform.python_version()}
    for mod in ("torch", "torchvision", "ultralytics", "numpy", "cv2"):
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            versions[mod] = None
    manifest_csv = Path(dataset or yaml_path.parent) / "manifest.csv"
    info = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "run": run_dir.name,
        "base": base,
        "base_sha256": file_sha256(base) if Path(base).is_file() else None,
        # grown_from / scale / function_preserving_maxdiff of a freekiki_sizes init
        "base_freekiki": (_load_checkpoint(base) or {}).get("freekiki"),
        "data_yaml": str(yaml_path),
        "data_yaml_sha256": file_sha256(yaml_path),
        "dataset_manifest_sha256": file_sha256(manifest_csv) if manifest_csv.is_file() else None,
        "args": args,
        "sampling": sampling,
        "versions": versions,
    }
    path = Path(run_dir) / "train_manifest.json"
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Oversampling manifests (manifests/vNNN, immutable)
# --------------------------------------------------------------------------- #
MANIFESTS_DIR = "manifests"


def read_exclude_list(path) -> set[str]:
    """Image file names to leave out: a CSV with an ``image`` column or one name per line."""
    text = Path(path).read_text(encoding="utf-8")
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines and "image" in lines[0].split(","):
        names = [r["image"] for r in csv.DictReader(lines) if r.get("image")]
    else:
        names = lines
    return {Path(n).name for n in names}


def make_manifest(
    ws,
    *,
    t: float = 0.05,
    cap: float = 4.0,
    seed: int = 0,
    exclude=None,
    name: str | None = None,
) -> Path:
    """Build ``manifests/<name>/`` (default next ``vNNN``) by repeat-factor sampling.

    Read-only on the dataset. A manifest folder is never overwritten: a new
    parameter set gets a new version. The folder is written under a
    ``.<name>.partial`` name and renamed at the end, so an interrupted build
    never looks complete.
    """
    ws = Path(ws).expanduser().resolve()
    root = ws / MANIFESTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    if t <= 0 or cap < 1:
        raise ValueError(f"repeat-factor sampling needs t > 0 and cap >= 1 (got t={t}, cap={cap})")
    if not name:
        numbers = [int(p.name[1:]) for p in root.glob("v*") if p.name[1:].isdigit()]
        name = f"v{max(numbers, default=0) + 1:03d}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name.startswith("."):
        raise ValueError(f"Invalid manifest name: {name!r}")
    final = root / name
    if final.exists():
        raise FileExistsError(f"{final} exists; manifests are immutable, use a new --name.")
    names = read_exclude_list(exclude) if exclude else set()
    tmp = root / f".{name}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)  # leftover of an interrupted build of this same manifest
    info = diag.build_rfs_manifest(
        dataset_dir(ws), tmp, t=t, cap=cap, seed=seed, exclude=names, version=name, log=_log
    )
    if exclude:
        info["method"]["exclude"] = {
            "file": str(Path(exclude).resolve()),
            "sha256": file_sha256(exclude),
            "names": len(names),
            "matched_train_images": info["counts"]["excluded"],
        }
        (tmp / "manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    tmp.rename(final)
    _log(f"manifest -> {final}")
    _log(f"to train on it: uv run --no-sync vaila/freekiki.py train -w {ws} --manifest {name}")
    return final


def materialize_manifest(ws, manifest: str, run_dir: Path) -> tuple[Path, dict]:
    """Check ``manifests/<manifest>`` against the dataset and write the run's data.yaml.

    Writes ``run_dir/sampling/train.txt`` (absolute image paths) and
    ``run_dir/sampling/data.yaml`` (the dataset's data.yaml with ``train``
    pointing at that list). Raises when the list was edited or the dataset's
    train labels changed since the manifest was built.
    """
    ws = Path(ws)
    mdir = ws / MANIFESTS_DIR / manifest
    if not (mdir / "manifest.json").is_file():
        raise FileNotFoundError(f"No manifest {mdir}. Build one: freekiki.py manifest -w {ws}")
    info = json.loads((mdir / "manifest.json").read_text(encoding="utf-8"))
    list_path = mdir / "train_list.txt"
    if file_sha256(list_path) != info["train_list_sha256"]:
        raise RunStateError(f"{list_path} was modified after it was built (SHA-256 differs).")
    ds = dataset_dir(ws).resolve()
    if diag.train_labels_digest(ds) != info["inputs"]["train_labels_digest"]:
        raise RunStateError(
            f"Train labels changed since manifest {manifest} was built; build a new one."
        )
    out = Path(run_dir) / "sampling"
    out.mkdir(parents=True, exist_ok=True)
    images = [
        line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    (out / "train.txt").write_text(
        "\n".join((ds / rel).as_posix() for rel in images) + "\n", encoding="utf-8"
    )
    _log(
        "sampling list expanded to absolute paths; Ultralytics may rebuild "
        "datasets/.../labels/train.cache (derived file; images and labels stay as they are)"
    )
    text = refresh_yaml_path(ds / "data.yaml").read_text(encoding="utf-8")
    text, count = re.subn(
        r"(?m)^train:.*$", f'train: "{(out / "train.txt").resolve().as_posix()}"', text, count=1
    )
    if count == 0:
        raise ValueError(f"No 'train:' entry in {ds / 'data.yaml'}")
    yaml_path = out / "data.yaml"
    yaml_path.write_text(text, encoding="utf-8")
    sampling = {
        "manifest": manifest,
        "dir": str(mdir.resolve()),
        "train_list_sha256": info["train_list_sha256"],
        "method": info["method"],
        "entries": info["counts"]["entries_after"],
        "train_images": info["counts"]["kept_images"],
    }
    return yaml_path, sampling


# --------------------------------------------------------------------------- #
# Interrupted trainings (status / resume)
# --------------------------------------------------------------------------- #
RUNNING_MARKER = "freekiki_running.pid"


class RunStateError(Exception):
    """A train/resume request that does not fit the run's current state."""


@contextlib.contextmanager
def _running_marker(run_dir: Path):
    """Mark ``run_dir`` as being trained by this process (removed on any exit)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / RUNNING_MARKER
    marker.write_text(str(os.getpid()), encoding="utf-8")
    try:
        yield
    finally:
        marker.unlink(missing_ok=True)


def _is_running(run_dir: Path) -> bool:
    """True when the process named in the run's marker is still alive.

    A marker left by a killed process or a power loss points to a dead PID
    (or to an unrelated process after reboot, filtered by its command line).
    """
    import psutil

    try:
        pid = int((run_dir / RUNNING_MARKER).read_text(encoding="utf-8").strip())
        cmd = psutil.Process(pid).cmdline()
    except (OSError, ValueError, psutil.Error):
        return False
    is_freekiki = any(Path(a).name == "freekiki.py" or a == "vaila.freekiki" for a in cmd)
    return is_freekiki and ("train" in cmd or "resume" in cmd)


def _checkpoint_resumable(last_pt: Path) -> bool:
    """True when ``last.pt`` still carries optimizer state (training not finished)."""
    import torch

    ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
    return ckpt.get("epoch", -1) >= 0 and ckpt.get("optimizer") is not None


def run_state(ws, run_dir) -> dict:
    """Training progress of one run folder.

    ``state`` is ``registered`` (finished and logged), ``running`` (being trained
    by a live FreeKiki process), ``resumable`` (interrupted,
    continue with ``resume``), ``finished`` (training ended but not logged yet;
    ``resume`` only registers it) or ``no-checkpoint`` (stopped before the first
    epoch ended; train again).
    """
    ws, run_dir = Path(ws), Path(run_dir)
    args_yaml = run_dir / "args.yaml"
    args = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) if args_yaml.is_file() else {}
    results = run_dir / "results.csv"
    done = 0
    if results.is_file():
        done = max(0, len(results.read_text(encoding="utf-8").strip().splitlines()) - 1)
    registered = set()
    if (ws / REGISTRY_CSV).is_file():
        with (ws / REGISTRY_CSV).open(encoding="utf-8") as f:
            registered = {r["run"] for r in csv.DictReader(f)}
    last_pt = run_dir / "weights" / "last.pt"
    if run_dir.name in registered:
        state = "registered"
    elif _is_running(run_dir):
        state = "running"
    elif not last_pt.is_file():
        state = "no-checkpoint"
    else:
        state = "resumable" if _checkpoint_resumable(last_pt) else "finished"
    return {
        "run": run_dir.name,
        "state": state,
        "epochs_done": done,
        "epochs": int(args.get("epochs") or 0),
        "imgsz": int(args.get("imgsz") or 0),
        "base": Path(str(args.get("model", ""))).name,
    }


def list_runs(ws) -> list[dict]:
    """State of every run in ``<ws>/runs``, oldest first; logs a table."""
    runs_dir = Path(ws) / "runs"
    dirs = sorted(
        (d for d in runs_dir.glob("*") if d.is_dir()) if runs_dir.is_dir() else [],
        key=lambda d: d.stat().st_mtime,
    )
    rows = [run_state(ws, d) for d in dirs]
    for r in rows:
        _log(
            f"{r['run']:<28} {r['state']:<14} epochs {r['epochs_done']}/{r['epochs']} "
            f"imgsz {r['imgsz']} base {r['base']}"
        )
    if not rows:
        _log("no training runs yet")
    return rows


def run_backend(run_dir) -> str:
    """``heatmap`` when the run's ``args.yaml`` says so, else ``yolo``."""
    args_yaml = Path(run_dir) / "args.yaml"
    args = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) if args_yaml.is_file() else {}
    return "heatmap" if (args or {}).get("backend") == "heatmap" else "yolo"


def _run_sampling_manifest(run_dir: Path) -> str | None:
    """Name of the oversampling manifest a run was started with (None: plain train split)."""
    path = Path(run_dir) / "train_manifest.json"
    if not path.is_file():
        return None
    sampling = json.loads(path.read_text(encoding="utf-8")).get("sampling") or {}
    return sampling.get("manifest")


def resume(ws, *, name: str | None = None, device: str | None = None, batch=None) -> dict:
    """Continue an interrupted training from its ``weights/last.pt``.

    Without ``name`` the newest run that is ``resumable`` or ``finished`` (not
    registered) is used. The dataset path and run folder come from the current
    workspace, so a workspace copied to another machine can also be resumed.
    A run started with ``--manifest`` resumes on the same oversampled list.
    """
    ws = Path(ws).expanduser().resolve()
    rows = [r for r in list_runs(ws) if not name or r["run"] == name]
    if name and rows and rows[0]["state"] not in ("resumable", "finished"):
        raise RunStateError(
            f"Run {name} is '{rows[0]['state']}': "
            + {
                "running": "it is still training.",
                "registered": "it already finished; retrain with --base active instead.",
            }.get(rows[0]["state"], "no checkpoint (stopped before epoch 1 ended); train again.")
        )
    rows = [r for r in rows if r["state"] in ("resumable", "finished")]
    if not rows:
        raise RunStateError(
            f"No interrupted run{' named ' + name if name else ''} to resume in {ws / 'runs'}."
        )
    info = rows[-1]
    run_dir = ws / "runs" / info["run"]
    if info["state"] == "resumable" and run_backend(run_dir) == "heatmap":
        args = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8")) or {}
        _log(
            f"resume: heatmap run={info['run']} after epoch "
            f"{info['epochs_done']}/{info['epochs']} from {run_dir / 'weights' / 'last.pt'}"
        )
        with _running_marker(run_dir):
            _, _, train_images, val_fn = _heatmap_data(
                ws,
                run_dir,
                args.get("manifest") or _run_sampling_manifest(run_dir),
                float(args.get("fraction") or 1.0),
                int(args.get("seed") or 0),
            )
            _heatmap().train_heatmap(
                run_dir,
                train_images=train_images,
                flip_idx=load_schema()[1],
                val_fn=val_fn,
                batch=int(batch) if batch and float(batch) >= 1 else None,
                workers=None,
                device=device,
                resume=True,
                log=_log,
            )
    elif info["state"] == "resumable":
        try:
            from . import yolotrain
        except ImportError:
            import yolotrain  # ty: ignore[unresolved-import]
        from ultralytics import YOLO

        yaml_path = refresh_yaml_path(dataset_dir(ws) / "data.yaml")
        manifest = _run_sampling_manifest(run_dir)
        if manifest:  # keep training on the same oversampled list (paths re-checked/rewritten)
            yaml_path, _ = materialize_manifest(ws, manifest, run_dir)
        _log(
            f"resume: run={info['run']} after epoch {info['epochs_done']}/{info['epochs']} "
            f"from {run_dir / 'weights' / 'last.pt'}"
        )
        net = YOLO(str(run_dir / "weights" / "last.pt"))
        yolotrain._attach_yolo_progress_callbacks(net)
        extra: dict = {}
        if device:
            extra["device"] = device
        if batch is not None:
            extra["batch"] = batch
        with _running_marker(run_dir):
            net.train(resume=True, data=str(yaml_path), save_dir=str(run_dir), **extra)
    else:
        _log(f"resume: run {info['run']} already finished training; registering it")
    return register_run(ws, run_dir, base=info["base"], epochs=info["epochs"], imgsz=info["imgsz"])


# --------------------------------------------------------------------------- #
# Evaluation (labelled split)
# --------------------------------------------------------------------------- #
def read_label_keypoints(label_path: Path) -> tuple[Any, Any]:
    """Keypoints of the first field instance in a YOLO-pose label, normalised.

    Returns ``(xy (NKP, 2) in 0..1, visibility (NKP,))`` or ``(None, None)``.
    """
    import numpy as np

    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) == 5 + NKP * 3:
            kps = np.asarray(values[5:], dtype=float).reshape(NKP, 3)
            return kps[:, :2], kps[:, 2]
    return None, None


def model_imgsz(net, imgsz: int | None, fallback: int) -> int:
    """Image size for inference: explicit value, else the model's training size."""
    return int(imgsz or net.overrides.get("imgsz") or fallback)


def _heatmap():
    """``freekiki_heatmap`` (lazy: it imports torch/torchvision)."""
    try:
        from . import freekiki_heatmap as fh
    except ImportError:
        import freekiki_heatmap as fh  # ty: ignore[unresolved-import]
    return fh


class YoloPredictor:
    """Ultralytics pose model behind the predictor interface of :func:`load_predictor`."""

    backend = "yolo"

    def __init__(self, model_path: str, *, imgsz: int | None, fallback_imgsz: int, device=None):
        from ultralytics import YOLO

        self.net = YOLO(model_path)
        self.imgsz = model_imgsz(self.net, imgsz, fallback_imgsz)
        self.device = device

    def predict(self, frame) -> tuple[float, Any, Any]:
        results = self.net.predict(
            frame, imgsz=self.imgsz, conf=RAW_CONF, device=self.device, verbose=False
        )
        return best_instance(list(results)[0])


def load_predictor(
    model_path: str, *, imgsz: int | None = None, fallback_imgsz: int = 1280, device=None
):
    """One interface for both backends: ``.predict(frame BGR) -> (box conf, xy (49, 2), conf (49,))``.

    ``.backend`` is ``yolo`` or ``heatmap`` and ``.imgsz`` the network input
    width. Ultralytics predictions keep boxes down to ``RAW_CONF``; a heatmap
    model always returns all 49 peaks (thresholds are applied afterwards).
    A heatmap model has a fixed input size, so ``imgsz`` is ignored for it.
    """
    fh = _heatmap()
    if fh.model_backend(model_path) == "heatmap":
        predictor = fh.HeatmapPredictor.load(model_path, device)
        if imgsz and int(imgsz) != predictor.imgsz:
            _log(
                f"heatmap model has a fixed input {predictor.width}x{predictor.height}; imgsz ignored"
            )
        return predictor
    return YoloPredictor(model_path, imgsz=imgsz, fallback_imgsz=fallback_imgsz, device=device)


def file_sha256(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def pose_label_line(
    points, width: int, height: int, *, class_id: int = 0, bbox=None, pad: float = 0.0
) -> str:
    """Build one sparse YOLO pose instance; reject coordinates outside the image."""
    if width <= 0 or height <= 0 or not class_id >= 0:
        raise ValueError("Invalid image size or class")
    visible = []
    for point in points:
        if point is None:
            continue
        x, y = map(float, point)
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= width and 0 <= y <= height):
            raise ValueError(f"Point outside image: {point}")
        visible.append((x, y))
    if not visible:
        raise ValueError("At least one visible point is required")
    if bbox is None:
        xs, ys = zip(*visible, strict=True)
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        margin = pad * max(width, height)
        x0, y0 = max(0.0, x0 - margin), max(0.0, y0 - margin)
        x1, y1 = min(float(width), x1 + margin), min(float(height), y1 + margin)
        if x1 == x0:
            x0, x1 = max(0.0, x0 - 0.5), min(float(width), x1 + 0.5)
        if y1 == y0:
            y0, y1 = max(0.0, y0 - 0.5), min(float(height), y1 + 0.5)
        bbox = (x0, y0, x1 - x0, y1 - y0)
    bx, by, bw, bh = map(float, bbox)
    if not all(math.isfinite(v) for v in (bx, by, bw, bh)) or not (
        0 <= bx < width
        and 0 <= by < height
        and bw > 0
        and bh > 0
        and bx + bw <= width
        and by + bh <= height
    ):
        raise ValueError(f"Invalid bbox: {bbox}")
    parts = [
        str(class_id),
        f"{(bx + bw / 2) / width:.8f}",
        f"{(by + bh / 2) / height:.8f}",
        f"{bw / width:.8f}",
        f"{bh / height:.8f}",
    ]
    for point in points:
        if point is None:
            parts.extend(("0", "0", "0"))
        else:
            x, y = map(float, point)
            parts.extend((f"{x / width:.8f}", f"{y / height:.8f}", "2"))
    return " ".join(parts) + "\n"


def write_pose_pair(image_path: Path, label_path: Path, image, label: str) -> None:
    """Write one image/YOLO-pose label pair for either GetPixelVideo exporter."""
    import cv2

    image_path = Path(image_path)
    label_path = Path(label_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), image):
        raise OSError(f"Could not write {image_path}")
    label_path.write_text(label, encoding="utf-8")


def new_review_session(
    video, width: int, height: int, fps: float, *, workspace=None, session_path=None
) -> dict:
    """Create a video-bound review session; callers keep its in-memory frame map."""
    video = Path(video).expanduser().resolve()
    if not video.is_file() or width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Review needs an existing video and valid dimensions/FPS")
    names, flips = load_schema()
    if session_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        sid = f"{video.stem}_{hashlib.sha256(str(video).encode()).hexdigest()[:10]}_{stamp}"
        parent = Path(workspace) / "incoming" if workspace else video.parent
        session_path = parent / sid / "session.json"
    return {
        "format": 1,
        "session_id": Path(session_path).parent.name,
        "session_path": str(Path(session_path).expanduser().resolve()),
        "video": str(video),
        "video_size": video.stat().st_size,
        "video_mtime_ns": video.stat().st_mtime_ns,
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "schema": "soccerfield_kiki49",
        "names": names,
        "flip_idx": flips,
        "frames": {},
    }


def save_review_session(session: dict) -> Path:
    path = Path(session["session_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=".session-", suffix=".json", delete=False
    ) as f:
        tmp = Path(f.name)
        try:
            json.dump(session, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)
    if session.get("prediction_source"):
        folder = Path(session["prediction_source"])
        if folder.is_dir():
            (folder / "correction_session.txt").write_text(str(path.resolve()), encoding="utf-8")
    return path


def find_review_session(video, source=None, workspace=None) -> Path | None:
    """Find saved corrections for a video, including relocated dataset sessions."""
    video = Path(video).expanduser().resolve()
    folder = Path(source).expanduser().resolve() if source else video.parent
    if folder.is_file():
        if folder.name == "session.json":
            load_review_session(folder, video)
            return folder
        folder = folder.parent
    candidates = [folder / "session.json"]
    link = folder / "correction_session.txt"
    if link.is_file():
        candidates.append(Path(link.read_text(encoding="utf-8").strip()))
    candidates.extend(video.parent.glob(f"{video.stem}_*/session.json"))
    candidates.append(video.parent / f"freekiki_corrections_{video.stem}" / "session.json")
    if workspace:
        candidates.extend((Path(workspace) / "incoming").glob("*/session.json"))
    valid = []
    for candidate in set(candidates):
        if not candidate.is_file():
            continue
        try:
            load_review_session(candidate, video)
        except (ValueError, OSError, KeyError, TypeError):
            continue
        valid.append(candidate)
    return max(valid, key=lambda p: p.stat().st_mtime_ns) if valid else None


def load_review_session(path, video=None, width=None, height=None, fps=None) -> dict:
    path = Path(path).expanduser().resolve()
    if path.is_dir() and (path / "session.json").is_file():
        path = path / "session.json"
    session = json.loads(path.read_text(encoding="utf-8"))
    names, flips = load_schema()
    if (
        session.get("format") != 1
        or session.get("schema") != "soccerfield_kiki49"
        or session.get("names") != names
        or session.get("flip_idx") != flips
    ):
        raise ValueError("Review session schema mismatch")

    orig_video_path = Path(session.get("video", ""))
    target_video = Path(video).expanduser().resolve() if video is not None else None

    # Resolve video source candidate
    source = None
    if target_video is not None:
        source = target_video
    elif orig_video_path.is_file():
        source = orig_video_path.resolve()
    elif orig_video_path.name:
        for cand_dir in (path.parent, path.parent.parent, path.parent.parent.parent):
            candidate = cand_dir / orig_video_path.name
            if candidate.is_file():
                source = candidate.resolve()
                break

    if source is None or not source.is_file():
        raise ValueError("Review session video identity or metadata mismatch")

    # Verify identity: filename and size must match, or exact path match
    same_name = source.name == orig_video_path.name
    same_size = source.stat().st_size == session["video_size"]
    if not ((str(source) == session["video"] or same_name) and same_size):
        raise ValueError("Review session video identity or metadata mismatch")

    if (
        (width is not None and int(width) != session["width"])
        or (height is not None and int(height) != session["height"])
        or (fps is not None and not math.isclose(float(fps), session["fps"], rel_tol=1e-3))
    ):
        raise ValueError("Review session video identity or metadata mismatch")

    # If video path or mtime shifted (e.g. copied to another folder/machine), update session
    if str(source) != session["video"] or source.stat().st_mtime_ns != session.get(
        "video_mtime_ns"
    ):
        session["video"] = str(source)
        session["video_mtime_ns"] = source.stat().st_mtime_ns

    if any(len(r.get("points", [])) != NKP for r in session["frames"].values()):
        raise ValueError("Review session contains a frame with wrong point count")
    session["session_path"] = str(path)
    return session


def review_frame(session: dict, frame: int) -> dict:
    if frame < 0:
        raise ValueError("Negative frame index")
    return session["frames"].setdefault(
        str(frame),
        {
            "state": "UNLABELED",
            "points": [None] * NKP,
            "hidden": [],
            "point_sources": ["absent"] * NKP,
            "prediction": None,
            "model_sha256": "",
            "reviewed_at": "",
            "bbox": None,
        },
    )


def edit_review_point(session: dict, frame: int, index: int, point) -> None:
    if not 0 <= index < NKP:
        raise ValueError("Kiki49 index outside 0..48")
    row = review_frame(session, frame)
    if point is not None:
        try:
            x, y = map(float, point)
        except (TypeError, ValueError):
            point = None
        else:
            if not (math.isfinite(x) and math.isfinite(y)):
                point = None
            else:
                w, h = float(session["width"]), float(session["height"])
                x = max(0.0, min(w, x))
                y = max(0.0, min(h, y))
                point = [x, y]
                pose_label_line([point], session["width"], session["height"])
    before = row["points"][index]
    row["points"][index] = point
    row["point_sources"][index] = (
        "absent" if point is None else "corrected" if row["prediction"] else "manual"
    )
    if point != before or row["state"] == "UNLABELED":
        row["state"] = "DRAFT_MANUAL"
        row["reviewed_at"] = ""


def apply_review_prediction(
    session: dict, frame: int, xy, conf, model_sha256: str, *, kp_conf: float = 0.5, box_conf=None
) -> None:
    row = review_frame(session, frame)
    if row["state"] in ("HUMAN_REVIEWED", "EXPORTED"):
        raise ValueError("Reviewed frame must be explicitly reopened before prediction")
    if xy is not None and (len(xy) != NKP or len(conf) != NKP):
        raise ValueError("Prediction must contain 49 points")
    prediction = []
    points = []
    for i in range(NKP):
        if (
            xy is None
            or xy[i] is None
            or conf is None
            or conf[i] is None
            or not all(math.isfinite(float(v)) for v in (*xy[i], conf[i]))
        ):
            prediction.append(None)
            points.append(None)
            continue
        x, y, c = float(xy[i][0]), float(xy[i][1]), float(conf[i])
        prediction.append([x, y, c])
        points.append(
            [x, y]
            if c >= kp_conf and 0 <= x <= session["width"] and 0 <= y <= session["height"]
            else None
        )
    row.update(
        state="AI_DRAFT",
        points=points,
        prediction=prediction,
        point_sources=["predicted" if p is not None else "absent" for p in points],
        model_sha256=model_sha256,
        box_conf=box_conf,
        reviewed_at="",
    )


def review_completeness(session: dict, row: dict) -> dict:
    """:func:`freekiki_diag.label_completeness` of one review frame."""
    return diag.label_completeness(
        row["points"], row.get("hidden", []), session["width"], session["height"]
    )


def completeness_problem(session: dict, frame: int, row: dict) -> str | None:
    """Why a frame cannot become a training label, or None when it is complete."""
    comp = review_completeness(session, row)
    if comp["status"] == "complete":
        return None
    if comp["status"] == "incomplete":
        missing = ", ".join(f"p{i} {session['names'][i]}" for i in comp["suspects"])
        return f"frame {frame}: {missing} probably visible - label or hide (Del)"
    return (
        f"frame {frame}: completeness not verifiable ({comp['status']}, "
        f"{comp['n_planar']} planar points) - label >= 4 spread pitch points"
    )


def review_suggestions(session: dict, frame: int, *, min_conf: float = SUGGEST_MIN_CONF) -> dict:
    """Ghost positions for the missing points of a review frame.

    For every point neither labelled nor hidden: the AI prediction kept below
    ``kp_conf`` (conf >= ``min_conf``) and/or the projection of the labelled
    points' field homography. When both exist and disagree by more than
    ``SUGGEST_AGREE_PX`` px@1920 the geometry wins (the AI may have swapped
    identities). Returns ``{"status", "suspects", "suggestions": [{index, xy,
    source, conf}]}`` with source ai | geometry | ai+geometry.
    """
    row = session["frames"].get(str(frame))
    if row is None:
        return {"status": "few_points", "suspects": [], "suggestions": []}
    comp = review_completeness(session, row)
    projected = dict(comp["projected"])
    if comp["status"] in ("complete", "incomplete"):
        # Flags / post tops: the camera of the labelled points places them too.
        G = _geom()
        pts = [p if p is not None else (math.nan, math.nan) for p in row["points"]]
        labelled = [p is not None for p in row["points"]]
        model = G.fit_field_camera(pts, labelled, session["width"], session["height"])
        if model["status"] == "ok" and model.get("P") is not None:
            proj, inside = G.project_field(model, session["width"], session["height"])
            _, planar = G._field()
            for i in range(NKP):
                if not planar[i] and inside[i] and row["points"][i] is None:
                    projected[i] = [float(proj[i, 0]), float(proj[i, 1])]
    w, h = float(session["width"]), float(session["height"])
    scale = diag.REF_WIDTH / max(1.0, w)
    hidden = set(row.get("hidden", []))
    prediction = row.get("prediction") or [None] * NKP
    suggestions = []
    for i in range(NKP):
        if row["points"][i] is not None or i in hidden:
            continue
        p = prediction[i]
        ai = (
            [float(p[0]), float(p[1])]
            if p is not None and float(p[2]) >= min_conf and 0 <= p[0] <= w and 0 <= p[1] <= h
            else None
        )
        geo = projected.get(i)
        conf = float(p[2]) if p is not None and ai is not None else None
        if ai is not None and geo is not None:
            if math.dist(ai, geo) * scale <= SUGGEST_AGREE_PX:
                suggestions.append({"index": i, "xy": ai, "source": "ai+geometry", "conf": conf})
            else:
                suggestions.append({"index": i, "xy": geo, "source": "geometry", "conf": None})
        elif ai is not None:
            suggestions.append({"index": i, "xy": ai, "source": "ai", "conf": conf})
        elif geo is not None:
            suggestions.append({"index": i, "xy": geo, "source": "geometry", "conf": None})
    return {"status": comp["status"], "suspects": comp["suspects"], "suggestions": suggestions}


def discard_review_frame(session: dict, frame: int) -> str:
    """Toggle a bad frame out of the dataset (state ``DISCARDED``) and back.

    Discarding clears its points; the frame is never exported, PageDown and
    *Next draft* skip it, and an earlier export of it is removed from the
    session folder (derived files only; the video is untouched). Pressed
    again, the frame returns to the network draft (``AI_DRAFT``) or to
    ``UNLABELED``. Returns the new state.
    """
    row = review_frame(session, frame)
    if row["state"] == "DISCARDED":
        prediction = row.get("prediction")
        row["points"] = [
            [p[0], p[1]] if p is not None and p[2] >= row.get("kp_conf", 0.5) else None
            for p in (prediction or [None] * NKP)
        ]
        row["point_sources"] = ["predicted" if p is not None else "absent" for p in row["points"]]
        row["state"] = "AI_DRAFT" if prediction else "UNLABELED"
        return row["state"]
    if row["state"] == "EXPORTED":
        root = Path(session["session_path"]).parent
        origin = hashlib.sha256(session["video"].encode()).hexdigest()[:10]
        stem = f"{Path(session['video']).stem}_{origin}_f{int(frame):08d}"
        (root / "images" / f"{stem}.png").unlink(missing_ok=True)
        (root / "labels" / f"{stem}.txt").unlink(missing_ok=True)
    row.update(
        state="DISCARDED",
        points=[None] * NKP,
        point_sources=["absent"] * NKP,
        hidden=[],
        reviewed_at="",
    )
    return "DISCARDED"


def mark_reviewed(session: dict, frame: int) -> None:
    row = review_frame(session, frame)
    if row["state"] == "UNLABELED" or not any(p is not None for p in row["points"]):
        raise ValueError("Mark at least one visible point before review")
    if len(row["points"]) != NKP:
        raise ValueError("Expected exactly 49 points")
    pose_label_line(row["points"], session["width"], session["height"], bbox=row["bbox"])
    problem = completeness_problem(session, frame, row)
    if problem:
        raise ValueError(problem)
    row["state"] = "HUMAN_REVIEWED"
    row["reviewed_at"] = datetime.now().astimezone().isoformat()


_DETECT_RUN_GLOBS = ("freekiki_predict_*", "processed_freekiki_*")
_DETECT_RUN_STAMP = re.compile(r"(\d{8}_\d{6})$")


def _detect_run_children(folder: Path) -> list[Path]:
    """Per-video detect folders, newest timestamp first.

    New runs are ``freekiki_predict_<stem>_<timestamp>``. Older runs used
    ``processed_freekiki_<stem>_<timestamp>``.
    """
    children = {
        child.resolve()
        for pattern in _DETECT_RUN_GLOBS
        for child in folder.glob(pattern)
        if child.is_dir()
    }
    return sorted(
        children,
        key=lambda path: (
            (m.group(1) if (m := _DETECT_RUN_STAMP.search(path.name)) else ""),
            path.name,
        ),
        reverse=True,
    )


def _prediction_dir_for_video(batch_dir: Path, video: str) -> Path:
    """Newest detect-run child of a batch whose README names ``video``."""
    for child in _detect_run_children(batch_dir):
        readme = child / "README.txt"
        if not readme.is_file():
            continue
        for line in readme.read_text(encoding="utf-8").splitlines():
            if line.startswith("video: ") and Path(line[7:]).resolve() == Path(video):
                return child
    raise ValueError(f"No detect output for {Path(video).name} in {batch_dir}")


def load_raw_review_predictions(
    session: dict, directory, *, kp_conf: float = 0.5, frames=None
) -> int:
    """Load detect's raw CSV only after checking its README and video metadata.

    ``directory`` is one detect output or a detect batch folder; for a batch the
    newest output of the session's video is used. ``frames`` (a set of frame
    indices) drafts only those frames; None drafts every predicted frame.
    """
    import cv2

    directory = Path(directory)
    if not (directory / "README.txt").is_file():
        directory = _prediction_dir_for_video(directory, session["video"])
    readme = (directory / "README.txt").read_text(encoding="utf-8")
    video_line = next(
        (line[7:] for line in readme.splitlines() if line.startswith("video: ")), None
    )
    model_line = next(
        (line[7:] for line in readme.splitlines() if line.startswith("model: ")), None
    )
    hash_line = next(
        (line[14:] for line in readme.splitlines() if line.startswith("model_sha256: ")), None
    )
    dimensions = next(
        (line[12:] for line in readme.splitlines() if line.startswith("dimensions: ")), None
    )
    if video_line is None or Path(video_line).resolve() != Path(session["video"]):
        raise ValueError("Prediction CSV belongs to a different video")
    if dimensions and dimensions != f"{session['width']}x{session['height']}":
        raise ValueError("Prediction CSV dimensions differ from review session")
    cap = cv2.VideoCapture(str(session["video"]))
    try:
        if (
            not cap.isOpened()
            or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) != session["width"]
            or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) != session["height"]
        ):
            raise ValueError("Prediction video dimensions differ from review session")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    model_sha = hash_line or (
        file_sha256(model_line) if model_line and Path(model_line).is_file() else ""
    )
    if not model_sha:
        raise ValueError("Prediction provenance has no model hash")
    rows = list(csv.DictReader((directory / "field_kps_raw.csv").open(encoding="utf-8")))
    seen = set()
    for row in rows:
        frame = int(row["frame"])
        if frame in seen or not 0 <= frame < total:
            raise ValueError(f"Duplicate or invalid prediction frame: {frame}")
        seen.add(frame)
    for row in rows:
        frame = int(row["frame"])
        if frames is not None and frame not in frames:
            continue
        if review_frame(session, frame)["state"] != "UNLABELED":
            continue
        xy, conf = [], []
        for i in range(NKP):
            if row[f"p{i}_x"] == "" or row[f"p{i}_y"] == "":
                xy.append((float("nan"), float("nan")))
                conf.append(float("nan"))
            else:
                xy.append((float(row[f"p{i}_x"]), float(row[f"p{i}_y"])))
                conf.append(float(row[f"p{i}_conf"]))
        apply_review_prediction(
            session,
            frame,
            xy,
            conf,
            model_sha,
            kp_conf=kp_conf,
            box_conf=float(row["box_conf"]) if row["box_conf"] else None,
        )
    return len(rows)


DETECT_README_TITLE = "FreeKiki field keypoints"


def detect_output_dir(path) -> Path | None:
    """The detect output folder of ``path`` (the folder or a CSV inside it), else None."""
    path = Path(path).expanduser()
    folder = path if path.is_dir() else path.parent
    readme = folder / "README.txt"
    try:
        head = readme.read_text(encoding="utf-8").lstrip()
    except OSError:
        return None
    return folder.resolve() if head.startswith(DETECT_README_TITLE) else None


def run_video(run_dir) -> Path:
    """Original video of a detect run: README ``video:``, else the same name next to the run."""
    run_dir = Path(run_dir)
    video = _read_readme_video(run_dir)
    if video is None:
        raise ValueError(f"No 'video:' line in {run_dir / 'README.txt'}")
    if video.is_file():
        return video.resolve()
    for folder in (run_dir.parent, run_dir.parent.parent):
        if (folder / video.name).is_file():
            return (folder / video.name).resolve()
    raise ValueError(f"Original video not found: {video.name}; move it next to the run folder")


def detect_runs(path) -> list[tuple[Path, Path]]:
    """``(run_dir, video)`` of a detect run, or of every video of a detect batch.

    ``path`` may be a run or batch folder, or any file inside one. A batch
    gives the newest run of each video, sorted by video name. Runs whose
    video is missing are skipped; ``[]`` when nothing is a detect output.
    """
    run = detect_output_dir(path)
    if run is not None:
        return [(run, run_video(run))]
    path = Path(path).expanduser()
    folder = path if path.is_dir() else path.parent
    newest: dict[Path, Path] = {}
    for child in _detect_run_children(folder):
        if detect_output_dir(child) is None:
            continue
        try:
            newest.setdefault(run_video(child), child.resolve())
        except ValueError as exc:
            _log(f"skip {child.name}: {exc}")
    return sorted(((r, v) for v, r in newest.items()), key=lambda rv: rv[1].name)


def freekiki_run_options(run_dir) -> dict:
    """getpixelvideo options that open a detect run in correction mode."""
    readme = (Path(run_dir) / "README.txt").read_text(encoding="utf-8").splitlines()
    model = next((line[7:] for line in readme if line.startswith("model: ")), "")
    ws = workspace_for_model_file(model) if model else None
    return {
        "workspace": str(ws) if ws else None,
        "predictions": str(run_dir),
        "session": None,
    }


def relocate_review_session(session: dict, folder) -> Path:
    """Move a review session (and its future exports) into ``folder``, then save it.

    ``folder`` must be empty or already hold this session; its name becomes the
    session id (``ingest`` checks that the folder and the id agree).
    """
    folder = Path(folder).expanduser().resolve()
    target = folder / "session.json"
    if Path(session["session_path"]).resolve() != target and folder.is_dir():
        if target.is_file():
            other = json.loads(target.read_text(encoding="utf-8"))
            if Path(other.get("video", "")).name != Path(session["video"]).name:
                raise ValueError(f"{folder} holds the session of another video")
        elif any(folder.iterdir()):
            raise ValueError(f"Choose an empty folder for the dataset: {folder}")
    old_folder = Path(session["session_path"]).resolve().parent
    session["session_path"] = str(target)
    session["session_id"] = folder.name
    if session.get("dataset_folder"):
        session["dataset_folder"] = str(folder)
    path = save_review_session(session)
    if old_folder != folder and (old_folder / MONTAGE_CSV).is_file():
        shutil.copy2(old_folder / MONTAGE_CSV, folder / MONTAGE_CSV)  # keeps match per frame
    return path


def default_match_id(session: dict) -> str:
    """Match id for ``ingest`` when none is given: the video stem, lowercase ASCII."""
    return _match_from_name(Path(session["video"]).stem)


def add_corrections(ws, folders, *, match_id: str | None = None) -> list[dict]:
    """Ingest corrected-dataset folders (getpixelvideo "Save dataset") into train.

    Idempotent: frames already ingested are skipped. Any incomplete frame, a
    val/test/hard match or a near-duplicate raises before anything is copied.
    """
    reports = []
    for folder in folders:
        folder = Path(folder).expanduser().resolve()
        session = load_review_session(folder / "session.json")
        is_montage = (folder / MONTAGE_CSV).is_file()
        match = match_id or (None if is_montage else default_match_id(session))
        _log(f"corrections: {folder} (match {match or 'per montage frame'})")
        reports.append(ingest_reviewed(ws, folder, match, split="train", commit=True))
    return reports


def _read_readme_video(directory: Path) -> Path | None:
    for line in (directory / "README.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("video: "):
            return Path(line[7:])
    return None


def parse_keypoints(text) -> tuple[int, ...]:
    """``"p5,p29,39"`` -> ``(5, 29, 39)``; rejects indices outside p0..p48."""
    out = []
    for tok in str(text).replace(" ", "").split(","):
        if not tok:
            continue
        num = tok.lower().removeprefix("p")
        if not num.isdigit() or not 0 <= int(num) < NKP:
            raise ValueError(f"Unknown keypoint {tok!r} (use p0..p{NKP - 1})")
        out.append(int(num))
    if not out:
        raise ValueError("No keypoint given (e.g. --need p5,p29,p39,p47)")
    return tuple(dict.fromkeys(out))


def _linspace(a: float, b: float, n: int) -> list[float]:
    return [a] if n <= 1 else [a + (b - a) * k / (n - 1) for k in range(n)]


def _pick_spread(tiers: dict[int, list[dict]], per_video: int, min_gap: int) -> list[dict]:
    """Up to ``per_video`` frames, best tier first, spread over the clip, ``min_gap`` apart."""
    chosen: list[dict] = []
    for tier in sorted(tiers):
        cands = sorted(tiers[tier], key=lambda r: r["frame"])
        budget = per_video - len(chosen)
        if budget <= 0 or not cands:
            continue
        # Spread first (evenly spaced picks), then fill the gaps left by min_gap.
        spread = sorted({round(x) for x in _linspace(0, len(cands) - 1, budget)})
        order = [cands[i] for i in spread] + [c for i, c in enumerate(cands) if i not in spread]
        for c in order:
            if len(chosen) >= per_video:
                break
            if all(abs(c["frame"] - s["frame"]) >= min_gap for s in chosen):
                chosen.append(c)
    return sorted(chosen, key=lambda r: r["frame"])


def queue_frames(status_rows, raw_rows, *, per_video: int, min_gap: int, kp_conf: float):
    """Frames of one detect output worth labelling, in priority tiers.

    Tier 0: not calibratable (homography not ok or < 4 accepted points).
    Tier 1: a rare point (``RARE_KPS``) predicted with conf in
    [``SUGGEST_MIN_CONF``, ``kp_conf``) - the model half-sees it.
    Tier 2: the rest. Inside a tier the frames are spread evenly over the clip;
    selected frames are at least ``min_gap`` apart, at most ``per_video``.
    Returns ``[{frame, tier, reason, ...}]`` sorted by frame.
    """
    raw = {int(r["frame"]): r for r in raw_rows}
    tiers: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    for st in status_rows:
        frame = int(st["frame"])
        rare = {}
        for i in RARE_KPS:
            text = raw.get(frame, {}).get(f"p{i}_conf", "")
            if text:
                rare[f"p{i}"] = float(text)
        low = [k for k, c in rare.items() if SUGGEST_MIN_CONF <= c < kp_conf]
        if st["homography"] != "ok" or int(st["n_accepted"] or 0) < 4:
            tier, reason = 0, f"not calibratable ({st['homography']}, {st['n_accepted']} kps)"
        elif low:
            tier, reason = 1, "low-confidence rare " + " ".join(low)
        else:
            tier, reason = 2, "calibrated"
        tiers[tier].append(
            {
                "frame": frame,
                "tier": tier,
                "reason": reason,
                "n_accepted": st["n_accepted"],
                "homography": st["homography"],
                **{k: round(c, 4) for k, c in rare.items()},
            }
        )
    return _pick_spread(tiers, per_video, min_gap)


def _raw_frame(row: dict):
    """``(xy (49, 2), conf (49,), box_conf)`` of one ``field_kps_raw.csv`` row, or None."""
    import numpy as np

    if not row or row.get("p0_x", "") == "":
        return None
    xy = np.array([[float(row[f"p{i}_x"]), float(row[f"p{i}_y"])] for i in range(NKP)])
    kc = np.array([float(row[f"p{i}_conf"]) for i in range(NKP)])
    box = float(row["box_conf"]) if row.get("box_conf") else float("nan")
    return xy, kc, box


def queue_need_frames(
    status_rows,
    raw_rows,
    need,
    width: float,
    height: float,
    *,
    per_video: int,
    min_gap: int,
    settings: dict | None = None,
    half_seen: bool = False,
):
    """Frames where the needed keypoints are in the picture, for labelling.

    The field camera of each frame (``freekiki_geom``, fitted to the points
    the network accepted) says whether a needed point lies inside the image,
    even when the network does not find it:
      tier 0  camera: needed point inside the image, network misses it;
      tier 1  no camera, but the network half-sees a needed point
              (conf >= ``SUGGEST_MIN_CONF``);
      tier 2  camera: needed point inside, network already accepts it.
    Frames with none of the needed points are skipped, and tier 1 only comes
    with ``half_seen`` (reviewers discarded 105 of 105 such frames: without a
    camera they are mostly adverts, replays and close-ups). Selection as
    :func:`queue_frames` (spread, ``min_gap`` apart, at most ``per_video``).
    """
    G = _geom()
    names, _ = load_schema()
    raw = {int(r["frame"]): r for r in raw_rows}
    tiers: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    for st in status_rows:
        frame = int(st["frame"])
        parsed = _raw_frame(raw.get(frame, {}))
        if parsed is None:
            continue
        xy, kc, _ = parsed
        codes = [st.get(f"p{i}", "") for i in range(NKP)]
        model = G.fit_field_camera(xy, [c == "D" for c in codes], width, height, settings)
        has_camera = model["status"] == "ok"
        inside = G.project_field(model, width, height)[1] if has_camera else [False] * NKP
        seen = [i for i in need if inside[i]]
        missing = [i for i in seen if codes[i] != "D"]
        half = [i for i in need if codes[i] != "D" and kc[i] >= SUGGEST_MIN_CONF]

        def label(idx):
            return " ".join(f"p{i}" for i in idx)

        if missing:
            tier, reason = 0, f"camera: {label(missing)} in picture, network misses"
        elif not has_camera and half:
            if not half_seen:
                continue
            tier, reason = 1, f"no camera; network half-sees {label(half)}"
        elif seen:
            tier, reason = 2, f"camera: {label(seen)} in picture, network accepts"
        else:
            continue
        tiers[tier].append(
            {
                "frame": frame,
                "tier": tier,
                "reason": reason,
                "camera": model["source"] if has_camera else model["status"],
                "need_inside": label(seen),
                "need_missing": label(missing),
                "n_accepted": st.get("n_accepted", ""),
                **{f"p{i}_conf": round(float(kc[i]), 4) for i in need},
                "names": " | ".join(names[i] for i in (missing or seen or half)),
            }
        )
    return _pick_spread(tiers, per_video, min_gap)


MONTAGE_CSV = "montage_frames.csv"
MONTAGE_FPS = 5.0  # montage frames are independent pictures; the rate only sets timestamps


def _match_from_name(name: str) -> str:
    """Match id from a file stem: lowercase ASCII, ``_`` separated.

    Cuts of one video (``<stem>_frame_<a>_to_<b>``, vailá Cut Video) share
    the match of their source.
    """
    stem = unicodedata.normalize("NFKD", str(name).lower())
    stem = "".join(c for c in stem if not unicodedata.combining(c))
    stem = re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", stem)).strip("_")
    return re.sub(r"_frame_\d+_to_\d+$", "", stem) or "match"


class _MontageWriter:
    """H.264 all-intra writer (exact frame seeking, high quality) with an OpenCV fallback."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        import cv2

        self.path, self.proc, self.cv = Path(path), None, None
        try:
            try:
                from .ffmpeg_utils import get_ffmpeg_path
            except ImportError:
                from ffmpeg_utils import get_ffmpeg_path  # ty: ignore[unresolved-import]
            cmd = [
                get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
                "-crf", "14", "-g", "1", "-bf", "0", "-pix_fmt", "yuv420p", str(self.path),
            ]  # fmt: skip
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except (OSError, RuntimeError, FileNotFoundError):
            self.cv = cv2.VideoWriter(
                str(self.path),
                cv2.VideoWriter_fourcc(*"mp4v"),  # ty: ignore[unresolved-attribute]
                fps,
                (width, height),
            )

    def write(self, frame) -> None:
        if self.proc is not None:
            assert self.proc.stdin is not None
            self.proc.stdin.write(frame.tobytes())
        elif self.cv is not None:
            self.cv.write(frame)

    def close(self) -> None:
        if self.proc is not None:
            assert self.proc.stdin is not None
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg could not write the montage: {self.path}")
        elif self.cv is not None:
            self.cv.release()


def build_montage(ws, items, *, kp_conf: float, folder=None) -> Path:
    """One review video made of the queued frames of many videos.

    ``items``: ``[(detect_run_dir, video, picked_rows)]``. Every frame is read
    in order (exact indices), letterboxed to the most common size, and written
    to ``<folder>/montage.mp4`` (H.264, every frame a key frame). The
    network predictions are moved to montage pixels and drafted in
    ``session.json``; ``montage_frames.csv`` maps each montage frame to its
    source video, frame and ``match`` (one match per source video by default:
    edit that column to merge clips of the same match). ``ingest`` then
    groups the frames by that column. Returns the session path.
    """
    import cv2
    import numpy as np

    ws = Path(ws).expanduser().resolve()
    items = [(Path(r), Path(v), picked) for r, v, picked in items if picked]
    if not items:
        raise ValueError("No queued frame to put in a montage")
    if folder is None:
        folder = ws / "incoming" / f"montage_{datetime.now():%Y%m%d_%H%M%S}"
        n = 1
        while folder.exists():  # two montages in the same second
            folder = folder.with_name(f"{folder.name.split('-')[0]}-{n}")
            n += 1
    folder = Path(folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=False)
    sizes: Counter = Counter()
    for _, video, picked in items:
        cap = cv2.VideoCapture(str(video))
        sizes[
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        ] += len(picked)
        cap.release()
    W, H = sizes.most_common(1)[0][0]
    W, H = W + W % 2, H + H % 2  # H.264 4:2:0 needs even sizes
    out_video = folder / "montage.mp4"
    writer = _MontageWriter(out_video, W, H, MONTAGE_FPS)
    rows, drafts = [], []
    try:
        for run, video, picked in items:
            with (run / "field_kps_raw.csv").open(encoding="utf-8") as f:
                raw = {int(r["frame"]): r for r in csv.DictReader(f)}
            readme = (run / "README.txt").read_text(encoding="utf-8").splitlines()
            sha = next((ln[14:] for ln in readme if ln.startswith("model_sha256: ")), "")
            info = {int(r["frame"]): r for r in picked}
            cap = cv2.VideoCapture(str(video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            idx = 0
            try:
                for target in sorted(info):
                    while idx < target and cap.grab():
                        idx += 1
                    ok, img = cap.read() if idx == target else (False, None)
                    idx += 1
                    if not ok or img is None:
                        _log(f"montage: skip {video.name} frame {target} (not readable)")
                        continue
                    h0, w0 = img.shape[:2]
                    scale = min(W / w0, H / h0)
                    nw, nh = round(w0 * scale), round(h0 * scale)
                    ox, oy = (W - nw) // 2, (H - nh) // 2
                    canvas = np.zeros((H, W, 3), np.uint8)
                    canvas[oy : oy + nh, ox : ox + nw] = (
                        img
                        if (nw, nh) == (w0, h0)
                        else cv2.resize(
                            img,
                            (nw, nh),
                            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
                        )
                    )
                    writer.write(canvas)
                    k = len(rows)
                    drafts.append((k, _raw_frame(raw.get(target, {})), scale, ox, oy, sha))
                    rows.append(
                        {
                            "montage_frame": k,
                            "match": _match_from_name(video.stem),
                            "video": str(video),
                            "frame": target,
                            "time_s": round(target / fps, 3) if fps else "",
                            "scale": round(scale, 6),
                            "offset_x": ox,
                            "offset_y": oy,
                            "tier": info[target].get("tier", ""),
                            "reason": info[target].get("reason", ""),
                            "run": str(run),
                        }
                    )
            finally:
                cap.release()
    finally:
        writer.close()
    cap = cv2.VideoCapture(str(out_video))
    n_written = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n_written != len(rows):
        raise RuntimeError(f"Montage has {n_written} frames, expected {len(rows)}: {out_video}")
    session = new_review_session(out_video, W, H, MONTAGE_FPS, session_path=folder / "session.json")
    session["dataset_folder"] = str(folder)  # F9 exports here, next to montage_frames.csv
    for k, parsed, scale, ox, oy, sha in drafts:
        if parsed is None:
            continue
        xy, kc, box = parsed
        apply_review_prediction(
            session,
            k,
            xy * scale + (ox, oy),
            kc,
            sha,
            kp_conf=kp_conf,
            box_conf=None if not math.isfinite(box) else box,
        )
    diag.write_csv(folder / MONTAGE_CSV, rows)
    return save_review_session(session)


def already_queued(ws) -> set[tuple[Path, int]]:
    """Source ``(video, frame)`` pairs already labelled or put in a montage.

    Read from the dataset manifest (``source_video`` / ``source_frame``,
    ``video`` / ``frame``) and from every ``incoming/*/montage_frames.csv``,
    so a new queue never offers the same picture twice.
    """
    ws = Path(ws)
    out: set[tuple[Path, int]] = set()

    def add(video, frame):
        if video and str(frame).strip().lstrip("-").isdigit():
            out.add((Path(video).expanduser().resolve(), int(frame)))

    manifest = dataset_dir(ws) / "manifest.csv"
    if manifest.is_file():
        with manifest.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                add(r.get("source_video"), r.get("source_frame", ""))
                add(r.get("video"), r.get("frame", ""))
    for path in (ws / "incoming").glob(f"*/{MONTAGE_CSV}"):
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                add(r.get("video"), r.get("frame", ""))
    return out


def build_label_queue(
    ws,
    batch,
    *,
    per_video: int | None = None,
    min_gap: int | None = None,
    kp_conf: float | None = None,
    need=None,
    montage: bool = False,
    half_seen: bool = False,
) -> list[Path]:
    """Review sessions with only the frames worth labelling drafted.

    ``batch``: one or more detect outputs / batch folders. Without ``need``
    the frames come from :func:`queue_frames` (default 25 per video); with
    ``need`` (keypoint indices, e.g. 5, 29, 39, 47) from
    :func:`queue_need_frames` (default 5 per video: many videos, few frames
    each). One session per video under ``<ws>/incoming/<session>/`` with
    ``queue.csv``, or with ``montage`` a single session of all the frames
    (:func:`build_montage`). getpixelvideo's PageUp/PageDown walks exactly
    the queued frames; the split (train or the hard holdout) is chosen at
    ``ingest``.
    """
    import cv2

    ws = Path(ws).expanduser().resolve()
    batches = [batch] if isinstance(batch, (str, Path)) else list(batch)
    settings = load_settings(ws)
    kp_conf = float(settings["detect"]["kp_conf"] if kp_conf is None else kp_conf)
    need = None if need is None else tuple(need)
    per_video = int(per_video or (5 if need else 25))
    min_gap = int(min_gap if min_gap is not None else (30 if need else 5))
    taken = already_queued(ws)  # (video, frame) labelled or in an earlier montage
    outputs = []
    for b in batches:
        b = Path(b).expanduser().resolve()
        outputs += (
            [b]
            if (b / "README.txt").is_file()
            else [p for p in _detect_run_children(b) if (p / "README.txt").is_file()]
        )
    if not outputs:
        raise ValueError(f"No detect output (README.txt) in {', '.join(map(str, batches))}")
    names, _ = load_schema()
    sessions, items, done = [], [], set()
    for out in outputs:
        video = _read_readme_video(out)
        if video is None or not video.is_file():
            _log(f"skip {out.name}: video not found ({video})")
            continue
        if video.resolve() in done:
            _log(f"skip {out.name}: a newer run of {video.name} is already queued")
            continue
        done.add(video.resolve())
        with (out / "field_kps_status.csv").open(encoding="utf-8") as f:
            status_rows = list(csv.DictReader(f))
        with (out / "field_kps_raw.csv").open(encoding="utf-8") as f:
            raw_rows = list(csv.DictReader(f))
        cap = cv2.VideoCapture(str(video))
        try:
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
        finally:
            cap.release()
        skip = {f for v, f in taken if v == video.resolve()}
        if skip:
            status_rows = [r for r in status_rows if int(r["frame"]) not in skip]
        if need:
            picked = queue_need_frames(
                status_rows, raw_rows, need, w, h, per_video=per_video, min_gap=min_gap,
                settings=settings["geometry"], half_seen=half_seen,
            )  # fmt: skip
            tiers = Counter(r["tier"] for r in picked)
            summary = (
                f"camera-visible & missed {tiers[0]}, no camera & half-seen {tiers[1]}, "
                f"already accepted {tiers[2]}"
            )
        else:
            picked = queue_frames(
                status_rows, raw_rows, per_video=per_video, min_gap=min_gap, kp_conf=kp_conf
            )
            tiers = Counter(r["tier"] for r in picked)
            summary = f"not calibratable {tiers[0]}, rare low-conf {tiers[1]}, other {tiers[2]}"
        if not picked:
            _log(f"queue {video.name}: no frame to label")
            continue
        if montage:
            items.append((out, video, picked))
            _log(f"queue {video.name}: {len(picked)} frames ({summary}) -> montage")
            continue
        session = new_review_session(video, w, h, fps, workspace=ws)
        load_raw_review_predictions(
            session, out, kp_conf=kp_conf, frames={r["frame"] for r in picked}
        )
        path = save_review_session(session)
        diag.write_csv(path.parent / "queue.csv", picked)
        _log(f"queue {video.name}: {len(picked)} frames ({summary}) -> {path}")
        sessions.append(path)
    if montage and items:
        items.sort(key=lambda item: item[1].name)
        path = build_montage(ws, items, kp_conf=kp_conf)
        _log(
            f"montage: {sum(len(p) for _, _, p in items)} frames from {len(items)} videos -> "
            f"{path.parent / 'montage.mp4'} (map: {path.parent / MONTAGE_CSV})"
        )
        sessions.append(path)
    if need:
        _log("needed keypoints: " + ", ".join(f"p{i} {names[i]}" for i in need))
    for path in sessions:
        print_gui_cli_mirror(
            "vaila/getpixelvideo",
            [
                "uv", "run", "--no-sync", "vaila/getpixelvideo.py", "--freekiki",
                "--freekiki-workspace", str(ws), "--freekiki-session", str(path),
            ],
            note="Review the queued frames (then ingest --split train|hard):",
        )  # fmt: skip
    _log(
        "review: PageDown/PageUp = next/previous queued frame, F10 = accept ghost, "
        "Del = hide (not visible), F3 = Frame OK, F9 = save dataset"
    )
    if montage and items:
        _log(
            f"ingest: uv run --no-sync vaila/freekiki.py ingest -w {ws} --src {sessions[-1].parent} "
            f"(one match per source video from {MONTAGE_CSV}; edit its 'match' column to merge "
            "clips of the same match)"
        )
    return sessions


REVIEWED_FIELDS = (
    "image",
    "label",
    "image_sha256",
    "label_sha256",
    "video",
    "frame",
    "timestamp",
    "group",
    "session",
    "reviewed_at",
    "model_sha256",
    "n_visible",
    "n_corrected",
    "n_ai",
)


def export_dataset_yaml(session: dict, root: Path) -> Path:
    names, flips = load_schema()
    yaml_dict = {
        "path": str(root.resolve()),
        "train": "images",
        "val": "images",
        "test": "images",
        "kpt_shape": [NKP, 3],
        "flip_idx": flips,
        "names": {0: "football_pitch"},
        "kpt_names": {0: names},
    }
    yaml_path = root / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False)
    return yaml_path


def export_wide_markers_csv(
    session: dict, root: Path, total_video_frames: int | None = None
) -> list[Path]:
    header = ["frame"]
    for i in range(NKP):
        header.extend([f"p{i}_x", f"p{i}_y"])

    max_frame = max((int(k) for k in session.get("frames", {})), default=-1)
    n_frames = (
        total_video_frames
        if total_video_frames is not None and total_video_frames > 0
        else (max_frame + 1)
    )

    rows = []
    for f in range(n_frames):
        row = [str(f)]
        frame_data = session.get("frames", {}).get(str(f))
        pts = frame_data.get("points") if frame_data else None
        for i in range(NKP):
            if (
                pts
                and i < len(pts)
                and pts[i] is not None
                and pts[i][0] is not None
                and pts[i][1] is not None
            ):
                x, y = pts[i][0], pts[i][1]
                row.extend([f"{float(x):.2f}", f"{float(y):.2f}"])
            else:
                row.extend(["", ""])
        rows.append(row)

    written = []
    video_stem = Path(session.get("video", "video")).stem
    csv_paths = [
        root / f"{video_stem}_markers.csv",
        root / "field_kps_getpixelvideo.csv",
    ]
    video_path = Path(session.get("video", ""))
    if video_path.parent.is_dir() and video_path.parent.resolve() != root.resolve():
        csv_paths.append(video_path.parent / f"{video_stem}_markers.csv")

    for cp in csv_paths:
        try:
            with cp.open("w", newline="", encoding="utf-8") as f_out:
                writer = csv.writer(f_out)
                writer.writerow(header)
                writer.writerows(rows)
            written.append(cp)
        except OSError:
            pass
    return written


def export_reviewed_session(session: dict, *, mode: str = "only_correct") -> Path:
    """Write frames into incoming/<session>/, never a split.

    mode:
      'only_correct' (default): exports only human-confirmed complete frames.
      'full': exports all frames with keypoints (human-reviewed/corrected frames
              plus uncorrected/predicted AI draft frames).
    """
    import cv2

    mode = str(mode).strip().lower()
    if mode in ("lite", "only_correct", "only", "correct"):
        mode = "only_correct"
    elif mode in ("full", "all"):
        mode = "full"
    else:
        raise ValueError(
            f"Unknown export mode: '{mode}' (expected 'full' or 'lite'/'only_correct')"
        )

    root = Path(session["session_path"]).parent
    origin = hashlib.sha256(session["video"].encode()).hexdigest()[:10]

    # Only complete frames become labels in only_correct mode: an empty point is written as "not
    # visible", so a partial frame teaches the network to ignore visible points.
    # Complete manual drafts are promoted; incomplete reviewed frames go back to
    # DRAFT_MANUAL (PageUp/PageDown finds them) and their stale export is removed.
    incomplete = []
    for key, row in session.get("frames", {}).items():
        if row.get("state") not in ("DRAFT_MANUAL", "HUMAN_REVIEWED", "EXPORTED") or not any(
            p is not None for p in row.get("points", [])
        ):
            continue
        problem = completeness_problem(session, int(key), row)
        if problem is None:
            if row["state"] == "DRAFT_MANUAL":
                row["state"] = "HUMAN_REVIEWED"
                if not row.get("reviewed_at"):
                    row["reviewed_at"] = datetime.now().astimezone().isoformat()
            continue
        incomplete.append({"frame": int(key), "state": row["state"], "problem": problem})
        if row["state"] != "DRAFT_MANUAL":
            row["state"] = "DRAFT_MANUAL"
            row["reviewed_at"] = ""
        if mode != "full":
            stem = f"{Path(session['video']).stem}_{origin}_f{int(key):08d}"
            (root / "images" / f"{stem}.png").unlink(missing_ok=True)
            (root / "labels" / f"{stem}.txt").unlink(missing_ok=True)
    incomplete.sort(key=lambda r: r["frame"])
    if mode != "full" and incomplete:
        root.mkdir(parents=True, exist_ok=True)
        diag.write_csv(root / "incomplete_frames.csv", incomplete)
        for r in incomplete[:10]:
            _log(f"not exported: {r['problem']}")
        _log(
            f"{len(incomplete)} incomplete frame(s) left as DRAFT -> {root / 'incomplete_frames.csv'}"
        )
    else:
        (root / "incomplete_frames.csv").unlink(missing_ok=True)

    if mode == "full":
        export_frames = [
            (int(k), v)
            for k, v in session.get("frames", {}).items()
            if any(p is not None for p in v.get("points", []))
        ]
        if not export_frames:
            save_review_session(session)
            raise ValueError("No frames with keypoints to export in full mode")
    else:
        export_frames = [
            (int(k), v)
            for k, v in session.get("frames", {}).items()
            if v.get("state") in ("HUMAN_REVIEWED", "EXPORTED")
        ]
        if not export_frames:
            save_review_session(session)
            raise ValueError(
                "No complete human-reviewed frames to export"
                + (
                    f" ({len(incomplete)} incomplete; first: {incomplete[0]['problem']})"
                    if incomplete
                    else ""
                )
            )

    export_frames.sort(key=lambda r: r[0])
    cap = cv2.VideoCapture(session["video"])
    if not cap.isOpened():
        raise ValueError("Could not open review video")
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for frame, row in export_frames:
            label = pose_label_line(
                row["points"], session["width"], session["height"], bbox=row.get("bbox")
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok or image.shape[1] != session["width"] or image.shape[0] != session["height"]:
                raise ValueError(f"Could not read matching video frame {frame}")
            stem = f"{Path(session['video']).stem}_{origin}_f{frame:08d}"
            img = root / "images" / f"{stem}.png"
            txt = root / "labels" / f"{stem}.txt"
            if (
                mode != "full"
                and img.exists()
                and txt.exists()
                and txt.read_text(encoding="utf-8") != label
            ):
                raise ValueError(f"Existing export has different label: {stem}")
            write_pose_pair(img, txt, image, label)
            point_sources = row.get("point_sources") or [
                "predicted" if p is not None else "absent" for p in row["points"]
            ]
            rows.append(
                {
                    "image": f"images/{img.name}",
                    "label": f"labels/{txt.name}",
                    "image_sha256": file_sha256(img),
                    "label_sha256": file_sha256(txt),
                    "video": session["video"],
                    "frame": frame,
                    "timestamp": f"{frame / session['fps']:.6f}",
                    "group": "",  # supplied as --match-id during ingest
                    "session": session["session_id"],
                    "reviewed_at": row.get("reviewed_at", ""),
                    "model_sha256": row.get("model_sha256", ""),
                    "n_visible": sum(p is not None for p in row["points"]),
                    "n_corrected": point_sources.count("corrected"),
                    "n_ai": point_sources.count("predicted"),
                }
            )
            row["state"] = "EXPORTED"
    finally:
        cap.release()
    session["export_mode"] = mode
    with (root / "reviewed_frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEWED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    save_review_session(session)
    write_review_diagnostics(session)
    export_dataset_yaml(session, root)
    export_wide_markers_csv(session, root, total_video_frames=total_video_frames)
    return root


def write_review_diagnostics(session: dict) -> Path:
    root = Path(session["session_path"]).parent
    names, _ = load_schema()
    rows = []
    diag_len = math.hypot(session["width"], session["height"])
    for frame, review in sorted(session["frames"].items(), key=lambda kv: int(kv[0])):
        if review["state"] not in ("HUMAN_REVIEWED", "EXPORTED") or review["prediction"] is None:
            continue
        for i, (human, pred) in enumerate(zip(review["points"], review["prediction"], strict=True)):
            error = math.dist(human, pred[:2]) if human is not None and pred is not None else None
            status = (
                "prediction_absent"
                if pred is None
                else "human_absent"
                if human is None
                else "accepted"
                if review["point_sources"][i] == "predicted"
                else "corrected"
            )
            rows.append(
                {
                    "video": session["video"],
                    "frame": frame,
                    "point": i,
                    "name": names[i],
                    "model_sha256": review.get("model_sha256", ""),
                    "human_x": human[0] if human else "",
                    "human_y": human[1] if human else "",
                    "ai_x": pred[0] if pred else "",
                    "ai_y": pred[1] if pred else "",
                    "confidence": pred[2] if pred else "",
                    "error_px": error if error is not None else "",
                    "error_diagonal": error / diag_len if error is not None else "",
                    "status": status,
                }
            )
    fields = (
        "video",
        "frame",
        "point",
        "name",
        "model_sha256",
        "human_x",
        "human_y",
        "ai_x",
        "ai_y",
        "confidence",
        "error_px",
        "error_diagonal",
        "status",
    )
    path = root / "human_vs_ai.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for key, filename in (
        ("point", "human_vs_ai_by_point.csv"),
        ("video", "human_vs_ai_by_video.csv"),
    ):
        groups = {}
        for row in rows:
            groups.setdefault(row[key], []).append(row)
        with (root / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    key,
                    "n",
                    "accepted",
                    "corrected",
                    "prediction_absent",
                    "human_absent",
                    "mean_error_px",
                ]
            )
            for group, items in groups.items():
                errors = [float(r["error_px"]) for r in items if r["error_px"] != ""]
                writer.writerow(
                    [group, len(items)]
                    + [
                        sum(r["status"] == s for r in items)
                        for s in ("accepted", "corrected", "prediction_absent", "human_absent")
                    ]
                    + [sum(errors) / len(errors) if errors else ""]
                )
    return path


def manifest_index(ws) -> dict[str, dict]:
    """``image stem -> manifest row`` (source, group, ...); empty without manifest."""
    path = dataset_dir(ws) / "manifest.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return {Path(r["image"]).stem: r for r in csv.DictReader(f)}


def best_instance(result) -> tuple[float, Any, Any]:
    """``(box conf, xy (49, 2), conf (49,))`` of the highest-confidence box, NaN/None if none."""
    import numpy as np

    if result.keypoints is None or not len(result.keypoints) or result.boxes is None:
        return float("nan"), None, None
    best = int(result.boxes.conf.argmax())
    xy = result.keypoints.xy[best].cpu().numpy()
    kc = (
        result.keypoints.conf[best].cpu().numpy()
        if result.keypoints.conf is not None
        else np.ones(NKP)
    )
    return float(result.boxes.conf[best]), xy, kc


def collect_predictions(predictor, images, lbl_dir, manifest) -> dict:
    """Raw best-instance predictions + labels of every image (no threshold applied).

    ``predictor``: see :func:`load_predictor`.
    """
    import cv2
    import numpy as np

    keep: dict[str, list] = {k: [] for k in PRED_KEYS}
    for n, img_path in enumerate(images, 1):
        gt_xy, gt_vis = read_label_keypoints(lbl_dir / f"{img_path.stem}.txt")
        frame = cv2.imread(str(img_path)) if gt_xy is not None else None
        if frame is None:
            continue
        h, w = frame.shape[:2]
        box_conf, xy, kc = predictor.predict(frame)
        row = manifest.get(img_path.stem, {})
        keep["images"].append(img_path.name)
        keep["groups"].append(row.get("group", img_path.stem))
        keep["sources"].append(row.get("source", "unknown"))
        keep["width"].append(w)
        keep["height"].append(h)
        keep["box_conf"].append(box_conf)
        keep["pred_xy"].append(np.full((NKP, 2), np.nan) if xy is None else xy)
        keep["pred_kc"].append(np.full(NKP, np.nan) if kc is None else kc)
        keep["gt_xy"].append(gt_xy * (w, h))
        keep["gt_vis"].append(gt_vis)
        if n % 500 == 0:
            _log(f"  {n}/{len(images)} images")
    return {k: np.asarray(v) for k, v in keep.items()}


def load_predictions(eval_dir) -> dict:
    import numpy as np

    with np.load(Path(eval_dir) / "predictions.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_scores(out: Path, scores: dict) -> None:
    """per_keypoint.csv, per_source.csv, failures.csv, calibration_per_image.csv."""
    diag.write_csv(out / "per_keypoint.csv", scores["per_keypoint"])
    diag.write_csv(out / "per_source.csv", scores["per_source"])
    diag.write_csv(out / "failures.csv", scores["failures"] or [{"image": "", "kp": ""}])
    per_image = scores["calib"].get("per_image") or []
    if per_image:
        diag.write_csv(
            out / "calibration_per_image.csv",
            [
                {"image": img, "gt_status": g, "pred_status": p, "err_px": diag._num(e, 2)}
                for img, (g, p, e) in zip(scores["images"], per_image, strict=True)
            ],
        )


METRIC_DEFINITIONS = {
    "units": "px@1920: pixel distances rescaled to a 1920-px-wide image",
    "match": "labelled and predicted (box conf >= det_conf and kp conf >= kp_conf); "
    "tp if distance <= match_px, else mislocalized (counts as one FP and one FN)",
    "precision": "tp / (tp + mislocalized + fp)",
    "recall": "tp / (tp + mislocalized + fn) = tp / n_ref",
    "err_*": "distance statistics over tp only; err_any_* over tp + mislocalized",
    "pck<t>_pred": "(tp+mislocalized with distance <= t) / (tp + mislocalized)",
    "pck<t>_all": "(tp+mislocalized with distance <= t) / n_ref (misses are failures)",
    "calib.ok_rate": "images whose predicted planar keypoints give a valid homography "
    "(RANSAC, >= min_inliers, spread, RMSE, orientation) / images whose labels do; "
    "self-consistency only: a wrong but plausible layout also passes",
    "calib.correct_rate": "images whose predicted homography is valid AND reprojects the "
    "labelled planar points within correct_px (median) / images whose labels give a valid "
    "homography; this is the rate the promotion gate uses",
    "calib.err_*": "median distance of labelled planar points to their projection "
    "through the predicted homography, per image (valid images only)",
    "undefined": "blank / null, never zero",
}


def evaluate(
    ws,
    *,
    model: str = "active",
    split: str = "val",
    imgsz: int | None = None,
    batch: int = 8,
    device: str | None = None,
    det_conf: float | None = None,
    kp_conf: float | None = None,
    match_px: float = 25.0,
    pck: tuple = (5, 10, 25),
    max_images: int = 0,
    geom_settings: dict | None = None,
) -> Path:
    """Measure a model on a labelled split (default ``val``; ``test`` is for final reports).

    Two views of quality:
      * Ultralytics validation (pose/box mAP, precision, recall, OKS);
      * FreeKiki per-keypoint tables with an explicit distance gate
        (see ``METRIC_DEFINITIONS``) and the predicted field homography.

    Raw best-instance predictions are saved at a low confidence
    (``predictions.npz``) so thresholds can be swept offline (``sweep``).
    Writes ``<ws>/outputs/processed_freekiki_eval_<split>_<ts>/`` and appends a
    row to ``models/evaluations_v2.csv``. A heatmap model has no Ultralytics
    validation: its pose mAP fields stay blank and only the FreeKiki tables
    (the same for both backends) are written.
    """
    ws = Path(ws).expanduser().resolve()
    settings = load_settings(ws)
    det_conf = float(settings["detect"]["conf"] if det_conf is None else det_conf)
    kp_conf = float(settings["detect"]["kp_conf"] if kp_conf is None else kp_conf)
    model_path = resolve_model(ws, model)
    _log(f"model: {describe_model(ws, model, model_path)}")
    predictor = load_predictor(
        model_path, imgsz=imgsz, fallback_imgsz=settings["detect"]["imgsz"], device=device
    )
    imgsz = predictor.imgsz
    yaml_path = refresh_yaml_path(dataset_dir(ws) / "data.yaml")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not data.get(split):
        raise ValueError(f"data.yaml has no '{split}' split")
    img_dir = dataset_dir(ws) / str(data[split])
    lbl_dir = dataset_dir(ws) / str(data[split]).replace("images", "labels", 1)
    out = ws / "outputs" / f"processed_freekiki_eval_{split}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    _log(f"evaluate: model={model_path} ({predictor.backend}) split={split} imgsz={imgsz} -> {out}")
    if split in ("test", "hard"):
        _log(f"note: {split} split - report only, do not choose thresholds/hyper-parameters on it")

    ultra: dict = {}
    if isinstance(predictor, YoloPredictor):
        val_yaml, val_split = yaml_path, split
        if split not in ("train", "val", "test"):
            # Ultralytics resolves only train/val/test paths: score <split> as "val".
            val_yaml, val_split = out / f"data_{split}.yaml", "val"
            val_yaml.write_text(
                yaml.safe_dump(data | {"val": str(img_dir)}, sort_keys=False), encoding="utf-8"
            )
        val = predictor.net.val(
            data=str(val_yaml),
            split=val_split,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(out),
            name="ultralytics_val",
            verbose=False,
        )
        ultra = {k: round(float(v), 4) for k, v in val.results_dict.items()}
    else:
        _log("heatmap backend: no Ultralytics mAP (not comparable); FreeKiki tables only")

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if max_images:
        images = images[:max_images]
    pred = collect_predictions(predictor, images, lbl_dir, manifest_index(ws))
    np_savez(out / "predictions.npz", pred)
    scores = diag.score_predictions(
        pred, det_conf=det_conf, kp_conf=kp_conf, match_px=match_px, pck=pck
    )
    scores["images"] = pred["images"].tolist()
    write_scores(out, scores)
    geometry = evaluate_geometry(
        pred, out, det_conf=det_conf, kp_conf=kp_conf, match_px=match_px, pck=pck,
        settings=settings["geometry"] | dict(geom_settings or {}),
    )  # fmt: skip
    overall, calib = (
        scores["overall"],
        {k: v for k, v in scores["calib"].items() if k != "per_image"},
    )
    summary = {
        "eval_schema": diag.EVAL_SCHEMA,
        "model": model_path,
        "model_sha256": file_sha256(model_path),
        "backend": predictor.backend,
        "split": split,
        "n_images": len(pred["images"]),
        "images_sha1": diag.names_digest(pred["images"].tolist()),
        "imgsz": imgsz,
        "raw_conf": RAW_CONF,
        "det_conf": det_conf,
        "kp_conf": kp_conf,
        "match_px": match_px,
        "pck": list(pck),
        "oks_sigmas": data.get("kpt_oks_sigmas") or "uniform (Ultralytics default 1/nkpt)",
        "ultralytics": ultra,
        "keypoints": overall,
        "calib": calib,
        "geometry": geometry,
        "definitions": METRIC_DEFINITIONS,
    }
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    t_mid = f"pck{pck[len(pck) // 2]:g}_all"
    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_path,
        "model_sha256": summary["model_sha256"][:16],
        "split": split,
        "n_images": summary["n_images"],
        "imgsz": imgsz,
        "det_conf": det_conf,
        "kp_conf": kp_conf,
        "match_px": match_px,
        "pose_map50": ultra.get(MAP50_COL, ""),
        "pose_map50_95": ultra.get(MAP50_95_COL, ""),
        "kp_recall": overall["recall"],
        "kp_precision": overall["precision"],
        "err_median_px": overall["err_median"],
        t_mid: overall.get(t_mid),
        "calib_ok_rate": calib.get("ok_rate"),
        "calib_correct_rate": calib.get("correct_rate"),
        "calib_err_median_px": calib.get("err_median_px"),
        "report": out.relative_to(ws).as_posix(),
    }
    _append_csv(ws / "models" / "evaluations_v2.csv", row)

    _log(
        f"pose mAP50={row['pose_map50']} mAP50-95={row['pose_map50_95']} | keypoints "
        f"(det>={det_conf}, kp>={kp_conf}, match<={match_px} px@1920): "
        f"recall={overall['recall']} precision={overall['precision']} "
        f"median error={overall['err_median']} px@1920 "
        + " ".join(f"PCK{t:g}={overall.get(f'pck{t:g}_all')}" for t in pck)
        + f" | homography of {calib.get('n_images_gt_calibratable')} calibratable: "
        f"valid {calib.get('ok_rate')}, correct (<= {calib.get('correct_px')} px@1920 vs labels) "
        f"{calib.get('correct_rate')}, median {calib.get('err_median_px')} px "
        f"({overall['images']} images)"
    )
    ranked = sorted(
        (k for k in scores["per_keypoint"] if k["n_ref"]),
        key=lambda k: (k["recall"] or 0.0, k["kp"]),
    )
    _log("hardest keypoints (lowest recall; swaps = labelled point covered by another index):")
    for k in ranked[:8]:
        _log(
            f"  {k['kp']:>3s} {k['name']:<32s} n={k['n_ref']:<5d} groups={k['n_groups_ref']:<4d} "
            f"recall={k['recall']} mis={k['n_mislocalized']} fn={k['n_fn']} "
            f"median_err={k['err_median']} swaps(mirror/rot180/other)="
            f"{k['swap_mirror']}/{k['swap_rot180']}/{k['swap_other']}"
        )
    critical = ("p5", "p29", "p39", "p47")
    net_kp = {k["kp"]: k for k in scores["per_keypoint"]}
    _log(
        "field geometry (same labels; tune min_conf / fix_px on val only): "
        "recall / precision / PCK10 | p5 p29 p39 p47 recall"
    )
    for name, block, per_kp in [("network", overall, net_kp)] + [
        (mode, geometry[mode]["keypoints"], geometry[mode]["per_keypoint"])
        for mode in ("fill", "fix")
    ]:
        _log(
            f"  {name:<8s} {block['recall']} / {block['precision']} / {block.get('pck10_all')} | "
            + " ".join(str(per_kp[k]["recall"]) for k in critical)
            + ("" if name == "network" else f" | camera {geometry[name]['camera_ok_rate']:.0%}")
        )
    _log(f"evaluate done -> {out}")
    return out


def evaluate_geometry(pred, out: Path, *, det_conf, kp_conf, match_px, pck, settings) -> dict:
    """Score the network output after field-geometry fill and fix (``freekiki_geom``).

    Writes ``per_keypoint_geom_<mode>.csv``; returns, per mode, the overall
    keypoint table, a per-keypoint recall/precision map and the camera stats.
    """
    G = _geom()
    result: dict = {"settings": G.GEOM_DEFAULTS | dict(settings)}
    for mode in ("fill", "fix"):
        gpred, stats = G.refine_predictions(
            pred, det_conf=det_conf, kp_conf=kp_conf, mode=mode, settings=settings
        )
        gscores = diag.score_predictions(
            gpred, det_conf=det_conf, kp_conf=kp_conf, match_px=match_px, pck=pck
        )
        diag.write_csv(out / f"per_keypoint_geom_{mode}.csv", gscores["per_keypoint"])
        result[mode] = stats | {
            "keypoints": gscores["overall"],
            "calib": {k: v for k, v in gscores["calib"].items() if k != "per_image"},
            "per_keypoint": {
                k["kp"]: {"recall": k["recall"], "precision": k["precision"], "n_ref": k["n_ref"]}
                for k in gscores["per_keypoint"]
            },
        }
    return result


def _append_csv(path: Path, row: dict) -> None:
    """Append one row; a header change starts a new ``<name>_<ts>.csv`` beside it."""
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != list(row):
            path = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
    new_file = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def np_savez(path: Path, arrays: dict) -> None:
    import numpy as np

    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    np.savez_compressed(path, **arrays)


def sweep(
    ws,
    eval_dir,
    *,
    det_confs=(0.05, 0.1, 0.25, 0.5),
    kp_confs=(0.1, 0.2, 0.3, 0.5, 0.7),
    match_px: float = 25.0,
    allow_test: bool = False,
) -> Path:
    """Offline det_conf x kp_conf grid on a saved evaluation (``predictions.npz``).

    Shows the recall / false-positive trade-off per threshold pair. Refuses a
    ``test`` evaluation unless ``allow_test`` (choose thresholds on ``val``).
    """
    ws = Path(ws).expanduser().resolve()
    eval_dir = Path(eval_dir).expanduser().resolve()
    summary = json.loads((eval_dir / "eval_summary.json").read_text(encoding="utf-8"))
    if summary.get("split") in ("test", "hard") and not allow_test:
        raise ValueError(f"sweep on the {summary['split']} split refused: choose thresholds on val")
    pred = load_predictions(eval_dir)
    out = (
        ws
        / "outputs"
        / f"processed_freekiki_sweep_{summary['split']}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for dc in det_confs:
        for kc in kp_confs:
            s = diag.score_predictions(pred, det_conf=dc, kp_conf=kc, match_px=match_px)
            o, c = s["overall"], s["calib"]
            rows.append(
                {k: o.get(k) for k in SWEEP_COLS}
                | {
                    "calib_ok_rate": c.get("ok_rate"),
                    "calib_correct_rate": c.get("correct_rate"),
                    "calib_err_median_px": c.get("err_median_px"),
                }
            )
            _log(
                f"det>={dc} kp>={kc}: recall={o['recall']} precision={o['precision']} "
                f"fp={o['n_fp']} mis={o['n_mislocalized']} pck10_all={o.get('pck10_all')} "
                f"calib_ok={c.get('ok_rate')} calib_correct={c.get('correct_rate')}"
            )
    diag.write_csv(out / "sweep.csv", rows)
    (out / "README.txt").write_text(
        f"Threshold sweep of {eval_dir} (split {summary['split']}, model {summary['model']}).\n"
        f"match_px = {match_px} px@1920. Metrics: see eval_summary.json 'definitions'.\n",
        encoding="utf-8",
    )
    _log(f"sweep done -> {out / 'sweep.csv'}")
    return out


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def audit(ws, *, dup_bits: int = 10, hash_images: bool = True) -> Path:
    """Read-only dataset audit into ``outputs/processed_freekiki_audit_<ts>/``."""
    ws = Path(ws).expanduser().resolve()
    out = ws / "outputs" / f"processed_freekiki_audit_{datetime.now():%Y%m%d_%H%M%S}"
    diag.audit_dataset(dataset_dir(ws), out, dup_bits=dup_bits, hash_images=hash_images, log=_log)
    return out


def _eval_for(ws, ref: str, *, split: str, like: dict | None = None) -> Path:
    """An evaluation folder as-is, or the (cached or new) evaluation of a model."""
    path = Path(ref).expanduser()
    if (path / "eval_summary.json").is_file():
        return path.resolve()
    model = resolve_model(ws, ref)
    found = find_eval(ws, model, split=split, like=like)
    if found:
        return found
    kwargs = {} if like is None else {k: like[k] for k in ("det_conf", "kp_conf", "match_px")}
    return evaluate(ws, model=model, split=split, **kwargs)


def compare(ws, *, baseline: str | None = None, candidate: str, promote: bool = False) -> dict:
    """Gate ``candidate`` against ``baseline`` (eval folders, models or slots) on the same split.

    ``baseline`` defaults to the model in the candidate's size slot, or to the
    active model while that slot is empty. With ``promote`` the candidate is
    copied into its slot (``models/freekiki_<slot>.pt``, previous file backed up)
    only when every gate check passes. The decision is always logged.
    """
    ws = Path(ws).expanduser().resolve()
    settings = load_settings(ws)
    gate = settings["promotion"]
    split = gate.get("split", "val")
    cand_dir = _eval_for(ws, candidate, split=split)
    like = json.loads((cand_dir / "eval_summary.json").read_text(encoding="utf-8"))
    found = checkpoint_slot(like.get("model") or "")
    slot, arch = found or base_slot(candidate, settings["models"]["default"])
    target = slot_path(ws, settings, slot)
    if baseline is None:
        baseline = str(target) if target.is_file() else "active"
    base_dir = _eval_for(ws, baseline, split=like["split"], like=like)
    decision = diag.compare_evals(base_dir, cand_dir, gate)
    (cand_dir / "promotion_decision.json").write_text(json.dumps(decision, indent=2), "utf-8")
    for c in decision["checks"]:
        _log(
            f"  {'ok  ' if c['pass'] else 'FAIL'} {c['check']:<22s} "
            f"{c['baseline']} -> {c['candidate']} (tolerance {c['tolerance']})"
        )
    for note in decision["notes"]:
        _log(f"  note: {note}")
    promoted = bool(promote and decision["promote"])
    if promoted:
        if target.is_file():
            shutil.copy2(
                target, target.with_name(f"{target.stem}_before_{datetime.now():%Y%m%d_%H%M%S}.pt")
            )
        values = {c["check"]: c["candidate"] for c in decision["checks"]}
        _install_slot(
            ws,
            settings,
            slot,
            Path(decision["candidate_model"]),
            {
                "arch": arch,
                "run": Path(decision["candidate_model"]).stem,
                "pose_map50_95": values.get("pose_map50_95") or 0.0,
                "pck10_all": values.get("overall_pck10_all", ""),
            },
        )
    log_promotion(
        ws,
        {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "candidate": decision["candidate_model"],
            "mode": "compare --promote" if promote else "compare",
            "promoted": promoted,
            "reasons": " | ".join(decision["reasons"]) or "gate passed",
        },
    )
    verdict = "passes" if decision["promote"] else "FAILS"
    _log(
        f"compare ({decision['split']}): candidate {verdict} the gate"
        + (f" -> PROMOTED to {slot_file(slot)}" if promoted else "")
        + ("" if decision["promote"] else " | " + "; ".join(decision["reasons"]))
    )
    return decision


def bench(
    ws,
    *,
    batches=(2, 8, 16),
    workers=(8,),
    fraction: float = 0.02,
    imgsz: int | None = None,
    model: str | None = None,
    device: str | None = None,
    warmup: int = 5,
) -> Path:
    """Short throughput / peak-VRAM benchmark of training batch x dataloader workers.

    One epoch on ``fraction`` of the train split per setting, no validation, no
    saved weights. Throughput counts only the batches after ``warmup`` (images
    per second of forward + backward + dataloading); peak VRAM is
    ``torch.cuda.max_memory_reserved``. Speed only: says nothing about quality.

    Ultralytics halves the batch after a CUDA OOM in the first epoch and
    retries, so the batch actually trained (``batch_effective``) is read from
    the trainer and only the batches of the last attempt are timed.
    """
    import time

    import torch
    from ultralytics import YOLO

    ws = Path(ws).expanduser().resolve()
    defaults = load_settings(ws)["train"]
    imgsz = int(imgsz or defaults["imgsz"])
    model_path = resolve_model(ws, model or defaults["base"])
    yaml_path = refresh_yaml_path(dataset_dir(ws) / "data.yaml")
    out = ws / "outputs" / f"processed_freekiki_bench_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for nw in workers:
        for bs in batches:
            stamps: list[tuple[float, int]] = []
            net = YOLO(model_path)
            net.add_callback(
                "on_train_batch_end",
                lambda t, s=stamps: s.append((time.perf_counter(), int(t.batch_size))),
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            row: dict = {"batch": bs, "workers": nw, "imgsz": imgsz, "fraction": fraction}
            try:
                net.train(
                    data=str(yaml_path),
                    epochs=1,
                    batch=bs,
                    workers=nw,
                    imgsz=imgsz,
                    device=device,
                    fraction=fraction,
                    val=False,
                    save=False,
                    plots=False,
                    project=str(out),
                    name=f"b{bs}_w{nw}",
                    exist_ok=True,
                    verbose=False,
                )
                row["status"] = "ok"
            except torch.cuda.OutOfMemoryError:
                row["status"] = "oom"
            eff = stamps[-1][1] if stamps else None
            if eff is not None and eff != bs and row["status"] == "ok":
                row["status"] = "oom_reduced"
            row["batch_effective"] = eff
            last_attempt = [ts for ts, b in stamps if b == eff]
            timed = last_attempt[warmup:]
            if eff and len(timed) >= 2:
                sec = (timed[-1] - timed[0]) / (len(timed) - 1)
                row |= {"s_per_batch": round(sec, 4), "images_per_s": round(eff / sec, 2)}
            row["batches_timed"] = max(0, len(timed) - 1)
            if torch.cuda.is_available():
                row["peak_vram_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
                row["peak_vram_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            rows.append(row)
            _log(f"bench: {row}")
            del net
    diag.write_csv(out / "bench.csv", rows)
    (out / "README.txt").write_text(
        "FreeKiki training benchmark (vailá): speed only, not quality.\n"
        f"model: {model_path}\ndata: {yaml_path}\nimgsz: {imgsz}\nfraction: {fraction}\n"
        f"images_per_s = batch_effective / mean seconds between on_train_batch_end after {warmup} "
        "warm-up batches (forward + backward + dataloading).\n"
        "status oom_reduced: Ultralytics hit CUDA OOM and halved the batch; batch_effective is\n"
        "what really trained, so the requested batch does not fit this GPU.\n"
        "peak_vram_* = torch.cuda.max_memory_reserved / allocated (GiB) during the run.\n"
        "Effective batch stays nbs=64 through gradient accumulation (Ultralytics), so a\n"
        "larger batch changes speed and BatchNorm statistics, not the optimizer step size.\n",
        encoding="utf-8",
    )
    _log(f"bench done -> {out / 'bench.csv'}")
    return out


def _geom():
    """``freekiki_geom`` (lazy: it pulls the DLT modules only when needed)."""
    try:
        from . import freekiki_geom
    except ImportError:
        import freekiki_geom  # ty: ignore[unresolved-import]
    return freekiki_geom


GEOM_ACCEPTED = ("D", "Dg", "G", "Gx")  # codes written to the geometry CSV


def _draw_geom(frame, xy, gcodes: list[str]):
    """Geometry points on top of the overlay: G yellow (filled), Gx orange (replaced)."""
    import cv2

    colors = {"G": (0, 255, 255), "Gx": (0, 140, 255)}
    for i, code in enumerate(gcodes):
        if code in colors and all(map(math.isfinite, xy[i])):
            p = (int(xy[i][0]), int(xy[i][1]))
            cv2.circle(frame, p, 6, colors[code], 2)
            cv2.putText(frame, f"p{i}{code[1:]}", (p[0] + 6, p[1] + 14), 0, 0.4, colors[code], 1)
    return frame


def keypoints_row(frame: int, xy, kconf, kp_conf: float) -> list:
    """One getpixelvideo row: ``frame, p0_x, p0_y, ...``; blanks below ``kp_conf``/NaN."""
    row: list = [frame]
    for i in range(NKP):
        c = float(kconf[i]) if kconf is not None else float("nan")
        if xy is None or not c >= kp_conf or not all(map(math.isfinite, xy[i])):
            row += ["", ""]
        else:
            row += [f"{float(xy[i][0]):.2f}", f"{float(xy[i][1]):.2f}"]
    return row


def getpixelvideo_header() -> list[str]:
    header = ["frame"]
    for i in range(NKP):
        header += [f"p{i}_x", f"p{i}_y"]
    return header


def _draw_overlay(frame, xy, kconf, kp_conf: float, bones: list[tuple[int, int]]):
    import cv2

    ok = [xy is not None and float(kconf[i]) >= kp_conf for i in range(NKP)]
    for a, b in bones:
        if ok[a] and ok[b]:
            pa = (int(xy[a][0]), int(xy[a][1]))
            pb = (int(xy[b][0]), int(xy[b][1]))
            cv2.line(frame, pa, pb, (255, 255, 255), 1, cv2.LINE_AA)
    for i in range(NKP):
        if ok[i]:
            p = (int(xy[i][0]), int(xy[i][1]))
            cv2.circle(frame, p, 4, (0, 255, 0), -1)
            cv2.putText(frame, f"p{i}", (p[0] + 5, p[1] - 5), 0, 0.4, (0, 255, 255), 1)
    return frame


def _draw_diagnostic(frame, xy, kc, codes: list[str], names: list[str], header: str):
    """Accepted points green (index, name, conf); rejected points with conf >= 0.1 red."""
    import cv2

    for i in range(NKP):
        if xy is None or not math.isfinite(float(kc[i])) or (codes[i] != "D" and kc[i] < 0.1):
            continue
        p = (int(xy[i][0]), int(xy[i][1]))
        color = (0, 255, 0) if codes[i] == "D" else (0, 0, 255)
        cv2.circle(frame, p, 5, color, -1 if codes[i] == "D" else 1)
        label = f"p{i} {names[i]} {float(kc[i]):.2f}" + ("" if codes[i] == "D" else f" {codes[i]}")
        cv2.putText(frame, label, (p[0] + 6, p[1] - 6), 0, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (p[0] + 6, p[1] - 6), 0, 0.45, color, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(frame, header, (8, 21), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# Per-keypoint status codes of detect (field_kps_status.csv)
POINT_CODES = {
    "D": "detected: box conf >= conf and keypoint conf >= kp_conf",
    "N": "no field instance in the frame",
    "Rb": "rejected: best box conf < conf",
    "Rk": "rejected: keypoint conf < kp_conf",
    "Ro": "rejected: outside the image",
    "Rd": "rejected: same pixel (< 3 px@1920) as a higher-confidence index",
    "I": "interpolated (field_kps_filled_*.csv only; gap <= --fill-gaps frames, same shot)",
}


def point_codes(xy, kc, box_conf: float, conf: float, kp_conf: float, size) -> list[str]:
    """Status code of every keypoint in one frame (see ``POINT_CODES``)."""
    import numpy as np

    if xy is None or not math.isfinite(box_conf):
        return ["N"] * NKP
    if box_conf < conf:
        return ["Rb"] * NKP
    w, h = size
    codes = []
    for i in range(NKP):
        x, y = float(xy[i][0]), float(xy[i][1])
        if not float(kc[i]) >= kp_conf:
            codes.append("Rk")
        elif not (0 <= x < w and 0 <= y < h):
            codes.append("Ro")
        else:
            codes.append("D")
    accepted = np.array([c == "D" for c in codes])
    for a, b in diag.duplicate_points(xy, accepted, w):
        codes[b if kc[a] >= kc[b] else a] = "Rd"
    return codes


def video_quality(
    xy_seq: list, kc_seq: list, kp_conf: float, *, width=1920.0, cuts=None, calib=None
) -> dict:
    """Label-free indicators of a detection run (one entry per processed frame).

    NOT accuracy: without labels none of these says whether a point is right.

    * ``detection_rate``: frames with an accepted field instance;
    * ``mean_visible_kps``: accepted keypoints (conf >= ``kp_conf``) per frame;
    * ``min4_kps_rate``: frames with >= 4 accepted keypoints (a count only);
    * ``calib_ok_rate``: frames whose accepted planar keypoints give a valid
      homography (``freekiki_diag.fit_field_homography``: RANSAC inliers,
      spread, reprojection RMSE, orientation); ``calib_status`` counts;
    * ``mean_kp_conf``: mean confidence of the accepted keypoints;
    * ``cuts``: detected shot changes;
    * displacement / residual (``freekiki_diag.temporal_metrics``): residual
      is the motion left after removing the camera motion between frames.
    """
    import numpy as np

    n = len(kc_seq)
    pts = np.full((n, NKP, 2), np.nan)
    visible = np.zeros((n, NKP), dtype=bool)
    confs = []
    for t, (xy, kc) in enumerate(zip(xy_seq, kc_seq, strict=True)):
        if xy is None or kc is None:
            continue
        vis = np.nan_to_num(np.asarray(kc, dtype=float), nan=-1.0) >= kp_conf
        visible[t] = vis
        pts[t, vis] = np.asarray(xy)[vis]
        confs.extend(np.asarray(kc)[vis].tolist())
    counts = visible.sum(axis=1)
    cuts = np.zeros(n, dtype=bool) if cuts is None else np.asarray(cuts, dtype=bool)
    _, _, xyz = diag.load_field_points()
    detected = sum(kc is not None for kc in kc_seq)
    out = {
        "frames": n,
        "detection_rate": round(detected / n, 4) if n else 0.0,
        "mean_visible_kps": round(float(counts.mean()), 2) if n else 0.0,
        "min4_kps_rate": round(float((counts >= 4).mean()), 4) if n else 0.0,
        "mean_kp_conf": round(float(np.mean(confs)), 4) if confs else None,
        "cuts": int(cuts.sum()),
    }
    if calib is not None:
        out["calib_ok_rate"] = round(sum(s == "ok" for s in calib) / n, 4) if n else 0.0
        out["calib_status"] = dict(sorted(Counter(calib).items()))
    out |= diag.temporal_metrics(pts, cuts, xyz[:, 2] == 0, width)
    return out


def detect_video(
    ws,
    video,
    *,
    model: str = "active",
    output_dir=None,
    stride: int = 1,
    start: int = 0,
    max_frames: int = 0,
    conf: float | None = None,
    kp_conf: float | None = None,
    imgsz: int | None = None,
    device: str | None = None,
    overlay: bool = True,
    fill_gaps: int = 0,
    diag_frames: int = 12,
    cut_threshold: float = 0.4,
    geom: str | None = None,
) -> Path:
    """Detect the 49 field keypoints in a video; returns the output folder.

    Outputs (``README.txt`` explains them): accepted points in getpixelvideo
    format, raw best-instance predictions, per-point status codes with the
    rejection reason, per-frame issues (cut, homography status), label-free
    ``quality.json`` and a few annotated ``diag_frames/``. ``fill_gaps`` > 0
    writes separate ``field_kps_filled_*`` CSVs with short gaps (same shot
    only) linearly interpolated and marked ``I``. ``geom`` (fill | fix | off,
    default from settings) adds the field-geometry CSVs (``freekiki_geom``):
    the network CSVs above are never changed by it.
    """
    import cv2
    import numpy as np

    ws = Path(ws).expanduser().resolve()
    video = Path(video).expanduser().resolve()
    all_settings = load_settings(ws)
    defaults = all_settings["detect"]
    geom_settings = all_settings["geometry"]
    geom_mode = str(geom or geom_settings.get("mode", "fill"))
    G = _geom() if geom_mode != "off" else None
    conf = float(defaults["conf"] if conf is None else conf)
    kp_conf = float(defaults["kp_conf"] if kp_conf is None else kp_conf)
    stride = max(1, int(stride))
    model_path = resolve_model(ws, model)
    model_info = describe_model(ws, model, model_path)
    _log(f"model: {model_info}")
    predictor = load_predictor(
        model_path, imgsz=imgsz, fallback_imgsz=defaults["imgsz"], device=device
    )
    imgsz = predictor.imgsz
    base_out = Path(output_dir).expanduser() if output_dir else video.parent
    out = base_out / f"freekiki_predict_{video.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    (out / "diag_frames").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    bones = load_bones()
    names, _, xyz = diag.load_field_points()
    planar, world = xyz[:, 2] == 0, xyz[:, :2]
    writer = None
    if overlay:
        writer = cv2.VideoWriter(
            str(out / f"{video.stem}_freekiki_overlay.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),  # ty: ignore[unresolved-attribute]
            fps / stride,
            (width, height),
        )
    _log(f"detect: {video.name} ({total} frames) model={model_path} -> {out}")

    n_expected = max(1, (min(total - start, max_frames or total) + stride - 1) // stride)
    diag_at = set(np.linspace(0, n_expected - 1, max(0, diag_frames)).round().astype(int).tolist())
    snapshot_at = n_expected // 2
    frames, raw_rows, status_rows, issue_rows = [], [], [], []
    pix_rows, conf_rows, xy_seq, kc_seq, cuts, calib = [], [], [], [], [], []
    all_codes: Counter = Counter()
    geom_rows, geom_status, geom_kps, dlt2d_rows, dlt3d_rows = [], [], [], [], []
    geom_codes: Counter = Counter()
    geom_sources: Counter = Counter()
    geom_fills: Counter = Counter()
    geom_fixes: Counter = Counter()
    prev_sig = None
    frame_idx, processed, cut_shots = start, 0, 0
    try:
        while True:
            if (frame_idx - start) % stride:
                if not cap.grab():
                    break
                frame_idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            box_conf, xy, kc = predictor.predict(frame)
            sig = diag.frame_signature(frame)
            cut = diag.is_cut(prev_sig, sig, cut_threshold)
            prev_sig = sig
            codes = point_codes(xy, kc, box_conf, conf, kp_conf, (width, height))
            all_codes.update(codes)
            acc = np.array([c == "D" for c in codes])
            kc_acc = None if kc is None or box_conf < conf else np.where(acc, kc, np.nan)
            xy_acc = None if kc_acc is None else xy
            fit = diag.frame_homography(xy_acc, kc_acc, kp_conf, planar, world, width)
            gxy, gcodes = None, []
            if G is not None:
                gxy, gkc, gcodes, gmodel = (
                    G.refine_keypoints(
                        xy, kc, codes, width, height, mode=geom_mode, kp_conf=kp_conf,
                        settings=geom_settings,
                    )
                    if xy is not None and kc_acc is not None
                    else (xy, kc, list(codes), None)
                )  # fmt: skip
                gacc = np.array([c in GEOM_ACCEPTED for c in gcodes])
                gkc_acc = None if gxy is None else np.where(gacc, gkc, np.nan)
                geom_rows.append(keypoints_row(frame_idx, gxy, gkc_acc, kp_conf))
                src = gmodel["source"] if gmodel and gmodel["status"] == "ok" else "none"
                geom_status.append(
                    [
                        frame_idx,
                        src,
                        "" if gmodel is None else gmodel["focal_px"] or "",
                        "" if gmodel is None else gmodel["cam_height_m"] or "",
                        "" if gmodel is None else gmodel["rmse_px"] or "",
                    ]
                    + gcodes
                )
                d2, d3 = G.dlt_coefficients(gmodel)
                dlt2d_rows.append(
                    [frame_idx] + ([""] * 8 if d2 is None else [f"{v:.10g}" for v in d2])
                )
                dlt3d_rows.append(
                    [frame_idx] + ([""] * 11 if d3 is None else [f"{v:.10g}" for v in d3])
                )
                geom_codes.update(gcodes)
                geom_sources[src] += 1
                geom_fills.update(f"p{i}" for i, c in enumerate(gcodes) if c == "G")
                geom_fixes.update(f"p{i}" for i, c in enumerate(gcodes) if c == "Gx")
                geom_kps.append(int(gacc.sum()))
            frames.append(frame_idx)
            xy_seq.append(xy_acc)
            kc_seq.append(kc_acc)
            cuts.append(cut)
            calib.append(fit["status"])
            pix_rows.append(keypoints_row(frame_idx, xy_acc, kc_acc, kp_conf))
            conf_rows.append(
                [frame_idx]
                + (
                    [f"{float(c):.3f}" for c in kc]
                    if kc is not None and box_conf >= conf
                    else [""] * NKP
                )
            )
            raw = [frame_idx, "" if not math.isfinite(box_conf) else f"{box_conf:.4f}"]
            for i in range(NKP):
                raw += (
                    ["", "", ""]
                    if xy is None
                    else [f"{xy[i][0]:.2f}", f"{xy[i][1]:.2f}", f"{float(kc[i]):.4f}"]
                )
            raw_rows.append(raw)
            status_rows.append(
                [
                    frame_idx,
                    int(cut),
                    int(acc.sum()),
                    fit["status"],
                    fit["n_inliers"],
                    fit["rmse_px"],
                ]
                + codes
            )
            issues = []
            if cut:
                issues.append("cut")
            if codes[0] == "N":
                issues.append("no_instance")
            elif codes[0] == "Rb":
                issues.append("box_below_conf")
            if fit["status"] != "ok":
                issues.append(f"homography_{fit['status']}")
            if "Rd" in codes:
                issues.append("duplicate_points")
            if issues:
                issue_rows.append(
                    {
                        "frame": frame_idx,
                        "issues": ";".join(issues),
                        "box_conf": diag._num(box_conf, 3),
                        "n_accepted": int(acc.sum()),
                        "n_rejected_kp_conf": codes.count("Rk"),
                        "homography_n_points": fit["n_points"],
                        "homography_inliers": fit["n_inliers"],
                        "homography_rmse_px": fit["rmse_px"],
                        "minor_std_m": fit["minor_std_m"],
                    }
                )
            if processed in diag_at or (cut and cut_shots < 8):
                cut_shots += int(cut)
                header = (
                    f"frame {frame_idx} | box {box_conf:.2f} | accepted {int(acc.sum())} | "
                    f"H {fit['status']}" + (" | CUT" if cut else "")
                )
                cv2.imwrite(
                    str(out / "diag_frames" / f"frame_{frame_idx:06d}.jpg"),
                    _draw_diagnostic(frame.copy(), xy, kc, codes, names, header),
                )
            if writer is not None or processed == snapshot_at:
                drawn = _draw_overlay(frame, xy_acc, kc_acc, kp_conf, bones)
                if G is not None and gxy is not None:
                    drawn = _draw_geom(drawn, gxy, gcodes)
                if writer is not None:
                    writer.write(drawn)
                if processed == snapshot_at:
                    cv2.imwrite(str(out / f"{video.stem}_freekiki_snapshot.png"), drawn)
            processed += 1
            frame_idx += 1
            if processed % 100 == 0:
                _log(f"  frame {frame_idx}/{total}")
            if max_frames and processed >= max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    def write_rows(name: str, header: list, rows: list) -> None:
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    write_rows("field_kps_getpixelvideo.csv", getpixelvideo_header(), pix_rows)
    write_rows("field_kps_conf.csv", ["frame"] + [f"p{i}_conf" for i in range(NKP)], conf_rows)
    write_rows(
        "field_kps_raw.csv",
        ["frame", "box_conf"] + [f"p{i}_{k}" for i in range(NKP) for k in ("x", "y", "conf")],
        raw_rows,
    )
    write_rows(
        "field_kps_status.csv",
        ["frame", "cut", "n_accepted", "homography", "inliers", "rmse_px"]
        + [f"p{i}" for i in range(NKP)],
        status_rows,
    )
    diag.write_csv(out / "frame_issues.csv", issue_rows or [{"frame": "", "issues": ""}])
    if G is not None:
        write_rows("field_kps_geom_getpixelvideo.csv", getpixelvideo_header(), geom_rows)
        write_rows(
            "field_kps_geom_status.csv",
            ["frame", "camera", "focal_px", "cam_height_m", "rmse_px"]
            + [f"p{i}" for i in range(NKP)],
            geom_status,
        )
        # Per-frame field calibration for rec2d.py / rec3d.py (blank row = no camera).
        write_rows(f"{video.stem}.dlt2d", ["frame"] + [f"p{j}" for j in range(1, 9)], dlt2d_rows)
        write_rows(f"{video.stem}.dlt3d", ["frame"] + [f"p{j}" for j in range(1, 12)], dlt3d_rows)
    if fill_gaps > 0:
        pts = (
            np.stack(
                [
                    np.full((NKP, 2), np.nan)
                    if x is None
                    else np.where((np.nan_to_num(k, nan=-1.0) >= kp_conf)[:, None], x, np.nan)
                    for x, k in zip(xy_seq, kc_seq, strict=True)
                ]
            )
            if xy_seq
            else np.zeros((0, NKP, 2))
        )
        filled, interp = diag.fill_short_gaps(pts, np.asarray(cuts), fill_gaps)
        ones = np.ones(NKP)
        write_rows(
            "field_kps_filled_getpixelvideo.csv",
            getpixelvideo_header(),
            [keypoints_row(f, p, ones, 0.5) for f, p in zip(frames, filled, strict=True)],
        )
        write_rows(
            "field_kps_filled_source.csv",
            ["frame"] + [f"p{i}" for i in range(NKP)],
            [
                [f]
                + [
                    "I" if interp[t, i] else ("D" if np.isfinite(pts[t, i, 0]) else "")
                    for i in range(NKP)
                ]
                for t, f in enumerate(frames)
            ],
        )
    (out / "README.txt").write_text(
        "FreeKiki field keypoints (vailá)\n"
        f"video: {video}\nmodel: {model_path}\nmodel_sha256: {file_sha256(model_path)}\n"
        f"model_info: {model_info}\n"
        f"dimensions: {width}x{height}\nframes: {processed} (start={start}, "
        f"stride={stride})\ndetection (box) conf: {conf}\nkeypoint conf: {kp_conf}\n"
        f"imgsz: {imgsz}\nfill gaps: {fill_gaps} frames\ngeometry: {geom_mode}\n"
        "schema: vaila/models/soccerfield_kiki.csv (p0..p48, 0-based)\n\n"
        "field_kps_getpixelvideo.csv  accepted points only (status D)\n"
        "field_kps_conf.csv           keypoint conf of the accepted box (blank: no box)\n"
        f"field_kps_raw.csv            best instance down to box conf {RAW_CONF} (no filter)\n"
        "field_kps_status.csv         per point code, cut flag, homography status\n"
        "frame_issues.csv             frames with a cut / no box / no valid homography\n"
        "diag_frames/                 annotated frames (green accepted, red rejected)\n"
        "quality.json                 label-free indicators - NOT accuracy\n"
        + (
            ""
            if G is None
            else "field_kps_geom_getpixelvideo.csv  network points + field-geometry fills/fixes\n"
            "field_kps_geom_status.csv    camera (dlt3d | homography | planar | none), focal,\n"
            "                             camera height, fit RMSE and per-point codes G/Gx/Dg\n"
            f"{video.stem}.dlt2d / .dlt3d   per-frame field calibration (8 / 11 DLT coefficients;\n"
            "                             blank = no camera) for rec2d.py / rec3d.py. World: metres,\n"
            "                             origin centre spot, x right goal, y top touchline, z up\n"
            "geometry: a camera fitted to the accepted points says where every keypoint\n"
            "must be; a geometry point may be occluded (see freekiki_geom.py).\n"
        )
        + "\nstatus codes:\n"
        + "".join(f"  {k:<3s} {v}\n" for k, v in POINT_CODES.items())
        + ("" if G is None else "".join(f"  {k:<3s} {v}\n" for k, v in G.GEOM_CODES.items()))
        + "\nhomography: world (metres, z = 0 points) -> image, RANSAC; ok needs >= 6 inliers,\n"
        "field spread >= 1 m on the minor axis, RMSE <= 6 px@1920 and the non-mirrored\n"
        "orientation (see freekiki_diag.fit_field_homography).\n",
        encoding="utf-8",
    )
    quality = (
        {"video": video.name, "model": model_path, "conf": conf, "kp_conf": kp_conf}
        | video_quality(xy_seq, kc_seq, kp_conf, width=width, cuts=cuts, calib=calib)
        | {"point_codes": dict(sorted(all_codes.items())), "note": "label-free, not accuracy"}
    )
    if G is not None:
        n_geom = max(1, len(geom_kps))
        quality |= {
            "geom_mode": geom_mode,
            "geom_camera_ok_rate": round(
                sum(v for k, v in geom_sources.items() if k != "none") / n_geom, 4
            ),
            "geom_camera_sources": dict(sorted(geom_sources.items())),
            "geom_mean_kps": round(sum(geom_kps) / n_geom, 2),
            "geom_fills": dict(geom_fills.most_common()),
            "geom_fixes": dict(geom_fixes.most_common()),
            "geom_flagged": geom_codes.get("Dg", 0),
            "geom_dlt2d_rate": round(sum(r[1] != "" for r in dlt2d_rows) / n_geom, 4),
            "geom_dlt3d_rate": round(sum(r[1] != "" for r in dlt3d_rows) / n_geom, 4),
        }
    (out / "quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    _log(
        f"detect done: {processed} frames | detected {quality['detection_rate']:.0%} | "
        f"accepted kps/frame {quality['mean_visible_kps']} | >=4 kps "
        f"{quality['min4_kps_rate']:.0%} | homography ok {quality['calib_ok_rate']:.0%} | "
        f"cuts {quality['cuts']} | residual median {quality['residual_median_px']} px@1920 "
        f"(camera-motion removed) -> {out}"
    )
    if G is not None:
        rare = " ".join(f"p{i}={geom_fills.get(f'p{i}', 0)}" for i in (5, 29, 39, 47))
        _log(
            f"geometry {geom_mode}: camera {quality['geom_camera_ok_rate']:.0%} "
            f"{quality['geom_camera_sources']} | kps/frame {quality['geom_mean_kps']} | "
            f"filled {sum(geom_fills.values())} ({rare}) | fixed {sum(geom_fixes.values())} | "
            f"flagged {quality['geom_flagged']} | DLT2D {quality['geom_dlt2d_rate']:.0%} "
            f"DLT3D {quality['geom_dlt3d_rate']:.0%} -> {video.stem}.dlt2d/.dlt3d"
        )
    return out


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def workspace_for_model_file(model_file: str | Path) -> Path | None:
    """Find the owning workspace when a model file is selected in the GUI."""
    path = Path(model_file).expanduser()
    if not path.is_file():
        return None
    return next((parent for parent in path.parents if (parent / CONFIG_NAME).is_file()), None)


def validate_detection_workspace(ws, model: str) -> None:
    """Fail before creating output when detection lacks a workspace or active model."""
    if not str(ws).strip():
        raise FileNotFoundError("Choose a FreeKiki workspace folder containing freekiki.toml.")
    load_settings(Path(ws).expanduser())
    if not model.strip():
        raise ValueError("Choose 'active', a model slot or a model .pt file for detection.")
    if model == "active" or _slot_name(model):
        resolve_model(ws, model)


def detect_videos(ws, source, *, output_dir=None, **kwargs) -> Path:
    """Detect on one video or on every video of a folder.

    A folder writes ``processed_freekiki_batch_<ts>/`` (inside ``output_dir``,
    default the folder itself) with one ``freekiki_predict_<stem>_<ts>/``
    sub-folder per video and ``quality_summary.csv`` comparing them.
    Returns the output folder.
    """
    validate_detection_workspace(ws, kwargs.get("model", "active"))
    source = Path(source).expanduser().resolve()
    if source.is_file():
        return detect_video(ws, source, output_dir=output_dir, **kwargs)
    videos = sorted(p for p in source.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise FileNotFoundError(f"No videos ({', '.join(sorted(VIDEO_EXTS))}) in {source}")
    base_out = Path(output_dir).expanduser() if output_dir else source
    batch = base_out / f"processed_freekiki_batch_{datetime.now():%Y%m%d_%H%M%S}"
    batch.mkdir(parents=True, exist_ok=True)
    _log(f"batch: {len(videos)} videos -> {batch}")
    rows = []
    failed = []
    for video in videos:
        try:
            out = detect_video(ws, video, output_dir=batch, **kwargs)
        except (OSError, ValueError, RuntimeError) as exc:
            # One bad video must not lose the others (hours of GPU on a long batch).
            _log(f"ERROR on {video.name}: {type(exc).__name__}: {exc} - continuing")
            failed.append({"video": video.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append(json.loads((out / "quality.json").read_text(encoding="utf-8")))
    if failed:
        diag.write_csv(batch / "failed_videos.csv", failed)
    if rows:
        with (batch / "quality_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(
                {k: json.dumps(v) if isinstance(v, dict) else v for k, v in r.items()} for r in rows
            )
    _log(
        f"batch done: {len(rows)}/{len(videos)} videos, summary -> {batch / 'quality_summary.csv'}"
        + (f" | {len(failed)} failed -> {batch / 'failed_videos.csv'}" if failed else "")
    )
    return batch


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="freekiki", description="FreeKiki: 49-point soccer-field keypoints (train/detect)."
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("gui", help="Open the FreeKiki window (default).")
    for cmd, help_text in (
        ("init", "Create the workspace layout."),
        ("import-dataset", "Copy a kiki49 YOLO-pose build into the workspace."),
        ("queue", "Pick the frames worth labelling from a detect batch (review sessions)."),
        ("ingest", "Validate complete reviewed frames; --commit appends them to train or hard."),
        (
            "extend",
            "Convert an external field dataset (Roboflow, SoccerNet-GSR, SoccerNet rescue) to "
            "kiki49 labels in a staging folder; --src <staging> --commit appends it to train.",
        ),
        ("check", "Validate the workspace dataset."),
        ("manifest", "Build an oversampled train list (manifests/vNNN) for rare keypoints."),
        ("train", "Train or retrain (--base active / a slot) the field-keypoint network."),
        ("status", "List training runs and which ones can be resumed."),
        ("resume", "Continue an interrupted training from its last saved epoch."),
        ("evaluate", "Measure a model on a labelled split (default val; quality report)."),
        ("sweep", "det_conf x kp_conf grid on a saved evaluation (val only by default)."),
        ("audit", "Read-only dataset audit (points, visibility, leakage, geometry)."),
        ("compare", "Gate a candidate model/evaluation against the baseline on val."),
        ("export", "Export reviewed session frames to images, labels, data.yaml and markers CSV."),
        ("models", "List the model slots (freekiki_n..x, freekiki_hm_*); --default sets 'active'."),
        ("sizes", "Parameter overlap of a model with every YOLO26 scale (read only)."),
        ("grow", "Deepen a trained m into a function-preserving l initialisation."),
        ("bench", "Short training speed / peak-VRAM benchmark (batch x workers)."),
        ("detect", "Detect field keypoints in a video or in every video of a folder."),
    ):
        p = sub.add_parser(cmd, help=help_text)
        if cmd == "export":
            p.add_argument("-w", "--workspace", help="Optional FreeKiki workspace folder.")
            p.add_argument(
                "--session", required=True, help="Review session JSON or session folder."
            )
            p.add_argument("--video", help="Optional video file path if moved.")
            p.add_argument(
                "--mode",
                choices=("full", "only_correct", "lite"),
                default="only_correct",
                help="Export mode: full (all frames with keypoints) or only_correct/lite (human-reviewed only).",
            )
        else:
            p.add_argument("-w", "--workspace", required=True, help="FreeKiki workspace folder.")
        if cmd == "import-dataset":
            p.add_argument("--src", required=True, help="kiki49 dataset folder (has data.yaml).")
        elif cmd == "ingest":
            p.add_argument("--src", required=True, help="Exported review session directory.")
            p.add_argument(
                "--match-id",
                help="Match/sequence shared across cameras and cuts (a montage session takes one "
                "match per frame from montage_frames.csv).",
            )
            p.add_argument(
                "--split",
                choices=INGEST_SPLITS,
                default="train",
                help="train, or hard = labelled holdout of difficult footage (never trained on).",
            )
            p.add_argument(
                "--commit", action="store_true", help="Publish validated pairs to the split."
            )
        elif cmd == "extend":
            p.add_argument("--source", choices=EXTERNAL_SOURCES, help="Dataset to convert.")
            p.add_argument(
                "--project",
                action="append",
                default=[],
                help="Roboflow workspace/project[@version] (repeatable).",
            )
            p.add_argument(
                "--search",
                action="append",
                default=[],
                help='Roboflow Universe query, e.g. "class:pitch" (repeatable).',
            )
            p.add_argument("--limit", type=int, help="Cap images per project / clips / images.")
            p.add_argument("--model", default="l", help="Network for alignment / rescue.")
            p.add_argument("--kp-conf", type=float, help="Keypoint acceptance (default: settings).")
            p.add_argument(
                "--root", help="Folder of the downloaded source (default: FIFA sources)."
            )
            p.add_argument("--src", help="Staging folder to append to train (with --commit).")
            p.add_argument("--commit", action="store_true", help="Publish the staged frames.")
        elif cmd == "queue":
            p.add_argument(
                "--batch",
                dest="batch_dir",
                nargs="+",
                required=True,
                help="detect output or batch folder(s).",
            )
            p.add_argument(
                "--need",
                help="Keypoints the frames must show, e.g. p5,p29,p39,p47 (the field camera "
                "says when they are in the picture).",
            )
            p.add_argument(
                "--half-seen",
                action="store_true",
                help="With --need, also frames without a field camera where the network half-sees "
                "a needed point (rarely useful: mostly adverts, replays, close-ups).",
            )
            p.add_argument(
                "--montage",
                action="store_true",
                help="One review video of all queued frames (montage.mp4 + montage_frames.csv).",
            )
            p.add_argument(
                "--per-video",
                type=int,
                help="Max queued frames per clip (default 25; 5 with --need).",
            )
            p.add_argument(
                "--min-gap", type=int, help="Min frames between picks (default 5; 30 with --need)."
            )
            p.add_argument("--kp-conf", type=float, help="Default: detect kp_conf setting.")
        elif cmd == "train":
            p.add_argument("--base", help="yolo26*-pose.pt, 'active', a slot (m, l) or a .pt.")
            p.add_argument("--epochs", type=int)
            p.add_argument("--imgsz", type=int)
            p.add_argument("--batch", type=float, help="Batch size; -1 = AutoBatch.")
            p.add_argument("--device")
            p.add_argument("--name", help="Run name (default kiki49_<timestamp>).")
            p.add_argument("--fraction", type=float, default=1.0, help="Train subset (smoke).")
            p.add_argument("--patience", type=int)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument(
                "--workers", type=int, help="Dataloader workers (default Ultralytics 8; see bench)."
            )
            p.add_argument("--manifest", help="Train on manifests/<name> (e.g. v001).")
            p.add_argument(
                "--add-dataset",
                action="append",
                default=[],
                metavar="DIR",
                help="Corrected dataset saved by getpixelvideo (F9); added to train first. Repeatable.",
            )
            p.add_argument(
                "--match-id", help="Match of the --add-dataset folders (default: video name)."
            )
            p.add_argument(
                "--backend",
                choices=("yolo", "heatmap"),
                default="yolo",
                help="yolo = Ultralytics pose; heatmap = vailá ResNet heatmap (torchvision only).",
            )
            p.add_argument(
                "--backbone",
                default="resnet50",
                help="Heatmap backbone (resnet18/34/50/101/152, ImageNet weights from torch cache).",
            )
            p.add_argument(
                "--no-pretrained",
                dest="pretrained",
                action="store_false",
                help="Heatmap: start from random weights (no ImageNet checkpoint needed).",
            )
            p.add_argument("--lr", type=float, help="Heatmap AdamW learning rate (default 1e-3).")
        elif cmd == "manifest":
            p.add_argument("--rfs-t", type=float, default=0.05, help="Repeat-factor threshold t.")
            p.add_argument("--cap", type=float, default=4.0, help="Max repeat factor per image.")
            p.add_argument("--seed", type=int, default=0, help="Seed of the fractional rounding.")
            p.add_argument("--exclude", help="Images to leave out (CSV 'image' column or lines).")
            p.add_argument("--name", help="Manifest name (default next vNNN).")
        elif cmd == "resume":
            p.add_argument("--name", help="Run to resume (default: newest interrupted run).")
            p.add_argument("--device")
            p.add_argument("--batch", type=float, help="Override batch (e.g. after OOM).")
        elif cmd == "evaluate":
            p.add_argument("--model", default="active", help="'active', a slot or a .pt path.")
            p.add_argument(
                "--split",
                default="val",
                choices=("val", "test", "train", "hard"),
                help="val to choose/compare; test and hard (labelled holdout) only to report.",
            )
            p.add_argument("--imgsz", type=int)
            p.add_argument("--batch", type=int, default=8)
            p.add_argument("--device")
            p.add_argument("--det-conf", type=float, help="Box threshold (default: detect conf).")
            p.add_argument("--kp-conf", type=float)
            p.add_argument("--match-px", type=float, default=25.0, help="Match radius, px@1920.")
            p.add_argument("--pck", default="5,10,25", help="PCK radii, px@1920.")
            p.add_argument("--max-images", type=int, default=0, help="0 = all images.")
            p.add_argument(
                "--geom-min-conf",
                type=float,
                help="Geometry fill needs this raw network conf (default 0.05; tune on val).",
            )
            p.add_argument(
                "--geom-fix-px", type=float, help="Geometry fix distance, px@1920 (default 20)."
            )
        elif cmd == "sweep":
            p.add_argument("--eval-dir", required=True, help="Folder written by evaluate.")
            p.add_argument("--det-confs", default="0.05,0.1,0.25,0.5")
            p.add_argument("--kp-confs", default="0.1,0.2,0.3,0.5,0.7")
            p.add_argument("--match-px", type=float, default=25.0)
            p.add_argument("--allow-test", action="store_true", help="Permit a test evaluation.")
        elif cmd == "audit":
            p.add_argument("--dup-bits", type=int, default=10, help="dHash near-duplicate radius.")
            p.add_argument("--no-hash", action="store_true", help="Skip the image hashing.")
        elif cmd == "compare":
            p.add_argument(
                "--baseline", help="Eval folder, model or slot (default: the candidate's slot)."
            )
            p.add_argument("--candidate", required=True, help="Eval folder or model .pt.")
            p.add_argument(
                "--promote",
                action="store_true",
                help="Install into models/freekiki_<slot>.pt if the gate passes.",
            )
        elif cmd == "models":
            p.add_argument("--default", help="Slot that 'active' means (e.g. m, l, hm_m).")
        elif cmd == "sizes":
            p.add_argument("--src", default="active", help="'active', a slot or a .pt path.")
        elif cmd == "grow":
            p.add_argument("--src", default="m", help="Trained m model: slot, 'active' or .pt.")
            p.add_argument("--to", default="l", choices=("l",), help="Target scale (m -> l only).")
            p.add_argument(
                "--out", default="models/freekiki_l_init.pt", help="Output (workspace-relative)."
            )
        elif cmd == "bench":
            p.add_argument("--batches", default="2,8,16")
            p.add_argument("--workers", default="8")
            p.add_argument("--fraction", type=float, default=0.02)
            p.add_argument("--imgsz", type=int)
            p.add_argument("--model", help="Default: the train base.")
            p.add_argument("--device")
        elif cmd == "detect":
            p.add_argument("--video", required=True, help="Video file or folder of videos.")
            p.add_argument("--model", default="active")
            p.add_argument("--output-dir", help="Default: the video's folder.")
            p.add_argument("--stride", type=int, default=1)
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--max-frames", type=int, default=0)
            p.add_argument("--conf", type=float, help="Box (detection) threshold.")
            p.add_argument("--kp-conf", type=float, help="Keypoint threshold.")
            p.add_argument("--imgsz", type=int)
            p.add_argument("--device")
            p.add_argument("--no-overlay", action="store_true")
            p.add_argument(
                "--fill-gaps",
                type=int,
                default=0,
                help="Interpolate gaps <= N frames inside one shot (separate CSV, marked I).",
            )
            p.add_argument("--diag-frames", type=int, default=12, help="Annotated frames saved.")
            p.add_argument(
                "--geom",
                choices=("fill", "fix", "off"),
                help="Field geometry (default: settings, fill): fill missing points / also fix "
                "points that disagree with the camera / off. Writes field_kps_geom_*.csv.",
            )
    return parser


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in str(text).replace(" ", "").split(",") if v)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in (None, "gui"):
        run_freekiki()
        return 0
    if args.command == "export":
        session = load_review_session(args.session, video=args.video)
        out = export_reviewed_session(session, mode=args.mode)
        _log(f"session exported ({args.mode}): {out}")
        return 0
    ws = Path(args.workspace)
    batch = getattr(args, "batch", None)
    if batch is not None and float(batch).is_integer():
        batch = int(batch)
    if args.command == "init":
        init_workspace(ws)
    elif args.command == "import-dataset":
        import_dataset(ws, args.src)
    elif args.command == "ingest":
        ingest_reviewed(ws, args.src, args.match_id, split=args.split, commit=args.commit)
    elif args.command == "extend":
        if args.src:
            commit_external(ws, args.src, commit=args.commit)
        elif args.source:
            stage_external(
                ws,
                args.source,
                projects=args.project,
                search=args.search,
                limit=args.limit,
                model=args.model,
                kp_conf=args.kp_conf,
                root=args.root,
            )
        else:
            raise SystemExit("extend needs --source (stage) or --src (commit a staging folder)")
    elif args.command == "queue":
        build_label_queue(
            ws,
            args.batch_dir,
            per_video=args.per_video,
            min_gap=args.min_gap,
            kp_conf=args.kp_conf,
            need=parse_keypoints(args.need) if args.need else None,
            montage=args.montage,
            half_seen=args.half_seen,
        )
    elif args.command == "check":
        return 1 if check_dataset(ws) else 0
    elif args.command == "status":
        rows = list_runs(ws)
        todo = [r["run"] for r in rows if r["state"] in ("resumable", "finished")]
        if todo:
            _log(f"to continue: uv run vaila/freekiki.py resume -w {ws} --name {todo[-1]}")
    elif args.command == "resume":
        resume(ws, name=args.name, device=args.device, batch=batch)
    elif args.command == "train":
        if args.add_dataset:
            if args.manifest:
                raise ValueError(
                    "A manifest is a fixed train list without the new corrections: run "
                    "'train --add-dataset ...' without --manifest, or build a new manifest after "
                    "'ingest'."
                )
            add_corrections(ws, args.add_dataset, match_id=args.match_id)
        train(
            ws,
            base=args.base,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch,
            device=args.device,
            name=args.name,
            fraction=args.fraction,
            patience=args.patience,
            seed=args.seed,
            workers=args.workers,
            manifest=args.manifest,
            backend=args.backend,
            backbone=args.backbone,
            pretrained=args.pretrained,
            lr=args.lr,
        )
    elif args.command == "manifest":
        make_manifest(
            ws, t=args.rfs_t, cap=args.cap, seed=args.seed, exclude=args.exclude, name=args.name
        )
    elif args.command == "evaluate":
        evaluate(
            ws,
            model=args.model,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            det_conf=args.det_conf,
            kp_conf=args.kp_conf,
            match_px=args.match_px,
            pck=_floats(args.pck),
            max_images=args.max_images,
            geom_settings={
                k: v
                for k, v in (("min_conf", args.geom_min_conf), ("fix_px", args.geom_fix_px))
                if v is not None
            },
        )
    elif args.command == "sweep":
        sweep(
            ws,
            args.eval_dir,
            det_confs=_floats(args.det_confs),
            kp_confs=_floats(args.kp_confs),
            match_px=args.match_px,
            allow_test=args.allow_test,
        )
    elif args.command == "audit":
        audit(ws, dup_bits=args.dup_bits, hash_images=not args.no_hash)
    elif args.command == "compare":
        decision = compare(
            ws, baseline=args.baseline, candidate=args.candidate, promote=args.promote
        )
        return 0 if decision["promote"] else 2
    elif args.command == "models":
        list_models(ws, default=args.default)
    elif args.command == "sizes":
        model_sizes(ws, args.src)
    elif args.command == "grow":
        grow_model(ws, src=args.src, to=args.to, out=args.out)
    elif args.command == "bench":
        bench(
            ws,
            batches=tuple(int(b) for b in _floats(args.batches)),
            workers=tuple(int(w) for w in _floats(args.workers)),
            fraction=args.fraction,
            imgsz=args.imgsz,
            model=args.model,
            device=args.device,
        )
    elif args.command == "detect":
        detect_videos(
            ws,
            args.video,
            model=args.model,
            output_dir=args.output_dir,
            stride=args.stride,
            start=args.start,
            max_frames=args.max_frames,
            conf=args.conf,
            kp_conf=args.kp_conf,
            imgsz=args.imgsz,
            device=args.device,
            overlay=not args.no_overlay,
            fill_gaps=args.fill_gaps,
            diag_frames=args.diag_frames,
            geom=args.geom,
        )
    return 0


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
def _last_workspace() -> str:
    try:
        return str(toml.load(USER_SETTINGS).get("workspace", ""))
    except (OSError, toml.TomlDecodeError):
        return ""


def _remember_workspace(ws: str) -> None:
    try:
        USER_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        USER_SETTINGS.write_text(toml.dumps({"workspace": ws}), encoding="utf-8")
    except OSError:
        pass


def run_freekiki() -> None:
    """Open the FreeKiki window (workspace / dataset / train / detect)."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    try:
        from .dialogsuser import link_output_to_input
    except ImportError:
        from dialogsuser import link_output_to_input  # ty: ignore[unresolved-import]

    root = tk.Tk()
    root.title("FreeKiki - soccer-field keypoints (kiki49)")
    state: dict = {"proc": None}
    lines: queue.Queue[str] = queue.Queue()

    v = {
        "ws": tk.StringVar(value=_last_workspace()),
        "src": tk.StringVar(),
        "base": tk.StringVar(value="yolo26m-pose.pt"),
        "backend": tk.StringVar(value="yolo"),
        "epochs": tk.StringVar(value="150"),
        "imgsz": tk.StringVar(value="1280"),
        "batch": tk.StringVar(value="-1"),
        "device": tk.StringVar(value=""),
        "fraction": tk.StringVar(value="1.0"),
        "manifest": tk.StringVar(value=""),
        "video": tk.StringVar(),
        "out": tk.StringVar(),
        "model": tk.StringVar(value="active"),
        "stride": tk.StringVar(value="1"),
        "max_frames": tk.StringVar(value="0"),
        "conf": tk.StringVar(value="0.25"),
        "kp_conf": tk.StringVar(value="0.5"),
        "split": tk.StringVar(value="val"),
        "fill_gaps": tk.StringVar(value="0"),
        "geom": tk.StringVar(value="fill"),
        "corrections": tk.StringVar(),
        "ext_source": tk.StringVar(value="roboflow"),
        "ext_query": tk.StringVar(value="class:pitch"),
        "ext_src": tk.StringVar(),
        "match_id": tk.StringVar(),
    }
    overlay = tk.BooleanVar(value=True)
    link_output_to_input(v["video"], v["out"])

    def browse_dir(var, title):
        path = filedialog.askdirectory(title=title, initialdir=var.get() or None)
        if path:
            var.set(path)

    def browse_file(var, title, types):
        path = filedialog.askopenfilename(title=title, filetypes=types)
        if path:
            var.set(path)

    def browse_detection_model():
        path = filedialog.askopenfilename(title="Model weights", filetypes=[("PyTorch", "*.pt")])
        if path:
            v["model"].set(path)
            model_ws = workspace_for_model_file(path)
            if model_ws is not None:
                v["ws"].set(str(model_ws))

    def run_cli(args: list[str]) -> None:
        if state["proc"] is not None and state["proc"].poll() is None:
            messagebox.showwarning("FreeKiki", "A task is still running.", parent=root)
            return
        ws = v["ws"].get().strip()
        if not ws:
            messagebox.showerror("FreeKiki", "Choose a workspace folder first.", parent=root)
            return
        _remember_workspace(ws)
        argv = [args[0], "-w", ws, *args[1:]]
        print_gui_cli_mirror(
            "vaila/freekiki", ["uv", "run", "--no-sync", "vaila/freekiki.py", *argv]
        )
        log.insert("end", f"\n>> freekiki {' '.join(argv)}\n")
        proc = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), *argv],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        state["proc"] = proc

        def pump():
            assert proc.stdout is not None
            for text in proc.stdout:
                lines.put(text)
            lines.put(f"[freekiki] finished (exit code {proc.wait()})\n")

        threading.Thread(target=pump, daemon=True).start()

    def poll():
        while not lines.empty():
            text = lines.get_nowait()
            print(text, end="", flush=True)
            log.insert("end", text)
            log.see("end")
        root.after(150, poll)

    def stop():
        if state["proc"] is not None and state["proc"].poll() is None:
            state["proc"].terminate()

    def opt(flag: str, key: str) -> list[str]:
        value = v[key].get().strip()
        return [flag, value] if value else []

    def do_train():
        base = v["base"].get().strip()
        heatmap = v["backend"].get() == "heatmap"
        # A YOLO base name means nothing to the heatmap backend (ImageNet start instead).
        args = ["train", "--backend", "heatmap"] if heatmap else ["train"]
        if base and not (heatmap and Path(base).name.startswith("yolo")):
            args += ["--base", base]
        run_cli(
            args
            + opt("--epochs", "epochs")
            + opt("--imgsz", "imgsz")
            + opt("--batch", "batch")
            + opt("--device", "device")
            + opt("--fraction", "fraction")
            + opt("--manifest", "manifest")
            + [
                arg
                for folder in v["corrections"].get().split(";")
                if folder.strip()
                for arg in ("--add-dataset", folder.strip())
            ]
            + (opt("--match-id", "match_id") if v["corrections"].get().strip() else [])
        )

    def smoke_preset():
        for key, value in (
            ("backend", "yolo"),
            ("base", "yolo26n-pose.pt"),
            ("epochs", "5"),
            ("imgsz", "640"),
            ("batch", "16"),
            ("fraction", "0.1"),
        ):
            v[key].set(value)

    def heatmap_preset():
        for key, value in (
            ("backend", "heatmap"),
            ("base", ""),
            ("epochs", "60"),
            ("imgsz", "1024"),
            ("batch", "16"),
            ("fraction", "1.0"),
        ):
            v[key].set(value)

    def full_preset():
        for key, value in (
            ("backend", "yolo"),
            ("base", "yolo26m-pose.pt"),
            ("epochs", "150"),
            ("imgsz", "1280"),
            ("batch", "-1"),
            ("fraction", "1.0"),
        ):
            v[key].set(value)

    def do_evaluate():
        run_cli(
            ["evaluate", "--model", v["model"].get().strip()]
            + opt("--split", "split")
            + opt("--det-conf", "conf")
            + opt("--kp-conf", "kp_conf")
            + opt("--device", "device")
        )

    def do_sweep():
        ws = v["ws"].get().strip()
        path = filedialog.askdirectory(
            title="Evaluation folder (processed_freekiki_eval_val_*)",
            initialdir=str(Path(ws) / "outputs") if ws else None,
        )
        if path:
            run_cli(["sweep", "--eval-dir", path])

    def do_compare():
        model = v["model"].get().strip()
        if model in ("", "active"):
            messagebox.showerror(
                "FreeKiki", "Set Model (section 5) to the candidate .pt to compare.", parent=root
            )
            return
        run_cli(["compare", "--candidate", model])

    def do_detect():
        if not v["video"].get().strip():
            messagebox.showerror("FreeKiki", "Choose a video or a folder of videos.", parent=root)
            return
        ws = v["ws"].get().strip()
        model = v["model"].get().strip()
        try:
            validate_detection_workspace(ws, model)
        except (FileNotFoundError, ValueError, toml.TomlDecodeError) as exc:
            messagebox.showerror(
                "FreeKiki",
                "In section 5, choose the FreeKiki workspace folder containing "
                "freekiki.toml (and a promoted model when Model is 'active' or a slot).\n\n"
                f"{exc}",
                parent=root,
            )
            return
        args = ["detect", "--video", v["video"].get().strip(), "--model", v["model"].get().strip()]
        args += opt("--output-dir", "out") + opt("--stride", "stride")
        args += opt("--max-frames", "max_frames") + opt("--conf", "conf")
        args += opt("--kp-conf", "kp_conf") + opt("--device", "device")
        args += opt("--fill-gaps", "fill_gaps") + opt("--geom", "geom")
        if not overlay.get():
            args.append("--no-overlay")
        run_cli(args)

    def open_help():
        webbrowser.open(HELP_HTML.resolve().as_uri())

    def add_corrections_folder():
        path = filedialog.askdirectory(
            title="Corrected dataset (getpixelvideo -> Save dataset F9)",
            initialdir=v["out"].get() or None,
        )
        if path:
            current = [p for p in v["corrections"].get().split(";") if p.strip()]
            v["corrections"].set(";".join([*current, path]))

    def do_extend_stage():
        args = ["extend", "--source", v["ext_source"].get()]
        if v["ext_source"].get() == "roboflow":
            # "a/b@3" entries are projects, anything else is a Universe search query.
            for item in (q.strip() for q in v["ext_query"].get().split(";")):
                if item:
                    args += ["--project" if "/" in item and " " not in item else "--search", item]
        run_cli(args)

    def do_extend_commit(commit: bool):
        src = v["ext_src"].get().strip()
        if not src:
            src = filedialog.askdirectory(
                title="Staging folder (incoming/external_*)",
                initialdir=str(Path(v["ws"].get().strip() or ".") / "incoming"),
            )
            v["ext_src"].set(src or "")
        if src:
            run_cli(["extend", "--src", src] + (["--commit"] if commit else []))

    def do_correct():
        video, ws = v["video"].get().strip(), v["ws"].get().strip()
        if not video or not Path(video).exists() or not ws:
            messagebox.showerror(
                "FreeKiki",
                "Choose the workspace and the video or folder (section 5), then Detect it first.",
                parent=root,
            )
            return
        out = v["out"].get().strip() or str(
            Path(video) if Path(video).is_dir() else Path(video).parent
        )
        gpv = Path(__file__).resolve().parent / "getpixelvideo.py"
        if Path(video).is_file():
            argv = ["-f", video, "--freekiki", "--freekiki-workspace", ws]
            argv += ["--freekiki-predictions", out]
        else:  # a folder of videos: FreeKiki Load on the newest batch, one video after the other
            batches = sorted(Path(out).glob("processed_freekiki_batch_*"))
            argv = ["--freekiki-run", str(batches[-1] if batches else out)]
        print_gui_cli_mirror("vaila/getpixelvideo", ["uv", "run", "--no-sync", str(gpv), *argv])
        log.insert("end", f"\n>> getpixelvideo {' '.join(argv)}\n")
        subprocess.Popen([sys.executable, str(gpv), *argv])  # own window, runs alongside

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)

    def row(parent, r, label, var, browse=None, width=48):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(parent, textvariable=var, width=width).grid(row=r, column=1, sticky="we")
        if browse:
            ttk.Button(parent, text="Browse", command=browse).grid(row=r, column=2, padx=4)

    box = ttk.LabelFrame(frm, text="1. Workspace (portable folder)", padding=6)
    box.pack(fill="x", pady=4)
    row(box, 0, "Workspace", v["ws"], lambda: browse_dir(v["ws"], "FreeKiki workspace"))
    ttk.Button(box, text="Init workspace", command=lambda: run_cli(["init"])).grid(
        row=1, column=1, sticky="w", pady=2
    )

    box = ttk.LabelFrame(frm, text="2. Dataset (kiki49 YOLO-pose build)", padding=6)
    box.pack(fill="x", pady=4)
    row(box, 0, "Source", v["src"], lambda: browse_dir(v["src"], "kiki49 dataset folder"))
    bar = ttk.Frame(box)
    bar.grid(row=1, column=1, sticky="w", pady=2)
    ttk.Button(
        bar,
        text="Import (copy)",
        command=lambda: run_cli(["import-dataset", "--src", v["src"].get().strip()]),
    ).pack(side="left")
    ttk.Button(bar, text="Check dataset", command=lambda: run_cli(["check"])).pack(
        side="left", padx=6
    )

    box = ttk.LabelFrame(frm, text="3. Train / Retrain", padding=6)
    box.pack(fill="x", pady=4)
    ttk.Label(box, text="Base model").grid(row=0, column=0, sticky="w", padx=4)
    ttk.Combobox(box, textvariable=v["base"], values=BASE_MODELS, width=46).grid(
        row=0, column=1, sticky="we"
    )
    ttk.Button(
        box,
        text="Browse",
        command=lambda: browse_file(v["base"], "Base weights", [("PyTorch", "*.pt")]),
    ).grid(row=0, column=2, padx=4)
    grid = ttk.Frame(box)
    grid.grid(row=1, column=0, columnspan=3, sticky="w", pady=2)
    for i, (label, key) in enumerate(
        (("Epochs", "epochs"), ("imgsz", "imgsz"), ("Batch", "batch"))
        + (("Device", "device"), ("Fraction", "fraction"), ("Manifest", "manifest"))
    ):
        ttk.Label(grid, text=label).grid(row=0, column=2 * i, padx=(4, 2))
        ttk.Entry(grid, textvariable=v[key], width=7).grid(row=0, column=2 * i + 1)
    ttk.Label(grid, text="Backend").grid(row=0, column=12, padx=(4, 2))
    ttk.Combobox(
        grid, textvariable=v["backend"], values=("yolo", "heatmap"), width=8, state="readonly"
    ).grid(row=0, column=13)
    bar = ttk.Frame(box)
    bar.grid(row=2, column=1, sticky="w", pady=2)
    ttk.Button(bar, text="Train", command=do_train).pack(side="left")
    ttk.Button(bar, text="Smoke preset (~5 min)", command=smoke_preset).pack(side="left", padx=6)
    ttk.Button(bar, text="Full preset (hours)", command=full_preset).pack(side="left")
    ttk.Button(bar, text="Heatmap preset", command=heatmap_preset).pack(side="left", padx=6)
    ttk.Button(
        bar,
        text="Bench batch/VRAM (~5 min)",
        command=lambda: run_cli(["bench"] + opt("--imgsz", "imgsz") + opt("--device", "device")),
    ).pack(side="left", padx=6)
    bar = ttk.Frame(box)
    bar.grid(row=3, column=1, sticky="w", pady=2)
    ttk.Button(bar, text="Runs status", command=lambda: run_cli(["status"])).pack(side="left")
    ttk.Button(
        bar,
        text="Resume interrupted",
        command=lambda: run_cli(["resume"] + opt("--device", "device")),
    ).pack(side="left", padx=6)
    ttk.Button(bar, text="Build oversampling manifest", command=lambda: run_cli(["manifest"])).pack(
        side="left"
    )
    corr = ttk.Frame(box)
    corr.grid(row=5, column=0, columnspan=3, sticky="we", pady=2)
    ttk.Label(corr, text="Corrections").pack(side="left", padx=(4, 2))
    ttk.Entry(corr, textvariable=v["corrections"], width=52).pack(
        side="left", fill="x", expand=True
    )
    ttk.Button(corr, text="Add folder...", command=add_corrections_folder).pack(side="left", padx=4)
    ttk.Label(corr, text="Match id").pack(side="left", padx=(8, 2))
    ttk.Entry(corr, textvariable=v["match_id"], width=16).pack(side="left")
    ext = ttk.Frame(box)
    ext.grid(row=6, column=0, columnspan=3, sticky="we", pady=2)
    ttk.Label(ext, text="External data").pack(side="left", padx=(4, 2))
    ttk.Combobox(
        ext, textvariable=v["ext_source"], values=EXTERNAL_SOURCES, width=16, state="readonly"
    ).pack(side="left")
    ttk.Entry(ext, textvariable=v["ext_query"], width=28).pack(side="left", padx=4)
    ttk.Button(ext, text="Stage", command=do_extend_stage).pack(side="left")
    ttk.Entry(ext, textvariable=v["ext_src"], width=28).pack(side="left", padx=(8, 2))
    ttk.Button(ext, text="Preview", command=lambda: do_extend_commit(False)).pack(side="left")
    ttk.Button(ext, text="Commit", command=lambda: do_extend_commit(True)).pack(side="left", padx=4)
    ttk.Label(
        box,
        text="Retrain = Base model 'active' (default slot), a slot (m, l) or any trained .pt: "
        "AdamW lr0=1e-4, 1 warmup epoch, cosine, mosaic off, backbone unfrozen.\n"
        "Manifest = e.g. v001 from 'Build oversampling manifest' (rare points repeated; "
        "empty = plain train split).\n"
        "Stopped / crash / power loss? Every finished epoch is saved: press Resume interrupted.\n"
        "Backend heatmap = vailá ResNet heatmap net (torch/torchvision only, no Ultralytics); "
        "never auto-promoted: Evaluate + Compare on val first.",
    ).grid(row=4, column=0, columnspan=3, sticky="w", padx=4)

    box = ttk.LabelFrame(frm, text="4. Evaluate quality (labelled split)", padding=6)
    box.pack(fill="x", pady=4)
    bar = ttk.Frame(box)
    bar.grid(row=0, column=0, columnspan=3, sticky="w")
    ttk.Label(bar, text="Split").pack(side="left", padx=(4, 2))
    ttk.Combobox(
        bar, textvariable=v["split"], values=("val", "test", "hard"), width=5, state="readonly"
    ).pack(side="left")
    ttk.Button(bar, text="Evaluate model", command=do_evaluate).pack(side="left", padx=6)
    ttk.Button(bar, text="Sweep thresholds (val)", command=do_sweep).pack(side="left")
    ttk.Button(bar, text="Compare with its slot", command=do_compare).pack(side="left", padx=6)
    ttk.Button(bar, text="Model slots", command=lambda: run_cli(["models"])).pack(side="left")
    ttk.Button(bar, text="Audit dataset", command=lambda: run_cli(["audit"])).pack(side="left")
    ttk.Label(
        box,
        text="Uses Model / Conf / KP conf below. Choose thresholds and compare models on val;\n"
        "test only for the final report. Reports in <workspace>/outputs.",
    ).grid(row=1, column=0, columnspan=3, sticky="w", padx=4)

    box = ttk.LabelFrame(frm, text="5. Detect field keypoints (video or folder)", padding=6)
    box.pack(fill="x", pady=4)
    row(
        box,
        0,
        "Workspace (freekiki.toml)",
        v["ws"],
        lambda: browse_dir(v["ws"], "FreeKiki workspace (contains freekiki.toml)"),
    )
    video_types = [("Video", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All", "*.*")]
    row(box, 1, "Video", v["video"], lambda: browse_file(v["video"], "Video", video_types))
    ttk.Button(box, text="Folder", command=lambda: browse_dir(v["video"], "Folder of videos")).grid(
        row=1, column=3
    )
    row(box, 2, "Output dir", v["out"], lambda: browse_dir(v["out"], "Output folder"))
    row(
        box,
        3,
        "Model",
        v["model"],
        browse_detection_model,
    )
    grid = ttk.Frame(box)
    grid.grid(row=4, column=0, columnspan=3, sticky="w", pady=2)
    for i, (label, key) in enumerate(
        (
            ("Stride", "stride"),
            ("Max frames", "max_frames"),
            ("Conf", "conf"),
            ("KP conf", "kp_conf"),
            ("Fill gaps", "fill_gaps"),
        )
    ):
        ttk.Label(grid, text=label).grid(row=0, column=2 * i, padx=(4, 2))
        ttk.Entry(grid, textvariable=v[key], width=7).grid(row=0, column=2 * i + 1)
    ttk.Checkbutton(grid, text="Overlay MP4", variable=overlay).grid(row=0, column=10, padx=6)
    ttk.Label(grid, text="Geometry").grid(row=0, column=11, padx=(8, 2))
    ttk.Combobox(
        grid, textvariable=v["geom"], values=("fill", "fix", "off"), width=5, state="readonly"
    ).grid(row=0, column=12)
    bar = ttk.Frame(box)
    bar.grid(row=5, column=1, sticky="w", pady=2)
    ttk.Button(bar, text="Detect", command=do_detect).pack(side="left")
    ttk.Button(bar, text="Correct in getpixelvideo", command=do_correct).pack(side="left", padx=6)
    ttk.Label(
        box,
        text="Correct: fix wrong points, mark missing ones (Del = not visible), F3 frame OK, "
        "F9 Save dataset (Full or Only Correct mode).\nThen put that folder in section 3 'Corrections' and Train "
        "(Base 'active' = fine-tune).",
    ).grid(row=6, column=0, columnspan=3, sticky="w", padx=4)

    bar = ttk.Frame(frm)
    bar.pack(fill="x", pady=4)
    ttk.Button(bar, text="Help", command=open_help).pack(side="left")
    ttk.Button(bar, text="Stop task", command=stop).pack(side="left", padx=6)
    ttk.Button(bar, text="Close", command=lambda: (stop(), root.destroy())).pack(side="right")

    log = tk.Text(frm, height=14, width=100)
    log.pack(fill="both", expand=True)
    log.insert("end", "FreeKiki log - each action runs as a CLI subprocess (>> printed).\n")

    poll()
    root.mainloop()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RunStateError as exc:
        _log(f"ERROR: {exc}")
        sys.exit(2)
