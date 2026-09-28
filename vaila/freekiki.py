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
Version: 0.4.5
Created: 25 September 2026
Update Date: 28 September 2026

Description:
    FreeKiki (Soccer Tools) trains, retrains and runs a YOLO-pose network that
    finds the 49 soccer-field keypoints of ``vaila/models/soccerfield_kiki.csv``
    (pitch lines, goal posts/nets, corner flags, center) in broadcast video.

    Everything lives in one portable *workspace* folder, so it can be copied to
    another machine and pointed at for new trainings:

        <workspace>/
          freekiki.toml          settings + active model (paths relative)
          spec/                  soccerfield_kiki.csv + soccerfield_kiki49.json
          datasets/kiki49/       imported YOLO-pose dataset (data.yaml, images, labels)
          manifests/vNNN/        oversampled train lists (immutable, see ``manifest``)
          runs/<name>/           Ultralytics training runs
          models/registry.csv    every finished run with its best-epoch metrics
          models/active.pt       best model so far (used by Detect / Retrain)
          models/promotion_log.csv  every promotion decision and its reasons
          outputs/               evaluations, sweeps, audits, benchmarks

    Retrain starts from ``models/active.pt``. That continued fine-tune uses
    AdamW at ``lr0=1e-4`` (cosine down to ``1e-6``), one warmup epoch and
    ``mosaic=0``. The checkpoint already finished its previous run with mosaic
    off and a learning rate near ``2e-4``; ``lr0=0.001`` plus mosaic dropped
    pose mAP50-95 from 0.857 to 0.601 in one epoch. The backbone stays
    trainable: the set is large and already the same field domain. A new run
    replaces ``active.pt`` only when it passes the promotion gate
    (``[promotion]`` in freekiki.toml, mode ``gate``): candidate and active are
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
    uv run vaila/freekiki.py audit  -w WORKSPACE                       # read-only dataset audit
    uv run vaila/freekiki.py bench  -w WORKSPACE --batches 2,8,16      # speed / peak VRAM
    uv run vaila/freekiki.py detect -w WORKSPACE --video match.mp4 [--output-dir DIR]
    uv run vaila/freekiki.py detect -w WORKSPACE --video FOLDER_OF_VIDEOS [--fill-gaps 3]

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
import threading
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
ACTIVE_MODEL = "models/active.pt"
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
)
DEFAULT_SETTINGS = {
    "workspace": {"schema": "soccerfield_kiki49", "dataset": f"datasets/{DATASET_NAME}"},
    "active": {"model": ACTIVE_MODEL, "run": "", "pose_map50_95": 0.0},
    "train": {
        "base": "yolo26m-pose.pt",
        "epochs": 150,
        "imgsz": 1280,
        "batch": -1,
        "patience": 30,
    },
    "detect": {"conf": 0.25, "kp_conf": 0.5, "imgsz": 1280},
    "promotion": dict(diag.PROMOTION_DEFAULTS),
}
# Continued fine-tune from models/active.pt. The 150-epoch run ended with
# mosaic off and param-group lrs near 1.7e-4..5e-4 (pose mAP50-95 0.857,
# pose precision 0.954). AdamW lr0=0.001 with mosaic=1.0 then scored 0.601 /
# 0.871 after one epoch. Stay under that final lr and keep mosaic off; the
# rare-point signal comes from the repeat-factor list. Backbone stays
# trainable (large set, same domain).
ACTIVE_FINETUNE_ARGS = {
    "optimizer": "AdamW",
    "lr0": 1e-4,
    "lrf": 0.01,
    "warmup_epochs": 1.0,
    "cos_lr": True,
    "mosaic": 0.0,
}
MAP50_COL = "metrics/mAP50(P)"
MAP50_95_COL = "metrics/mAP50-95(P)"
BOX_MAP50_95_COL = "metrics/mAP50-95(B)"
PROMOTION_LOG = "models/promotion_log.csv"
# evaluate: raw predictions are kept down to this box confidence so the
# det_conf / kp_conf thresholds can be applied (and swept) offline.
RAW_CONF = 0.01
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
    return [r["point_name"] for r in rows], [int(r["flip_idx"]) for r in rows]


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
    for section, values in DEFAULT_SETTINGS.items():
        merged = dict(values)
        merged.update(settings.get(section, {}))
        settings[section] = merged
    return settings


def save_settings(ws: Path, settings: dict) -> None:
    (Path(ws) / CONFIG_NAME).write_text(toml.dumps(settings), encoding="utf-8")


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
    for split in ("train", "val", "test"):
        entry = data.get(split)
        if not entry:
            if split != "test":
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
    """
    best = {"best_epoch": "", "pose_map50": 0.0, "pose_map50_95": 0.0, "fitness": 0.0}
    if not Path(results_csv).is_file():
        return best
    top = float("-inf")
    with Path(results_csv).open(encoding="utf-8") as f:
        for raw in csv.DictReader(f):
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


def promotion_decision(ws, candidate: str, settings: dict) -> dict:
    """Gate a candidate against the active model on the same split and thresholds."""
    gate = settings["promotion"]
    split = gate.get("split", "val")
    cand_dir = find_eval(ws, candidate, split=split) or evaluate(ws, model=candidate, split=split)
    like = json.loads((cand_dir / "eval_summary.json").read_text(encoding="utf-8"))
    active = str(Path(ws) / settings["active"]["model"])
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


def register_run(ws, run_dir, *, base: str, epochs: int, imgsz: int) -> dict:
    """Copy a run's best.pt into ``models/``, log it, promote to active if the policy allows.

    ``settings['promotion']['mode']``:
      * ``gate`` (default): candidate and active are evaluated on the same
        split (``val``) and compared with :func:`freekiki_diag.compare_evals`
        (pose mAP, PCK with misses, homography rate, critical keypoints);
      * ``map``: best-epoch validation pose mAP50-95 only (legacy);
      * ``never``: keep the active model (use ``compare --promote`` later).
    The first model of a workspace is always promoted. Every decision is
    appended to ``models/promotion_log.csv``.
    """
    ws = Path(ws)
    run_dir = Path(run_dir)
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        raise FileNotFoundError(f"Training produced no best.pt: {best_pt}")
    settings = load_settings(ws)
    metrics = read_best_metrics(run_dir / "results.csv")
    model_rel = f"models/{DATASET_NAME}_{run_dir.name}.pt"
    shutil.copy2(best_pt, ws / model_rel)
    active = settings["active"]
    active_path = ws / active["model"]
    mode = str(settings["promotion"].get("mode", "gate"))
    reasons: list[str] = []
    if not active_path.is_file():
        promoted, reasons = True, ["first model of the workspace"]
    elif mode == "map":
        promoted = metrics["pose_map50_95"] > float(active.get("pose_map50_95", 0.0))
        reasons = [f"pose mAP50-95 {active.get('pose_map50_95')} -> {metrics['pose_map50_95']}"]
    elif mode == "gate":
        decision = promotion_decision(ws, str(ws / model_rel), settings)
        promoted, reasons = bool(decision["promote"]), decision["reasons"] or ["gate passed"]
    else:
        promoted, reasons = False, [f"promotion mode '{mode}'"]
    if promoted:
        shutil.copy2(best_pt, active_path)
        active.update(
            {"run": run_dir.name, "pose_map50_95": metrics["pose_map50_95"], "model": ACTIVE_MODEL}
        )
        save_settings(ws, settings)
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
    }
    registry = ws / REGISTRY_CSV
    header = REGISTRY_FIELDS
    if registry.is_file():  # keep an existing registry's columns
        with registry.open(encoding="utf-8") as f:
            header = next(csv.reader(f), []) or REGISTRY_FIELDS
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
    _log(
        f"run {run_dir.name}: pose mAP50-95={metrics['pose_map50_95']:.4f} ({mode}) "
        f"-> {'PROMOTED to ' + ACTIVE_MODEL if promoted else 'kept previous active model'}"
        + ("" if promoted else " | " + "; ".join(reasons))
    )
    return row


def resolve_model(ws, model: str) -> str:
    """``active`` -> workspace model; a workspace-relative/absolute path; or a named YOLO .pt."""
    ws = Path(ws)
    if model == "active":
        path = ws / load_settings(ws)["active"]["model"]
        if not path.is_file():
            raise FileNotFoundError(f"No active model yet ({path}). Train one first.")
        return str(path)
    for candidate in (Path(model).expanduser(), ws / model):
        if candidate.is_file():
            return str(candidate.resolve())
    return model  # named Ultralytics weights, resolved by yolotrain


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
) -> dict:
    """Train (from a base model) or retrain (``base='active'``) on the workspace dataset.

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
    base = base or defaults["base"]
    epochs = int(epochs or defaults["epochs"])
    imgsz = int(imgsz or defaults["imgsz"])
    batch = defaults["batch"] if batch is None else batch
    patience = int(defaults["patience"] if patience is None else patience)
    name = name or f"{DATASET_NAME}_{datetime.now():%Y%m%d_%H%M%S}"
    if (ws / "runs" / name).is_dir():
        info = run_state(ws, ws / "runs" / name)
        if info["state"] in ("running", "resumable"):
            raise RunStateError(
                f"Run '{name}' is {info['state']} (epoch {info['epochs_done']}/{info['epochs']}); "
                f"a new Train would overwrite it. Use: resume --name {name}"
            )
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
    if base == "active":
        extra.update(ACTIVE_FINETUNE_ARGS)
        _log(
            "fine-tune from active: AdamW lr0=1e-4 lrf=0.01 warmup_epochs=1 "
            "cos_lr mosaic=0 freeze=none"
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
    for mod in ("torch", "ultralytics", "numpy", "cv2"):
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
    if info["state"] == "resumable":
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


def file_sha256(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


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


def collect_predictions(net, images, lbl_dir, manifest, *, imgsz, device, raw_conf) -> dict:
    """Raw best-instance predictions + labels of every image (no threshold applied)."""
    import cv2
    import numpy as np

    keep: dict[str, list] = {k: [] for k in PRED_KEYS}
    for n, img_path in enumerate(images, 1):
        gt_xy, gt_vis = read_label_keypoints(lbl_dir / f"{img_path.stem}.txt")
        frame = cv2.imread(str(img_path)) if gt_xy is not None else None
        if frame is None:
            continue
        h, w = frame.shape[:2]
        result: Any = list(
            net.predict(frame, imgsz=imgsz, device=device, conf=raw_conf, verbose=False)
        )[0]
        box_conf, xy, kc = best_instance(result)
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
) -> Path:
    """Measure a model on a labelled split (default ``val``; ``test`` is for final reports).

    Two views of quality:
      * Ultralytics validation (pose/box mAP, precision, recall, OKS);
      * FreeKiki per-keypoint tables with an explicit distance gate
        (see ``METRIC_DEFINITIONS``) and the predicted field homography.

    Raw best-instance predictions are saved at a low confidence
    (``predictions.npz``) so thresholds can be swept offline (``sweep``).
    Writes ``<ws>/outputs/processed_freekiki_eval_<split>_<ts>/`` and appends a
    row to ``models/evaluations_v2.csv``.
    """
    from ultralytics import YOLO

    ws = Path(ws).expanduser().resolve()
    settings = load_settings(ws)
    det_conf = float(settings["detect"]["conf"] if det_conf is None else det_conf)
    kp_conf = float(settings["detect"]["kp_conf"] if kp_conf is None else kp_conf)
    model_path = resolve_model(ws, model)
    net = YOLO(model_path)
    imgsz = model_imgsz(net, imgsz, settings["detect"]["imgsz"])
    yaml_path = refresh_yaml_path(dataset_dir(ws) / "data.yaml")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not data.get(split):
        raise ValueError(f"data.yaml has no '{split}' split")
    img_dir = dataset_dir(ws) / str(data[split])
    lbl_dir = dataset_dir(ws) / str(data[split]).replace("images", "labels", 1)
    out = ws / "outputs" / f"processed_freekiki_eval_{split}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    _log(f"evaluate: model={model_path} split={split} imgsz={imgsz} -> {out}")
    if split == "test":
        _log("note: test split - report only, do not choose thresholds/hyper-parameters on it")

    val = net.val(
        data=str(yaml_path),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(out),
        name="ultralytics_val",
        verbose=False,
    )
    ultra = {k: round(float(v), 4) for k, v in val.results_dict.items()}

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if max_images:
        images = images[:max_images]
    pred = collect_predictions(
        net, images, lbl_dir, manifest_index(ws), imgsz=imgsz, device=device, raw_conf=RAW_CONF
    )
    np_savez(out / "predictions.npz", pred)
    scores = diag.score_predictions(
        pred, det_conf=det_conf, kp_conf=kp_conf, match_px=match_px, pck=pck
    )
    scores["images"] = pred["images"].tolist()
    write_scores(out, scores)
    overall, calib = (
        scores["overall"],
        {k: v for k, v in scores["calib"].items() if k != "per_image"},
    )
    summary = {
        "eval_schema": diag.EVAL_SCHEMA,
        "model": model_path,
        "model_sha256": file_sha256(model_path),
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
    _log(f"evaluate done -> {out}")
    return out


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
    if summary.get("split") == "test" and not allow_test:
        raise ValueError("sweep on the test split refused: choose thresholds on val")
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


def compare(ws, *, baseline: str = "active", candidate: str, promote: bool = False) -> dict:
    """Gate ``candidate`` against ``baseline`` (eval folders or models) on the same split.

    With ``promote`` the candidate model is copied to ``models/active.pt`` only
    when every gate check passes. The decision is always logged.
    """
    ws = Path(ws).expanduser().resolve()
    settings = load_settings(ws)
    gate = settings["promotion"]
    split = gate.get("split", "val")
    cand_dir = _eval_for(ws, candidate, split=split)
    like = json.loads((cand_dir / "eval_summary.json").read_text(encoding="utf-8"))
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
        active_path = ws / settings["active"]["model"]
        backup = active_path.with_name(f"active_before_{datetime.now():%Y%m%d_%H%M%S}.pt")
        shutil.copy2(active_path, backup)
        shutil.copy2(decision["candidate_model"], active_path)
        cand_map = next(
            (c["candidate"] for c in decision["checks"] if c["check"] == "pose_map50_95"), None
        )
        settings["active"].update(
            {"run": Path(decision["candidate_model"]).stem, "model": ACTIVE_MODEL}
            | ({} if cand_map is None else {"pose_map50_95": cand_map})
        )
        save_settings(ws, settings)
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
        + (" -> PROMOTED" if promoted else "")
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
) -> Path:
    """Detect the 49 field keypoints in a video; returns the output folder.

    Outputs (``README.txt`` explains them): accepted points in getpixelvideo
    format, raw best-instance predictions, per-point status codes with the
    rejection reason, per-frame issues (cut, homography status), label-free
    ``quality.json`` and a few annotated ``diag_frames/``. ``fill_gaps`` > 0
    writes separate ``field_kps_filled_*`` CSVs with short gaps (same shot
    only) linearly interpolated and marked ``I``.
    """
    import cv2
    import numpy as np
    from ultralytics import YOLO

    ws = Path(ws).expanduser().resolve()
    video = Path(video).expanduser().resolve()
    defaults = load_settings(ws)["detect"]
    conf = float(defaults["conf"] if conf is None else conf)
    kp_conf = float(defaults["kp_conf"] if kp_conf is None else kp_conf)
    stride = max(1, int(stride))
    model_path = resolve_model(ws, model)
    net = YOLO(model_path)
    imgsz = model_imgsz(net, imgsz, defaults["imgsz"])
    base_out = Path(output_dir).expanduser() if output_dir else video.parent
    out = base_out / f"processed_freekiki_{video.stem}_{datetime.now():%Y%m%d_%H%M%S}"
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
            results = net.predict(frame, imgsz=imgsz, conf=RAW_CONF, device=device, verbose=False)
            box_conf, xy, kc = best_instance(list(results)[0])
            sig = diag.frame_signature(frame)
            cut = diag.is_cut(prev_sig, sig, cut_threshold)
            prev_sig = sig
            codes = point_codes(xy, kc, box_conf, conf, kp_conf, (width, height))
            all_codes.update(codes)
            acc = np.array([c == "D" for c in codes])
            kc_acc = None if kc is None or box_conf < conf else np.where(acc, kc, np.nan)
            xy_acc = None if kc_acc is None else xy
            fit = diag.frame_homography(xy_acc, kc_acc, kp_conf, planar, world, width)
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
        f"video: {video}\nmodel: {model_path}\nframes: {processed} (start={start}, "
        f"stride={stride})\ndetection (box) conf: {conf}\nkeypoint conf: {kp_conf}\n"
        f"imgsz: {imgsz}\nfill gaps: {fill_gaps} frames\n"
        "schema: vaila/models/soccerfield_kiki.csv (p0..p48, 0-based)\n\n"
        "field_kps_getpixelvideo.csv  accepted points only (status D)\n"
        "field_kps_conf.csv           keypoint conf of the accepted box (blank: no box)\n"
        f"field_kps_raw.csv            best instance down to box conf {RAW_CONF} (no filter)\n"
        "field_kps_status.csv         per point code, cut flag, homography status\n"
        "frame_issues.csv             frames with a cut / no box / no valid homography\n"
        "diag_frames/                 annotated frames (green accepted, red rejected)\n"
        "quality.json                 label-free indicators - NOT accuracy\n\n"
        "status codes:\n"
        + "".join(f"  {k:<3s} {v}\n" for k, v in POINT_CODES.items())
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
    (out / "quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    _log(
        f"detect done: {processed} frames | detected {quality['detection_rate']:.0%} | "
        f"accepted kps/frame {quality['mean_visible_kps']} | >=4 kps "
        f"{quality['min4_kps_rate']:.0%} | homography ok {quality['calib_ok_rate']:.0%} | "
        f"cuts {quality['cuts']} | residual median {quality['residual_median_px']} px@1920 "
        f"(camera-motion removed) -> {out}"
    )
    return out


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def detect_videos(ws, source, *, output_dir=None, **kwargs) -> Path:
    """Detect on one video or on every video of a folder.

    A folder writes ``processed_freekiki_batch_<ts>/`` (inside ``output_dir``,
    default the folder itself) with one sub-folder per video and
    ``quality_summary.csv`` comparing them. Returns the output folder.
    """
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
    for video in videos:
        out = detect_video(ws, video, output_dir=batch, **kwargs)
        rows.append(json.loads((out / "quality.json").read_text(encoding="utf-8")))
    with (batch / "quality_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v) if isinstance(v, dict) else v for k, v in r.items()} for r in rows
        )
    _log(f"batch done: {len(videos)} videos, summary -> {batch / 'quality_summary.csv'}")
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
        ("check", "Validate the workspace dataset."),
        ("manifest", "Build an oversampled train list (manifests/vNNN) for rare keypoints."),
        ("train", "Train or retrain (--base active) the field-keypoint network."),
        ("status", "List training runs and which ones can be resumed."),
        ("resume", "Continue an interrupted training from its last saved epoch."),
        ("evaluate", "Measure a model on a labelled split (default val; quality report)."),
        ("sweep", "det_conf x kp_conf grid on a saved evaluation (val only by default)."),
        ("audit", "Read-only dataset audit (points, visibility, leakage, geometry)."),
        ("compare", "Gate a candidate model/evaluation against the baseline on val."),
        ("bench", "Short training speed / peak-VRAM benchmark (batch x workers)."),
        ("detect", "Detect field keypoints in a video or in every video of a folder."),
    ):
        p = sub.add_parser(cmd, help=help_text)
        p.add_argument("-w", "--workspace", required=True, help="FreeKiki workspace folder.")
        if cmd == "import-dataset":
            p.add_argument("--src", required=True, help="kiki49 dataset folder (has data.yaml).")
        elif cmd == "train":
            p.add_argument("--base", help="yolo26*-pose.pt, 'active' or a .pt path.")
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
            p.add_argument("--model", default="active", help="'active' or a .pt path.")
            p.add_argument(
                "--split",
                default="val",
                choices=("val", "test", "train"),
                help="val to choose/compare; test only for the final report.",
            )
            p.add_argument("--imgsz", type=int)
            p.add_argument("--batch", type=int, default=8)
            p.add_argument("--device")
            p.add_argument("--det-conf", type=float, help="Box threshold (default: detect conf).")
            p.add_argument("--kp-conf", type=float)
            p.add_argument("--match-px", type=float, default=25.0, help="Match radius, px@1920.")
            p.add_argument("--pck", default="5,10,25", help="PCK radii, px@1920.")
            p.add_argument("--max-images", type=int, default=0, help="0 = all images.")
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
            p.add_argument("--baseline", default="active", help="Eval folder, model or 'active'.")
            p.add_argument("--candidate", required=True, help="Eval folder or model .pt.")
            p.add_argument(
                "--promote", action="store_true", help="Copy to active.pt if the gate passes."
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
    return parser


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in str(text).replace(" ", "").split(",") if v)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in (None, "gui"):
        run_freekiki()
        return 0
    ws = Path(args.workspace)
    batch = getattr(args, "batch", None)
    if batch is not None and float(batch).is_integer():
        batch = int(batch)
    if args.command == "init":
        init_workspace(ws)
    elif args.command == "import-dataset":
        import_dataset(ws, args.src)
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
        run_cli(
            ["train", "--base", v["base"].get().strip()]
            + opt("--epochs", "epochs")
            + opt("--imgsz", "imgsz")
            + opt("--batch", "batch")
            + opt("--device", "device")
            + opt("--fraction", "fraction")
            + opt("--manifest", "manifest")
        )

    def smoke_preset():
        for key, value in (
            ("base", "yolo26n-pose.pt"),
            ("epochs", "5"),
            ("imgsz", "640"),
            ("batch", "16"),
            ("fraction", "0.1"),
        ):
            v[key].set(value)

    def full_preset():
        for key, value in (
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
        run_cli(["compare", "--baseline", "active", "--candidate", model])

    def do_detect():
        if not v["video"].get().strip():
            messagebox.showerror("FreeKiki", "Choose a video or a folder of videos.", parent=root)
            return
        args = ["detect", "--video", v["video"].get().strip(), "--model", v["model"].get().strip()]
        args += opt("--output-dir", "out") + opt("--stride", "stride")
        args += opt("--max-frames", "max_frames") + opt("--conf", "conf")
        args += opt("--kp-conf", "kp_conf") + opt("--device", "device")
        args += opt("--fill-gaps", "fill_gaps")
        if not overlay.get():
            args.append("--no-overlay")
        run_cli(args)

    def open_help():
        webbrowser.open(HELP_HTML.resolve().as_uri())

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
    bar = ttk.Frame(box)
    bar.grid(row=2, column=1, sticky="w", pady=2)
    ttk.Button(bar, text="Train", command=do_train).pack(side="left")
    ttk.Button(bar, text="Smoke preset (~5 min)", command=smoke_preset).pack(side="left", padx=6)
    ttk.Button(bar, text="Full preset (hours)", command=full_preset).pack(side="left")
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
    ttk.Label(
        box,
        text="Retrain = Base model 'active': AdamW lr0=1e-4, 1 warmup epoch, cosine, "
        "mosaic off, backbone unfrozen.\n"
        "Manifest = e.g. v001 from 'Build oversampling manifest' (rare points repeated; "
        "empty = plain train split).\n"
        "Stopped / crash / power loss? Every finished epoch is saved: press Resume interrupted.",
    ).grid(row=4, column=0, columnspan=3, sticky="w", padx=4)

    box = ttk.LabelFrame(frm, text="4. Evaluate quality (labelled split)", padding=6)
    box.pack(fill="x", pady=4)
    bar = ttk.Frame(box)
    bar.grid(row=0, column=0, columnspan=3, sticky="w")
    ttk.Label(bar, text="Split").pack(side="left", padx=(4, 2))
    ttk.Combobox(
        bar, textvariable=v["split"], values=("val", "test"), width=5, state="readonly"
    ).pack(side="left")
    ttk.Button(bar, text="Evaluate model", command=do_evaluate).pack(side="left", padx=6)
    ttk.Button(bar, text="Sweep thresholds (val)", command=do_sweep).pack(side="left")
    ttk.Button(bar, text="Compare with active", command=do_compare).pack(side="left", padx=6)
    ttk.Button(bar, text="Audit dataset", command=lambda: run_cli(["audit"])).pack(side="left")
    ttk.Label(
        box,
        text="Uses Model / Conf / KP conf below. Choose thresholds and compare models on val;\n"
        "test only for the final report. Reports in <workspace>/outputs.",
    ).grid(row=1, column=0, columnspan=3, sticky="w", padx=4)

    box = ttk.LabelFrame(frm, text="5. Detect field keypoints (video or folder)", padding=6)
    box.pack(fill="x", pady=4)
    video_types = [("Video", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All", "*.*")]
    row(box, 0, "Video", v["video"], lambda: browse_file(v["video"], "Video", video_types))
    ttk.Button(box, text="Folder", command=lambda: browse_dir(v["video"], "Folder of videos")).grid(
        row=0, column=3
    )
    row(box, 1, "Output dir", v["out"], lambda: browse_dir(v["out"], "Output folder"))
    row(
        box,
        2,
        "Model",
        v["model"],
        lambda: browse_file(v["model"], "Model weights", [("PyTorch", "*.pt")]),
    )
    grid = ttk.Frame(box)
    grid.grid(row=3, column=0, columnspan=3, sticky="w", pady=2)
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
    ttk.Button(box, text="Detect", command=do_detect).grid(row=4, column=1, sticky="w", pady=2)

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
