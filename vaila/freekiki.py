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
Update Date: 25 September 2026

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
          runs/<name>/           Ultralytics training runs
          models/registry.csv    every finished run with its pose mAP
          models/active.pt       best model so far (used by Detect / Retrain)

    Retrain starts from ``models/active.pt`` (transfer learning) and a new run
    only replaces ``active.pt`` when its validation pose mAP50-95 is better.

    A training stopped by the Stop button, a crash or a power loss keeps
    ``runs/<name>/weights/last.pt`` (saved after every epoch); ``resume``
    continues it from the last completed epoch and then registers it.

    Evaluate measures a model on the labelled test split (never seen in
    training): pose mAP plus per-keypoint recall, false positives and pixel
    error, logged in ``models/evaluations.csv``.

    Detect writes, per video, a getpixelvideo-compatible wide CSV
    (``frame,p0_x,p0_y,...,p48_x,p48_y``; 0-based, blank when a keypoint is
    below the confidence threshold), a per-keypoint confidence CSV and an
    optional overlay MP4.

Usage:
    uv run vaila/freekiki.py                       # GUI
    uv run vaila/freekiki.py init   -w WORKSPACE
    uv run vaila/freekiki.py import-dataset -w WORKSPACE --src /path/kiki49_dataset
    uv run vaila/freekiki.py check  -w WORKSPACE
    uv run vaila/freekiki.py train  -w WORKSPACE --base yolo26m-pose.pt --epochs 150 --imgsz 1280
    uv run vaila/freekiki.py train  -w WORKSPACE --base active          # retrain / fine-tune
    uv run vaila/freekiki.py status -w WORKSPACE                       # runs + resumable ones
    uv run vaila/freekiki.py resume -w WORKSPACE [--name RUN]          # continue after stop/power loss
    uv run vaila/freekiki.py evaluate -w WORKSPACE                     # quality on test split
    uv run vaila/freekiki.py detect -w WORKSPACE --video match.mp4 [--output-dir DIR]
    uv run vaila/freekiki.py detect -w WORKSPACE --video FOLDER_OF_VIDEOS

    On NVIDIA/CUDA machines use ``uv run --no-sync`` (see CLAUDE.md).

License:
    GNU Affero General Public License v3.0 (AGPLv3).
================================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

import toml
import yaml

try:
    from .cli_highlight import print_gui_cli_mirror
except ImportError:
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
}
MAP50_COL = "metrics/mAP50(P)"
MAP50_95_COL = "metrics/mAP50-95(P)"


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
    """Best epoch (by pose mAP50-95) from an Ultralytics ``results.csv``."""
    best = {"best_epoch": "", "pose_map50": 0.0, "pose_map50_95": 0.0}
    if not Path(results_csv).is_file():
        return best
    with Path(results_csv).open(encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
            try:
                score = float(row.get(MAP50_95_COL, "nan"))
            except ValueError:
                continue
            if score == score and score >= best["pose_map50_95"]:
                best = {
                    "best_epoch": row.get("epoch", ""),
                    "pose_map50": float(row.get(MAP50_COL) or 0.0),
                    "pose_map50_95": score,
                }
    return best


def register_run(ws, run_dir, *, base: str, epochs: int, imgsz: int) -> dict:
    """Copy a run's best.pt into ``models/``, log it, promote to active if better."""
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
    promoted = (not active_path.is_file()) or metrics["pose_map50_95"] > float(
        active.get("pose_map50_95", 0.0)
    )
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
    new_file = not registry.is_file()
    with registry.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    _log(
        f"run {run_dir.name}: pose mAP50-95={metrics['pose_map50_95']:.4f} "
        f"-> {'PROMOTED to ' + ACTIVE_MODEL if promoted else 'kept previous active model'}"
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
) -> dict:
    """Train (from a base model) or retrain (``base='active'``) on the workspace dataset."""
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
    model_path = resolve_model(ws, base)
    _log(f"train: base={model_path} epochs={epochs} imgsz={imgsz} batch={batch} run={name}")
    extra: dict = {"patience": patience}
    if fraction < 1.0:
        extra["fraction"] = fraction
    with _running_marker(ws / "runs" / name):
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


def resume(ws, *, name: str | None = None, device: str | None = None, batch=None) -> dict:
    """Continue an interrupted training from its ``weights/last.pt``.

    Without ``name`` the newest run that is ``resumable`` or ``finished`` (not
    registered) is used. The dataset path and run folder come from the current
    workspace, so a workspace copied to another machine can also be resumed.
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


def keypoint_errors(pred_xy, pred_conf, gt_xy, gt_vis, width: int, kp_conf: float) -> dict:
    """Compare one prediction with one label (pixels of the image).

    Errors are rescaled to a 1920-px-wide frame so images of different
    resolutions are comparable. Returns per-keypoint arrays:
    ``labelled`` (bool), ``found`` (labelled and predicted), ``false_pos``
    (predicted but not labelled) and ``err_px`` (NaN unless found).
    """
    import numpy as np

    labelled = np.asarray(gt_vis) > 0
    predicted = np.zeros(NKP, dtype=bool) if pred_conf is None else np.asarray(pred_conf) >= kp_conf
    found = labelled & predicted
    err = np.full(NKP, np.nan)
    if pred_xy is not None and found.any():
        d = np.linalg.norm(np.asarray(pred_xy) - np.asarray(gt_xy), axis=1)
        err[found] = d[found] * (1920.0 / max(1, int(width)))
    return {"labelled": labelled, "found": found, "false_pos": predicted & ~labelled, "err_px": err}


def summarize_keypoint_errors(per_image: list[dict]) -> tuple[dict, list[dict]]:
    """Aggregate :func:`keypoint_errors` results into overall + per-keypoint tables."""
    import numpy as np

    names, _ = load_schema()
    lab = np.array([p["labelled"] for p in per_image]).reshape(-1, NKP)
    fnd = np.array([p["found"] for p in per_image]).reshape(-1, NKP)
    fps = np.array([p["false_pos"] for p in per_image]).reshape(-1, NKP)
    err = np.array([p["err_px"] for p in per_image], dtype=float).reshape(-1, NKP)

    def stats(lab_n: int, fnd_n: int, fp_n: int, errors) -> dict:
        errors = errors[np.isfinite(errors)]
        return {
            "labelled": lab_n,
            "recall": round(fnd_n / lab_n, 4) if lab_n else None,
            "false_pos": fp_n,
            "precision": round(fnd_n / (fnd_n + fp_n), 4) if fnd_n + fp_n else None,
            "median_err_px": round(float(np.median(errors)), 2) if errors.size else None,
            "pck10": round(float((errors <= 10).mean()), 4) if errors.size else None,
            "pck25": round(float((errors <= 25).mean()), 4) if errors.size else None,
        }

    per_kp = [
        {"kp": f"p{i}", "name": names[i]}
        | stats(int(lab[:, i].sum()), int(fnd[:, i].sum()), int(fps[:, i].sum()), err[:, i])
        for i in range(NKP)
    ]
    overall = {"images": len(per_image)} | stats(
        int(lab.sum()), int(fnd.sum()), int(fps.sum()), err.ravel()
    )
    return overall, per_kp


def model_imgsz(net, imgsz: int | None, fallback: int) -> int:
    """Image size for inference: explicit value, else the model's training size."""
    return int(imgsz or net.overrides.get("imgsz") or fallback)


def evaluate(
    ws,
    *,
    model: str = "active",
    split: str = "test",
    imgsz: int | None = None,
    batch: int = 8,
    device: str | None = None,
    kp_conf: float | None = None,
    max_images: int = 0,
) -> Path:
    """Measure a model on a labelled split (default ``test``, never used in training).

    Two views of quality:
      * Ultralytics validation (pose/box mAP, precision, recall);
      * per-keypoint pixel error / recall / false positives, the numbers that
        matter for camera calibration.

    Writes ``<ws>/outputs/processed_freekiki_eval_<split>_<ts>/`` and appends a
    row to ``models/evaluations.csv`` so models can be compared over time.
    """
    import cv2
    from ultralytics import YOLO

    ws = Path(ws).expanduser().resolve()
    settings = load_settings(ws)
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
    per_image = []
    for n, img_path in enumerate(images, 1):
        gt_xy, gt_vis = read_label_keypoints(lbl_dir / f"{img_path.stem}.txt")
        if gt_xy is None:
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        result: Any = list(net.predict(frame, imgsz=imgsz, device=device, verbose=False))[0]
        pxy = pkc = None
        if result.keypoints is not None and len(result.keypoints) and result.boxes is not None:
            best = int(result.boxes.conf.argmax())
            pxy = result.keypoints.xy[best].cpu().numpy()
            if result.keypoints.conf is not None:
                pkc = result.keypoints.conf[best].cpu().numpy()
        per_image.append(keypoint_errors(pxy, pkc, gt_xy * (w, h), gt_vis, w, kp_conf))
        if n % 500 == 0:
            _log(f"  {n}/{len(images)} images")
    overall, per_kp = summarize_keypoint_errors(per_image)

    with (out / "per_keypoint.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_kp[0]))
        writer.writeheader()
        writer.writerows(per_kp)
    summary = {
        "model": model_path,
        "split": split,
        "imgsz": imgsz,
        "kp_conf": kp_conf,
        "ultralytics": ultra,
        "keypoints": overall,
    }
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_path,
        "split": split,
        "imgsz": imgsz,
        "pose_map50": ultra.get(MAP50_COL, ""),
        "pose_map50_95": ultra.get(MAP50_95_COL, ""),
        "kp_recall": overall["recall"],
        "kp_precision": overall["precision"],
        "median_err_px": overall["median_err_px"],
        "pck10": overall["pck10"],
        "pck25": overall["pck25"],
        "report": out.relative_to(ws).as_posix(),
    }
    evals = ws / "models" / "evaluations.csv"
    new_file = not evals.is_file()
    with evals.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)

    _log(
        f"pose mAP50={row['pose_map50']} mAP50-95={row['pose_map50_95']} | keypoints: "
        f"recall={overall['recall']} precision={overall['precision']} "
        f"median error={overall['median_err_px']} px@1920 PCK10={overall['pck10']} "
        f"PCK25={overall['pck25']} ({overall['images']} images)"
    )
    ranked = sorted(
        (k for k in per_kp if k["labelled"]), key=lambda k: (k["recall"] or 0.0, k["kp"])
    )
    _log("hardest keypoints (lowest recall):")
    for k in ranked[:8]:
        _log(
            f"  {k['kp']:>3s} {k['name']:<42s} n={k['labelled']:<5d} recall={k['recall']} "
            f"median_err={k['median_err_px']} px"
        )
    _log(f"evaluate done -> {out}")
    return out


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def keypoints_row(frame: int, xy, kconf, kp_conf: float) -> list:
    """One getpixelvideo row: ``frame, p0_x, p0_y, ...``; blanks below ``kp_conf``."""
    row: list = [frame]
    for i in range(NKP):
        if xy is None or kconf is None or float(kconf[i]) < kp_conf:
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


def video_quality(xy_seq: list, kc_seq: list, kp_conf: float) -> dict:
    """Label-free quality indicators of a detection run (one entry per processed frame).

    * ``detection_rate``: frames where a field instance was found;
    * ``mean_visible_kps``: keypoints above ``kp_conf`` per frame;
    * ``calib_ready_rate``: frames with >= 4 visible keypoints (minimum for a
      homography / DLT2D);
    * ``mean_kp_conf``: mean confidence of the visible keypoints;
    * ``jitter_px``: median |x[t+1] - 2x[t] + x[t-1]| of keypoints visible in
      three consecutive processed frames (smooth camera motion ~0, flicker high).
    """
    import numpy as np

    n = len(kc_seq)
    pts = np.full((n, NKP, 2), np.nan)
    visible = np.zeros((n, NKP), dtype=bool)
    confs = []
    for t, (xy, kc) in enumerate(zip(xy_seq, kc_seq, strict=True)):
        if xy is None or kc is None:
            continue
        vis = np.asarray(kc) >= kp_conf
        visible[t] = vis
        pts[t, vis] = np.asarray(xy)[vis]
        confs.extend(np.asarray(kc)[vis].tolist())
    counts = visible.sum(axis=1)
    jitter = np.array([])
    if n >= 3:
        acc = np.linalg.norm(pts[2:] - 2 * pts[1:-1] + pts[:-2], axis=2)
        jitter = acc[np.isfinite(acc)]
    detected = sum(kc is not None for kc in kc_seq)
    return {
        "frames": n,
        "detection_rate": round(detected / n, 4) if n else 0.0,
        "mean_visible_kps": round(float(counts.mean()), 2) if n else 0.0,
        "calib_ready_rate": round(float((counts >= 4).mean()), 4) if n else 0.0,
        "mean_kp_conf": round(float(np.mean(confs)), 4) if confs else 0.0,
        "jitter_px": round(float(np.median(jitter)), 2) if jitter.size else None,
    }


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
) -> Path:
    """Detect the 49 field keypoints in a video; returns the output folder."""
    import cv2
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
    out.mkdir(parents=True, exist_ok=True)

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
    writer = None
    if overlay:
        writer = cv2.VideoWriter(
            str(out / f"{video.stem}_freekiki_overlay.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),  # ty: ignore[unresolved-attribute]
            fps / stride,
            (width, height),
        )
    _log(f"detect: {video.name} ({total} frames) model={model_path} -> {out}")

    pix_rows, conf_rows, xy_seq, kc_seq = [], [], [], []
    snapshot_at = (min(total - start, max_frames or total) // stride) // 2 if total else 0
    frame_idx, processed = start, 0
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
            results = net.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)
            result: Any = list(results)[0]
            xy = kc = None
            kps = result.keypoints
            if kps is not None and len(kps) and result.boxes is not None:
                best = int(result.boxes.conf.argmax())
                xy = kps.xy[best].cpu().numpy()
                kc = kps.conf[best].cpu().numpy() if kps.conf is not None else None
            if kc is None:
                xy = None
            xy_seq.append(xy)
            kc_seq.append(kc)
            pix_rows.append(keypoints_row(frame_idx, xy, kc, kp_conf))
            conf_rows.append(
                [frame_idx] + ([f"{float(c):.3f}" for c in kc] if kc is not None else [""] * NKP)
            )
            if writer is not None or processed == snapshot_at:
                drawn = _draw_overlay(frame, xy, kc, kp_conf, bones)
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

    with (out / "field_kps_getpixelvideo.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(getpixelvideo_header())
        w.writerows(pix_rows)
    with (out / "field_kps_conf.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame"] + [f"p{i}_conf" for i in range(NKP)])
        w.writerows(conf_rows)
    (out / "README.txt").write_text(
        "FreeKiki field keypoints (vailá)\n"
        f"video: {video}\nmodel: {model_path}\nframes: {processed} (start={start}, "
        f"stride={stride})\ndetection conf: {conf}\nkeypoint conf: {kp_conf}\nimgsz: {imgsz}\n"
        "schema: vaila/models/soccerfield_kiki.csv (p0..p48, 0-based)\n",
        encoding="utf-8",
    )
    quality = {"video": video.name, "model": model_path} | video_quality(xy_seq, kc_seq, kp_conf)
    (out / "quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    _log(
        f"detect done: {processed} frames | detected {quality['detection_rate']:.0%} | "
        f"visible kps/frame {quality['mean_visible_kps']} | calib-ready (>=4 kps) "
        f"{quality['calib_ready_rate']:.0%} | kp conf {quality['mean_kp_conf']} | "
        f"jitter {quality['jitter_px']} px -> {out}"
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
        writer.writerows(rows)
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
        ("train", "Train or retrain (--base active) the field-keypoint network."),
        ("status", "List training runs and which ones can be resumed."),
        ("resume", "Continue an interrupted training from its last saved epoch."),
        ("evaluate", "Measure a model on the labelled test split (quality report)."),
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
        elif cmd == "resume":
            p.add_argument("--name", help="Run to resume (default: newest interrupted run).")
            p.add_argument("--device")
            p.add_argument("--batch", type=float, help="Override batch (e.g. after OOM).")
        elif cmd == "evaluate":
            p.add_argument("--model", default="active", help="'active' or a .pt path.")
            p.add_argument("--split", default="test", choices=("test", "val", "train"))
            p.add_argument("--imgsz", type=int)
            p.add_argument("--batch", type=int, default=8)
            p.add_argument("--device")
            p.add_argument("--kp-conf", type=float)
            p.add_argument("--max-images", type=int, default=0, help="0 = all images.")
        elif cmd == "detect":
            p.add_argument("--video", required=True, help="Video file or folder of videos.")
            p.add_argument("--model", default="active")
            p.add_argument("--output-dir", help="Default: the video's folder.")
            p.add_argument("--stride", type=int, default=1)
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--max-frames", type=int, default=0)
            p.add_argument("--conf", type=float)
            p.add_argument("--kp-conf", type=float)
            p.add_argument("--imgsz", type=int)
            p.add_argument("--device")
            p.add_argument("--no-overlay", action="store_true")
    return parser


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
        )
    elif args.command == "evaluate":
        evaluate(
            ws,
            model=args.model,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            kp_conf=args.kp_conf,
            max_images=args.max_images,
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
        "video": tk.StringVar(),
        "out": tk.StringVar(),
        "model": tk.StringVar(value="active"),
        "stride": tk.StringVar(value="1"),
        "max_frames": tk.StringVar(value="0"),
        "conf": tk.StringVar(value="0.25"),
        "kp_conf": tk.StringVar(value="0.5"),
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
            + opt("--kp-conf", "kp_conf")
            + opt("--device", "device")
        )

    def do_detect():
        if not v["video"].get().strip():
            messagebox.showerror("FreeKiki", "Choose a video or a folder of videos.", parent=root)
            return
        args = ["detect", "--video", v["video"].get().strip(), "--model", v["model"].get().strip()]
        args += opt("--output-dir", "out") + opt("--stride", "stride")
        args += opt("--max-frames", "max_frames") + opt("--conf", "conf")
        args += opt("--kp-conf", "kp_conf") + opt("--device", "device")
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
        + (("Device", "device"), ("Fraction", "fraction"))
    ):
        ttk.Label(grid, text=label).grid(row=0, column=2 * i, padx=(4, 2))
        ttk.Entry(grid, textvariable=v[key], width=7).grid(row=0, column=2 * i + 1)
    bar = ttk.Frame(box)
    bar.grid(row=2, column=1, sticky="w", pady=2)
    ttk.Button(bar, text="Train", command=do_train).pack(side="left")
    ttk.Button(bar, text="Smoke preset (~5 min)", command=smoke_preset).pack(side="left", padx=6)
    ttk.Button(bar, text="Full preset (hours)", command=full_preset).pack(side="left")
    bar = ttk.Frame(box)
    bar.grid(row=3, column=1, sticky="w", pady=2)
    ttk.Button(bar, text="Runs status", command=lambda: run_cli(["status"])).pack(side="left")
    ttk.Button(
        bar,
        text="Resume interrupted",
        command=lambda: run_cli(["resume"] + opt("--device", "device")),
    ).pack(side="left", padx=6)
    ttk.Label(
        box,
        text="Retrain = Base model 'active' (transfer from best model so far).\n"
        "Stopped / crash / power loss? Every finished epoch is saved: press Resume interrupted.",
    ).grid(row=4, column=0, columnspan=3, sticky="w", padx=4)

    box = ttk.LabelFrame(frm, text="4. Evaluate quality (labelled test split)", padding=6)
    box.pack(fill="x", pady=4)
    ttk.Button(box, text="Evaluate model", command=do_evaluate).grid(row=0, column=1, sticky="w")
    ttk.Label(
        box, text="Uses the Model below; report in <workspace>/outputs + models/evaluations.csv."
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
        )
    ):
        ttk.Label(grid, text=label).grid(row=0, column=2 * i, padx=(4, 2))
        ttk.Entry(grid, textvariable=v[key], width=7).grid(row=0, column=2 * i + 1)
    ttk.Checkbutton(grid, text="Overlay MP4", variable=overlay).grid(row=0, column=8, padx=6)
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
