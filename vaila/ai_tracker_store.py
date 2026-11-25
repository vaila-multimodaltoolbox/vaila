"""
Where the AI Track network lives: a folder chosen by the user, outside the vailá install.

Project: vailá Multimodal Toolbox
Script: ai_tracker_store.py - Home folder, versioned atomic saves and sample bank of AI Track

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 24 November 2026
Update Date: 24 November 2026
Version: 0.4.19

Description:
    The AI Track network (online discriminators, fine-tuned backbones and the human-placed
    training samples) is small but valuable: it improves with every manual marker. It must
    survive deleting or reinstalling vailá, so it lives in a folder the user chooses.

    The folder (the *home*) is chosen, in this order, by

    1. ``--ai-models DIR`` (CLI; the GUI calls :func:`set_session_home`);
    2. the environment variable ``VAILA_AI_MODELS_HOME``;
    3. ``~/.vaila/ai_models.json`` (written by :func:`set_home` when the user chooses);
    4. the legacy folder inside the vailá checkout, ``vaila/models/ai_tracker`` (no backup).

    Resolution never prompts. :func:`ensure_home` is the only function that may ask, and
    only through a ``prompt_fn`` the GUI supplies. An explicitly chosen home that cannot be
    created is an error (``ValueError``), never a silent fallback to another folder.

    Layout of ``<home>/ai_tracker/`` (``root`` below)::

        manifest.json           active backbones, sha256/size of every file, HF sync info
        README.txt
        backbones/              <variant>_finetuned_<ts>.pth (the official *_imagenet.pth
                                is NOT copied: it is re-downloadable)
        discriminators/         discriminator_<profile>__<key>.npz, key = classic | <variant>-<sha8>
        samples/                <session ts>_<video stem>.npz (local only, never uploaded)
        history/<file name>/    up to HISTORY_KEEP previous versions of each file
        logs/

    Frozen public API (other modules build on these names and signatures)::

        ENV_HOME, CONFIG_FILE, SUGGESTED_HOME, SUBDIR, LEGACY_DIR, HISTORY_KEEP
        StorePaths(root, source, backed_up)
        set_session_home(path | None) -> None
        resolve_home(cli_home=None) -> StorePaths
        is_configured() -> bool
        set_home(path, *, migrate=True) -> StorePaths
        ensure_home(prompt_fn=None, cli_home=None) -> StorePaths
        layout(root) -> dict[str, Path]  (keys backbones, discriminators, samples, history, logs)
        migrate_legacy(root) -> list[Path]
        file_sha256(path) -> str
        feature_key(variant, weights_path) -> str
        discriminator_path(root, profile="default", key="classic") -> Path
        save_npz_atomic(path, *, compressed=False, **arrays) -> Path
        atomic_copy(src, dst) -> Path
        history_versions(path) -> list[Path]  (newest first)
        restore(path, version_path) -> Path
        read_manifest(root) -> dict, write_manifest(root, data), update_manifest_file(root, path, **info)
        active_backbone(root, variant) -> Path | None, set_active_backbone(root, variant, path)
        samples_dir(root) -> Path, load_samples(path) -> dict[str, np.ndarray]

    The *root* of a file is the nearest parent folder named ``ai_tracker`` (or the legacy
    folder). Files outside any root (for example a test writing to ``tmp/x.npz``) keep their
    ``history/`` next to the file.

License: AGPLv3
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

ENV_HOME = "VAILA_AI_MODELS_HOME"
CONFIG_FILE = Path.home() / ".vaila" / "ai_models.json"
SUGGESTED_HOME = Path.home() / "vaila_ai_models"
SUBDIR = "ai_tracker"
LEGACY_DIR = Path(__file__).resolve().parent / "models" / SUBDIR
HISTORY_KEEP = 10
VAILA_VERSION = "0.4.19"

CLASSIC_KEY = "classic"
_CLASSIC_DIM = 808  # AITracker classical feature length (768 colour grid + 40 region stats)
_RESNET50_DEEP_DIM = 2048
_LAYOUT_DIRS = ("backbones", "discriminators", "samples", "history", "logs")

_session_home: Path | None = None
_SHA_CACHE: dict[tuple[str, int, int], str] = {}


class StorePaths(NamedTuple):
    """Result of :func:`resolve_home`.

    ``root`` is the ``<home>/ai_tracker`` folder (or the legacy folder), ``source`` is
    ``"cli" | "env" | "config" | "legacy"`` and ``backed_up`` is False only for ``legacy``
    (the network then lives inside the vailá checkout, with no copy outside it).
    """

    root: Path
    source: str
    backed_up: bool


# ---------------------------------------------------------------- home resolution


def set_session_home(path: str | Path | None) -> None:
    """Process-wide override of the home (``getpixelvideo --ai-models``); None clears it."""
    global _session_home
    _session_home = Path(path).expanduser() if path else None


def _env_home() -> Path | None:
    value = os.environ.get(ENV_HOME, "").strip()
    return Path(value).expanduser() if value else None


def _config_home() -> Path | None:
    try:
        data = json.loads(Path(CONFIG_FILE).read_text(encoding="utf-8"))
        value = str(data.get("home", "") or "").strip()
    except (OSError, ValueError, AttributeError):
        return None
    return Path(value).expanduser() if value else None


def _chosen_home(cli_home: str | Path | None) -> tuple[Path | None, str]:
    if cli_home:
        return Path(cli_home).expanduser(), "cli"
    if _session_home is not None:
        return _session_home, "cli"
    env = _env_home()
    if env is not None:
        return env, "env"
    cfg = _config_home()
    if cfg is not None:
        return cfg, "config"
    return None, "legacy"


def layout(root: str | Path) -> dict[str, Path]:
    """Creates and returns the sub-folders of a root: backbones, discriminators, samples,
    history, logs."""
    base = Path(root)
    out = {name: base / name for name in _LAYOUT_DIRS}
    for folder in out.values():
        folder.mkdir(parents=True, exist_ok=True)
    return out


def _open_root(home: Path, source: str) -> StorePaths:
    root = home / SUBDIR
    try:
        layout(root)
    except OSError as err:
        raise ValueError(
            f"AI Track models folder {home} (from {source}) cannot be created: {err}. "
            f"Choose another with --ai-models DIR, ${ENV_HOME} or the Net button."
        ) from err
    readme = root / "README.txt"
    if not readme.exists():  # never overwrite; the legacy folder never comes through here
        with contextlib.suppress(OSError):  # a courtesy: the folder works without it
            _write_text_atomic(readme, _README)
    return StorePaths(root, source, True)


def resolve_home(cli_home: str | Path | None = None) -> StorePaths:
    """Resolves the AI Track root; never prompts.

    Order: ``cli_home`` or the session home, then ``$VAILA_AI_MODELS_HOME``, then
    ``~/.vaila/ai_models.json``, then the legacy checkout folder. A chosen home is created
    on the way; one that cannot be created raises ``ValueError``.
    """
    home, source = _chosen_home(cli_home)
    if home is None:
        return StorePaths(Path(LEGACY_DIR), "legacy", False)
    return _open_root(home, source)


def is_configured() -> bool:
    """True when a home was chosen (cli/session, env var or config file)."""
    return _chosen_home(None)[1] != "legacy"


_README = """AI Track models folder (vaila - Multimodal Toolbox)
====================================================

This folder holds the AI Track network of vaila. It lives outside the vaila install on
purpose: deleting or reinstalling vaila does not touch it.

  discriminators/  online discriminators, one file per profile and feature space
                   (classic = 808 values; <variant>-<sha8> = classic + deep backbone)
  backbones/       fine-tuned backbones (<variant>_finetuned_<timestamp>.pth)
  samples/         the points you placed by hand (image crops). Local only: never uploaded
  history/         up to 10 previous versions of every file, newest first
  logs/            logs of the retraining jobs
  manifest.json    active backbones, sha256 and size of every file, Hugging Face sync info

Every write is atomic (temporary file, then rename) and keeps the previous version in
history/. Copy this whole folder to another disk or computer to carry the network with you.
Choose the folder with --ai-models DIR, the VAILA_AI_MODELS_HOME variable or the Net button
of getpixelvideo.
"""


def set_home(path: str | Path, *, migrate: bool = True) -> StorePaths:
    """Chooses ``path`` as the AI Track home: writes ``~/.vaila/ai_models.json``, creates the
    layout and README.txt, and (``migrate``) copies the legacy files into it."""
    home = Path(path).expanduser().absolute()
    paths = _open_root(home, "config")
    cfg = Path(CONFIG_FILE)
    try:
        cfg.parent.mkdir(parents=True, exist_ok=True)
        _write_text_atomic(cfg, json.dumps({"home": str(home)}, indent=2) + "\n")
    except OSError as err:
        raise ValueError(f"cannot write the AI Track settings ({err})") from err
    if migrate:
        migrate_legacy(paths.root)
    else:
        write_manifest(paths.root, read_manifest(paths.root))
    return paths


def ensure_home(
    prompt_fn: Callable[[Path], str | Path | None] | None = None,
    cli_home: str | Path | None = None,
) -> StorePaths:
    """Returns the home, asking once through ``prompt_fn(SUGGESTED_HOME)`` when none is set.

    ``prompt_fn`` returns a folder (it becomes the home via :func:`set_home`) or None
    (cancelled: the legacy folder is used for this session, nothing is written, and the
    question comes back next session). Without ``prompt_fn`` it never asks.
    """
    if _chosen_home(cli_home)[1] != "legacy":
        return resolve_home(cli_home)
    if prompt_fn is not None:
        answer = prompt_fn(SUGGESTED_HOME)
        if answer:
            return set_home(answer)
    return StorePaths(Path(LEGACY_DIR), "legacy", False)


# ---------------------------------------------------------------- hashing and keys


def file_sha256(path: str | Path) -> str:
    """sha256 hex digest of a file, cached by (path, size, mtime_ns) so a 98 MB backbone is
    not re-hashed at every AI Track start."""
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    cached = _SHA_CACHE.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def feature_key(variant: str | None, weights_path: str | Path | None) -> str:
    """Identifies the feature space of a discriminator: ``"classic"`` without a backbone,
    else ``"<variant>-<sha256[:8] of the weights file>"`` (``-unknown`` if the file is gone)."""
    if not variant:
        return CLASSIC_KEY
    digest = ""
    if weights_path:
        try:
            digest = file_sha256(weights_path)[:8]
        except OSError:
            digest = ""
    return f"{variant}-{digest or 'unknown'}"


def _safe(text: str, default: str) -> str:
    cleaned = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(text))
    return cleaned.strip(".") or default


def discriminator_path(root: str | Path, profile: str = "default", key: str = CLASSIC_KEY) -> Path:
    """``root/discriminators/discriminator_<safe profile>__<safe key>.npz``."""
    name = f"discriminator_{_safe(profile, 'default')}__{_safe(key, CLASSIC_KEY)}.npz"
    return Path(root) / "discriminators" / name


def samples_dir(root: str | Path) -> Path:
    """``root/samples`` (created)."""
    folder = Path(root) / "samples"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def load_samples(path: str | Path) -> dict[str, np.ndarray]:
    """Reads a sample bank written by ``AITracker.save_samples`` (no pickle).

    Keys (N = number of samples): ``patches`` (N,h,w,3 uint8), ``contexts`` (N,H,W,3 uint8,
    twice the block window, centred on the sample), ``labels`` (N, +1 marker / -1 background),
    ``frames``, ``markers``, ``block_w``, ``block_h`` (N ints), ``points`` (N,2 float32),
    ``videos``, ``shapes`` (N strings), ``session`` (0-d string).
    """
    with np.load(Path(path), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


# ---------------------------------------------------------------- atomic writes + history


def _find_root(path: Path) -> Path:
    """Nearest parent named SUBDIR (or the legacy folder); else the file's own folder."""
    legacy = Path(LEGACY_DIR)
    try:
        legacy_r = legacy.resolve()
    except OSError:
        legacy_r = legacy
    for parent in path.parents:
        if parent.name == SUBDIR:
            return parent
        try:
            if parent.resolve() == legacy_r:
                return parent
        except OSError:
            continue
    return path.parent


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _history_dir(path: Path) -> Path:
    return _find_root(path) / "history" / path.name


def history_versions(path: str | Path) -> list[Path]:
    """Previous versions of a file kept in ``history/<file name>/``, newest first."""
    folder = _history_dir(Path(path))
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if p.is_file()), key=lambda p: p.name, reverse=True)


def _archive_current(path: Path) -> Path | None:
    """Copies the current file into history/ and trims history to HISTORY_KEEP versions."""
    if not path.is_file():
        return None
    folder = _history_dir(path)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()
    dest = folder / f"{stamp}{path.suffix}"
    n = 0
    while dest.exists():
        n += 1
        dest = folder / f"{stamp}_{n}{path.suffix}"
    shutil.copy2(path, dest)
    for old in history_versions(path)[HISTORY_KEEP:]:
        old.unlink(missing_ok=True)
    return dest


def _tmp_in(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=folder)
    os.close(fd)
    return Path(tmp)


def _install(tmp: Path, dst: Path) -> Path:
    """History of the old file, then atomic rename of ``tmp`` onto ``dst``."""
    try:
        _archive_current(dst)
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dst


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = _tmp_in(path.parent, path.name)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_npz_atomic(path: str | Path, *, compressed: bool = False, **arrays: Any) -> Path:
    """Writes ``arrays`` to ``path`` through a temp file + ``os.replace``; the previous
    version goes to ``root/history/<name>/<timestamp>.<ext>`` (newest HISTORY_KEEP kept)."""
    dst = Path(path)
    tmp = _tmp_in(dst.parent, dst.name)
    try:
        with tmp.open("wb") as fh:
            (np.savez_compressed if compressed else np.savez)(fh, **arrays)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return _install(tmp, dst)


def atomic_copy(src: str | Path, dst: str | Path) -> Path:
    """Copies ``src`` to ``dst`` atomically, archiving an existing ``dst`` in history/."""
    target = Path(dst)
    tmp = _tmp_in(target.parent, target.name)
    try:
        shutil.copyfile(src, tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return _install(tmp, target)


def restore(path: str | Path, version_path: str | Path) -> Path:
    """Puts a history version back as ``path``; the current file goes to history first."""
    return atomic_copy(version_path, path)


# ---------------------------------------------------------------- manifest


def _manifest_file(root: str | Path) -> Path:
    return Path(root) / "manifest.json"


def _empty_manifest() -> dict[str, Any]:
    return {
        "version": VAILA_VERSION,
        "updated": "",
        "active_backbones": {},
        "files": {},
        "hf": {"repo": "", "revision": "", "synced": ""},
    }


def read_manifest(root: str | Path) -> dict[str, Any]:
    """The manifest of a root (a fresh skeleton when absent or unreadable)."""
    base = _empty_manifest()
    try:
        data = json.loads(_manifest_file(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return base
    if isinstance(data, dict):
        base.update(data)
    return base


def write_manifest(root: str | Path, data: dict[str, Any]) -> Path:
    """Atomic write of ``root/manifest.json``; stamps ``updated`` and fills ``version``."""
    out = dict(data)
    out.setdefault("version", VAILA_VERSION)
    out["updated"] = datetime.now().isoformat(timespec="seconds")
    out.setdefault("active_backbones", {})
    out.setdefault("files", {})
    out.setdefault("hf", {"repo": "", "revision": "", "synced": ""})
    path = _manifest_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(path, json.dumps(out, indent=2, sort_keys=True) + "\n")
    return path


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def update_manifest_file(root: str | Path, path: str | Path, **info: Any) -> dict[str, Any]:
    """Records ``path`` in the manifest: sha256 and bytes are measured, ``info`` (kind,
    n_samples, feat_dim, key, ...) is merged. Returns the entry."""
    base = Path(root)
    target = Path(path)
    manifest = read_manifest(base)
    rel = _relative(base, target)
    entry = dict(manifest["files"].get(rel, {}))
    if target.is_file():
        entry["sha256"] = file_sha256(target)
        entry["bytes"] = target.stat().st_size
    entry.update({k: v for k, v in info.items() if v is not None})
    manifest["files"][rel] = entry
    write_manifest(base, manifest)
    return entry


def active_backbone(root: str | Path, variant: str) -> Path | None:
    """The fine-tuned backbone marked active for ``variant`` in the manifest, if the file exists."""
    rel = read_manifest(root)["active_backbones"].get(variant)
    if not rel:
        return None
    candidate = Path(rel)
    if not candidate.is_absolute():
        candidate = Path(root) / rel
    return candidate if candidate.is_file() else None


def set_active_backbone(root: str | Path, variant: str, path: str | Path) -> None:
    """Marks ``path`` as the active backbone of ``variant`` (relative to root when inside)."""
    base = Path(root)
    manifest = read_manifest(base)
    manifest["active_backbones"][variant] = _relative(base, Path(path))
    write_manifest(base, manifest)
    if Path(path).is_file():
        update_manifest_file(base, path, kind="backbone", variant=variant)


# ---------------------------------------------------------------- legacy migration


def _npz_feat_dim(path: Path) -> int | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            if "feat_dim" in data.files:
                return int(data["feat_dim"])
            return int(np.asarray(data["w"]).shape[0])
    except Exception:
        return None


def _npz_n_samples(path: Path) -> float | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            return float(data["n_samples"])
    except Exception:
        return None


def _legacy_discriminator_target(src: Path, dim: int, resnet_key: str) -> str:
    profile = src.stem.removeprefix("discriminator_")
    if dim == _CLASSIC_DIM:
        key = CLASSIC_KEY
    elif dim == _CLASSIC_DIM + _RESNET50_DEEP_DIM:
        key = resnet_key
    else:
        return f"discriminator_{_safe(profile, 'default')}__dim{dim}.npz"
    return discriminator_path(".", _safe(profile, "default"), key).name


def migrate_legacy(root: str | Path) -> list[Path]:
    """COPIES (never moves or deletes) the legacy flat files into ``root``'s sub-folders.

    * ``LEGACY_DIR/discriminator_<p>.npz``: 808 values -> ``classic``; 808 + 2048 values ->
      ``resnet50-<sha8 of LEGACY_DIR/resnet50_imagenet.pth>`` (``resnet50-imagenet`` if that
      file is missing); other widths -> ``discriminator_<p>__dim<N>.npz``;
    * ``*_finetuned_*.pth`` -> ``backbones/``;
    * ``*_imagenet.pth`` is never copied (official, re-downloadable).

    Files already present (same sha256) are skipped; a different file already at the
    destination is left alone (it holds newer work). Works with ``root == LEGACY_DIR``
    (flat files into its own sub-folders). Returns the paths created.
    """
    base = Path(root)
    legacy = Path(LEGACY_DIR)
    dirs = layout(base)
    copied: list[Path] = []
    if not legacy.is_dir():
        return copied

    official = legacy / "resnet50_imagenet.pth"
    try:
        resnet_key = feature_key("resnet50", official) if official.is_file() else ""
    except OSError:
        resnet_key = ""
    if not resnet_key or resnet_key.endswith("-unknown"):
        resnet_key = "resnet50-imagenet"

    def _copy(src: Path, dst: Path, **info: Any) -> None:
        if dst.is_file():
            return  # same sha: nothing to do; different: keep the newer local work
        atomic_copy(src, dst)
        copied.append(dst)
        update_manifest_file(base, dst, **info)

    for src in sorted(legacy.glob("discriminator_*.npz")):
        dim = _npz_feat_dim(src)
        if dim is None:
            continue
        name = _legacy_discriminator_target(src, dim, resnet_key)
        key = name.split("__", 1)[1].removesuffix(".npz")
        _copy(
            src,
            dirs["discriminators"] / name,
            kind="discriminator",
            feat_dim=dim,
            key=key,
            n_samples=_npz_n_samples(src),
        )
    for src in sorted(legacy.glob("*_finetuned_*.pth")):
        _copy(src, dirs["backbones"] / src.name, kind="backbone")
    return copied
