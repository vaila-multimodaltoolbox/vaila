"""
Share and manage the AI Track network: status, Hugging Face push/pull, local import, restore.

Project: vailá Multimodal Toolbox
Script: ai_tracker_hub.py - AI Track models folder: status, Hugging Face sync, dialog and CLI

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 24 November 2026
Update Date: 24 November 2026
Version: 0.4.19

Description:
    The AI Track network lives in a folder chosen by the user (see ``ai_tracker_store``).
    This module is the one place that moves it between computers:

    * :func:`status` / :func:`format_status`: what the folder holds (EN or PT text);
    * :func:`push`: sends the discriminators and the active fine-tuned backbones to a public
      Hugging Face model repository (update in place; the HF commit history keeps old
      versions). It never sends ``samples/`` (crops of people from the user's videos),
      ``history/``, ``logs/`` or the official ``*_imagenet.pth`` weights, and it always asks
      for confirmation first;
    * :func:`pull` / :func:`import_folder`: install a downloaded repository (or a local
      folder, for example a USB stick) after checking the sha256 of every file; a file that
      is replaced goes to ``history/`` first;
    * :func:`restore_latest`: puts the newest ``history/`` version of a file back;
    * :func:`start_retrain`: launches ``python -m vaila.ai_tracker_train --from-samples`` in the
      background;
    * :func:`ai_models_dialog`: the Tkinter window behind the *Net* button of getpixelvideo.

    Command line (every GUI action prints its ``>>`` equivalent)::

        uv run --no-sync python -m vaila.ai_tracker_hub [--ai-models DIR] status
        uv run --no-sync python -m vaila.ai_tracker_hub set-home DIR
        uv run --no-sync python -m vaila.ai_tracker_hub pull [--repo R] [--revision X]
        uv run --no-sync python -m vaila.ai_tracker_hub push [--repo R] [--yes] [--dry-run]
        uv run --no-sync python -m vaila.ai_tracker_hub import SRC
        uv run --no-sync python -m vaila.ai_tracker_hub restore FILE
        uv run --no-sync python -m vaila.ai_tracker_hub retrain
        uv run --no-sync python -m vaila.ai_tracker_hub gui

    Logging in for ``push``: ``uv run --no-sync hf auth login`` (or the ``HF_TOKEN`` variable).
    ``pull`` needs no login (the repository is public).

License: AGPLv3
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from . import ai_tracker_store as _store
    from .ai_tracker_store import (
        SUBDIR,
        SUGGESTED_HOME,
        atomic_copy,
        file_sha256,
        history_versions,
        layout,
        read_manifest,
        resolve_home,
        restore,
        set_home,
        set_session_home,
        update_manifest_file,
        write_manifest,
    )
except ImportError:
    import ai_tracker_store as _store  # type: ignore[no-redef]  # ty: ignore[unresolved-import]
    from ai_tracker_store import (  # type: ignore[no-redef]  # ty: ignore[unresolved-import]
        SUBDIR,
        SUGGESTED_HOME,
        atomic_copy,
        file_sha256,
        history_versions,
        layout,
        read_manifest,
        resolve_home,
        restore,
        set_home,
        set_session_home,
        update_manifest_file,
        write_manifest,
    )

DEFAULT_REPO = os.environ.get("VAILA_AI_MODELS_REPO") or "paulosantiago/vaila-ai-tracker"
UPLOAD_PATTERNS = ["manifest.json", "README.md", "discriminators/*.npz", "backbones/*.pth"]
_DELETE_PATTERNS = ["discriminators/*", "backbones/*"]
_ENTRY_FIELDS = ("sha256", "bytes", "kind", "variant", "key", "feat_dim", "n_samples")
_ALLOWED_REL = re.compile(r"^(discriminators/[^/]+\.npz|backbones/[^/]+\.pth)$")
_LOGIN_HELP = "uv run --no-sync hf auth login"
_REPO_ROOT = Path(__file__).resolve().parent.parent


class HubLoginError(RuntimeError):
    """No Hugging Face login (or the token was refused)."""


class HubImportError(RuntimeError):
    """The folder is not an AI Track repository or a file failed its sha256 check."""


# ---------------------------------------------------------------- helpers


def _hf() -> tuple[Any, Any]:
    """Lazy import of ``(HfApi, snapshot_download)``; tests monkeypatch this one function."""
    from huggingface_hub import HfApi, snapshot_download

    return HfApi, snapshot_download


def _stored_token() -> str | None:
    """Token saved by ``hf auth login`` (None if there is none); tests may patch this."""
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def _token(token: str | None = None) -> str | None:
    if token:
        return token
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(var, "").strip()
        if value:
            return value
    return _stored_token()


def has_login(token: str | None = None) -> bool:
    """True when a Hugging Face token is available (argument, HF_TOKEN or ``hf auth login``)."""
    return bool(_token(token))


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _root(root: str | Path | None) -> Path:
    return Path(root) if root else resolve_home().root


def _mirror(*parts: str, module: str = "ai_tracker_hub") -> str:
    """Prints (and returns) the CLI line equivalent to a GUI action."""
    cmd = "uv run --no-sync python -m vaila." + module
    line = f">> vaila/{module}: {cmd} " + " ".join(parts)
    line = line.rstrip()
    print(line, flush=True)
    return line


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _npz_info(path: Path) -> dict[str, Any]:
    """feat_dim and n_samples of a discriminator from the npz header (no pickle)."""
    out: dict[str, Any] = {}
    try:
        with np.load(path, allow_pickle=False) as data:
            if "feat_dim" in data.files:
                out["feat_dim"] = int(data["feat_dim"])
            elif "w" in data.files:
                out["feat_dim"] = int(np.asarray(data["w"]).shape[0])
            if "n_samples" in data.files:
                out["n_samples"] = float(data["n_samples"])
    except Exception:
        pass
    return out


def _key_of(path: Path) -> str:
    stem = path.stem
    return stem.split("__", 1)[1] if "__" in stem else "classic"


def _is_official(path: Path | str) -> bool:
    return str(path).endswith("_imagenet.pth")


def _backbone_path(root: Path, rel: str) -> Path:
    cand = Path(rel)
    return cand if cand.is_absolute() else root / rel


def _official_sha(variant: str) -> str:
    """sha256 of the official ImageNet weights if the file is on this computer ("" otherwise)."""
    for folder in (Path(_store.LEGACY_DIR),):
        cand = folder / f"{variant}_imagenet.pth"
        if cand.is_file():
            try:
                return file_sha256(cand)
            except OSError:
                return ""
    return ""


# ---------------------------------------------------------------- status


def status(root: str | Path | None = None) -> dict[str, Any]:
    """Describes the AI Track folder (no side effects besides creating the layout)."""
    if root is None:
        paths = resolve_home()
        base, source, backed_up = paths.root, paths.source, paths.backed_up
    else:
        base, source, backed_up = Path(root), "path", True
    layout(base)
    manifest = read_manifest(base)
    entries = manifest.get("files", {})
    discs = []
    for path in sorted((base / "discriminators").glob("*.npz")):
        entry = entries.get(_rel(base, path), {})
        info = _npz_info(path)
        discs.append(
            {
                "file": path.name,
                "key": entry.get("key") or _key_of(path),
                "feat_dim": entry.get("feat_dim", info.get("feat_dim")),
                "n_samples": entry.get("n_samples", info.get("n_samples")),
                "date": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="minutes"),
            }
        )
    backbones = [p.name for p in sorted((base / "backbones").glob("*.pth"))]
    n_files = n_samples = n_markers = 0
    for path in sorted((base / "samples").glob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as data:
                labels = np.asarray(data["labels"])
        except Exception:
            continue
        n_files += 1
        n_samples += int(labels.size)
        n_markers += int((labels > 0).sum())
    return {
        "root": str(base),
        "source": source,
        "backed_up": bool(backed_up),
        "warning": None if backed_up else "no_backup",
        "active_backbones": dict(manifest.get("active_backbones", {})),
        "backbones": backbones,
        "discriminators": discs,
        "n_sample_files": n_files,
        "n_samples": n_samples,
        "n_markers": n_markers,
        "last_retrain": manifest.get("last_retrain") or None,
        "hf": dict(manifest.get("hf", {})),
    }


_TEXT: dict[str, dict[str, Any]] = {
    "en": {
        "folder": "Folder",
        "source": "Chosen by",
        "no_backup": "WARNING: this is the folder inside the vaila install; deleting or "
        "reinstalling vaila loses it. Use 'Change folder...' to move it out.",
        "backbones": "Active backbones",
        "none": "(none)",
        "discs": "Discriminators",
        "samples": "Stored samples",
        "samples_fmt": "{files} file(s), {n} crops, {m} manual markers",
        "last": "Last retrain",
        "never": "never",
        "hf": "Hugging Face",
        "hf_none": "never synced",
        "samples_n": "samples",
        "src": {
            "cli": "--ai-models",
            "env": "VAILA_AI_MODELS_HOME",
            "config": "saved setting (~/.vaila/ai_models.json)",
            "legacy": "default folder inside vaila",
            "path": "given folder",
        },
    },
    "pt": {
        "folder": "Pasta",
        "source": "Escolhida por",
        "no_backup": "ATENCAO: esta e a pasta dentro da instalacao do vaila; apagar ou "
        "reinstalar o vaila a perde. Use 'Trocar pasta...' para leva-la para fora.",
        "backbones": "Backbones ativos",
        "none": "(nenhum)",
        "discs": "Discriminadores",
        "samples": "Amostras guardadas",
        "samples_fmt": "{files} arquivo(s), {n} recortes, {m} marcadores manuais",
        "last": "Ultimo retreino",
        "never": "nunca",
        "hf": "Hugging Face",
        "hf_none": "nunca sincronizado",
        "samples_n": "amostras",
        "src": {
            "cli": "--ai-models",
            "env": "VAILA_AI_MODELS_HOME",
            "config": "configuracao salva (~/.vaila/ai_models.json)",
            "legacy": "pasta padrao dentro do vaila",
            "path": "pasta informada",
        },
    },
}


def format_status(info: dict[str, Any], lang: str = "en") -> str:
    """Human-readable text of :func:`status` (``lang`` = "en" or "pt")."""
    t = _TEXT["pt" if str(lang).lower().startswith("pt") else "en"]
    lines = [
        f"{t['folder']}: {info['root']}",
        f"{t['source']}: {t['src'].get(info['source'], info['source'])}",
    ]
    if info.get("warning") == "no_backup":
        lines.append(t["no_backup"])
    active = info.get("active_backbones") or {}
    lines.append(
        f"{t['backbones']}: " + (", ".join(f"{k} -> {v}" for k, v in active.items()) or t["none"])
    )
    lines.append(f"{t['discs']}:")
    discs = info.get("discriminators") or []
    if not discs:
        lines.append(f"  {t['none']}")
    for d in discs:
        n = d.get("n_samples")
        n_txt = f"{int(n)} {t['samples_n']}" if n is not None else "? " + t["samples_n"]
        lines.append(
            f"  {d['file']}  key={d['key']}  dim={d.get('feat_dim')}  {n_txt}  {d['date']}"
        )
    lines.append(
        f"{t['samples']}: "
        + t["samples_fmt"].format(
            files=info.get("n_sample_files", 0),
            n=info.get("n_samples", 0),
            m=info.get("n_markers", 0),
        )
    )
    last = info.get("last_retrain")
    if last:
        before, after = last.get("before"), last.get("after")
        score = ""
        if before is not None or after is not None:
            score = f"  {last.get('metric', 'metric')}: {before} -> {after}"
        lines.append(
            f"{t['last']}: {last.get('status')}  {last.get('finished') or last.get('started')}"
            f"  ({last.get('mode')}, {last.get('device')}, {last.get('n_samples')} "
            f"{t['samples_n']}){score}"
        )
        if last.get("reason"):
            lines.append(f"  {last['reason']}")
    else:
        lines.append(f"{t['last']}: {t['never']}")
    hf = info.get("hf") or {}
    if hf.get("repo") and hf.get("synced"):
        rev = f"  rev {str(hf['revision'])[:10]}" if hf.get("revision") else ""
        lines.append(f"{t['hf']}: {hf['repo']}{rev}  {hf['synced']}")
    else:
        lines.append(f"{t['hf']}: {t['hf_none']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- what is sent to HF


def _plan(root: Path) -> dict[str, tuple[Path, str, str]]:
    """``{upload rel path: (source file, kind, variant)}``; never samples/history/logs/imagenet."""
    plan: dict[str, tuple[Path, str, str]] = {}
    for path in sorted((root / "discriminators").glob("*.npz")):
        plan[f"discriminators/{path.name}"] = (path, "discriminator", "")
    manifest = read_manifest(root)
    for variant, rel in sorted(manifest.get("active_backbones", {}).items()):
        cand = _backbone_path(root, str(rel))
        if cand.is_file() and cand.suffix == ".pth" and not _is_official(cand):
            plan[f"backbones/{cand.name}"] = (cand, "backbone", str(variant))
    return plan


def upload_plan(root: str | Path | None = None) -> list[str]:
    """Names of the files :func:`push` would send (nothing is copied)."""
    return sorted([*_plan(_root(root)), "README.md", "manifest.json"])


def _card(entries: dict[str, dict[str, Any]], official: dict[str, dict[str, str]]) -> str:
    backbone_by_sha8 = {
        str(e.get("sha256", ""))[:8]: rel for rel, e in entries.items() if e["kind"] == "backbone"
    }
    rows = []
    for rel, e in sorted(entries.items()):
        if e["kind"] != "discriminator":
            continue
        key = str(e.get("key", "classic"))
        if key == "classic":
            needs = "none (classic features only)"
        else:
            variant, _, sha8 = key.rpartition("-")
            if sha8 in backbone_by_sha8:
                needs = f"fine-tuned backbone included: `{backbone_by_sha8[sha8]}`"
            else:
                full = official.get(variant, {}).get("sha256", "")
                needs = (
                    f"official ImageNet weights of `{variant}` "
                    f"(sha256 {full or sha8 + '...'}), downloaded by vaila on first Deep use"
                )
        n = e.get("n_samples")
        rows.append(
            f"| `{rel}` | `{key}` | {e.get('feat_dim', '?')} | "
            f"{int(n) if n is not None else '?'} | {needs} |"
        )
    table = "\n".join(
        [
            "| file | feature key | values | training samples | backbone needed |",
            "| --- | --- | --- | --- | --- |",
            *rows,
        ]
    )
    return f"""---
license: agpl-3.0
library_name: vaila
tags:
  - vaila
  - tracking
  - biomechanics
  - markerless
---

# vaila AI Track network

Small online discriminators (and, when present, fine-tuned backbones) used by the **AI Track**
tool of the video marker editor (`getpixelvideo`) of *vaila*, the open-source multimodal
toolbox for biomechanical analysis. AI Track follows a point that a person marked by hand: the
discriminator scores image patches around the previous position as "the marker" or "background".

## Files

{table}

* `manifest.json` lists every file with its sha256 and size. vaila checks them on download.
* The feature key says which feature space a discriminator was trained in. `classic` is 808
  hand-crafted colour and region values. `<variant>-<sha8>` is the same 808 values plus the
  deep features of the backbone `<variant>` whose weights file has a sha256 starting with
  `<sha8>`. A discriminator is only valid with that exact backbone.
* The official ImageNet weights (`*_imagenet.pth`) are not repeated here: they come from
  torchvision and vaila fetches them on first use. Fine-tuned backbones are included.

## How to use

In *vaila*: open `getpixelvideo`, press the **Net** button, then **Download from Hugging Face**
(or **Use a local folder...** for a downloaded copy). From a terminal:

```bash
uv run --no-sync python -m vaila.ai_tracker_hub pull
```

Existing local files are kept in a `history/` folder before they are replaced.

## Training data

The networks were trained on the markers placed by hand by the repository owner in their own
videos. The image crops used for training are **not published** (they contain people).

## Limitations

* Trained on few videos of one person's work: expect it to help on similar footage and to be
  less reliable elsewhere. It does not replace checking the tracked points.
* A discriminator never ships with the backbone it was not trained with; use the matching one.
* Licensed AGPL-3.0, like vaila.
"""


def build_upload_folder(root: str | Path, staging: str | Path) -> list[str]:
    """Copies only the allowed files into ``staging`` and writes a filtered ``manifest.json``
    plus the model card ``README.md``. Returns the relative paths, sorted."""
    base = Path(root)
    out = Path(staging)
    out.mkdir(parents=True, exist_ok=True)
    local = read_manifest(base)
    entries: dict[str, dict[str, Any]] = {}
    active: dict[str, str] = {}
    for rel, (src, kind, variant) in _plan(base).items():
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        known = local.get("files", {}).get(_rel(base, src), {})
        entry: dict[str, Any] = {k: known[k] for k in _ENTRY_FIELDS if k in known}
        entry["kind"] = kind
        entry["sha256"] = file_sha256(dst)
        entry["bytes"] = dst.stat().st_size
        if kind == "discriminator":
            entry.setdefault("key", _key_of(src))
            for k, v in _npz_info(src).items():
                entry.setdefault(k, v)
        else:
            entry["variant"] = variant
            active[variant] = rel
        entries[rel] = entry
    official = {}
    for e in entries.values():
        if e["kind"] == "discriminator" and e.get("key", "classic") != "classic":
            variant = str(e["key"]).rpartition("-")[0]
            sha = _official_sha(variant)
            if sha:
                official[variant] = {"file": f"{variant}_imagenet.pth", "sha256": sha}
    manifest = {
        "version": local.get("version"),
        "active_backbones": active,
        "files": entries,
        "official_backbones": official,
    }
    write_manifest(out, manifest)
    (out / "README.md").write_text(_card(entries, official), encoding="utf-8")
    return sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())


def push(
    root: str | Path | None = None,
    repo_id: str = DEFAULT_REPO,
    *,
    token: str | None = None,
    confirm: Callable[[str, list[str]], bool] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Sends the network to a public Hugging Face model repository, replacing it in place.

    ``confirm(repo_id, files)`` must return True or nothing is sent (``dry_run`` only builds
    the folder). Returns ``{"repo", "files", "revision", "dry_run"}``, plus ``"aborted": True``
    when the user declined. Raises :class:`HubLoginError` without a login."""
    base = _root(root)
    with tempfile.TemporaryDirectory(prefix="vaila_ai_hub_") as tmp:
        files = build_upload_folder(base, tmp)
        result: dict[str, Any] = {
            "repo": repo_id,
            "files": files,
            "revision": "",
            "dry_run": bool(dry_run),
        }
        if dry_run:
            return result
        tok = _token(token)
        if not tok:
            raise HubLoginError(
                f"No Hugging Face login. Run `{_LOGIN_HELP}` (or set HF_TOKEN) and try again."
            )
        if confirm is None or not confirm(repo_id, files):
            result["aborted"] = True
            return result
        api_cls, _ = _hf()
        api = api_cls(token=tok)
        try:
            api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
            info = api.upload_folder(
                repo_id=repo_id,
                folder_path=tmp,
                repo_type="model",
                commit_message=f"vaila AI Track update {_now()}",
                delete_patterns=list(_DELETE_PATTERNS),
            )
        except Exception as err:
            code = getattr(getattr(err, "response", None), "status_code", None)
            if code in (401, 403):
                raise HubLoginError(
                    f"Hugging Face refused the token ({code}). Run `{_LOGIN_HELP}` again."
                ) from err
            raise
        revision = str(getattr(info, "oid", "") or "")
        result["revision"] = revision
    manifest = read_manifest(base)
    manifest["hf"] = {"repo": repo_id, "revision": revision, "synced": _now()}
    write_manifest(base, manifest)
    return result


# ---------------------------------------------------------------- install (pull / import)


def _source_base(src: Path) -> Path:
    for cand in (src, src / SUBDIR):
        if (cand / "manifest.json").is_file():
            return cand
    raise HubImportError(f"{src} has no manifest.json: not an AI Track folder or repository.")


def _needs_official(manifest: dict[str, Any], root: Path, installed: list[str]) -> list[str]:
    out = []
    for rel in installed:
        if not rel.startswith("discriminators/"):
            continue
        key = str(manifest["files"].get(rel, {}).get("key") or _key_of(Path(rel)))
        if key == "classic":
            continue
        variant, _, sha8 = key.rpartition("-")
        present = any(
            e.get("kind") == "backbone" and str(e.get("sha256", "")).startswith(sha8)
            for e in manifest["files"].values()
            if sha8
        )
        official = _official_sha(variant)
        if present or (official and official.startswith(sha8)):
            continue
        out.append(
            f"{Path(rel).name} needs the official ImageNet weights of {variant} "
            "(not on this computer); vaila fetches them on the first Deep use."
        )
    return out


def import_folder(src: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    """Installs a downloaded repository, a home folder or its ``ai_tracker`` subfolder.

    Every file listed in the source manifest is checked against its sha256 first; one mismatch
    aborts with :class:`HubImportError` and nothing is installed. Returns
    ``{"installed": [...], "skipped": [...], "warnings": [...]}`` (relative paths)."""
    base = _root(root)
    layout(base)
    folder = _source_base(Path(src))
    try:
        remote = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise HubImportError(f"unreadable manifest.json in {folder}: {err}") from err
    if not isinstance(remote, dict) or not isinstance(remote.get("files", {}), dict):
        raise HubImportError(f"manifest.json in {folder} is not an AI Track manifest.")
    warnings: list[str] = []
    todo: list[tuple[str, Path, dict[str, Any]]] = []
    skipped: list[str] = []
    for rel, entry in sorted(remote.get("files", {}).items()):
        if not _ALLOWED_REL.match(rel) or ".." in Path(rel).parts:
            skipped.append(rel)
            warnings.append(f"{rel}: not a discriminator or backbone, ignored.")
            continue
        path = folder / rel
        if not path.is_file():
            skipped.append(rel)
            warnings.append(f"{rel}: listed in the manifest but missing, ignored.")
            continue
        if file_sha256(path) != str(entry.get("sha256", "")):
            raise HubImportError(f"{rel}: sha256 does not match the manifest; nothing installed.")
        todo.append((rel, path, entry))
    installed: list[str] = []
    manifest = read_manifest(base)
    for rel, path, entry in todo:
        dst = base / rel
        if dst.is_file() and file_sha256(dst) == file_sha256(path):
            skipped.append(rel)
        else:
            atomic_copy(path, dst)
            installed.append(rel)
        merged = {k: entry[k] for k in _ENTRY_FIELDS if k in entry}
        merged["sha256"] = file_sha256(dst)
        merged["bytes"] = dst.stat().st_size
        manifest["files"][rel] = merged
    for variant, rel in remote.get("active_backbones", {}).items():
        if isinstance(rel, str) and _ALLOWED_REL.match(rel) and (base / rel).is_file():
            manifest["active_backbones"][variant] = rel
    write_manifest(base, manifest)
    warnings.extend(_needs_official(manifest, base, [r for r, _, _ in todo]))
    return {"installed": installed, "skipped": skipped, "warnings": warnings}


def pull(
    root: str | Path | None = None,
    repo_id: str = DEFAULT_REPO,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Downloads the repository (public, no login) and installs it through :func:`import_folder`."""
    base = _root(root)
    _, snapshot_download = _hf()
    with tempfile.TemporaryDirectory(prefix="vaila_ai_pull_") as tmp:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            repo_type="model",
            local_dir=tmp,
            allow_patterns=list(UPLOAD_PATTERNS),
            token=token,
        )
        result = import_folder(tmp, base)
    manifest = read_manifest(base)
    manifest["hf"] = {"repo": repo_id, "revision": revision or "", "synced": _now()}
    write_manifest(base, manifest)
    result["repo"] = repo_id
    return result


# ---------------------------------------------------------------- restore and retrain


def _root_of(path: Path) -> Path | None:
    legacy = Path(_store.LEGACY_DIR)
    for parent in path.resolve().parents:
        if parent.name == SUBDIR or parent == legacy.resolve():
            return parent
    return None


def restore_latest(path: str | Path) -> Path | None:
    """Puts the newest ``history/`` version of ``path`` back (the current one goes to history
    first, so a second call toggles). Returns the version used, None when there is none."""
    target = Path(path)
    versions = history_versions(target)
    if not versions:
        return None
    restore(target, versions[0])
    base = _root_of(target)
    if base is not None:
        update_manifest_file(base, target)
    return versions[0]


def start_retrain(
    root: str | Path,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
) -> tuple[Any, Path]:
    """Starts the retraining in the background; stdout/stderr go to ``root/logs/retrain_<ts>.out``.

    Returns ``(process, log_path)`` immediately. The job writes ``manifest["last_retrain"]``."""
    base = Path(root)
    log = layout(base)["logs"] / f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.out"
    argv = [
        sys.executable,
        "-m",
        "vaila.ai_tracker_train",
        "--from-samples",
        str(base),
        "--mode",
        "both",
        "--device",
        "auto",
    ]
    _mirror(
        "--from-samples",
        shlex.quote(str(base)),
        "--mode both --device auto",
        module="ai_tracker_train",
    )
    with log.open("wb") as fh:
        proc = popen(argv, stdout=fh, stderr=subprocess.STDOUT, cwd=str(_REPO_ROOT))
    return proc, log


def gpu_memory_text() -> str:
    """Free GPU memory as text, without initialising CUDA in this process."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return f"GPU memory free: {free / 2**30:.1f} of {total / 2**30:.1f} GiB"
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        ).stdout.strip()
        free_mb, total_mb = (float(v) for v in out.splitlines()[0].split(","))
        return f"GPU memory free: {free_mb / 1024:.1f} of {total_mb / 1024:.1f} GiB"
    except Exception:
        return "GPU memory: unknown (no NVIDIA GPU found; the retrain will use the CPU)"


def open_folder(path: str | Path) -> None:
    """Opens a folder in the system file manager."""
    p = str(path)
    if sys.platform.startswith("win"):
        print(f">> vaila/ai_tracker_hub: explorer {p}", flush=True)
        os.startfile(p)  # type: ignore[attr-defined]
    else:
        cmd = ["open", p] if sys.platform == "darwin" else ["xdg-open", p]
        print(f">> vaila/ai_tracker_hub: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
        subprocess.Popen(cmd)


# ---------------------------------------------------------------- Tkinter dialog


class _ModelsDialog:
    """The window behind the Net button. ``self.win`` is a Toplevel (or its own Tk)."""

    def __init__(self, parent: Any = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.own_root = parent is None
        self.win = tk.Tk() if parent is None else tk.Toplevel(parent)
        self.win.title("AI Track models")
        self.win.minsize(640, 460)
        self._proc: Any = None
        self._log: Path | None = None
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._busy = False
        self.repo_var = tk.StringVar(value=DEFAULT_REPO)

        frame = ttk.Frame(self.win, padding=8)
        frame.pack(fill="both", expand=True)
        self.text = tk.Text(frame, width=96, height=18, wrap="word", font=("TkFixedFont", 9))
        self.text.pack(fill="both", expand=True)
        self.text.configure(state="disabled")

        repo_row = ttk.Frame(frame)
        repo_row.pack(fill="x", pady=(6, 0))
        ttk.Label(repo_row, text="Hugging Face repo:").pack(side="left")
        ttk.Entry(repo_row, textvariable=self.repo_var).pack(side="left", fill="x", expand=True)

        self.buttons: dict[str, Any] = {}
        grid = ttk.Frame(frame)
        grid.pack(fill="x", pady=(6, 0))
        specs = [
            ("Change folder...", self.change_folder),
            ("Open folder", self.open_folder),
            ("Download from Hugging Face", self.download),
            ("Use a local folder...", self.use_local),
            ("Retrain (GPU)", self.retrain),
            ("Send to Hugging Face...", self.send),
            ("Restore previous version...", self.restore_version),
            ("Refresh", self.refresh),
        ]
        for i, (label, cmd) in enumerate(specs):
            btn = ttk.Button(grid, text=label, command=cmd)
            btn.grid(row=i // 4, column=i % 4, sticky="ew", padx=2, pady=2)
            self.buttons[label] = btn
        for c in range(4):
            grid.columnconfigure(c, weight=1)
        self.status_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.status_var, anchor="w").pack(fill="x", pady=(6, 0))
        self.refresh()

    # -- helpers
    def root(self) -> Path:
        return resolve_home().root

    def _set_text(self, content: str) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", content)
        self.text.configure(state="disabled")

    def refresh(self) -> None:
        try:
            self._set_text(format_status(status(), "en"))
        except ValueError as err:
            self._set_text(f"Cannot open the AI Track folder:\n{err}")

    def _say(self, message: str) -> None:
        self.status_var.set(message)
        print(f"vaila/ai_tracker_hub: {message}", flush=True)

    def _alive(self) -> bool:
        try:
            return bool(self.win.winfo_exists())
        except self.tk.TclError:
            return False

    def _run_worker(self, label: str, work: Callable[[], Any]) -> None:
        """Runs ``work`` in a thread; the result comes back to the Tk thread through after()."""
        if self._busy:
            return
        self._busy = True
        self._say(f"{label}...")

        def target() -> None:
            try:
                self._queue.put(("ok", work()))
            except Exception as err:
                self._queue.put(("error", err))

        threading.Thread(target=target, daemon=True).start()
        self._poll_worker(label)

    def _poll_worker(self, label: str) -> None:
        if not self._alive():
            return
        try:
            kind, payload = self._queue.get_nowait()
        except queue.Empty:
            self.win.after(200, lambda: self._poll_worker(label))
            return
        from tkinter import messagebox

        self._busy = False
        self.refresh()
        if kind == "error":
            self._say(f"{label} failed: {payload}")
            messagebox.showerror("AI Track models", f"{label} failed:\n{payload}", parent=self.win)
            return
        self._say(f"{label} finished.")
        msg = self._summarise(payload)
        if msg:
            messagebox.showinfo("AI Track models", msg, parent=self.win)

    @staticmethod
    def _summarise(result: Any) -> str:
        if not isinstance(result, dict):
            return ""
        if "installed" in result:
            parts = [
                f"Installed {len(result['installed'])}, unchanged {len(result['skipped'])}.",
                *result.get("warnings", []),
            ]
            return "\n".join(parts)
        if "files" in result:
            return f"Sent {len(result['files'])} file(s) to {result['repo']}."
        return ""

    # -- actions
    def change_folder(self) -> None:
        from tkinter import filedialog, messagebox

        chosen = filedialog.askdirectory(
            parent=self.win, initialdir=str(SUGGESTED_HOME.parent), title="AI Track models folder"
        )
        if not chosen:
            return
        _mirror("set-home", shlex.quote(chosen))
        try:
            set_home(chosen)
        except ValueError as err:
            messagebox.showerror("AI Track models", str(err), parent=self.win)
            return
        self.refresh()

    def open_folder(self) -> None:
        try:
            open_folder(self.root())
        except Exception as err:
            self._say(f"cannot open the folder: {err}")

    def download(self) -> None:
        repo = self.repo_var.get().strip() or DEFAULT_REPO
        base = self.root()
        _mirror("pull", "--repo", shlex.quote(repo))
        self._run_worker("Download from Hugging Face", lambda: pull(base, repo))

    def use_local(self) -> None:
        from tkinter import filedialog

        chosen = filedialog.askdirectory(parent=self.win, title="Folder with an AI Track network")
        if not chosen:
            return
        base = self.root()
        _mirror("import", shlex.quote(chosen))
        self._run_worker("Import", lambda: import_folder(chosen, base))

    def send(self) -> None:
        from tkinter import messagebox

        repo = self.repo_var.get().strip() or DEFAULT_REPO
        if not has_login():
            messagebox.showinfo(
                "AI Track models",
                f"Log in to Hugging Face first, in a terminal:\n\n  {_LOGIN_HELP}\n\n"
                "or set the HF_TOKEN variable, then press the button again.",
                parent=self.win,
            )
            return
        base = self.root()
        files = upload_plan(base)
        ok = messagebox.askyesno(
            "Send to Hugging Face",
            f"Replace the public repository\n  {repo}\nwith these files?\n\n  "
            + "\n  ".join(files)
            + "\n\nSamples (crops of people), history and logs are never sent.",
            parent=self.win,
        )
        if not ok:
            return
        _mirror("push", "--repo", shlex.quote(repo), "--yes")
        self._run_worker("Send to Hugging Face", lambda: push(base, repo, confirm=lambda *_: True))

    def _pick(self, title: str, items: list[str]) -> str | None:
        tk = self.tk
        top = tk.Toplevel(self.win)
        top.title(title)
        box = tk.Listbox(top, width=70, height=min(12, max(3, len(items))), exportselection=False)
        for it in items:
            box.insert("end", it)
        box.pack(padx=8, pady=8)
        chosen: list[str] = []

        def ok() -> None:
            sel = box.curselection()
            if sel:
                chosen.append(items[sel[0]])
            top.destroy()

        tk.Button(top, text="OK", command=ok).pack(pady=(0, 8))
        top.transient(self.win)
        top.grab_set()
        self.win.wait_window(top)
        return chosen[0] if chosen else None

    def restore_version(self) -> None:
        from tkinter import messagebox

        base = self.root()
        files = [
            p
            for p in [
                *sorted((base / "discriminators").glob("*.npz")),
                *(base / "backbones").glob("*.pth"),
            ]
            if history_versions(p)
        ]
        if not files:
            messagebox.showinfo(
                "AI Track models", "No file has a previous version yet.", parent=self.win
            )
            return
        names = [p.relative_to(base).as_posix() for p in files]
        picked = self._pick("Restore previous version", names)
        if not picked:
            return
        if not messagebox.askyesno(
            "AI Track models",
            f"Put the newest previous version of\n  {picked}\nback? The current one is kept in history.",
            parent=self.win,
        ):
            return
        _mirror("restore", shlex.quote(str(base / picked)))
        used = restore_latest(base / picked)
        self._say(f"restored {picked} from {used.name if used else '?'}")
        self.refresh()

    def retrain(self) -> None:
        from tkinter import messagebox

        if self._proc is not None and self._proc.poll() is None:
            messagebox.showinfo(
                "AI Track models", "A retraining is already running.", parent=self.win
            )
            return
        base = self.root()
        ok = messagebox.askyesno(
            "Retrain (GPU)",
            f"{gpu_memory_text()}\n\nRetrain the backbone and the discriminator from the stored "
            "samples now? It can take minutes and uses the GPU (close SAM or other heavy jobs "
            "first). The new network only replaces the current one if it does not do worse "
            "on held-out videos.",
            parent=self.win,
        )
        if not ok:
            return
        try:
            self._proc, self._log = start_retrain(base)
        except Exception as err:
            messagebox.showerror(
                "AI Track models", f"Cannot start the retraining:\n{err}", parent=self.win
            )
            return
        self._say(f"retraining started, log: {self._log}")
        self.win.after(1500, self._poll_retrain)

    def _poll_retrain(self) -> None:
        if not self._alive() or self._proc is None:
            return
        code = self._proc.poll()
        if code is None:
            self.win.after(1500, self._poll_retrain)
            return
        from tkinter import messagebox

        self.refresh()
        last = status(self.root()).get("last_retrain") or {}
        outcome = last.get("status") or ("error" if code else "finished")
        self._say(f"retraining ended: {outcome} (exit {code})")
        detail = f"\n{last['reason']}" if last.get("reason") else ""
        messagebox.showinfo(
            "AI Track models",
            f"Retraining ended: {outcome} (exit code {code}).{detail}\nLog: {self._log}",
            parent=self.win,
        )
        self._proc = None


def ai_models_dialog(parent: Any = None) -> None:
    """Opens the AI Track models window (Tkinter only).

    With ``parent`` it is a ``Toplevel`` of that window (one Tk root rule) and returns at
    once; without, it creates its own ``Tk`` and runs ``mainloop``."""
    dialog = _ModelsDialog(parent)
    if dialog.own_root:
        dialog.win.mainloop()


# ---------------------------------------------------------------- CLI


def _confirm_terminal(repo_id: str, files: list[str]) -> bool:
    print(f"Replace the public repository {repo_id} with these files?")
    for name in files:
        print(f"  {name}")
    print("Samples, history and logs are never sent.")
    try:
        return input("Send? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m vaila.ai_tracker_hub",
        description="Manage and share the AI Track network (folder, Hugging Face, restore).",
    )
    p.add_argument("--ai-models", metavar="DIR", help="AI Track models home for this run")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="show the folder, networks, samples and sync state")
    sp = sub.add_parser("set-home", help="choose the AI Track models folder")
    sp.add_argument("dir")
    sp = sub.add_parser("pull", help="download from Hugging Face (public, no login)")
    sp.add_argument("--repo", default=DEFAULT_REPO)
    sp.add_argument("--revision")
    sp = sub.add_parser("push", help="send to Hugging Face (needs `hf auth login`)")
    sp.add_argument("--repo", default=DEFAULT_REPO)
    sp.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    sp.add_argument("--dry-run", action="store_true", help="only list what would be sent")
    sp = sub.add_parser("import", help="install a downloaded repository or a local folder")
    sp.add_argument("src")
    sp = sub.add_parser("restore", help="put the newest previous version of a file back")
    sp.add_argument("file")
    sub.add_parser("retrain", help="retrain from the stored samples in the background job")
    sub.add_parser("gui", help="open the models window")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.ai_models:
        set_session_home(args.ai_models)
    try:
        if args.cmd == "status":
            print(format_status(status(), "en"))
        elif args.cmd == "set-home":
            paths = set_home(args.dir)
            print(f"AI Track folder: {paths.root}")
        elif args.cmd == "pull":
            res = pull(None, args.repo, revision=args.revision)
            print(
                f"Pulled {args.repo}: installed {len(res['installed'])}, "
                f"unchanged {len(res['skipped'])}"
            )
            for line in res["warnings"]:
                print(f"  warning: {line}")
        elif args.cmd == "push":
            res = push(
                None,
                args.repo,
                confirm=(lambda *_: True) if args.yes else _confirm_terminal,
                dry_run=args.dry_run,
            )
            if res.get("aborted"):
                print("Nothing was sent.")
                return 1
            verb = "Would send" if res["dry_run"] else "Sent"
            print(f"{verb} {len(res['files'])} file(s) to {res['repo']}:")
            for name in res["files"]:
                print(f"  {name}")
        elif args.cmd == "import":
            res = import_folder(args.src)
            print(f"Installed {len(res['installed'])}, unchanged {len(res['skipped'])}")
            for line in res["warnings"]:
                print(f"  warning: {line}")
        elif args.cmd == "restore":
            used = restore_latest(args.file)
            if used is None:
                print(f"No previous version of {args.file}.")
                return 1
            print(f"Restored {args.file} from {used.name}")
        elif args.cmd == "retrain":
            proc, log = start_retrain(resolve_home().root)
            print(f"Started (pid {getattr(proc, 'pid', '?')}); log: {log}")
        elif args.cmd == "gui":
            ai_models_dialog()
    except (HubLoginError, HubImportError, ValueError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
