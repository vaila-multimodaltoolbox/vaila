"""
Tests of vaila/ai_tracker_hub.py: upload folder, push, pull, import, restore, retrain, CLI.

No network ever: ``_hf()`` is replaced by fakes that record their calls. The AI Track home is
isolated by tests/conftest.py (VAILA_AI_MODELS_HOME).

Author: Paulo Roberto Pereira Santiago
Update Date: 24 November 2026
Version: 0.4.19
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vaila import ai_tracker_hub as hub
from vaila import ai_tracker_store as store


def _make_root(base: Path, *, seed: int = 0) -> Path:
    """A root with a classic and a deep discriminator, a fine-tuned backbone, samples,
    history, logs and an official ImageNet file."""
    root = base / "ai_tracker"
    dirs = store.layout(root)
    rng = np.random.default_rng(seed)
    classic = store.discriminator_path(root, "default", "classic")
    store.save_npz_atomic(
        classic, w=rng.normal(size=808), b=np.float32(0.1), n_samples=120.0, feat_dim=808
    )
    ft = dirs["backbones"] / "resnet50_finetuned_20261124_100000.pth"
    ft.write_bytes(rng.bytes(2048))
    key = store.feature_key("resnet50", ft)
    deep = store.discriminator_path(root, "default", key)
    store.save_npz_atomic(
        deep, w=rng.normal(size=2856), b=np.float32(0.2), n_samples=300.0, feat_dim=2856
    )
    store.update_manifest_file(root, classic, kind="discriminator", key="classic")
    store.update_manifest_file(root, deep, kind="discriminator", key=key)
    store.set_active_backbone(root, "resnet50", ft)
    # things that must never be uploaded
    labels = np.array([1, -1, -1, 1], dtype=np.int8)
    np.savez(dirs["samples"] / "20261124_vid.npz", labels=labels, patches=np.zeros((4, 2, 2, 3)))
    (dirs["history"] / "old.txt").write_text("old")
    (dirs["logs"] / "retrain_1.out").write_text("log")
    (dirs["backbones"] / "resnet50_imagenet.pth").write_bytes(b"official")
    return root


class FakeApi:
    calls: list[tuple[str, dict[str, Any]]] = []

    def __init__(self, token: str | None = None) -> None:
        FakeApi.calls.append(("init", {"token": token}))

    def create_repo(self, repo_id: str, **kw: Any) -> None:
        FakeApi.calls.append(("create_repo", {"repo_id": repo_id, **kw}))

    def upload_folder(self, **kw: Any) -> Any:
        folder = Path(kw["folder_path"])
        kw = {
            **kw,
            "uploaded": sorted(
                p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()
            ),
        }
        FakeApi.calls.append(("upload_folder", kw))

        class Info:
            oid = "abc123def456"

        return Info()


@pytest.fixture
def fake_hf(monkeypatch):
    FakeApi.calls = []
    snaps: list[dict[str, Any]] = []
    source: dict[str, Path | None] = {"dir": None}

    def snapshot_download(**kw: Any) -> str:
        snaps.append(kw)
        if source["dir"] is not None:
            shutil.copytree(source["dir"], kw["local_dir"], dirs_exist_ok=True)
        return str(kw["local_dir"])

    monkeypatch.setattr(hub, "_hf", lambda: (FakeApi, snapshot_download))
    monkeypatch.setattr(hub, "_stored_token", lambda: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    return {"snaps": snaps, "source": source}


@pytest.fixture
def root(tmp_path):
    return _make_root(tmp_path / "home")


# ---------------------------------------------------------------- upload folder


def test_upload_folder_contents_and_card(root, tmp_path):
    staging = tmp_path / "stage"
    files = hub.build_upload_folder(root, staging)
    assert files == sorted(
        [
            "README.md",
            "backbones/resnet50_finetuned_20261124_100000.pth",
            "discriminators/discriminator_default__classic.npz",
            *[
                f"discriminators/{p.name}"
                for p in (root / "discriminators").glob("*__resnet50-*.npz")
            ],
            "manifest.json",
        ]
    )
    assert not any(f.startswith(("samples", "history", "logs")) for f in files)
    assert not any(f.endswith("_imagenet.pth") for f in files)

    manifest = json.loads((staging / "manifest.json").read_text())
    assert set(manifest["files"]) == {f for f in files if f.endswith((".npz", ".pth"))}
    assert manifest["active_backbones"] == {
        "resnet50": "backbones/resnet50_finetuned_20261124_100000.pth"
    }
    blob = json.dumps(manifest)
    assert str(tmp_path) not in blob and "samples/" not in blob and "last_retrain" not in blob
    for rel, entry in manifest["files"].items():
        assert entry["sha256"] == store.file_sha256(staging / rel)
        assert set(entry) <= set(hub._ENTRY_FIELDS)

    card = (staging / "README.md").read_text()
    assert card.startswith("---\nlicense: agpl-3.0\n")
    for tag in ("vaila", "tracking", "biomechanics", "markerless"):
        assert f"  - {tag}" in card
    assert "library_name: vaila" in card
    assert "not published" in card and "classic" in card and "pull" in card
    assert "fine-tuned backbone included" in card
    assert str(tmp_path) not in card and not re.search(r"[\w.]+@[\w.]+\.\w+", card)
    assert "/home/" not in card


def test_upload_plan_matches_folder(root, tmp_path):
    assert hub.upload_plan(root) == hub.build_upload_folder(root, tmp_path / "s")


# ---------------------------------------------------------------- push


def test_push_aborts_without_confirmation(root, fake_hf):
    for confirm in (None, lambda repo, files: False):
        res = hub.push(root, "u/r", token="t", confirm=confirm)
        assert res["aborted"] is True and res["revision"] == ""
    assert FakeApi.calls == []
    assert read_hf(root)["synced"] == ""


def read_hf(root):
    return store.read_manifest(root)["hf"]


def test_push_sends_and_records_manifest(root, fake_hf):
    seen = {}

    def confirm(repo, files):
        seen["args"] = (repo, list(files))
        return True

    res = hub.push(root, "u/r", token="tok", confirm=confirm)
    names = dict(FakeApi.calls)
    assert names["init"]["token"] == "tok"
    assert names["create_repo"]["private"] is False
    assert names["create_repo"]["exist_ok"] is True
    assert names["create_repo"]["repo_type"] == "model"
    up = names["upload_folder"]
    assert up["repo_id"] == "u/r" and up["repo_type"] == "model"
    assert up["delete_patterns"] == ["discriminators/*", "backbones/*"]
    assert up["commit_message"].startswith("vaila AI Track update ")
    assert up["uploaded"] == res["files"] == seen["args"][1]
    assert not any(f.startswith(("samples", "history", "logs")) for f in up["uploaded"])
    assert res["revision"] == "abc123def456" and res["dry_run"] is False
    hf = read_hf(root)
    assert hf["repo"] == "u/r" and hf["revision"] == "abc123def456" and hf["synced"]


def test_push_dry_run_sends_nothing(root, fake_hf):
    res = hub.push(root, "u/r", dry_run=True)
    assert res["dry_run"] is True and "manifest.json" in res["files"]
    assert FakeApi.calls == []


def test_push_without_login_explains(root, fake_hf):
    with pytest.raises(hub.HubLoginError, match="hf auth login"):
        hub.push(root, "u/r", confirm=lambda *_: True)
    assert FakeApi.calls == []


def test_push_token_from_environment(root, fake_hf, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "envtok")
    hub.push(root, "u/r", confirm=lambda *_: True)
    assert FakeApi.calls[0] == ("init", {"token": "envtok"})


# ---------------------------------------------------------------- import / pull


def _repo_copy(root: Path, dest: Path) -> Path:
    hub.build_upload_folder(root, dest)
    return dest


def test_pull_uses_allow_patterns_and_installs(root, fake_hf, tmp_path):
    remote = _repo_copy(root, tmp_path / "remote")
    fake_hf["source"]["dir"] = remote
    dest = tmp_path / "new" / "ai_tracker"
    res = hub.pull(dest, "u/r", revision="v1")
    kw = fake_hf["snaps"][0]
    assert kw["repo_id"] == "u/r" and kw["revision"] == "v1" and kw["repo_type"] == "model"
    assert kw["allow_patterns"] == hub.UPLOAD_PATTERNS
    assert sorted(res["installed"]) == sorted(
        f for f in hub.upload_plan(root) if f.endswith((".npz", ".pth"))
    )
    for rel in res["installed"]:
        assert store.file_sha256(dest / rel) == store.file_sha256(root / rel)
    man = store.read_manifest(dest)
    assert man["active_backbones"]["resnet50"].startswith("backbones/resnet50_finetuned_")
    assert man["hf"]["repo"] == "u/r" and man["hf"]["revision"] == "v1"
    assert store.active_backbone(dest, "resnet50") is not None


def test_import_accepts_home_and_subfolder(root, tmp_path):
    for src in (root, root.parent):
        dest = tmp_path / f"d{src.name}" / "ai_tracker"
        res = hub.import_folder(src, dest)
        assert len(res["installed"]) == 3
        assert list((dest / "samples").glob("*")) == []


def test_import_rejects_corrupted_file_and_installs_nothing(root, tmp_path):
    remote = _repo_copy(root, tmp_path / "remote")
    victim = remote / "backbones" / "resnet50_finetuned_20261124_100000.pth"
    victim.write_bytes(b"corrupted")
    dest = tmp_path / "dest" / "ai_tracker"
    with pytest.raises(hub.HubImportError, match="sha256"):
        hub.import_folder(remote, dest)
    assert list((dest / "discriminators").glob("*")) == []
    assert list((dest / "backbones").glob("*")) == []


def test_import_requires_a_manifest(tmp_path):
    (tmp_path / "x").mkdir()
    with pytest.raises(hub.HubImportError, match="manifest"):
        hub.import_folder(tmp_path / "x", tmp_path / "d" / "ai_tracker")


def test_import_ignores_unexpected_paths(root, tmp_path):
    remote = _repo_copy(root, tmp_path / "remote")
    manifest = json.loads((remote / "manifest.json").read_text())
    (remote / "evil.npz").write_bytes(b"x")
    manifest["files"]["../evil.npz"] = {"sha256": "0"}
    manifest["files"]["samples/a.npz"] = {"sha256": "0"}
    (remote / "manifest.json").write_text(json.dumps(manifest))
    dest = tmp_path / "dest" / "ai_tracker"
    res = hub.import_folder(remote, dest)
    assert "../evil.npz" in res["skipped"] and "samples/a.npz" in res["skipped"]
    assert not (dest.parent / "evil.npz").exists()


def test_import_keeps_previous_file_in_history_and_local_manifest(root, tmp_path):
    remote = _repo_copy(root, tmp_path / "remote")
    dest = _make_root(tmp_path / "other", seed=7)
    classic = dest / "discriminators" / "discriminator_default__classic.npz"
    before = store.file_sha256(classic)
    manifest = store.read_manifest(dest)
    manifest["last_retrain"] = {"status": "activated", "mode": "both"}
    manifest["hf"] = {"repo": "me/mine", "revision": "r1", "synced": "2026-11-20T10:00:00"}
    store.write_manifest(dest, manifest)

    res = hub.import_folder(remote, dest)
    assert "discriminators/discriminator_default__classic.npz" in res["installed"]
    assert store.file_sha256(classic) == store.file_sha256(
        root / "discriminators" / "discriminator_default__classic.npz"
    )
    versions = store.history_versions(classic)
    assert versions and store.file_sha256(versions[0]) == before
    merged = store.read_manifest(dest)
    assert merged["last_retrain"]["status"] == "activated"
    assert merged["hf"]["repo"] == "me/mine"
    entry = merged["files"]["discriminators/discriminator_default__classic.npz"]
    assert entry["sha256"] == store.file_sha256(classic)
    # a second import changes nothing
    again = hub.import_folder(remote, dest)
    assert again["installed"] == []


def test_import_warns_when_official_weights_missing(root, tmp_path, monkeypatch):
    remote = _repo_copy(root, tmp_path / "remote")
    # drop the fine-tuned backbone: the deep discriminator now needs the official weights
    manifest = json.loads((remote / "manifest.json").read_text())
    rel = "backbones/resnet50_finetuned_20261124_100000.pth"
    (remote / rel).unlink()
    del manifest["files"][rel]
    manifest["active_backbones"] = {}
    (remote / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(store, "LEGACY_DIR", tmp_path / "no_legacy")
    res = hub.import_folder(remote, tmp_path / "dest" / "ai_tracker")
    assert any("official ImageNet" in w for w in res["warnings"])


# ---------------------------------------------------------------- restore


def test_restore_latest(root):
    path = root / "discriminators" / "discriminator_default__classic.npz"
    old = store.file_sha256(path)
    store.save_npz_atomic(path, w=np.ones(808), b=np.float32(0), n_samples=1.0, feat_dim=808)
    assert store.file_sha256(path) != old
    used = hub.restore_latest(path)
    assert used is not None
    assert store.file_sha256(path) == old
    assert store.read_manifest(root)["files"][path.relative_to(root).as_posix()]["sha256"] == old
    assert hub.restore_latest(root / "discriminators" / "none.npz") is None


# ---------------------------------------------------------------- retrain


def test_start_retrain_builds_frozen_argv(root, capsys):
    calls = []

    class FakeProc:
        pid = 1

    def fake_popen(argv, **kw):
        calls.append((argv, kw))
        return FakeProc()

    proc, log = hub.start_retrain(root, popen=fake_popen)
    argv, kw = calls[0]
    assert argv == [
        sys.executable,
        "-m",
        "vaila.ai_tracker_train",
        "--from-samples",
        str(root),
        "--mode",
        "both",
        "--device",
        "auto",
    ]
    assert isinstance(proc, FakeProc)
    assert log.parent == root / "logs" and re.fullmatch(r"retrain_\d{8}_\d{6}\.out", log.name)
    assert log.exists() and kw["cwd"] == str(Path(hub.__file__).resolve().parent.parent)
    out = capsys.readouterr().out
    assert ">> vaila/ai_tracker_train: uv run --no-sync python -m vaila.ai_tracker_train" in out


# ---------------------------------------------------------------- status


def test_status_and_format(root):
    info = hub.status(root)
    assert info["n_sample_files"] == 1 and info["n_samples"] == 4 and info["n_markers"] == 2
    assert len(info["discriminators"]) == 2
    deep = [d for d in info["discriminators"] if d["key"].startswith("resnet50-")][0]
    assert deep["feat_dim"] == 2856 and deep["n_samples"] == 300
    en = hub.format_status(info, "en")
    pt = hub.format_status(info, "pt")
    assert "Discriminators" in en and "Discriminadores" in pt
    assert "never synced" in en and "nunca sincronizado" in pt
    assert "WARNING" not in en


def test_status_legacy_warns(monkeypatch, tmp_path):
    monkeypatch.delenv(store.ENV_HOME)
    monkeypatch.setattr(store, "LEGACY_DIR", tmp_path / "legacy_ai")
    info = hub.status()
    assert info["source"] == "legacy" and info["warning"] == "no_backup"
    assert "WARNING" in hub.format_status(info, "en")
    assert "ATENCAO" in hub.format_status(info, "pt")


# ---------------------------------------------------------------- CLI


def test_cli_status_set_home_push_dry_run_import(tmp_path, capsys, fake_hf, monkeypatch):
    monkeypatch.delenv(store.ENV_HOME)
    home = tmp_path / "cli_home"
    assert hub.main(["set-home", str(home)]) == 0
    root = home / "ai_tracker"
    assert root.is_dir() and store.CONFIG_FILE.is_file()
    _make_root(home)
    capsys.readouterr()
    assert hub.main(["status"]) == 0
    assert str(root) in capsys.readouterr().out
    assert hub.main(["push", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Would send" in out and "manifest.json" in out
    assert FakeApi.calls == []

    other = tmp_path / "other_home"
    assert hub.main(["--ai-models", str(other), "import", str(root)]) == 0
    assert len(list((other / "ai_tracker" / "discriminators").glob("*.npz"))) >= 2
    store.set_session_home(None)


def test_cli_push_prompt_declined(tmp_path, monkeypatch, capsys, fake_hf):
    _make_root(Path(os.environ[store.ENV_HOME]))
    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert hub.main(["push"]) == 1
    assert "Nothing was sent" in capsys.readouterr().out
    assert FakeApi.calls == []


def test_cli_restore_and_errors(tmp_path, capsys):
    root = _make_root(Path(os.environ[store.ENV_HOME]))
    path = root / "discriminators" / "discriminator_default__classic.npz"
    assert hub.main(["restore", str(path)]) == 1  # nothing in history yet
    assert hub.main(["import", str(tmp_path / "nowhere")]) == 2
    assert "error:" in capsys.readouterr().err


# ---------------------------------------------------------------- dialog


def _display_or_skip():
    import tkinter as tk

    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    probe.destroy()


def test_dialog_builds_and_destroys_headless_safe():
    _display_or_skip()
    _make_root(Path(os.environ[store.ENV_HOME]))
    dlg = hub._ModelsDialog(None)
    try:
        dlg.win.update()
        dlg.refresh()
        content = dlg.text.get("1.0", "end")
        assert "Discriminators" in content
        for label in (
            "Change folder...",
            "Open folder",
            "Download from Hugging Face",
            "Use a local folder...",
            "Retrain (GPU)",
            "Send to Hugging Face...",
            "Restore previous version...",
        ):
            assert label in dlg.buttons
        assert dlg.repo_var.get() == hub.DEFAULT_REPO
    finally:
        dlg.win.destroy()


def test_dialog_download_and_send_actions(tmp_path, fake_hf, monkeypatch):
    import time
    from tkinter import filedialog, messagebox

    _display_or_skip()
    remote = _repo_copy(_make_root(tmp_path / "remote_home"), tmp_path / "remote")
    fake_hf["source"]["dir"] = remote
    shown: list[str] = []
    monkeypatch.setattr(messagebox, "showinfo", lambda *a, **k: shown.append(a[1]))
    monkeypatch.setattr(messagebox, "showerror", lambda *a, **k: shown.append("ERR " + a[1]))
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: True)
    monkeypatch.setattr(filedialog, "askdirectory", lambda **k: str(remote))
    dlg = hub._ModelsDialog(None)
    try:
        dlg.download()
        deadline = time.time() + 10
        while dlg._busy and time.time() < deadline:
            dlg.win.update()
            time.sleep(0.02)
        assert not dlg._busy and not any(m.startswith("ERR") for m in shown)
        assert "classic" in dlg.text.get("1.0", "end")
        assert fake_hf["snaps"][0]["repo_id"] == hub.DEFAULT_REPO
        # without a login the send button explains how to log in and sends nothing
        shown.clear()
        dlg.send()
        assert "hf auth login" in shown[0] and FakeApi.calls == []
        # with a token it sends after the yes/no
        monkeypatch.setenv("HF_TOKEN", "t")
        dlg.send()
        deadline = time.time() + 10
        while dlg._busy and time.time() < deadline:
            dlg.win.update()
            time.sleep(0.02)
        assert [n for n, _ in FakeApi.calls] == ["init", "create_repo", "upload_folder"]
        dlg.use_local()
        deadline = time.time() + 10
        while dlg._busy and time.time() < deadline:
            dlg.win.update()
            time.sleep(0.02)
        assert any("unchanged" in m for m in shown)
    finally:
        dlg.win.destroy()
