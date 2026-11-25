"""
Tests for vaila.ai_tracker_store: home resolution, legacy migration, atomic saves with
history, manifest and the feature-space key. Everything runs in tmp folders.

Author: Paulo Roberto Pereira Santiago
Update Date: 24 November 2026
Version: 0.4.19
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from vaila import ai_tracker_store as store


def _npz(path: Path, dim: int, n: float = 5.0) -> Path:
    np.savez(
        path,
        w=np.arange(dim, dtype=np.float32),
        b=np.float32(0.5),
        n_samples=np.float32(n),
        feat_dim=np.int32(dim),
    )
    return path


@pytest.fixture
def clean(tmp_path, monkeypatch):
    """No env var, no session home, a tmp config and an empty tmp legacy folder."""
    monkeypatch.delenv(store.ENV_HOME, raising=False)
    cfg = tmp_path / "cfg" / "ai_models.json"
    legacy = tmp_path / "legacy" / "ai_tracker"
    legacy.mkdir(parents=True)
    monkeypatch.setattr(store, "CONFIG_FILE", cfg)
    monkeypatch.setattr(store, "LEGACY_DIR", legacy)
    monkeypatch.setattr(store, "_session_home", None)
    return tmp_path, cfg, legacy


# ------------------------------------------------------------------ resolution


def test_legacy_when_nothing_is_chosen(clean) -> None:
    _, cfg, legacy = clean
    res = store.resolve_home()
    assert res == store.StorePaths(legacy, "legacy", False)
    assert store.is_configured() is False
    assert not cfg.exists()


def test_resolution_order_cli_session_env_config_legacy(clean, monkeypatch) -> None:
    tmp, cfg, _legacy = clean
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"home": str(tmp / "from_config")}))
    assert store.resolve_home().source == "config"
    assert store.resolve_home().root == tmp / "from_config" / "ai_tracker"

    monkeypatch.setenv(store.ENV_HOME, str(tmp / "from_env"))
    res = store.resolve_home()
    assert (res.source, res.root) == ("env", tmp / "from_env" / "ai_tracker")

    store.set_session_home(tmp / "from_session")
    res = store.resolve_home()
    assert (res.source, res.root) == ("cli", tmp / "from_session" / "ai_tracker")

    res = store.resolve_home(tmp / "from_cli")
    assert (res.source, res.root) == ("cli", tmp / "from_cli" / "ai_tracker")
    assert res.backed_up is True
    for name in ("backbones", "discriminators", "samples", "history", "logs"):
        assert (res.root / name).is_dir()

    store.set_session_home(None)
    assert store.resolve_home().source == "env"


def test_explicit_bad_home_is_an_error_not_a_fallback(clean) -> None:
    tmp, _cfg, _legacy = clean
    blocker = tmp / "a_file"
    blocker.write_text("x")
    with pytest.raises(ValueError, match="cannot be created"):
        store.resolve_home(blocker / "sub")
    store.set_session_home(blocker / "sub")
    with pytest.raises(ValueError):
        store.resolve_home()


def test_is_configured_for_each_source(clean, monkeypatch) -> None:
    tmp, cfg, _ = clean
    assert store.is_configured() is False
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"home": str(tmp / "h")}))
    assert store.is_configured() is True
    cfg.write_text("{not json")
    assert store.is_configured() is False
    monkeypatch.setenv(store.ENV_HOME, str(tmp / "e"))
    assert store.is_configured() is True
    monkeypatch.delenv(store.ENV_HOME)
    store.set_session_home(tmp / "s")
    assert store.is_configured() is True


def test_ensure_home_prompt_path_writes_config_only_on_choice(clean) -> None:
    tmp, cfg, _ = clean
    asked: list[Path] = []

    def pick(suggested: Path) -> Path:
        asked.append(suggested)
        return tmp / "chosen"

    res = store.ensure_home(prompt_fn=pick)
    assert asked == [store.SUGGESTED_HOME]
    assert res.source == "config" and res.root == tmp / "chosen" / "ai_tracker"
    assert json.loads(cfg.read_text()) == {"home": str(tmp / "chosen")}
    assert (res.root / "README.txt").is_file()
    # configured now: no second question
    store.ensure_home(prompt_fn=lambda _s: pytest.fail("asked again"))


def test_ensure_home_cancel_uses_legacy_and_writes_nothing(clean) -> None:
    _tmp, cfg, legacy = clean
    res = store.ensure_home(prompt_fn=lambda _s: None)
    assert res == store.StorePaths(legacy, "legacy", False)
    assert not cfg.exists()
    # and it asks again next session
    asked: list[Path] = []
    store.ensure_home(prompt_fn=lambda s: asked.append(s))
    assert asked == [store.SUGGESTED_HOME]
    assert store.ensure_home().source == "legacy"  # no prompt_fn: never asks


# ------------------------------------------------------------------ keys and hashing


def test_feature_key_classic_and_variant(tmp_path) -> None:
    assert store.feature_key(None, None) == "classic"
    assert store.feature_key("", tmp_path / "x.pth") == "classic"
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"abc" * 100)
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert store.feature_key("resnet50", weights) == f"resnet50-{sha[:8]}"
    assert store.feature_key("resnet50", tmp_path / "gone.pth") == "resnet50-unknown"
    weights.write_bytes(b"different")
    assert store.feature_key("resnet50", weights) != f"resnet50-{sha[:8]}"


def test_file_sha256_is_cached_by_size_and_mtime(tmp_path, monkeypatch) -> None:
    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * 1000)
    first = store.file_sha256(f)
    assert first == hashlib.sha256(b"x" * 1000).hexdigest()

    real_open = Path.open
    opened: list[Path] = []

    def spy(self, *a, **k):
        opened.append(self)
        return real_open(self, *a, **k)

    monkeypatch.setattr(Path, "open", spy)
    assert store.file_sha256(f) == first
    assert opened == []  # served from the cache, not re-read
    f.write_bytes(b"y" * 1001)
    assert store.file_sha256(f) == hashlib.sha256(b"y" * 1001).hexdigest()


def test_discriminator_path_names(tmp_path) -> None:
    p = store.discriminator_path(tmp_path, "my profile!", "resnet50-abc123de")
    assert p == tmp_path / "discriminators" / "discriminator_my_profile___resnet50-abc123de.npz"
    assert store.discriminator_path(tmp_path).name == "discriminator_default__classic.npz"


# ------------------------------------------------------------------ migration


def _legacy_files(legacy: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(legacy.iterdir())}


def test_migrate_legacy_by_feature_dim_copies_and_keeps_originals(clean) -> None:
    tmp, _cfg, legacy = clean
    official = legacy / "resnet50_imagenet.pth"
    official.write_bytes(b"R" * 5000)
    sha8 = hashlib.sha256(official.read_bytes()).hexdigest()[:8]
    _npz(legacy / "discriminator_default.npz", 808 + 2048, n=600)
    _npz(legacy / "discriminator_ball.npz", 808, n=40)
    _npz(legacy / "discriminator_odd.npz", 64)
    (legacy / "resnet50_finetuned_20260101_000000.pth").write_bytes(b"F" * 100)
    before = _legacy_files(legacy)

    root = tmp / "home" / "ai_tracker"
    copied = store.migrate_legacy(root)

    d = root / "discriminators"
    assert (d / f"discriminator_default__resnet50-{sha8}.npz").is_file()
    assert (d / "discriminator_ball__classic.npz").is_file()
    assert (d / "discriminator_odd__dim64.npz").is_file()
    assert (root / "backbones" / "resnet50_finetuned_20260101_000000.pth").is_file()
    assert not list((root / "backbones").glob("*_imagenet.pth"))
    assert len(copied) == 4
    assert _legacy_files(legacy) == before  # originals untouched, none moved

    man = store.read_manifest(root)
    key = f"discriminators/discriminator_default__resnet50-{sha8}.npz"
    assert man["files"][key]["feat_dim"] == 2856
    assert man["files"][key]["n_samples"] == 600
    assert man["files"][key]["key"] == f"resnet50-{sha8}"
    assert man["files"][key]["sha256"]


def test_migrate_without_official_weights_uses_imagenet_key(clean) -> None:
    tmp, _cfg, legacy = clean
    _npz(legacy / "discriminator_default.npz", 808 + 2048)
    root = tmp / "h" / "ai_tracker"
    store.migrate_legacy(root)
    assert (root / "discriminators" / "discriminator_default__resnet50-imagenet.npz").is_file()


def test_migrate_is_idempotent_and_keeps_newer_local_work(clean) -> None:
    tmp, _cfg, legacy = clean
    _npz(legacy / "discriminator_default.npz", 808, n=10)
    root = tmp / "h" / "ai_tracker"
    assert len(store.migrate_legacy(root)) == 1
    assert store.migrate_legacy(root) == []
    # the user keeps learning in the new home; a re-migration must not overwrite it
    target = root / "discriminators" / "discriminator_default__classic.npz"
    newer = _npz(tmp / "newer.npz", 808, n=999)
    store.atomic_copy(newer, target)
    assert store.migrate_legacy(root) == []
    assert float(np.load(target)["n_samples"]) == 999.0


def test_migrate_into_the_legacy_folder_itself(clean) -> None:
    _tmp, _cfg, legacy = clean
    _npz(legacy / "discriminator_default.npz", 808)
    copied = store.migrate_legacy(legacy)
    assert copied == [legacy / "discriminators" / "discriminator_default__classic.npz"]
    assert (legacy / "discriminator_default.npz").is_file()  # flat original stays


def test_set_home_migrates_and_writes_files(clean) -> None:
    tmp, cfg, legacy = clean
    _npz(legacy / "discriminator_default.npz", 808)
    res = store.set_home(tmp / "mine")
    assert json.loads(cfg.read_text())["home"] == str(tmp / "mine")
    assert (res.root / "README.txt").is_file()
    assert (res.root / "discriminators" / "discriminator_default__classic.npz").is_file()
    assert (res.root / "manifest.json").is_file()
    res2 = store.set_home(tmp / "other", migrate=False)
    assert not list((res2.root / "discriminators").glob("*.npz"))


def test_readme_written_for_env_and_cli_homes(clean, monkeypatch) -> None:
    tmp, _cfg, _legacy = clean
    monkeypatch.setenv(store.ENV_HOME, str(tmp / "from_env"))
    env_res = store.resolve_home()
    assert env_res.source == "env"
    assert (env_res.root / "README.txt").read_text(encoding="utf-8") == store._README
    monkeypatch.delenv(store.ENV_HOME)
    cli_res = store.resolve_home(tmp / "from_cli")
    assert cli_res.source == "cli"
    assert (cli_res.root / "README.txt").is_file()
    store.set_session_home(tmp / "from_session")
    sess_res = store.resolve_home()
    assert (sess_res.root / "README.txt").is_file()


def test_existing_readme_is_not_overwritten(clean) -> None:
    tmp, _cfg, _legacy = clean
    root = tmp / "mine" / store.SUBDIR
    root.mkdir(parents=True)
    (root / "README.txt").write_text("my notes", encoding="utf-8")
    store.resolve_home(tmp / "mine")
    store.set_home(tmp / "mine")
    assert (root / "README.txt").read_text(encoding="utf-8") == "my notes"


def test_legacy_root_gets_no_readme(clean) -> None:
    _tmp, _cfg, legacy = clean
    res = store.resolve_home()
    assert res.source == "legacy"
    assert not (legacy / "README.txt").exists()


# ------------------------------------------------------------------ atomic save + history


def test_save_npz_atomic_history_keeps_ten_and_restores(tmp_path) -> None:
    root = tmp_path / "ai_tracker"
    path = store.discriminator_path(root)
    path.parent.mkdir(parents=True)
    assert store.history_versions(path) == []
    store.save_npz_atomic(path, v=np.int32(0))
    assert store.history_versions(path) == []  # nothing to archive on the first write
    for i in range(1, 15):
        store.save_npz_atomic(path, v=np.int32(i))
    assert int(np.load(path)["v"]) == 14
    versions = store.history_versions(path)
    assert len(versions) == store.HISTORY_KEEP
    assert versions[0].parent == root / "history" / path.name
    assert [int(np.load(v)["v"]) for v in versions] == list(range(13, 3, -1))  # newest first
    assert not [p for p in path.parent.iterdir() if p.name.endswith(".tmp")]

    store.restore(path, versions[2])  # v == 11
    assert int(np.load(path)["v"]) == 11
    assert int(np.load(store.history_versions(path)[0])["v"]) == 14  # current went to history


def test_save_npz_atomic_compressed_and_failure_leaves_no_temp(tmp_path) -> None:
    root = tmp_path / "ai_tracker"
    path = root / "samples" / "s.npz"
    store.save_npz_atomic(path, compressed=True, a=np.zeros((50, 50), dtype=np.uint8))
    assert np.load(path)["a"].shape == (50, 50)

    class Boom:
        def __array__(self, *a, **k):
            raise RuntimeError("boom")

    with pytest.raises(Exception, match="boom"):
        store.save_npz_atomic(path, a=Boom())
    assert np.load(path)["a"].shape == (50, 50)  # previous file intact
    assert not [p for p in path.parent.iterdir() if p.name.endswith(".tmp")]


def test_atomic_copy_archives_previous_version(tmp_path) -> None:
    root = tmp_path / "ai_tracker"
    src1, src2 = tmp_path / "a.bin", tmp_path / "b.bin"
    src1.write_bytes(b"one")
    src2.write_bytes(b"two")
    dst = root / "backbones" / "x.pth"
    store.atomic_copy(src1, dst)
    store.atomic_copy(src2, dst)
    assert dst.read_bytes() == b"two"
    (only,) = store.history_versions(dst)
    assert only.read_bytes() == b"one" and only.suffix == ".pth"


def test_history_next_to_file_outside_any_root(tmp_path) -> None:
    path = tmp_path / "loose" / "x.npz"
    store.save_npz_atomic(path, v=np.int32(1))
    store.save_npz_atomic(path, v=np.int32(2))
    assert len(store.history_versions(path)) == 1
    assert store.history_versions(path)[0].parent == path.parent / "history" / "x.npz"


# ------------------------------------------------------------------ manifest


def test_manifest_round_trip_and_active_backbone(tmp_path) -> None:
    root = tmp_path / "ai_tracker"
    man = store.read_manifest(root)
    assert man["files"] == {} and man["active_backbones"] == {}
    assert set(man["hf"]) == {"repo", "revision", "synced"}

    store.write_manifest(root, {**man, "hf": {"repo": "u/r", "revision": "abc", "synced": "now"}})
    back = store.read_manifest(root)
    assert back["hf"]["repo"] == "u/r"
    assert back["version"] == store.VAILA_VERSION and back["updated"]

    bb = root / "backbones" / "resnet50_finetuned_1.pth"
    bb.parent.mkdir(parents=True)
    bb.write_bytes(b"w" * 10)
    assert store.active_backbone(root, "resnet50") is None
    store.set_active_backbone(root, "resnet50", bb)
    assert store.read_manifest(root)["active_backbones"]["resnet50"] == (
        "backbones/resnet50_finetuned_1.pth"
    )
    assert store.active_backbone(root, "resnet50") == bb
    bb.unlink()
    assert store.active_backbone(root, "resnet50") is None  # missing file: ignored

    entry = store.update_manifest_file(root, root / "samples" / "none.npz", kind="samples")
    assert entry == {"kind": "samples"}  # absent file: no sha, info kept
    ckpt = _npz(tmp_path / "ai_tracker" / "d.npz", 8)
    entry = store.update_manifest_file(root, ckpt, kind="discriminator", n_samples=3.0)
    assert entry["bytes"] == ckpt.stat().st_size and len(entry["sha256"]) == 64
    assert store.read_manifest(root)["files"]["d.npz"] == entry


def test_samples_dir_created(tmp_path) -> None:
    assert store.samples_dir(tmp_path / "ai_tracker").is_dir()
