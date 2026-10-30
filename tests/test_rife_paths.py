"""Where RIFE lives: chosen by the user, portable, never silently another copy.

Update Date: 30 October 2026
Version: 0.4.13
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from vaila import rife_paths

SPEC = importlib.util.spec_from_file_location(
    "setup_rife_paths_test", Path(__file__).resolve().parents[1] / "bin/setup_rife.py"
)
assert SPEC is not None and SPEC.loader is not None
setup_rife = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup_rife)


@pytest.fixture(autouse=True)
def no_env_home(monkeypatch):
    monkeypatch.delenv(rife_paths.ENV_HOME, raising=False)


def test_default_is_the_legacy_checkout_folders():
    paths = rife_paths.resolve_paths()
    assert (paths.checkout, paths.weights) == (
        rife_paths.LEGACY_CHECKOUT,
        rife_paths.LEGACY_WEIGHTS,
    )
    assert "checkout" in paths.source


def test_home_has_code_and_weights_side_by_side(tmp_path):
    paths = rife_paths.resolve_paths(home=tmp_path / "rife")
    assert paths.checkout == tmp_path / "rife" / "Practical-RIFE"
    assert paths.weights == tmp_path / "rife" / "train_log"


def test_environment_home_replaces_the_default(tmp_path, monkeypatch):
    monkeypatch.setenv(rife_paths.ENV_HOME, str(tmp_path / "env"))
    paths = rife_paths.resolve_paths()
    assert paths.checkout == tmp_path / "env" / "Practical-RIFE"
    assert rife_paths.ENV_HOME in paths.source


def test_blank_environment_is_ignored(monkeypatch):
    monkeypatch.setenv(rife_paths.ENV_HOME, "   ")
    assert rife_paths.resolve_paths().checkout == rife_paths.LEGACY_CHECKOUT


def test_explicit_home_beats_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(rife_paths.ENV_HOME, str(tmp_path / "env"))
    paths = rife_paths.resolve_paths(home=tmp_path / "chosen")
    assert paths.checkout == tmp_path / "chosen" / "Practical-RIFE"


def test_dir_and_weights_override_only_their_own_folder(tmp_path):
    paths = rife_paths.resolve_paths(rife_dir=tmp_path / "code", home=tmp_path / "home")
    assert paths.checkout == tmp_path / "code"
    assert paths.weights == tmp_path / "home" / "train_log"
    paths = rife_paths.resolve_paths(weights=tmp_path / "w")
    assert paths.checkout == rife_paths.LEGACY_CHECKOUT and paths.weights == tmp_path / "w"


def test_missing_message_tells_how_to_choose_a_folder(tmp_path):
    text = rife_paths.missing_message(rife_paths.resolve_paths(home=tmp_path))
    assert "--rife-home" in text and rife_paths.ENV_HOME in text and "--export" in text


def test_setup_script_and_package_agree_on_the_layout(tmp_path):
    checkout, weights, cache = setup_rife.layout(None, tmp_path / "h")
    assert (checkout, weights) == rife_paths.home_layout((tmp_path / "h").resolve())
    assert cache == (tmp_path / "h").resolve() / "downloads/rife425.zip"
    legacy = setup_rife.layout(tmp_path, None)
    assert legacy[0] == tmp_path.resolve() / ".local/third_party/Practical-RIFE"
    assert legacy[1] == tmp_path.resolve() / "vaila/models/rife/train_log"


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """A tiny verified install in a portable home, with the hashes of the fixture files."""
    files = {"flownet.pkl": b"fixture weights", "RIFE_HDv3.py": b"fixture model"}
    monkeypatch.setattr(
        setup_rife, "FILES", {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    )
    source = tmp_path / "source"
    checkout, weights = rife_paths.home_layout(source)
    (checkout / "model").mkdir(parents=True)
    (checkout / "model/warplayer.py").write_text("code", encoding="utf-8")
    (checkout / "model/__pycache__").mkdir()
    (checkout / "model/__pycache__/x.pyc").write_bytes(b"cache")
    weights.mkdir(parents=True)
    for name, data in files.items():
        (weights / name).write_bytes(data)
    monkeypatch.setattr(setup_rife, "ensure_checkout", lambda _path: None)
    return source, files


def test_export_copies_verified_files_without_network(installed, tmp_path, monkeypatch):
    source, files = installed
    monkeypatch.setattr(
        setup_rife, "download_archive", lambda *_a: pytest.fail("export must not download")
    )
    destination = tmp_path / "other disk" / "rife"
    result = setup_rife.export_install(None, source, destination, skip_check=True)
    checkout, weights = rife_paths.home_layout(destination)
    assert (checkout / "model/warplayer.py").read_text(encoding="utf-8") == "code"
    assert not (checkout / "model/__pycache__").exists()
    assert {p.name: p.read_bytes() for p in weights.iterdir()} == files
    stored = json.loads((destination / "setup_manifest.json").read_text(encoding="utf-8"))
    assert stored == result
    assert stored["weights"] == str(weights)
    assert stored["exported_from"] == str(source)
    assert stored["cuda_inference_verified"] is False
    assert not list(destination.glob("rife_*"))  # no staging folders left behind


def test_export_twice_is_idempotent(installed, tmp_path):
    source, _files = installed
    destination = tmp_path / "copy"
    setup_rife.export_install(None, source, destination, skip_check=True)
    setup_rife.export_install(None, source, destination, skip_check=True)


def test_export_never_overwrites_a_different_file(installed, tmp_path):
    source, _files = installed
    destination = tmp_path / "copy"
    _checkout, weights = rife_paths.home_layout(destination)
    weights.mkdir(parents=True)
    (weights / "flownet.pkl").write_bytes(b"someone else's weights")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        setup_rife.export_install(None, source, destination, skip_check=True)
    assert (weights / "flownet.pkl").read_bytes() == b"someone else's weights"
    assert not (weights / "RIFE_HDv3.py").exists()


def test_export_refuses_overlapping_destination(installed, tmp_path):
    source, _files = installed
    for destination in (source, source / "train_log", tmp_path):
        with pytest.raises(ValueError, match="overlaps"):
            setup_rife.export_install(None, source, destination, skip_check=True)


def test_export_stops_on_an_altered_source_weight(installed, tmp_path):
    source, _files = installed
    (source / "train_log/flownet.pkl").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="missing or altered"):
        setup_rife.export_install(None, source, tmp_path / "copy", skip_check=True)
    assert not (tmp_path / "copy").exists()


def test_export_without_installed_code_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="install it first"):
        setup_rife.export_install(None, tmp_path / "empty", tmp_path / "copy", skip_check=True)


def test_setup_into_a_home_keeps_everything_inside_it(installed, tmp_path, monkeypatch):
    source, _files = installed
    monkeypatch.setattr(
        setup_rife, "download_archive", lambda *_a: pytest.fail("a valid install needs no download")
    )
    result = setup_rife.setup(None, home=source, skip_check=True)
    assert result["weights"] == str((source / "train_log").resolve())
    assert (source / "setup_manifest.json").is_file()


def test_main_installs_into_the_environment_home(installed, monkeypatch):
    source, _files = installed
    monkeypatch.setenv(rife_paths.ENV_HOME, str(source))
    monkeypatch.setattr(sys, "argv", ["setup_rife.py", "--skip-check"])
    assert setup_rife.main() == 0
    assert (source / "setup_manifest.json").is_file()


def test_main_rejects_home_together_with_root(installed, monkeypatch, tmp_path):
    source, _files = installed
    monkeypatch.setattr(
        sys,
        "argv",
        ["setup_rife.py", "--home", str(source), "--root", str(tmp_path), "--skip-check"],
    )
    with pytest.raises(SystemExit):
        setup_rife.main()
