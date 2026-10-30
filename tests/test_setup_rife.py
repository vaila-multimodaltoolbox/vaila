"""Pinned RIFE installer integrity, safe extraction and idempotence.

Update Date: 29 October 2026
Version: 0.4.12
"""

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "setup_rife", Path(__file__).resolve().parents[1] / "bin/setup_rife.py"
)
assert SPEC is not None and SPEC.loader is not None
setup_rife = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup_rife)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    archive = tmp_path / "test.zip"
    files = {"flownet.pkl": b"fixture weights", "RIFE_HDv3.py": b"fixture model"}
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, data in files.items():
            zipped.writestr(f"train_log/{name}", data)
        zipped.writestr("../../escape.py", b"not extracted")
    monkeypatch.setattr(
        setup_rife,
        "FILES",
        {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    )
    monkeypatch.setattr(setup_rife, "ARCHIVE_SHA256", setup_rife.sha256(archive))
    return archive, files


def test_extract_only_verified_files(bundle, tmp_path):
    archive, files = bundle
    weights = tmp_path / "models/train_log"
    setup_rife.install_bundle(archive, weights)
    assert {p.name: p.read_bytes() for p in weights.iterdir()} == files
    assert not (tmp_path / "escape.py").exists()


def test_invalid_member_hash_does_not_replace_existing_files(bundle, tmp_path, monkeypatch):
    archive, files = bundle
    weights = tmp_path / "models/train_log"
    weights.mkdir(parents=True)
    (weights / "flownet.pkl").write_bytes(b"previous valid install")
    monkeypatch.setitem(setup_rife.FILES, "RIFE_HDv3.py", "wrong checksum")
    with pytest.raises(ValueError, match="model file checksum"):
        setup_rife.install_bundle(archive, weights)
    assert (weights / "flownet.pkl").read_bytes() == b"previous valid install"
    assert not (weights / "RIFE_HDv3.py").exists()


def test_existing_model_is_reused_without_network(bundle, tmp_path, monkeypatch):
    archive, files = bundle
    weights = tmp_path / "vaila/models/rife/train_log"
    setup_rife.install_bundle(archive, weights)
    monkeypatch.setattr(setup_rife, "ensure_checkout", lambda _: None)

    def no_network(*_args):
        pytest.fail("Existing verified model should not be downloaded")

    monkeypatch.setattr(setup_rife, "download_archive", no_network)
    result = setup_rife.setup(tmp_path, skip_check=True)
    assert result["cuda_inference_verified"] is False
    stored = json.loads((weights.parent / "setup_manifest.json").read_text())
    assert stored == result


def test_html_or_quota_response_never_becomes_model_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(
        setup_rife.urllib.request, "urlopen", lambda *_a, **_kw: io.BytesIO(b"<html>quota</html>")
    )
    destination = tmp_path / "cache/rife425.zip"
    with pytest.raises(ValueError, match="checksum mismatch"):
        setup_rife.download_archive(destination)
    assert not destination.exists()
