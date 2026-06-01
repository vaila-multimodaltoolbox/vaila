from __future__ import annotations

import hashlib
from io import BytesIO

import pytest

from vaila import crop_faces_atletas


def test_baixar_modelo_padrao_downloads_verified_model(tmp_path, monkeypatch):
    model_path = tmp_path / "models" / "crop_face" / "face_detector.task"
    model_bytes = b"verified-mediapipe-model"

    monkeypatch.setattr(crop_faces_atletas, "MODELO_PATH", model_path)
    monkeypatch.setattr(
        crop_faces_atletas,
        "MODELO_SHA256",
        hashlib.sha256(model_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        crop_faces_atletas,
        "urlopen",
        lambda url, timeout: BytesIO(model_bytes),
    )

    assert crop_faces_atletas.baixar_modelo_padrao() == model_path
    assert model_path.read_bytes() == model_bytes
    assert not model_path.with_suffix(".task.download").exists()


def test_baixar_modelo_padrao_removes_invalid_download(tmp_path, monkeypatch):
    model_path = tmp_path / "models" / "crop_face" / "face_detector.task"

    monkeypatch.setattr(crop_faces_atletas, "MODELO_PATH", model_path)
    monkeypatch.setattr(crop_faces_atletas, "MODELO_SHA256", "invalid-sha256")
    monkeypatch.setattr(
        crop_faces_atletas,
        "urlopen",
        lambda url, timeout: BytesIO(b"unexpected-model"),
    )

    with pytest.raises(RuntimeError, match="Não foi possível baixar automaticamente"):
        crop_faces_atletas.baixar_modelo_padrao()

    assert not model_path.exists()
    assert not model_path.with_suffix(".task.download").exists()


def test_main_download_model_exits_without_opening_gui(tmp_path, monkeypatch, capsys):
    model_path = tmp_path / "face_detector.task"
    model_path.write_bytes(b"model")

    monkeypatch.setattr(crop_faces_atletas, "baixar_modelo_padrao", lambda: model_path)
    monkeypatch.setattr(
        crop_faces_atletas,
        "run_crop_faces_atletas_gui",
        lambda: pytest.fail("GUI should not open"),
    )

    assert crop_faces_atletas.main(["--download-model"]) == 0
    assert str(model_path) in capsys.readouterr().out
