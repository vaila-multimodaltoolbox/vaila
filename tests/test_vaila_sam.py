"""Light tests for vaila_sam helpers (no GPU / no HF download)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

sam3_spec = importlib.util.find_spec("sam3")


@pytest.mark.skipif(sam3_spec is None, reason="sam3 optional extra not installed")
def test_resolve_bpe_path_exists() -> None:
    from vaila.vaila_sam import _resolve_bpe_path

    p = _resolve_bpe_path()
    assert p.is_file()


def test_resolve_checkpoint_rejects_sam3d_body_path(tmp_path: Path) -> None:
    from vaila.vaila_sam import _resolve_sam3_checkpoint_file

    p = tmp_path / "vaila" / "models" / "sam-3d-dinov3" / "assets" / "mhr_model.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="SAM 3D Body"):
        _resolve_sam3_checkpoint_file(p)


def test_composite_masks_bgr_empty() -> None:
    from vaila.vaila_sam import _composite_masks_bgr

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    out = _composite_masks_bgr(
        frame, np.zeros((0, 64, 64), dtype=bool), np.array([], dtype=np.int64)
    )
    assert out.shape == frame.shape


def test_sam3_build_oom_retry_attempts_extends_below_32() -> None:
    """When auto-VRAM caps at 32, OOM retry must still try 24→8 (regression)."""
    from vaila.vaila_sam import _sam3_build_oom_retry_attempts

    assert _sam3_build_oom_retry_attempts(32) == [32, 24, 16, 12, 8]
    chain0 = _sam3_build_oom_retry_attempts(0)
    assert chain0[0] == 0
    assert 16 in chain0
    none_chain = _sam3_build_oom_retry_attempts(None)
    assert none_chain[0] is None
    assert 32 in none_chain and 8 in none_chain


def test_sam3_prompt_presets_cover_sports_scenarios() -> None:
    """The GUI combobox must advertise at least a minimum set of open-vocabulary
    presets (see the plan: person/player/goalkeeper/referee/ball/...)."""
    from vaila.vaila_sam import SAM3_PROMPT_PRESETS

    required = {
        "person",
        "player",
        "goalkeeper",
        "referee",
        "ball",
        "soccer ball",
        "basketball",
        "crowd",
        "car",
        "bike",
    }
    missing = required - set(SAM3_PROMPT_PRESETS)
    assert not missing, f"missing prompt presets: {sorted(missing)}"
    assert len(SAM3_PROMPT_PRESETS) >= 11


def test_sam_video_dialog_has_help_button(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instantiate SamVideoDialog headlessly and confirm the Help button + combobox."""
    tkinter = pytest.importorskip("tkinter")
    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:
        pytest.skip(f"no display available for Tk: {exc}")
    root.withdraw()
    try:
        from vaila.vaila_sam import SAM3_PROMPT_PRESETS, SamVideoDialog

        dlg = SamVideoDialog(root)
        # walk all descendants and look for the Help button + the combobox
        stack = [dlg]
        texts: list[str] = []
        combobox_values: tuple[str, ...] = ()
        while stack:
            w = stack.pop()
            stack.extend(list(w.children.values()))
            cls = w.winfo_class()
            if cls == "TButton":
                texts.append(str(w.cget("text")))
            if cls == "TCombobox":
                combobox_values = tuple(w.cget("values"))
        assert "Help" in texts, f"Help button not found, saw buttons: {texts}"
        assert len(combobox_values) >= 11
        for preset in ("person", "player", "goalkeeper", "referee", "ball"):
            assert preset in combobox_values
        # minimal sanity that our module-level tuple was used
        assert set(combobox_values) >= set(SAM3_PROMPT_PRESETS[:5])
        dlg.destroy()
    finally:
        root.destroy()


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


SAM_SAMPLE_VIDEO = Path(__file__).resolve().parent / "SAM" / "test1000.mp4"


@pytest.mark.skipif(sam3_spec is None, reason="sam3 optional extra not installed")
@pytest.mark.skipif(not _cuda_available(), reason="SAM 3 video path requires CUDA")
@pytest.mark.skipif(
    os.environ.get("VAILA_TEST_SAM_GPU") != "1", reason="set VAILA_TEST_SAM_GPU=1 to run"
)
def test_sam3_smoke_sample_video(tmp_path: Path) -> None:
    """End-to-end SAM3 on tests/SAM/test1000.mp4 (slow; needs HF access or local sam3.pt)."""
    if not SAM_SAMPLE_VIDEO.is_file():
        pytest.skip(f"Add sample video at {SAM_SAMPLE_VIDEO} (see tests/SAM/README.md)")

    from vaila.vaila_sam import run_sam3_on_video

    out_dir = tmp_path / "sam_smoke"
    run_sam3_on_video(
        SAM_SAMPLE_VIDEO,
        out_dir,
        "person",
        frame_index=0,
        save_overlay_mp4=False,
        save_mask_png=False,
    )
    assert (out_dir / "README_sam.txt").is_file()
    assert (out_dir / "sam_frames_meta.csv").is_file()
