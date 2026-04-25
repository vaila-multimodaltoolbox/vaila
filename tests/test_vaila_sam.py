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

    assert _sam3_build_oom_retry_attempts(32) == [32, 24, 16, 12, 8, 4, 2, 1]
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


def test_split_video_into_chunks(tmp_path: Path) -> None:
    """_split_video_into_chunks produces chunk MP4s with correct frame ranges."""
    import cv2

    from vaila.vaila_sam import _split_video_into_chunks

    # Create a synthetic test video with 30 frames
    vid_path = tmp_path / "test_30frames.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid_path), fourcc, 30.0, (64, 64))
    for i in range(30):
        frame = np.full((64, 64, 3), i * 8, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    chunk_dir = tmp_path / "chunks"
    chunks = _split_video_into_chunks(vid_path, chunk_dir, chunk_size=10)

    assert len(chunks) == 3
    # First chunk: frames 0-9
    assert chunks[0][1] == 0
    assert chunks[0][2] == 10
    # Second chunk: frames 10-19
    assert chunks[1][1] == 10
    assert chunks[1][2] == 20
    # Third chunk: frames 20-29
    assert chunks[2][1] == 20
    assert chunks[2][2] == 30

    # Verify each chunk file exists and has correct frame count
    for chunk_path, start, end in chunks:
        assert Path(chunk_path).is_file()
        cap = cv2.VideoCapture(str(chunk_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert n == end - start


def test_merge_chunk_outputs(tmp_path: Path) -> None:
    """_merge_chunk_outputs merges masks and metadata from multiple chunks."""
    from vaila.vaila_sam import _merge_chunk_outputs

    video_path = tmp_path / "test.mp4"
    video_path.touch()
    final_out = tmp_path / "merged"
    final_out.mkdir()

    # Simulate two chunks with masks and metadata
    chunks = [
        (tmp_path / "c0.mp4", 0, 10),
        (tmp_path / "c1.mp4", 10, 20),
    ]
    chunk_dirs = []
    for ci, (_, start, end) in enumerate(chunks):
        d = tmp_path / f"chunk_out_{ci}"
        d.mkdir()
        chunk_dirs.append(d)
        masks_dir = d / "masks"
        masks_dir.mkdir()

        # Write some mask PNGs with chunk-local frame indices
        for local_fi in range(end - start):
            mask = np.full((64, 64), 255, dtype=np.uint8)
            import cv2

            cv2.imwrite(str(masks_dir / f"frame_{local_fi:06d}_obj_1.png"), mask)

        # Write metadata CSV
        header = "frame,box_x_1,box_y_1,box_w_1,box_h_1,prob_1"
        rows = []
        for local_fi in range(end - start):
            rows.append(f"{local_fi},10.0,20.0,30.0,40.0,0.95")
        (d / "sam_frames_meta.csv").write_text(
            header + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
        )

    _merge_chunk_outputs(
        chunks,
        chunk_dirs,
        final_out,
        video_path,
        save_overlay_mp4=False,
        save_mask_png=True,
    )

    # Check merged masks exist with global frame indices
    merged_masks = final_out / "masks"
    assert merged_masks.is_dir()
    # Should have 20 mask files (0-19)
    mask_files = sorted(merged_masks.glob("frame_*_obj_*.png"))
    assert len(mask_files) == 20
    # Verify first file is frame 0 and last is frame 19
    assert "frame_000000_obj_1.png" in mask_files[0].name
    assert "frame_000019_obj_1.png" in mask_files[-1].name

    # Check merged metadata CSV
    meta = final_out / "sam_frames_meta.csv"
    assert meta.is_file()
    lines = meta.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 21  # header + 20 data rows
    # First data row should be frame 0
    assert lines[1].startswith("0,")
    # Last data row should be frame 19
    assert lines[-1].startswith("19,")


def test_read_max_input_long_edge_cli_explicit() -> None:
    from vaila.vaila_sam import _read_max_input_long_edge

    assert _read_max_input_long_edge(0) == 0
    assert _read_max_input_long_edge(960) == 960


def test_maybe_downscale_video_long_edge_noop_small_video(tmp_path: Path) -> None:
    import cv2

    from vaila.vaila_sam import _maybe_downscale_video_long_edge

    vid = tmp_path / "small.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(vid), fourcc, 30.0, (320, 240))
    w.write(np.zeros((240, 320, 3), dtype=np.uint8))
    w.release()

    p, tmp, w0, h0, n = _maybe_downscale_video_long_edge(vid, tmp_path, 1280)
    assert tmp is None
    assert p == vid.resolve()
    assert w0 == 320 and h0 == 240
    assert n >= 1
