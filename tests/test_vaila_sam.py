"""Light tests for vaila_sam helpers (no GPU / no HF download)."""

from __future__ import annotations

import importlib.util
import json
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


def test_composite_masks_bgr_rich_draws_contour_and_label() -> None:
    from vaila.vaila_sam import _composite_masks_bgr

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((1, 64, 64), dtype=bool)
    mask[0, 16:48, 20:44] = True
    oids = np.array([3], dtype=np.int64)
    probs = np.array([0.9], dtype=np.float32)
    boxes = np.array([[20.0, 16.0, 24.0, 32.0]], dtype=np.float32)

    out_plain = _composite_masks_bgr(frame, mask, oids)
    out_rich = _composite_masks_bgr(
        frame,
        mask,
        oids,
        probs=probs,
        boxes_xywh=boxes,
        draw_box=True,
        draw_id=True,
        draw_contour=True,
        draw_centroid=True,
    )
    assert out_rich.shape == out_plain.shape
    # Rich overlay should add non-mask pixels (box/label/contour) compared to plain blending.
    diff = np.abs(out_rich.astype(np.int16) - out_plain.astype(np.int16)).sum(axis=2)
    assert int((diff > 0).sum()) > 0


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


def test_sam3_subsample_low_fps_signals_chunked_before_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Extreme temporal subsample should request chunking before VideoWriter."""
    import cv2

    import vaila.vaila_sam as sam

    calls = {"captures": 0, "writers": 0}

    class FakeCapture:
        def __init__(self, _path: str) -> None:
            calls["captures"] += 1

        def isOpened(self) -> bool:  # noqa: N802
            return True

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 30000.0
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 1920.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 1080.0
            return 0.0

        def release(self) -> None:
            return None

    def fail_writer(*_args: object, **_kwargs: object) -> object:
        calls["writers"] += 1
        pytest.fail("VideoWriter should not open after low-FPS chunking signal")

    monkeypatch.setattr(sam.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(sam, "_open_sam3_video_writer", fail_writer)

    with pytest.raises(sam._Sam3NeedsChunkedFallback, match=sam._SAM3_NEEDS_CHUNKING_SENTINEL):
        sam._maybe_subsample_video_for_vram(tmp_path / "long.mp4", tmp_path, 1)

    assert calls == {"captures": 1, "writers": 0}


def test_process_one_video_low_fps_signal_runs_chunked_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The retry wrapper routes the internal chunking signal to chunked fallback."""
    import vaila.vaila_sam as sam

    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    out_dir = tmp_path / "out"
    chunked_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_run(*_args: object, **_kwargs: object) -> None:
        raise sam._Sam3NeedsChunkedFallback(sam._sam3_needs_chunking_message("low fps"))

    def fake_chunked(*args: object, **kwargs: object) -> tuple[bool, str]:
        chunked_calls.append((args, kwargs))
        return True, "chunked ok"

    monkeypatch.setattr(sam, "run_sam3_on_video", fake_run)
    monkeypatch.setattr(sam, "_release_sam3_gpu_memory", lambda: None)
    monkeypatch.setattr(sam, "_process_video_chunked", fake_chunked)

    ok, msg = sam._process_one_video_with_oom_retry(
        video,
        out_dir,
        text_prompt="person",
        frame_index=0,
        checkpoint=None,
        max_input_frames=1,
        save_overlay_mp4=False,
        save_mask_png=False,
        frame_by_frame_fallback=False,
    )

    assert ok is True
    assert msg == "chunked ok"
    assert len(chunked_calls) == 1
    assert chunked_calls[0][0][0] == video


def test_process_one_video_low_fps_signal_no_chunked_returns_sentinel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Chunk subprocesses return a sentinel when recursive chunking is disabled."""
    import vaila.vaila_sam as sam

    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    out_dir = tmp_path / "out"

    def fake_run(*_args: object, **_kwargs: object) -> None:
        raise sam._Sam3NeedsChunkedFallback(sam._sam3_needs_chunking_message("low fps"))

    def fail_chunked(*_args: object, **_kwargs: object) -> tuple[bool, str]:
        pytest.fail("recursive chunked fallback should be disabled")

    monkeypatch.setattr(sam, "run_sam3_on_video", fake_run)
    monkeypatch.setattr(sam, "_release_sam3_gpu_memory", lambda: None)
    monkeypatch.setattr(sam, "_process_video_chunked", fail_chunked)

    ok, err = sam._process_one_video_with_oom_retry(
        video,
        out_dir,
        text_prompt="person",
        frame_index=0,
        checkpoint=None,
        max_input_frames=1,
        save_overlay_mp4=False,
        save_mask_png=False,
        frame_by_frame_fallback=False,
        no_chunked_fallback=True,
    )

    assert ok is False
    assert sam._SAM3_NEEDS_CHUNKING_SENTINEL in err
    assert sam._SAM3_NEEDS_CHUNKING_SENTINEL in (out_dir / "FAILED_sam.txt").read_text(
        encoding="utf-8"
    )


def test_sam3_auto_max_frames_upper_bound_scales_with_gpu_class() -> None:
    """Workstation GPUs must not be stuck at 128-frame SAM subsample cap (overlay sync)."""
    from vaila.vaila_sam import _sam3_auto_max_frames_upper_bound

    assert _sam3_auto_max_frames_upper_bound(8.0) == 128
    assert _sam3_auto_max_frames_upper_bound(10.5) == 512
    assert _sam3_auto_max_frames_upper_bound(15.0) == 1024
    assert _sam3_auto_max_frames_upper_bound(24.0) == 8192


def test_nearest_sess_idx_for_orig_frame_tie_breaks_forward() -> None:
    """VRAM subsample overlay must not systematically lag (tie → newer keyframe)."""
    from vaila.vaila_sam import _nearest_sess_idx_for_orig_frame

    a = np.array([0, 10, 20], dtype=np.int64)
    assert _nearest_sess_idx_for_orig_frame(15, a) == 2
    assert _nearest_sess_idx_for_orig_frame(12, a) == 1
    assert _nearest_sess_idx_for_orig_frame(0, a) == 0
    assert _nearest_sess_idx_for_orig_frame(100, a) == 2
    assert _nearest_sess_idx_for_orig_frame(7, a) == 1


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

    assert len(chunks) == 4
    # Default overlap is 2 frames: starts advance by chunk_size - overlap.
    assert chunks[0][1:] == (0, 10)
    assert chunks[1][1:] == (8, 18)
    assert chunks[2][1:] == (16, 26)
    assert chunks[3][1:] == (24, 30)

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

        # Tracks CSV (long)
        (d / "sam_tracks.csv").write_text(
            "frame,obj_id,x_px,y_px,w_px,h_px,score,area_px,n_polygons,largest_polygon_pts\n"
            + "\n".join(
                f"{local_fi},1,20,16,24,32,0.95,100,1,12" for local_fi in range(end - start)
            )
            + "\n",
            encoding="utf-8",
        )
        # Masks manifest
        (d / "sam_masks_manifest.csv").write_text(
            "frame,obj_id,area_px,mask_png\n"
            + "\n".join(
                f"{local_fi},1,100,masks/frame_{local_fi:06d}_obj_1.png"
                for local_fi in range(end - start)
            )
            + "\n",
            encoding="utf-8",
        )
        # Contours JSON (minimal schema)
        frames = []
        for local_fi in range(end - start):
            frames.append(
                {
                    "frame": local_fi,
                    "session_frame": local_fi,
                    "objects": [
                        {
                            "obj_id": 1,
                            "score": 0.95,
                            "bbox_xywh_px": [20, 16, 24, 32],
                            "area_px": 100,
                            "mask_png": f"masks/frame_{local_fi:06d}_obj_1.png",
                            "polygons": [[[20, 16], [44, 16], [44, 48], [20, 48]]],
                        }
                    ],
                }
            )

        (d / "sam_contours.json").write_text(
            json.dumps(
                {
                    "schema": "vaila_sam_contours_v1",
                    "video": "chunk.mp4",
                    "width": 64,
                    "height": 64,
                    "fps": 30.0,
                    "n_frames": end - start,
                    "object_ids": [1],
                    "frames": frames,
                },
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

    _merge_chunk_outputs(
        chunks,
        chunk_dirs,
        final_out,
        video_path,
        save_overlay_mp4=False,
        save_mask_png=True,
    )

    # Check merged masks exist with global frame indices. NOTE: the two chunks
    # don't share any frames in this test, so cross-chunk linking can't bridge
    # the local_oid=1 across them - each chunk's local ID 1 is mapped to a
    # fresh global ID (chunk 0 -> 0, chunk 1 -> 1).
    merged_masks = final_out / "masks"
    assert merged_masks.is_dir()
    mask_files = sorted(merged_masks.glob("frame_*_obj_*.png"))
    assert len(mask_files) == 20
    assert "frame_000000_obj_0.png" in mask_files[0].name
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

    # Check merged tracks + manifest + contours exist
    assert (final_out / "sam_tracks.csv").is_file()
    assert (final_out / "sam_masks_manifest.csv").is_file()
    c = final_out / "sam_contours.json"
    assert c.is_file()
    payload = json.loads(c.read_text(encoding="utf-8"))
    assert payload["schema"] == "vaila_sam_contours_v1"
    assert payload["frames"][0]["frame"] == 0
    assert payload["frames"][-1]["frame"] == 19
    # merged mask_png should point to global frame indices and remapped global IDs.
    # In this non-overlap test, the second chunk receives global ID 1.
    assert payload["frames"][-1]["objects"][0]["mask_png"].endswith("frame_000019_obj_1.png")


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


def test_merge_chunk_outputs_links_ids_across_overlap(tmp_path: Path) -> None:
    """Overlap frames link chunk-local IDs and duplicate overlap rows are dropped."""
    import cv2

    from vaila.vaila_sam import _merge_chunk_outputs

    video_path = tmp_path / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (64, 64))
    for i in range(6):
        writer.write(np.full((64, 64, 3), i * 20, dtype=np.uint8))
    writer.release()

    final_out = tmp_path / "merged_overlap"
    final_out.mkdir()
    chunks = [(tmp_path / "c0.mp4", 0, 4), (tmp_path / "c1.mp4", 2, 6)]
    chunk_dirs = []
    for ci, local_oid in enumerate((1, 7)):
        d = tmp_path / f"overlap_out_{ci}"
        d.mkdir()
        chunk_dirs.append(d)
        masks_dir = d / "masks"
        masks_dir.mkdir()
        n_local = chunks[ci][2] - chunks[ci][1]
        header = f"frame,box_x_{local_oid},box_y_{local_oid},box_w_{local_oid},box_h_{local_oid},prob_{local_oid}"
        meta_rows = []
        track_rows = [
            "frame,obj_id,x_px,y_px,w_px,h_px,score,area_px,n_polygons,largest_polygon_pts,cx_px,cy_px"
        ]
        manifest_rows = ["frame,obj_id,area_px,mask_png"]
        for local_fi in range(n_local):
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[16:48, 20:44] = 255
            cv2.imwrite(str(masks_dir / f"frame_{local_fi:06d}_obj_{local_oid}.png"), mask)
            meta_rows.append(f"{local_fi},0.3125,0.25,0.375,0.5,0.95")
            track_rows.append(f"{local_fi},{local_oid},20,16,24,32,0.95,768,1,4,32,32")
            manifest_rows.append(
                f"{local_fi},{local_oid},768,masks/frame_{local_fi:06d}_obj_{local_oid}.png"
            )
        (d / "sam_frames_meta.csv").write_text(
            header + "\n" + "\n".join(meta_rows) + "\n", encoding="utf-8"
        )
        (d / "sam_tracks.csv").write_text("\n".join(track_rows) + "\n", encoding="utf-8")
        (d / "sam_masks_manifest.csv").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")

    _merge_chunk_outputs(
        chunks,
        chunk_dirs,
        final_out,
        video_path,
        save_overlay_mp4=True,
        save_mask_png=True,
    )

    tracks = (final_out / "sam_tracks.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(tracks) == 7  # header + six unique global frames
    assert {line.split(",")[1] for line in tracks[1:]} == {"0"}
    assert [int(line.split(",")[0]) for line in tracks[1:]] == list(range(6))
    mask_files = sorted((final_out / "masks").glob("frame_*_obj_*.png"))
    assert len(mask_files) == 6
    assert all(path.name.endswith("_obj_0.png") for path in mask_files)
    overlay_files = list(final_out.glob("test_sam_overlay.*"))
    assert overlay_files, "expected merged overlay video (mp4 or avi)"
    assert overlay_files[0].suffix.lower() in {".mp4", ".avi"}


def test_delete_mask_artifacts(tmp_path):
    from vaila.vaila_sam import _delete_mask_artifacts

    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    (masks_dir / "frame_000000_obj_1.png").touch()
    manifest_csv = tmp_path / "sam_masks_manifest.csv"
    manifest_csv.touch()

    assert masks_dir.is_dir()
    assert manifest_csv.is_file()

    _delete_mask_artifacts(tmp_path)

    assert not masks_dir.exists()
    assert not manifest_csv.exists()


def test_sam3_writer_fallbacks_prefer_software_codecs() -> None:
    from vaila.vaila_sam import _SAM3_WRITER_FALLBACKS

    assert _SAM3_WRITER_FALLBACKS[0][0] == "mp4v"
    assert not any(fourcc == "avc1" for fourcc, _ext in _SAM3_WRITER_FALLBACKS)


def test_open_sam3_video_writer_ffmpeg_pipe_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When OpenCV codecs fail, fall back to ffmpeg libx264 pipe."""
    import numpy as np

    import vaila.vaila_sam as sam

    class FakeCvWriter:
        def isOpened(self) -> bool:  # noqa: N802
            return False

        def release(self) -> None:
            return None

    monkeypatch.setattr(sam.cv2, "VideoWriter", lambda *_a, **_k: FakeCvWriter())
    monkeypatch.setattr(sam, "_sam3_ffmpeg_available", lambda: True)

    target = tmp_path / "pipe_out.mp4"
    writer, actual = sam._open_sam3_video_writer(target, 30.0, (64, 48), purpose="unit test")
    try:
        assert isinstance(writer, sam._FfmpegPipeVideoWriter)
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    finally:
        writer.release()
    assert actual.suffix == ".mp4"
    assert actual.stat().st_size > 0


def test_open_sam3_video_writer_creates_file(tmp_path: Path) -> None:
    from vaila.vaila_sam import _open_sam3_video_writer

    target = tmp_path / "_sam3_subsample_input.mp4"
    writer, actual = _open_sam3_video_writer(
        target,
        30.0,
        (640, 480),
        purpose="unit test",
    )
    try:
        assert writer.isOpened()
        assert actual.parent == tmp_path.resolve()
        assert actual.suffix in {".mp4", ".avi"}
    finally:
        writer.release()
    assert actual.is_file()
    assert actual.stat().st_size > 0
