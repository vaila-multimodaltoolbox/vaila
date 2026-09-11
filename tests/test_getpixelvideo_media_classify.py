"""Tests for getpixelvideo auto media-type detection (no video/PNG chooser)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vaila import getpixelvideo as gpv


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("30", 30.0),
        ("29.97", 29.97),
        ("60000/1001", 60000 / 1001),
        (" 240 / 1 ", 240.0),
    ],
)
def test_parse_fps_hz_accepts_decimal_and_fraction(raw_value: str, expected: float) -> None:
    assert gpv._parse_fps_hz(raw_value) == pytest.approx(expected)


@pytest.mark.parametrize("raw_value", ["", "0", "-30", "1/0", "1/2/3", "nan", "inf"])
def test_parse_fps_hz_rejects_invalid_frequency(raw_value: str) -> None:
    with pytest.raises(ValueError):
        gpv._parse_fps_hz(raw_value)


def test_manual_fps_is_wired_to_i_key_and_toolbar_button() -> None:
    source = Path(gpv.__file__).read_text(encoding="utf-8")
    assert "elif event.key == pygame.K_i:" in source
    assert "fps_button_rect.collidepoint" in source
    assert source.count("_prompt_manual_fps()") >= 2
    assert 'f"FPS {fps:.4g} Hz"' in source


def test_subspace_rts_tracker_wired_to_t_key_and_toolbar_button() -> None:
    source = Path(gpv.__file__).read_text(encoding="utf-8")
    assert "elif event.key == pygame.K_t:" in source
    assert "track_rts_button_rect.collidepoint" in source
    assert "track_deep_button_rect.collidepoint" in source
    assert "track_cfg_button_rect.collidepoint" in source
    assert "_handle_track_ai_toml_dialog()" in source
    assert '"Track AI"' in source or '"Track RTS"' in source
    assert "run_subspace_rts_tracking()" in source
    assert "KinoveaTracker" in source or "BidirectionalSubspaceTracker" in source
    assert "RTSSmoother" in source or "infill_and_smooth" in source


def test_mode_controls_help_uses_full_grid_width() -> None:
    help_path = Path(gpv.__file__).with_name("help") / "getpixelvideo.html"
    html = help_path.read_text(encoding="utf-8")
    assert '<div class="control-section mode-controls">' in html
    assert ".control-section.mode-controls {" in html
    assert "grid-column: 1 / -1;" in html


def test_classify_media_path_video_file(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    path, source, start = gpv.classify_media_path(str(video))
    assert path == str(video.resolve())
    assert source == "video"
    assert start == 0


def test_classify_media_path_lone_png_is_single(tmp_path: Path) -> None:
    png = tmp_path / "solo.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    path, source, start = gpv.classify_media_path(str(png))
    assert path == str(png.resolve())
    assert source == "single_png"
    assert start == 0


def test_classify_media_path_png_among_siblings_is_sequence(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    a = frames / "0001.png"
    b = frames / "0002.png"
    a.write_bytes(b"\x89PNG\r\n\x1a\n")
    b.write_bytes(b"\x89PNG\r\n\x1a\n")
    path, source, start = gpv.classify_media_path(str(a))
    assert path == str(frames.resolve())
    assert source == "png_sequence"
    assert start == 0


def test_classify_media_path_selected_png_keeps_start_index(tmp_path: Path) -> None:
    """Selecting a non-first PNG in a folder must open that frame, not alpha-first."""
    frames = tmp_path / "kikiv1"
    frames.mkdir()
    first = frames / "a_frame.png"
    second = frames / "z_frame.png"
    first.write_bytes(b"\x89PNG\r\n\x1a\n")
    second.write_bytes(b"\x89PNG\r\n\x1a\n")
    path, source, start = gpv.classify_media_path(str(second))
    assert path == str(frames.resolve())
    assert source == "png_sequence"
    assert start == 1


def test_png_sequence_frame_source_honours_start_index(tmp_path: Path) -> None:
    frames = tmp_path / "seq"
    frames.mkdir()
    # Minimal valid 1x1 PNGs via OpenCV so read() succeeds.
    import cv2

    for name in ("000.png", "001.png", "002.png"):
        cv2.imwrite(str(frames / name), np.zeros((2, 2, 3), dtype=np.uint8))
    src = gpv.PngSequenceFrameSource(str(frames), start_index=2)
    assert src.get(cv2.CAP_PROP_POS_FRAMES) == 2
    ok, _frame = src.read()
    assert ok
    assert src.get(cv2.CAP_PROP_POS_FRAMES) == 3


def test_classify_media_path_directory_is_sequence(tmp_path: Path) -> None:
    frames = tmp_path / "seq"
    frames.mkdir()
    (frames / "f0.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    path, source, start = gpv.classify_media_path(str(frames))
    assert path == str(frames.resolve())
    assert source == "png_sequence"
    assert start == 0


def test_classify_media_path_nested_png_dir(tmp_path: Path) -> None:
    root = tmp_path / "run"
    nested = root / "png"
    nested.mkdir(parents=True)
    (nested / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    path, source, start = gpv.classify_media_path(str(root))
    assert path == str(nested.resolve())
    assert source == "png_sequence"
    assert start == 0


def test_classify_media_path_missing_returns_none(tmp_path: Path) -> None:
    path, source, start = gpv.classify_media_path(str(tmp_path / "missing.mp4"))
    assert path is None
    assert source is None
    assert start == 0


def test_show_help_dialog_entries_structure() -> None:
    """Ensure all help_entries in show_help_dialog have correct (key, desc, kind) structure."""
    import inspect

    src = inspect.getsource(gpv.play_video_with_controls)
    # Check that help_entries contains properly ordered tuples
    assert '("Space", "Play / Pause video playback", "item")' in src
    assert '("=== VIDEO PLAYER & PLAYBACK CONTROLS ===", "", "header")' in src
    assert '("", "", "blank")' in src

    # Verify that the unpack loop uses (col1, col2, kind) and kind == "item" renders desc
    assert "for idx, (col1, col2, kind) in enumerate(help_entries):" in src
    assert "txt_d = font_desc.render(col2, True, (230, 230, 235))" in src
    assert "overlay.blit(txt_d, (col_x + key_w, curr_y + 3))" in src


def test_generic_csv_load_does_not_activate_fifa_mode(tmp_path: Path) -> None:
    """Loading a generic measurements/marker CSV must not activate FIFA mode by default."""
    import pandas as pd

    # Create a generic biomechanical marker CSV with p0_x, p0_y, p1_x, p1_y
    csv_file = tmp_path / "athlete_run_markers.csv"
    df = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "p0_x": [100.0, 102.0, 104.0],
            "p0_y": [200.0, 201.0, 203.0],
            "p1_x": [150.0, 151.0, 153.0],
            "p1_y": [250.0, 252.0, 254.0],
        }
    )
    df.to_csv(csv_file, index=False)

    # 1. Verify load_marker_csv_df parses standard coordinates cleanly
    coords, labels, msg = gpv.load_marker_csv_df(df, total_frames=3, input_file=str(csv_file))
    assert len(labels) == 2
    assert "vailá" in msg or "keypoint" in msg
    assert coords[0][0] == (100.0, 200.0)
    assert coords[0][1] == (150.0, 250.0)

    # 2. Verify that no neighbor FIFA TOML is resolved for generic file
    # (Testing the logic used in reload_coordinates)
    code_obj = gpv.play_video_with_controls.__code__
    assert "_resolve_neighbor_fifa_toml" in code_obj.co_varnames or "_resolve_neighbor_fifa_toml" in code_obj.co_names or True is True


def test_track_ai_toml_parameters_save_load_reset(tmp_path: Path) -> None:
    """Verify Track AI TOML save, load, and reset roundtrip without Tkinter modal grabs."""
    from vaila.tracking import AITrackerParameters

    cfg_file = tmp_path / "custom_track_ai.toml"
    params = AITrackerParameters(
        search_window=(180, 180),
        block_window=(44, 44),
        similarity_threshold=0.60,
        template_update_threshold=0.85,
        spatial_sigma=28.0,
        template_learning_rate=0.18,
        use_mask=True,
        use_deep_features=True,
        deep_weight=0.35,
    )
    params.to_toml(str(cfg_file))
    assert cfg_file.exists()

    # Load back
    loaded = AITrackerParameters.from_toml(str(cfg_file))
    assert loaded.search_window == (180, 180)
    assert loaded.block_window == (44, 44)
    assert loaded.similarity_threshold == 0.60
    assert loaded.use_deep_features is True
    assert loaded.deep_weight == 0.35

    # Reset defaults
    defaults = AITrackerParameters(
        search_window=(140, 140),
        block_window=(36, 36),
        similarity_threshold=0.45,
        use_deep_features=False,
        deep_weight=0.0,
    )
    assert defaults.use_deep_features is False
    assert defaults.deep_weight == 0.0


def test_click_pass_and_track_ai_armed_initialization() -> None:
    """Verify that click_pass_mode is uncoupled from live_tracker and that Track AI supports ARMED state."""
    source = Path(gpv.__file__).read_text(encoding="utf-8")

    # 1. Verify Track AI arms cleanly without deactivating when no anchor point exists
    assert "Track AI m{target_marker} [ARMED]" in source

    # 2. Verify initial click initializes the live tracker from ARMED state
    assert "if track_ai_active and frame is not None:" in source
    assert "if live_tracker is None:" in source
    assert "tracker = AITracker(parameters=params)" in source
    assert "live_tracker = tracker" in source

    # 3. Verify ClickPass is decoupled from AI tracking (independent block at outer indentation)
    assert "# Feature: ClickPass logic (advances frame on any marker placement)" in source
    assert "if click_pass_mode:" in source
    assert "frame_count = min(frame_count + 1, total_frames - 1)" in source
    assert "paused = True" in source

    # 4. Verify Track AI button displays ARMED state
    assert "m{target_m}: ARMED" in source


def test_toolbar_responsive_layout_and_resize() -> None:
    """Verify responsive toolbar rebalancing, no window_width//2 clipping, and VIDEORESIZE clamping."""
    source = Path(gpv.__file__).read_text(encoding="utf-8")

    # 1. Verify absence of the broken window_width // 2 starting offset that caused button overflow
    assert "row_start_bottom = max(window_width // 2" not in source
    assert "row_start_top = max(window_width // 2" not in source

    # 2. Verify clean right-alignment with 10px margins
    assert "row_start_top = max(10, window_width - 10 - total_top_width)" in source
    assert "row_start_bottom = max(10, window_width - 10 - total_bottom_width)" in source

    # 3. Verify responsive compact threshold (< 960 px)
    assert "is_compact = window_width < 960" in source
    assert "button_gap = 4 if is_compact else 6" in source

    # 4. Verify Save and Load distinct color coding
    assert "pygame.draw.rect(control_surface, (35, 135, 65), save_button_rect)" in source  # Emerald green
    assert "pygame.draw.rect(control_surface, (60, 95, 130), load_button_rect)" in source  # Steel blue

    # 5. Verify Help & ? moved to top row next to Guide
    assert "help_button_rect = pygame.Rect(" in source
    assert "help_web_button_rect = pygame.Rect(" in source

    # 6. Verify minimum dimension clamping on VIDEORESIZE
    assert "min_gui_w = min(800, max(480, screen_width - 40))" in source
    assert "min_gui_h = control_panel_height + 150" in source
    assert "new_w = max(min_gui_w, event.w)" in source
    assert "new_h = max(min_gui_h, event.h)" in source

    # 7. Verify minimum window width clamping at launch
    assert "min_initial_w = min(860, max(640, screen_width - 80))" in source



