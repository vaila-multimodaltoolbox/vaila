"""Unit tests for getpixelvideo letterbox viewport math."""

from __future__ import annotations

from vaila.getpixelvideo import (
    GETPIXELVIDEO_UPDATE_DATE,
    GETPIXELVIDEO_VERSION,
    GETPIXELVIDEO_WINDOW_TITLE,
    compute_letterbox_pads,
    marked_count_hud_pos,
    screen_to_video_coords,
    video_to_screen_coords,
)


def test_build_stamp_matches_global_release() -> None:
    assert GETPIXELVIDEO_VERSION == "0.4.3"
    assert GETPIXELVIDEO_UPDATE_DATE == "16 September 2026"
    assert "0.4.3" in GETPIXELVIDEO_WINDOW_TITLE
    assert "16 September 2026" in GETPIXELVIDEO_WINDOW_TITLE
    assert GETPIXELVIDEO_WINDOW_TITLE.startswith("vailá getpixelvideo")


def test_letterbox_pads_portrait_in_landscape_window() -> None:
    # Portrait 1080x1920 fitted into a wider viewport → side margins only.
    zoomed_w, zoomed_h = 405, 720
    window_w, window_h = 860, 720
    pad_x, pad_y = compute_letterbox_pads(zoomed_w, zoomed_h, window_w, window_h)
    assert pad_x == (860 - 405) // 2
    assert pad_y == 0


def test_letterbox_pads_when_media_fills_viewport() -> None:
    assert compute_letterbox_pads(1920, 1080, 1920, 1080) == (0, 0)
    assert compute_letterbox_pads(2000, 1200, 800, 600) == (0, 0)


def test_screen_video_roundtrip_with_letterbox() -> None:
    zoom = 0.5
    crop_x = crop_y = 0
    pad_x, pad_y = 100, 40
    vx, vy = 200.0, 300.0
    sx, sy = video_to_screen_coords(
        vx,
        vy,
        zoom_level=zoom,
        crop_x=crop_x,
        crop_y=crop_y,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    assert (sx, sy) == (200, 190)  # 200*0.5+100, 300*0.5+40
    back = screen_to_video_coords(
        sx,
        sy,
        zoom_level=zoom,
        crop_x=crop_x,
        crop_y=crop_y,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    assert back == (vx, vy)


def test_left_margin_maps_to_negative_video_x() -> None:
    """Clicks in the left letterbox map to x < 0 (same idea as right margin)."""
    zoom = 0.5
    pad_x, pad_y = 227, 0  # e.g. 1080*0.5=540 in an ~994-wide viewport
    vx, vy = screen_to_video_coords(
        10,
        100,
        zoom_level=zoom,
        crop_x=0,
        crop_y=0,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    assert vx < 0
    assert vy == 200.0


def test_marked_count_hud_pos_top_right() -> None:
    x, y = marked_count_hud_pos(1280, text_width=280, text_height=14, margin_right=10)
    assert (x, y) == (1280 - 280 - 10, 4 + (20 - 14) // 2)


def test_marked_count_hud_pos_clears_toolbar() -> None:
    # Preferred right-align would sit under the toolbar; clear_left_of pushes past it.
    x, y = marked_count_hud_pos(
        900,
        text_width=280,
        text_height=14,
        margin_right=10,
        clear_left_of=700,
    )
    assert x == 700 + 8
    assert y == 7
