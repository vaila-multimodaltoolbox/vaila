"""Tests for interchangeable soccer-field coordinate frames."""

from __future__ import annotations

from vaila.field_coord_frames import (
    bl_to_center,
    bl_to_topleft,
    center_to_bl,
    convert_xy,
    enrich_metre_frames,
    topleft_to_bl,
)


def test_center_roundtrip():
    L, W = 105.0, 68.0
    # centre spot in bottom-left frame
    xc, yc = bl_to_center(L / 2, W / 2, L, W)
    assert abs(xc) < 1e-9 and abs(yc) < 1e-9
    bx, by = center_to_bl(0.0, 0.0, L, W)
    assert abs(bx - L / 2) < 1e-9 and abs(by - W / 2) < 1e-9


def test_topleft_roundtrip_and_skout_style():
    L, W = 105.0, 68.0
    # bottom-left corner → top-left frame should be (0, W)
    xtl, ytl = bl_to_topleft(0.0, 0.0, L, W)
    assert abs(xtl - 0.0) < 1e-9 and abs(ytl - W) < 1e-9
    # top-left corner of pitch in BL is (0, W) → TL (0, 0)
    xtl2, ytl2 = bl_to_topleft(0.0, W, L, W)
    assert abs(xtl2) < 1e-9 and abs(ytl2) < 1e-9
    bx, by = topleft_to_bl(xtl2, ytl2, L, W)
    assert abs(bx) < 1e-9 and abs(by - W) < 1e-9


def test_convert_xy_and_enrich():
    L, W = 105.0, 68.0
    x, y = 10.0, 20.0
    xc, yc = convert_xy(x, y, from_frame="bottom_left", to_frame="center", length_m=L, width_m=W)
    assert abs(xc - (10.0 - 52.5)) < 1e-9
    assert abs(yc - (20.0 - 34.0)) < 1e-9
    row = enrich_metre_frames(x, y, L, W)
    assert row["pos_x_m"] == 10.0
    assert abs(row["pos_x_center_m"] - xc) < 1e-9
    assert abs(row["pos_y_topleft_m"] - (W - y)) < 1e-9
