"""Interchangeable soccer-field coordinate frames used across *vailá* tools.

Update Date: 22 October 2026
Version: 0.4.10

Frames
------
- ``bottom_left`` (scout legacy): origin at bottom-left corner, X right, Y up.
  Range: x in [0, L], y in [0, W].
- ``center`` (FIFA / FreeKiki / drawsportsfields default): origin at centre spot.
  Range: x in [-L/2, L/2], y in [-W/2, W/2].
- ``top_left`` (Skout / image-style metres): origin at top-left corner, X right,
  Y **down**. Range: x in [0, L], y in [0, W].

Pixel coordinates are image-space (top-left, Y down) and are *not* convertible
to metres without a homography / DLT — callers keep them as parallel columns.
"""

from __future__ import annotations

from typing import Literal

CoordFrame = Literal["bottom_left", "center", "top_left"]


def bl_to_center(x: float, y: float, length_m: float, width_m: float) -> tuple[float, float]:
    """Bottom-left → centre-origin metres."""
    return float(x) - length_m / 2.0, float(y) - width_m / 2.0


def center_to_bl(x: float, y: float, length_m: float, width_m: float) -> tuple[float, float]:
    """Centre-origin → bottom-left metres."""
    return float(x) + length_m / 2.0, float(y) + width_m / 2.0


def bl_to_topleft(x: float, y: float, length_m: float, width_m: float) -> tuple[float, float]:
    """Bottom-left (Y up) → top-left (Y down) metres. ``length_m`` unused, kept for API symmetry."""
    _ = length_m
    return float(x), width_m - float(y)


def topleft_to_bl(x: float, y: float, length_m: float, width_m: float) -> tuple[float, float]:
    """Top-left (Y down) → bottom-left (Y up) metres."""
    _ = length_m
    return float(x), width_m - float(y)


def center_to_topleft(x: float, y: float, length_m: float, width_m: float) -> tuple[float, float]:
    """Centre → top-left (Y down) metres."""
    bx, by = center_to_bl(x, y, length_m, width_m)
    return bl_to_topleft(bx, by, length_m, width_m)


def topleft_to_center(x: float, y: float, length_m: float, width_m: float) -> tuple[float, float]:
    """Top-left (Y down) → centre metres."""
    bx, by = topleft_to_bl(x, y, length_m, width_m)
    return bl_to_center(bx, by, length_m, width_m)


def convert_xy(
    x: float,
    y: float,
    *,
    from_frame: CoordFrame,
    to_frame: CoordFrame,
    length_m: float,
    width_m: float,
) -> tuple[float, float]:
    """Convert a point between any two supported metre frames."""
    if from_frame == to_frame:
        return float(x), float(y)

    # Normalize to bottom_left first
    if from_frame == "bottom_left":
        bx, by = float(x), float(y)
    elif from_frame == "center":
        bx, by = center_to_bl(x, y, length_m, width_m)
    elif from_frame == "top_left":
        bx, by = topleft_to_bl(x, y, length_m, width_m)
    else:
        raise ValueError(f"Unknown from_frame: {from_frame!r}")

    if to_frame == "bottom_left":
        return bx, by
    if to_frame == "center":
        return bl_to_center(bx, by, length_m, width_m)
    if to_frame == "top_left":
        return bl_to_topleft(bx, by, length_m, width_m)
    raise ValueError(f"Unknown to_frame: {to_frame!r}")


def enrich_metre_frames(
    x_bl: float,
    y_bl: float,
    length_m: float,
    width_m: float,
) -> dict[str, float]:
    """From bottom-left click, return all interchangeable metre columns."""
    xc, yc = bl_to_center(x_bl, y_bl, length_m, width_m)
    xtl, ytl = bl_to_topleft(x_bl, y_bl, length_m, width_m)
    return {
        "pos_x_m": float(x_bl),
        "pos_y_m": float(y_bl),
        "pos_x_center_m": float(xc),
        "pos_y_center_m": float(yc),
        "pos_x_topleft_m": float(xtl),
        "pos_y_topleft_m": float(ytl),
        "field_length_m": float(length_m),
        "field_width_m": float(width_m),
    }
