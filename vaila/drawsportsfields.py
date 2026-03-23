"""
Project: vailá
Script: drawsportsfields.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 20 March 2025
Updated: 23 March 2026
Version: 0.0.2

Description:
    Unified sports-field/court visualization module.
    Draws soccer fields, tennis courts, basketball/volleyball/handball/futsal
    courts using matplotlib, with support for overlaying marker trajectories,
    scout events, KDE heatmaps, and configurable surface color schemes.

Usage:
    GUI:
        uv run drawsportsfields.py                    # soccer (default)
        uv run drawsportsfields.py --type tennis
        uv run drawsportsfields.py --type basketball
    CLI:
        uv run drawsportsfields.py --field <path_to_csv>
        uv run drawsportsfields.py --markers <csv> --heatmap
    Or import the functions:
        from vaila.drawsportsfields import plot_field, plot_court, run_drawsportsfields

Examples:
    uv run drawsportsfields.py
    uv run drawsportsfields.py --type tennis
    uv run drawsportsfields.py --type basketball
    uv run drawsportsfields.py --field <path_to_csv>
    uv run drawsportsfields.py --markers <csv> --heatmap

Requirements:
    - Python 3.12
    - pandas, matplotlib, numpy, tkinter, seaborn, rich

License:
    GNU Affero General Public License v3.0
"""

from __future__ import annotations

import math
import os
import tkinter as tk
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from tkinter import Button, Frame, filedialog, messagebox
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from rich import print

# Full FIFA layout reference names required by plot_field() (see models/soccerfield_ref3d.csv)
_SOCCER_FIELD_POINT_NAMES = frozenset(
    {
        "bottom_left_corner",
        "top_left_corner",
        "bottom_right_corner",
        "top_right_corner",
        "midfield_left",
        "midfield_right",
        "center_field",
        "center_circle_top_intersection",
        "center_circle_bottom_intersection",
        "left_goal_bottom_post",
        "left_goal_top_post",
        "right_goal_bottom_post",
        "right_goal_top_post",
        "left_penalty_area_top_left",
        "left_penalty_area_top_right",
        "left_penalty_area_bottom_left",
        "left_penalty_area_bottom_right",
        "left_goal_area_top_left",
        "left_goal_area_top_right",
        "left_goal_area_bottom_left",
        "left_goal_area_bottom_right",
        "left_penalty_spot",
        "left_penalty_arc_top",
        "left_penalty_arc_left_intersection",
        "left_penalty_arc_right_intersection",
        "right_penalty_area_top_left",
        "right_penalty_area_top_right",
        "right_penalty_area_bottom_left",
        "right_penalty_area_bottom_right",
        "right_goal_area_top_left",
        "right_goal_area_top_right",
        "right_goal_area_bottom_left",
        "right_goal_area_bottom_right",
        "right_penalty_spot",
        "right_penalty_arc_top",
        "right_penalty_arc_left_intersection",
        "right_penalty_arc_right_intersection",
    }
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Color Schemes Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Default colors for each sport
SPORT_COLORS = {
    "soccer": {
        "grass": {"playing": "forestgreen", "outer": "green", "lines": "white"},
        "synthetic": {"playing": "#1a5e1a", "outer": "#144d14", "lines": "white"},
        "dirt": {"playing": "#8b4513", "outer": "#6b3410", "lines": "white"},
    },
    "basketball": {
        "wood": {"playing": "#d4a574", "outer": "#b8895c", "lines": "white"},
        "fiba_blue": {"playing": "#2E6DB4", "outer": "#1A4872", "lines": "white"},
        "dark": {"playing": "#404040", "outer": "#262626", "lines": "white"},
    },
    "volleyball": {
        "wood": {"playing": "#f0e6d2", "outer": "#dcc9a8", "lines": "white"},
        "fivb_pro": {"playing": "#2E6DB4", "outer": "#FF8C00", "lines": "white"},  # Blue/Orange
        "synthetic": {"playing": "#e8dcc4", "outer": "#c9b896", "lines": "white"},
    },
    "handball": {
        "blue": {
            "playing": "#2E6DB4",
            "outer": "#1A4872",
            "lines": "white",
            "goal_area": "#b8860b",
        },
        "green": {
            "playing": "#3d7a4a",
            "outer": "#2a5233",
            "lines": "white",
            "goal_area": "#b8860b",
        },
        "wood": {
            "playing": "#d4a574",
            "outer": "#b8895c",
            "lines": "white",
            "goal_area": "#b8860b",
        },
    },
    "futsal": {
        "blue": {"playing": "#1f5fa8", "outer": "#123b6f", "lines": "white"},
        "green": {"playing": "#2d6b3a", "outer": "#1e4a28", "lines": "white"},
        "wood": {"playing": "#d4a574", "outer": "#b8895c", "lines": "white"},
    },
    "tennis": {
        "hard_blue": {"playing": "#2E6DB4", "outer": "#1A4872", "lines": "white"},
        "hard_green": {"playing": "#3A7D44", "outer": "#2C5F35", "lines": "white"},
        "clay": {"playing": "#C2603A", "outer": "#9E4D2E", "lines": "white"},
        "grass": {"playing": "#4B8B3B", "outer": "#3A6B2D", "lines": "white"},
    },
    "generic": {
        "green": {"playing": "forestgreen", "outer": "darkgreen", "lines": "white"},
        "blue": {"playing": "#2E6DB4", "outer": "#1A4872", "lines": "white"},
        "clay": {"playing": "#C2603A", "outer": "#9E4D2E", "lines": "white"},
    },
}


@dataclass(frozen=True)
class SportDef:
    """Registration record for a supported sport surface.

    To add a new sport:
      1. Write ``plot_<sport>_court(df, *, show_reference_points, show_axis_values,
         title, color_scheme) -> (fig, ax)``
      2. Add colour schemes to ``SPORT_COLORS["<sport>"]``
      3. Create ``vaila/models/<sport>_ref3d.csv``
      4. Register one entry in ``SPORT_REGISTRY``
    """

    label: str
    model_csv: str
    title: str
    plot_fn: Callable


# Populated after all plot_* functions are defined (see below SPORT_REGISTRY = {...}).
SPORT_REGISTRY: dict[str, SportDef] = {}

# Backward-compat alias (surface/lines/out keys expected by older callers)
COURT_COLORS = {
    k: {"surface": v["playing"], "lines": v["lines"], "out": v["outer"]}
    for k, v in SPORT_COLORS["tennis"].items()
}


def _get_colors(sport_key: str, scheme_name: str | None = None) -> dict[str, str]:
    """Helper to retrieve colors for a given sport and scheme."""
    sport_key = sport_key.lower()
    if sport_key not in SPORT_COLORS:
        sport_key = "generic"

    schemes = SPORT_COLORS[sport_key]
    if not scheme_name or scheme_name not in schemes:
        # Fallback to the first available scheme for that sport
        scheme_name = list(schemes.keys())[0]

    return schemes[scheme_name]


def draw_line(ax, point1, point2, **kwargs):
    """Draws a line between two points on the specified axis."""
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], **kwargs)


def draw_circle(ax, center, radius, **kwargs):
    """Draws a circle with the specified center and radius."""
    circle = patches.Circle(center, radius, **kwargs)
    ax.add_patch(circle)


def draw_rectangle(ax, bottom_left_corner, width, height, **kwargs):
    """Draws a rectangle with the specified dimensions."""
    rectangle = patches.Rectangle(bottom_left_corner, width, height, **kwargs)
    ax.add_patch(rectangle)


def draw_arc(ax, center, radius, theta1, theta2, **kwargs):
    """Draws an arc with the specified parameters."""
    arc = patches.Arc(center, 2 * radius, 2 * radius, theta1=theta1, theta2=theta2, **kwargs)
    ax.add_patch(arc)


def plot_court(
    df: pd.DataFrame,
    *,
    show_reference_points: bool = True,
    show_axis_values: bool = False,
    title: str = "Tennis Court (ITF 23.77×10.97 m)",
    color_scheme: str | None = None,
):
    """Plot an ITF-standard tennis court.

    Parameters
    ----------
    df : DataFrame
        Reference-point CSV with ``point_name``, ``x``, ``y``, ``point_number``.
    show_reference_points, show_axis_values : bool
        Toggle overlays.
    title : str
        Plot title.
    color_scheme : str | None
        Key into ``SPORT_COLORS["tennis"]`` (e.g. ``"hard_blue"``, ``"clay"``).
        ``None`` defaults to ``"hard_blue"``.

    Returns
    -------
    fig, ax
    """
    scheme_key = color_scheme or "hard_blue"
    sport_schemes = SPORT_COLORS.get("tennis", {})
    raw = sport_schemes.get(scheme_key) or sport_schemes.get("hard_blue") or {}
    colors = {
        "surface": raw.get("playing", "#2E6DB4"),
        "lines": raw.get("lines", "white"),
        "out": raw.get("outer", "#1A4872"),
    }
    line_color = colors["lines"]

    L = 23.77
    W = 10.97
    sa = 1.37
    sl = 6.40
    net_x = L / 2
    svc_left_x = net_x - sl
    svc_right_x = net_x + sl
    singles_y_low = sa
    singles_y_high = W - sa
    center_y = W / 2

    margin = 3.66
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-margin, L + margin)
    ax.set_ylim(-margin, W + margin)
    ax.set_aspect("equal")

    if show_axis_values:
        ax.grid(True, alpha=0.3, color="gray", linestyle="-", linewidth=0.5)
        ax.set_xlabel("X (meters)", fontsize=10)
        ax.set_ylabel("Y (meters)", fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.set_xticks(np.arange(-margin, L + margin + 1, 2))
        ax.set_yticks(np.arange(-margin, W + margin + 1, 2))
    else:
        ax.axis("off")

    # Background
    draw_rectangle(
        ax,
        (-margin, -margin),
        L + 2 * margin,
        W + 2 * margin,
        edgecolor="none",
        facecolor=colors["out"],
        zorder=0,
    )
    draw_rectangle(ax, (0, 0), L, W, edgecolor="none", facecolor=colors["surface"], zorder=0.5)

    lw = 2
    line_kw = {"color": line_color, "linewidth": lw, "zorder": 1}

    # Outer boundary (doubles)
    draw_line(ax, (0, 0), (L, 0), **line_kw)
    draw_line(ax, (0, W), (L, W), **line_kw)
    draw_line(ax, (0, 0), (0, W), **line_kw)
    draw_line(ax, (L, 0), (L, W), **line_kw)

    # Singles sidelines
    draw_line(ax, (0, singles_y_low), (L, singles_y_low), **line_kw)
    draw_line(ax, (0, singles_y_high), (L, singles_y_high), **line_kw)

    # Service lines
    draw_line(ax, (svc_left_x, singles_y_low), (svc_left_x, singles_y_high), **line_kw)
    draw_line(ax, (svc_right_x, singles_y_low), (svc_right_x, singles_y_high), **line_kw)

    # Center service line
    draw_line(ax, (svc_left_x, center_y), (net_x, center_y), **line_kw)
    draw_line(ax, (net_x, center_y), (svc_right_x, center_y), **line_kw)

    # Center marks on baselines
    mark = 0.20
    draw_line(ax, (0, center_y - mark), (0, center_y + mark), **dict(line_kw, linewidth=3))
    draw_line(ax, (L, center_y - mark), (L, center_y + mark), **dict(line_kw, linewidth=3))

    # Net
    draw_line(ax, (net_x, 0), (net_x, W), color=line_color, linewidth=4, zorder=2)
    draw_line(
        ax,
        (net_x, center_y - 0.15),
        (net_x, center_y + 0.15),
        color=line_color,
        linewidth=6,
        zorder=2,
    )

    # Reference point labels
    if show_reference_points and df is not None:
        points = {
            row["point_name"]: (row["x"], row["y"], row["point_number"]) for _, row in df.iterrows()
        }
        for _name, (x, y, num) in points.items():
            ax.text(
                x + 0.15,
                y + 0.15,
                str(num),
                color="black",
                fontsize=7,
                weight="bold",
                bbox={"facecolor": "white", "alpha": 0.7, "boxstyle": "round", "pad": 0.2},
                zorder=10,
            )

    style_label = scheme_key.replace("_", " ").title()
    ax.set_title(f"{title}  [{style_label}]", fontsize=11, pad=8)
    return fig, ax


def plot_field(df, show_reference_points=True, show_axis_values=False, color_scheme=None):
    """
    Plots a soccer field using coordinates from the DataFrame.

    Args:
        df: DataFrame with field point coordinates
        show_reference_points: Whether to show reference point numbers on the field
        show_axis_values: Whether to show numerical values on X and Y axes

    Returns:
        fig, ax: Matplotlib figure and axes with the drawn field
    """
    # Determine field dimensions from DataFrame
    min_x, max_x = df["x"].min(), df["x"].max()
    min_y, max_y = df["y"].min(), df["y"].max()
    field_width = max_x - min_x
    field_height = max_y - min_y

    # Define the margin around the field
    margin = 2

    # Use existing figsize, aspect ratio and dynamic limits will scale content
    fig, ax = plt.subplots(figsize=(10.5 + 0.4, 6.8 + 0.4))
    ax.set_xlim(min_x - margin - 1, max_x + margin + 1)
    ax.set_ylim(min_y - margin - 1, max_y + margin + 1)
    ax.set_aspect("equal")

    if show_axis_values:
        # Show axis values with grid
        ax.grid(True, alpha=0.3, color="gray", linestyle="-", linewidth=0.5)
        ax.set_xlabel("X (meters)", fontsize=10)
        ax.set_ylabel("Y (meters)", fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=8)
        # Set tick intervals based on field size
        x_ticks = np.arange(min_x - margin, max_x + margin + 1, 5)
        y_ticks = np.arange(min_y - margin, max_y + margin + 1, 5)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
    else:
        ax.axis("off")

    # Convert DataFrame to dictionary for easier access
    points = {
        row["point_name"]: (row["x"], row["y"], row["point_number"]) for _, row in df.iterrows()
    }

    # Get colors
    colors = _get_colors("soccer", color_scheme)
    playing_c = colors["playing"]
    outer_c = colors["outer"]
    line_c = colors["lines"]

    # Draw extended area (including margin around the field)
    draw_rectangle(
        ax,
        (min_x - margin, min_y - margin),
        field_width + 2 * margin,
        field_height + 2 * margin,
        edgecolor="none",
        facecolor=outer_c,
        zorder=0,
    )

    # Draw the main playing surface with slightly darker green
    draw_rectangle(
        ax,
        (min_x, min_y),  # Use min_x, min_y from DataFrame
        field_width,  # Use calculated field_width
        field_height,  # Use calculated field_height
        edgecolor="none",
        facecolor=playing_c,
        zorder=0.5,
    )

    # Draw perimeter lines (12cm = 0.12m width)
    # These use named points, should adapt if points are correct in CSV

    # Left sideline
    draw_line(
        ax,
        points["bottom_left_corner"][0:2],
        points["top_left_corner"][0:2],
        color=line_c,
        linewidth=2,
        zorder=1,
    )

    # Right sideline
    draw_line(
        ax,
        points["bottom_right_corner"][0:2],
        points["top_right_corner"][0:2],
        color=line_c,
        linewidth=2,
        zorder=1,
    )

    # Bottom goal line
    draw_line(
        ax,
        points["bottom_left_corner"][0:2],
        points["bottom_right_corner"][0:2],
        color=line_c,
        linewidth=2,
        zorder=1,
    )

    # Top goal line
    draw_line(
        ax,
        points["top_left_corner"][0:2],
        points["top_right_corner"][0:2],
        color=line_c,
        linewidth=2,
        zorder=1,
    )

    # Center line
    draw_line(
        ax,
        points["midfield_left"][0:2],
        points["midfield_right"][0:2],
        color=line_c,
        linewidth=2,
        zorder=1,
    )

    # Center circle - radius derived from points
    center_circle_radius = abs(
        points["center_circle_top_intersection"][1] - points["center_field"][1]
    )
    draw_circle(
        ax,
        points["center_field"][0:2],
        center_circle_radius,
        edgecolor=line_c,
        facecolor="none",
        linewidth=2,
        zorder=1,
    )

    # Center spot - fixed small radius
    center_spot_radius = 0.2
    draw_circle(
        ax,
        points["center_field"][0:2],
        center_spot_radius,
        edgecolor=line_c,
        facecolor=line_c,
        linewidth=2,
        zorder=1,
    )

    # Left penalty area - dimensions from points
    lp_bottom_left = points["left_penalty_area_bottom_left"]
    lp_width = points["left_penalty_area_top_left"][0] - points["left_penalty_area_bottom_left"][0]
    lp_height = (
        points["left_penalty_area_bottom_right"][1] - points["left_penalty_area_bottom_left"][1]
    )
    draw_rectangle(
        ax,
        lp_bottom_left[0:2],
        lp_width,
        lp_height,
        edgecolor=line_c,
        facecolor="none",
        linewidth=2,
        zorder=1,
    )

    # Left goal area - dimensions from points
    lg_bottom_left = points["left_goal_area_bottom_left"]
    lg_width = points["left_goal_area_top_left"][0] - points["left_goal_area_bottom_left"][0]
    lg_height = points["left_goal_area_bottom_right"][1] - points["left_goal_area_bottom_left"][1]
    draw_rectangle(
        ax,
        lg_bottom_left[0:2],
        lg_width,
        lg_height,
        edgecolor=line_c,
        facecolor="none",
        linewidth=2,
        zorder=1,
    )

    # Left penalty spot - fixed small radius
    draw_circle(
        ax,
        points["left_penalty_spot"][0:2],
        0.2,  # Standard penalty spot radius
        edgecolor=line_c,
        facecolor=line_c,
        linewidth=2,
        zorder=1,
    )

    # Left penalty arc - fully dynamic
    l_arc_center = points["left_penalty_spot"][0:2]
    # Calculate radius from spot to one of the intersection points
    l_arc_radius = math.hypot(
        points["left_penalty_arc_left_intersection"][0] - l_arc_center[0],
        points["left_penalty_arc_left_intersection"][1] - l_arc_center[1],
    )

    # Intersection line x-coordinate
    l_intersect_x = points["left_penalty_arc_left_intersection"][
        0
    ]  # Assumes both intersection points share this x

    # Determine angles using the provided intersection points
    # Ensure y-coordinates define bottom and top correctly for angle calculation
    y_intersect_1_l = points["left_penalty_arc_left_intersection"][1]
    y_intersect_2_l = points["left_penalty_arc_right_intersection"][1]

    # Sort y_coords to ensure bottom_angle corresponds to lower y and top_angle to higher y for the arc segment
    l_y_for_bottom_angle = min(y_intersect_1_l, y_intersect_2_l)
    l_y_for_top_angle = max(y_intersect_1_l, y_intersect_2_l)

    l_bottom_angle = math.degrees(
        math.atan2(l_y_for_bottom_angle - l_arc_center[1], l_intersect_x - l_arc_center[0])
    )
    l_top_angle = math.degrees(
        math.atan2(l_y_for_top_angle - l_arc_center[1], l_intersect_x - l_arc_center[0])
    )

    # Normalizing the angles for correct Arc drawing
    # No explicit normalization needed if relying on atan2 range and Arc interpretation.
    # Matplotlib's Arc draws counter-clockwise from theta1 to theta2.
    # For the left arc, the segment should be from the angle corresponding to the lower y-intersection point
    # to the angle corresponding to the higher y-intersection point.

    # Original code logic: theta1=bottom_angle, theta2=top_angle
    # Ensure correct order for drawing the visible segment.
    # After atan2, l_bottom_angle might be > l_top_angle if crossing 0/360 (e.g. 330 and 30)
    # We need the smaller sweep. The default ordering usually works if intersection points are consistent.
    if abs(l_top_angle - l_bottom_angle) > 180:  # If it's the major arc, swap them
        if l_bottom_angle < l_top_angle:
            l_bottom_angle += 360
        else:
            l_top_angle += 360

    # The original fixed code had bottom_angle potentially > top_angle after normalization (e.g., 315 and 45).
    # For Arc, theta1 to theta2 counter-clockwise. For left penalty arc, this implies bottom_angle is theta1.
    draw_arc(
        ax,
        l_arc_center,
        l_arc_radius,
        theta1=l_bottom_angle,
        theta2=l_top_angle,
        edgecolor=line_c,
        linewidth=2,
        zorder=1,
    )

    # Right penalty area - dimensions from points
    rp_anchor = points[
        "right_penalty_area_top_left"
    ]  # This point is (xmin_box, ymin_box) for this box
    rp_width = (
        points["right_penalty_area_bottom_right"][0] - points["right_penalty_area_top_left"][0]
    )
    rp_height = points["right_penalty_area_top_right"][1] - points["right_penalty_area_top_left"][1]
    draw_rectangle(
        ax,
        rp_anchor[0:2],
        rp_width,
        rp_height,
        edgecolor=line_c,
        facecolor="none",
        linewidth=2,
        zorder=1,
    )

    # Right goal area - dimensions from points
    rg_anchor = points["right_goal_area_top_left"]  # This point is (xmin_box, ymin_box)
    rg_width = points["right_goal_area_bottom_right"][0] - points["right_goal_area_top_left"][0]
    rg_height = points["right_goal_area_top_right"][1] - points["right_goal_area_top_left"][1]
    draw_rectangle(
        ax,
        rg_anchor[0:2],
        rg_width,
        rg_height,
        edgecolor=line_c,
        facecolor="none",
        linewidth=2,
        zorder=1,
    )

    # Right penalty spot
    draw_circle(
        ax,
        points["right_penalty_spot"][0:2],
        0.2,  # Standard penalty spot radius
        edgecolor=line_c,
        facecolor=line_c,
        linewidth=2,
        zorder=1,
    )

    # Right penalty arc - fully dynamic
    r_arc_center = points["right_penalty_spot"][0:2]
    r_arc_radius = math.hypot(
        points["right_penalty_arc_left_intersection"][0] - r_arc_center[0],
        points["right_penalty_arc_left_intersection"][1] - r_arc_center[1],
    )
    r_intersect_x = points["right_penalty_arc_left_intersection"][0]

    y_intersect_1_r = points["right_penalty_arc_left_intersection"][1]
    y_intersect_2_r = points["right_penalty_arc_right_intersection"][1]

    # For the right arc, angles are on the left side of the circle (typically 90 to 270 degrees)
    # The "top" intersection point (larger y) will have a smaller angle (e.g., 135 deg)
    # The "bottom" intersection point (smaller y) will have a larger angle (e.g., 225 deg)
    r_y_for_top_angle = max(
        y_intersect_1_r, y_intersect_2_r
    )  # Corresponds to theta1 in original right arc
    r_y_for_bottom_angle = min(
        y_intersect_1_r, y_intersect_2_r
    )  # Corresponds to theta2 in original right arc

    r_top_angle_calc = math.degrees(
        math.atan2(r_y_for_top_angle - r_arc_center[1], r_intersect_x - r_arc_center[0])
    )
    r_bottom_angle_calc = math.degrees(
        math.atan2(r_y_for_bottom_angle - r_arc_center[1], r_intersect_x - r_arc_center[0])
    )

    # Original code logic: theta1=top_angle, theta2=bottom_angle
    # Ensure correct order for drawing the visible segment.
    if abs(r_bottom_angle_calc - r_top_angle_calc) > 180:  # If it's the major arc, swap them
        if r_top_angle_calc < r_bottom_angle_calc:
            r_top_angle_calc += 360
        else:
            r_bottom_angle_calc += 360

    draw_arc(
        ax,
        r_arc_center,
        r_arc_radius,
        theta1=r_top_angle_calc,  # Theta1 is the angle for the "upper" intersection point of the arc segment
        theta2=r_bottom_angle_calc,  # Theta2 is the angle for the "lower" intersection point
        edgecolor=line_c,
        linewidth=2,
        zorder=1,
    )

    # Left goal line - thicker than the other lines
    draw_line(
        ax,
        points["left_goal_bottom_post"][0:2],
        points["left_goal_top_post"][0:2],
        color=line_c,
        linewidth=8,
        zorder=2,
    )

    # Right goal line - thicker than the other lines
    draw_line(
        ax,
        points["right_goal_bottom_post"][0:2],
        points["right_goal_top_post"][0:2],
        color=line_c,
        linewidth=8,
        zorder=2,
    )

    # Add point numbers to the field for reference (only if enabled)
    if show_reference_points:
        for _name, (x, y, num) in points.items():
            # Adjust text offset based on field size for better visibility
            text_offset_x = field_width * 0.005
            text_offset_y = field_height * 0.005
            # Ensure a minimum offset if field is very small
            offset_x = max(text_offset_x, 0.2)
            offset_y = max(text_offset_y, 0.2)
            ax.text(
                x + offset_x,
                y + offset_y,
                str(num),
                color="black",
                fontsize=8,
                weight="bold",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.7,
                    "boxstyle": "round",
                    "pad": 0.2,
                },  # added pad
                zorder=10,
            )

    return fig, ax


def plot_simple_field(
    df: pd.DataFrame,
    *,
    show_reference_points: bool = True,
    show_axis_values: bool = False,
    title: str = "Sports field (reference layout)",
    color_scheme: str | None = None,
    playing_facecolor: str | None = None,
    outer_facecolor: str | None = None,
):
    """Draw a rectangular pitch from CSV corners (non-soccer models).

    Expects at least ``bottom_left_corner``, ``top_left_corner``, ``bottom_right_corner``,
    ``top_right_corner``. Optionally ``midfield_left`` / ``midfield_right`` for a centre line,
    ``center_field`` + ``center_circle_top`` + ``center_circle_bottom`` for a centre circle,
    and volleyball-style ``attack_line_*`` pairs for vertical attack lines.
    """
    for col in ("x", "y", "z"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    min_x, max_x = df["x"].min(), df["x"].max()
    min_y, max_y = df["y"].min(), df["y"].max()
    field_width = max_x - min_x
    field_height = max_y - min_y
    margin = 2

    # Color resolution logic
    colors = _get_colors("generic", color_scheme)
    playing_c = playing_facecolor or colors["playing"]
    outer_c = outer_facecolor or colors["outer"]
    line_c = colors["lines"]

    fig, ax = plt.subplots(figsize=(10.5 + 0.4, 6.8 + 0.4))
    ax.set_xlim(min_x - margin - 1, max_x + margin + 1)
    ax.set_ylim(min_y - margin - 1, max_y + margin + 1)
    ax.set_aspect("equal")

    if show_axis_values:
        ax.grid(True, alpha=0.3, color="gray", linestyle="-", linewidth=0.5)
        ax.set_xlabel("X (meters)", fontsize=10)
        ax.set_ylabel("Y (meters)", fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=8)
    else:
        ax.axis("off")

    draw_rectangle(
        ax,
        (min_x - margin, min_y - margin),
        field_width + 2 * margin,
        field_height + 2 * margin,
        edgecolor="none",
        facecolor=outer_c,
        zorder=0,
    )
    draw_rectangle(
        ax,
        (min_x, min_y),
        field_width,
        field_height,
        edgecolor="none",
        facecolor=playing_c,
        zorder=0.5,
    )

    points = {
        row["point_name"]: (row["x"], row["y"], row["point_number"]) for _, row in df.iterrows()
    }

    req = ("bottom_left_corner", "top_left_corner", "bottom_right_corner", "top_right_corner")
    if not all(k in points for k in req):
        missing = [k for k in req if k not in points]
        raise ValueError(f"Simple field CSV missing required points: {missing}")

    lw = 2
    bl, tl, br, tr = (points[k][:2] for k in req)
    draw_line(ax, bl, tl, color=line_c, linewidth=lw, zorder=1)
    draw_line(ax, br, tr, color=line_c, linewidth=lw, zorder=1)
    draw_line(ax, bl, br, color=line_c, linewidth=lw, zorder=1)
    draw_line(ax, tl, tr, color=line_c, linewidth=lw, zorder=1)

    if "midfield_left" in points and "midfield_right" in points:
        draw_line(
            ax,
            points["midfield_left"][:2],
            points["midfield_right"][:2],
            color=line_c,
            linewidth=lw,
            zorder=1,
        )

    if (
        "center_field" in points
        and "center_circle_top" in points
        and "center_circle_bottom" in points
    ):
        cf = points["center_field"][:2]
        r = math.hypot(
            points["center_circle_top"][0] - cf[0], points["center_circle_top"][1] - cf[1]
        )
        if r > 0:
            draw_circle(ax, cf, r, edgecolor=line_c, facecolor="none", linewidth=lw, zorder=1)

    # Volleyball-style attack lines (vertical, full playing height)
    for prefix in ("attack_line_near", "attack_line_far"):
        k1, k2 = f"{prefix}_left", f"{prefix}_right"
        if k1 in points and k2 in points:
            xv = points[k1][0]
            if abs(points[k2][0] - xv) < 1e-6:
                draw_line(
                    ax,
                    (xv, min_y),
                    (xv, max_y),
                    color=line_c,
                    linewidth=lw,
                    linestyle="--",
                    zorder=1,
                )

    if show_reference_points:
        for _name, (x, y, num) in points.items():
            ox = max(field_width * 0.005, 0.15)
            oy = max(field_height * 0.005, 0.15)
            ax.text(
                x + ox,
                y + oy,
                str(num),
                color="black",
                fontsize=8,
                weight="bold",
                bbox={"facecolor": "white", "alpha": 0.7, "boxstyle": "round", "pad": 0.2},
                zorder=10,
            )

    ax.set_title(title, fontsize=11, pad=8)
    return fig, ax


def _ref_label(
    ax,
    points: dict,
    field_w: float,
    field_h: float,
    *,
    show: bool = True,
):
    """Draw numbered reference-point labels on *ax* when *show* is True."""
    if not show:
        return
    ox = max(field_w * 0.005, 0.15)
    oy = max(field_h * 0.005, 0.15)
    for _name, vals in points.items():
        x, y = vals[0], vals[1]
        num = vals[2] if len(vals) > 2 else ""
        ax.text(
            x + ox,
            y + oy,
            str(num),
            color="black",
            fontsize=8,
            weight="bold",
            bbox={"facecolor": "white", "alpha": 0.7, "boxstyle": "round", "pad": 0.2},
            zorder=10,
        )


def _setup_axes(ax, min_x, max_x, min_y, max_y, margin, *, show_axis_values: bool):
    """Common axes configuration for sport plot functions."""
    ax.set_xlim(min_x - margin - 1, max_x + margin + 1)
    ax.set_ylim(min_y - margin - 1, max_y + margin + 1)
    ax.set_aspect("equal")
    if show_axis_values:
        ax.grid(True, alpha=0.3, color="gray", linestyle="-", linewidth=0.5)
        ax.set_xlabel("X (meters)", fontsize=10)
        ax.set_ylabel("Y (meters)", fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=8)
    else:
        ax.axis("off")


def _parse_points(df: pd.DataFrame) -> dict[str, tuple[float, float, int]]:
    """Build ``{name: (x, y, number)}`` dict from *df*."""
    for col in ("x", "y", "z"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return {
        row["point_name"]: (float(row["x"]), float(row["y"]), int(row["point_number"]))
        for _, row in df.iterrows()
    }


# ---------------------------------------------------------------------------
# Basketball — FIBA 28 × 15 m
# ---------------------------------------------------------------------------


def plot_basketball_court(
    df: pd.DataFrame,
    *,
    show_reference_points: bool = True,
    show_axis_values: bool = False,
    title: str = "Basketball (FIBA 28×15 m)",
    color_scheme: str | None = None,
):
    """Draw a FIBA basketball court with paint, 3-point arcs and free-throw circles."""
    points = _parse_points(df)
    min_x, max_x = df["x"].min(), df["x"].max()
    min_y, max_y = df["y"].min(), df["y"].max()
    court_w = max_x - min_x
    court_h = max_y - min_y
    cx = min_x + court_w / 2
    cy = min_y + court_h / 2
    margin = 2
    lw = 2
    # Get colors
    colors = _get_colors("basketball", color_scheme)
    playing_c = colors["playing"]
    outer_c = colors["outer"]
    line_c = colors["lines"]

    fig, ax = plt.subplots(figsize=(12, 7))
    _setup_axes(ax, min_x, max_x, min_y, max_y, margin, show_axis_values=show_axis_values)

    draw_rectangle(
        ax,
        (min_x - margin, min_y - margin),
        court_w + 2 * margin,
        court_h + 2 * margin,
        edgecolor="none",
        facecolor=outer_c,
        zorder=0,
    )
    draw_rectangle(
        ax,
        (min_x, min_y),
        court_w,
        court_h,
        edgecolor="none",
        facecolor=playing_c,
        zorder=0.5,
    )

    # Boundary
    for p1, p2 in (
        ((min_x, min_y), (min_x, max_y)),
        ((max_x, min_y), (max_x, max_y)),
        ((min_x, min_y), (max_x, min_y)),
        ((min_x, max_y), (max_x, max_y)),
    ):
        draw_line(ax, p1, p2, color=line_c, linewidth=lw, zorder=1)

    # Center line
    draw_line(ax, (cx, min_y), (cx, max_y), color=line_c, linewidth=lw, zorder=1)

    # Center circle (r = 1.80 m)
    draw_circle(ax, (cx, cy), 1.80, edgecolor=line_c, facecolor="none", linewidth=lw, zorder=1)

    # FIBA constants
    paint_half_w = 2.45
    paint_depth = 5.80
    basket_offset = 1.575
    ft_radius = 1.80
    three_r = 6.75
    corner_y_dist = 0.90
    no_charge_r = 1.25
    backboard_offset = 1.20
    backboard_half = 0.90

    for side in ("left", "right"):
        if side == "left":
            end_x = min_x
            sign = 1
            basket_x = end_x + basket_offset
            ft_x = end_x + paint_depth
        else:
            end_x = max_x
            sign = -1
            basket_x = end_x - basket_offset
            ft_x = end_x - paint_depth

        # Paint area
        px = min(end_x, ft_x)
        draw_rectangle(
            ax,
            (px, cy - paint_half_w),
            abs(paint_depth),
            2 * paint_half_w,
            edgecolor=line_c,
            facecolor=outer_c,
            linewidth=lw,
            zorder=0.8,
        )

        # Free-throw semi-circle
        if side == "left":
            theta1, theta2 = 270, 90
        else:
            theta1, theta2 = 90, 270
        draw_arc(
            ax,
            (ft_x, cy),
            ft_radius,
            theta1,
            theta2,
            edgecolor=line_c,
            linewidth=lw,
            zorder=1,
        )
        # Dashed back half of free-throw circle
        back_t1, back_t2 = (90, 270) if side == "left" else (270, 90 + 360)
        draw_arc(
            ax,
            (ft_x, cy),
            ft_radius,
            back_t1,
            back_t2,
            edgecolor=line_c,
            linewidth=1,
            linestyle="--",
            zorder=1,
        )

        # Backboard
        bb_x = end_x + sign * backboard_offset
        draw_line(
            ax,
            (bb_x, cy - backboard_half),
            (bb_x, cy + backboard_half),
            color=line_c,
            linewidth=3,
            zorder=2,
        )

        # Basket ring
        draw_circle(
            ax,
            (basket_x, cy),
            0.225,
            edgecolor="#ff6600",
            facecolor="none",
            linewidth=2,
            zorder=2,
        )

        # No-charge zone (dashed semi-circle)
        draw_arc(
            ax,
            (basket_x, cy),
            no_charge_r,
            theta1,
            theta2,
            edgecolor=line_c,
            linewidth=1,
            linestyle="--",
            zorder=1,
        )

        # 3-point arc
        dy = cy - (min_y + corner_y_dist)
        dx_arc = math.sqrt(max(three_r**2 - dy**2, 0))
        phi = math.atan2(dy, dx_arc)
        if side == "left":
            angles = np.linspace(-phi, phi, 200)
        else:
            # Right side must open to the court interior (towards -x).
            angles = np.linspace(math.pi + phi, math.pi - phi, 200)
        arc_x = basket_x + three_r * np.cos(angles)
        arc_y = cy + three_r * np.sin(angles)
        ax.plot(arc_x, arc_y, color=line_c, linewidth=lw, zorder=1)

        # Corner straight sections
        arc_start_x = float(arc_x[0])
        arc_end_x = float(arc_x[-1])
        bottom_corner_y = min_y + corner_y_dist
        top_corner_y = max_y - corner_y_dist
        draw_line(
            ax,
            (end_x, bottom_corner_y),
            (arc_start_x, bottom_corner_y),
            color=line_c,
            linewidth=lw,
            zorder=1,
        )
        draw_line(
            ax,
            (end_x, top_corner_y),
            (arc_end_x, top_corner_y),
            color=line_c,
            linewidth=lw,
            zorder=1,
        )

    _ref_label(ax, points, court_w, court_h, show=show_reference_points)
    ax.set_title(title, fontsize=11, pad=8)
    return fig, ax


# ---------------------------------------------------------------------------
# Volleyball — FIVB 18 × 9 m
# ---------------------------------------------------------------------------


def plot_volleyball_court(
    df: pd.DataFrame,
    *,
    show_reference_points: bool = True,
    show_axis_values: bool = False,
    title: str = "Volleyball (FIVB 18×9 m)",
    color_scheme: str | None = None,
):
    """Draw a FIVB volleyball court with attack lines and net."""
    points = _parse_points(df)
    min_x, max_x = df["x"].min(), df["x"].max()
    min_y, max_y = df["y"].min(), df["y"].max()
    court_w = max_x - min_x
    court_h = max_y - min_y
    cx = min_x + court_w / 2
    margin = 2
    lw = 2
    # Get colors
    colors = _get_colors("volleyball", color_scheme)
    playing_c = colors["playing"]
    outer_c = colors["outer"]
    line_c = colors["lines"]

    fig, ax = plt.subplots(figsize=(12, 7))
    _setup_axes(ax, min_x, max_x, min_y, max_y, margin, show_axis_values=show_axis_values)

    draw_rectangle(
        ax,
        (min_x - margin, min_y - margin),
        court_w + 2 * margin,
        court_h + 2 * margin,
        edgecolor="none",
        facecolor=outer_c,
        zorder=0,
    )
    draw_rectangle(
        ax,
        (min_x, min_y),
        court_w,
        court_h,
        edgecolor="none",
        facecolor=playing_c,
        zorder=0.5,
    )

    # Boundary
    for p1, p2 in (
        ((min_x, min_y), (min_x, max_y)),
        ((max_x, min_y), (max_x, max_y)),
        ((min_x, min_y), (max_x, min_y)),
        ((min_x, max_y), (max_x, max_y)),
    ):
        draw_line(ax, p1, p2, color=line_c, linewidth=lw, zorder=1)

    # Center line / net
    draw_line(ax, (cx, min_y), (cx, max_y), color=line_c, linewidth=lw + 1, zorder=1)
    ax.plot(
        [cx, cx],
        [min_y - 0.5, max_y + 0.5],
        color="black",
        linewidth=1,
        linestyle="-",
        zorder=1.5,
        alpha=0.6,
    )
    ax.text(cx, max_y + 0.7, "NET", ha="center", fontsize=8, color="black", zorder=10)

    # Net posts
    ax.plot(cx, min_y - 0.3, "k^", markersize=6, zorder=2)
    ax.plot(cx, max_y + 0.3, "k^", markersize=6, zorder=2)

    # Attack lines (3 m from center line)
    attack_dist = 3.0
    for x_off in (cx - attack_dist, cx + attack_dist):
        draw_line(
            ax,
            (x_off, min_y),
            (x_off, max_y),
            color=line_c,
            linewidth=lw,
            linestyle="-",
            zorder=1,
        )

    # Half-court shading
    draw_rectangle(
        ax,
        (min_x, min_y),
        court_w / 2,
        court_h,
        edgecolor="none",
        facecolor=outer_c,
        alpha=0.3,
        zorder=0.6,
    )

    # Service zone marks (small dashes behind end lines)
    sz_len = 0.15
    for end_x in (min_x, max_x):
        sgn = -1 if end_x == min_x else 1
        draw_line(
            ax,
            (end_x + sgn * sz_len, min_y),
            (end_x + sgn * sz_len * 4, min_y),
            color=line_c,
            linewidth=1,
            zorder=1,
        )
        draw_line(
            ax,
            (end_x + sgn * sz_len, max_y),
            (end_x + sgn * sz_len * 4, max_y),
            color=line_c,
            linewidth=1,
            zorder=1,
        )

    _ref_label(ax, points, court_w, court_h, show=show_reference_points)
    ax.set_title(title, fontsize=11, pad=8)
    return fig, ax


# ---------------------------------------------------------------------------
# Handball — IHF 40 × 20 m
# ---------------------------------------------------------------------------


def plot_handball_court(
    df: pd.DataFrame,
    *,
    show_reference_points: bool = True,
    show_axis_values: bool = False,
    title: str = "Handball (IHF 40×20 m)",
    color_scheme: str | None = None,
):
    """Draw an IHF handball court with 6 m / 9 m arcs and goals."""
    points = _parse_points(df)
    min_x, max_x = df["x"].min(), df["x"].max()
    min_y, max_y = df["y"].min(), df["y"].max()
    court_w = max_x - min_x
    court_h = max_y - min_y
    cx = min_x + court_w / 2
    cy = min_y + court_h / 2
    margin = 2
    lw = 2
    # Get colors
    colors = _get_colors("handball", color_scheme)
    playing_c = colors["playing"]
    outer_c = colors["outer"]
    line_c = colors["lines"]
    goal_area_c = colors.get("goal_area", "#b8860b")

    fig, ax = plt.subplots(figsize=(12, 7))
    _setup_axes(ax, min_x, max_x, min_y, max_y, margin, show_axis_values=show_axis_values)

    draw_rectangle(
        ax,
        (min_x - margin, min_y - margin),
        court_w + 2 * margin,
        court_h + 2 * margin,
        edgecolor="none",
        facecolor=outer_c,
        zorder=0,
    )
    draw_rectangle(
        ax,
        (min_x, min_y),
        court_w,
        court_h,
        edgecolor="none",
        facecolor=playing_c,
        zorder=0.5,
    )

    # Boundary
    for p1, p2 in (
        ((min_x, min_y), (min_x, max_y)),
        ((max_x, min_y), (max_x, max_y)),
        ((min_x, min_y), (max_x, min_y)),
        ((min_x, max_y), (max_x, max_y)),
    ):
        draw_line(ax, p1, p2, color=line_c, linewidth=lw, zorder=1)

    # Center line
    draw_line(ax, (cx, min_y), (cx, max_y), color=line_c, linewidth=lw, zorder=1)

    # Goal & area dimensions
    goal_half = 1.5
    goal_area_r = 6.0
    free_throw_r = 9.0
    seven_m = 7.0
    four_m = 4.0
    gk_mark_half = 0.15

    for side in ("left", "right"):
        end_x = min_x if side == "left" else max_x
        sign = 1 if side == "left" else -1
        post_bottom = cy - goal_half
        post_top = cy + goal_half

        # Goal (thick red line on end line)
        draw_line(
            ax,
            (end_x, post_bottom),
            (end_x, post_top),
            color="#cc0000",
            linewidth=5,
            zorder=2,
        )
        # Goal depth indicator
        goal_depth = 1.0
        gd_x = end_x - sign * goal_depth
        draw_line(
            ax, (end_x, post_bottom), (gd_x, post_bottom), color="#cc0000", linewidth=2, zorder=2
        )
        draw_line(ax, (end_x, post_top), (gd_x, post_top), color="#cc0000", linewidth=2, zorder=2)
        draw_line(ax, (gd_x, post_bottom), (gd_x, post_top), color="#cc0000", linewidth=2, zorder=2)

        # 6 m goal-area line (two quarter-circle arcs + connecting segment)
        bottom_arc_center = (end_x, post_bottom)
        top_arc_center = (end_x, post_top)

        if side == "left":
            # Bottom post arc: from y going down along goal line to x going out
            t_b = np.linspace(-math.pi / 2, 0, 80)
            t_t = np.linspace(0, math.pi / 2, 80)
        else:
            t_b = np.linspace(-math.pi / 2, -math.pi, 80)
            t_t = np.linspace(math.pi, math.pi / 2, 80)

        arc_bx = bottom_arc_center[0] + goal_area_r * np.cos(t_b)
        arc_by = bottom_arc_center[1] + goal_area_r * np.sin(t_b)
        arc_tx = top_arc_center[0] + goal_area_r * np.cos(t_t)
        arc_ty = top_arc_center[1] + goal_area_r * np.sin(t_t)

        seg_x = end_x + sign * goal_area_r
        full_x = np.concatenate([arc_bx, [seg_x, seg_x], arc_tx])
        full_y = np.concatenate([arc_by, [post_bottom, post_top], arc_ty])
        ax.fill(full_x, full_y, facecolor=goal_area_c, alpha=0.35, edgecolor="none", zorder=0.7)
        ax.plot(full_x, full_y, color=line_c, linewidth=lw, zorder=1)

        # 9 m free-throw line (dashed, same shape)
        if side == "left":
            t_b9 = np.linspace(-math.pi / 2, 0, 80)
            t_t9 = np.linspace(0, math.pi / 2, 80)
        else:
            t_b9 = np.linspace(-math.pi / 2, -math.pi, 80)
            t_t9 = np.linspace(math.pi, math.pi / 2, 80)

        arc_bx9 = bottom_arc_center[0] + free_throw_r * np.cos(t_b9)
        arc_by9 = bottom_arc_center[1] + free_throw_r * np.sin(t_b9)
        arc_tx9 = top_arc_center[0] + free_throw_r * np.cos(t_t9)
        arc_ty9 = top_arc_center[1] + free_throw_r * np.sin(t_t9)

        seg9_x = end_x + sign * free_throw_r
        full9_x = np.concatenate([arc_bx9, [seg9_x, seg9_x], arc_tx9])
        full9_y = np.concatenate([arc_by9, [post_bottom, post_top], arc_ty9])
        # Clip to court boundaries
        mask = (full9_y >= min_y) & (full9_y <= max_y)
        ax.plot(
            full9_x[mask],
            full9_y[mask],
            color=line_c,
            linewidth=lw,
            linestyle="--",
            zorder=1,
        )

        # 7 m penalty mark
        mark_x = end_x + sign * seven_m
        draw_line(
            ax,
            (mark_x, cy - 0.5),
            (mark_x, cy + 0.5),
            color=line_c,
            linewidth=lw + 1,
            zorder=1,
        )

        # 4 m goalkeeper line
        gk_x = end_x + sign * four_m
        draw_line(
            ax,
            (gk_x, cy - gk_mark_half),
            (gk_x, cy + gk_mark_half),
            color=line_c,
            linewidth=lw + 1,
            zorder=1,
        )

    _ref_label(ax, points, court_w, court_h, show=show_reference_points)
    ax.set_title(title, fontsize=11, pad=8)
    return fig, ax


# ---------------------------------------------------------------------------
# Futsal — FIFA 40 × 20 m
# ---------------------------------------------------------------------------


def plot_futsal_court(
    df: pd.DataFrame,
    *,
    show_reference_points: bool = True,
    show_axis_values: bool = False,
    title: str = "Futsal (FIFA 40×20 m)",
    color_scheme: str | None = None,
):
    """Draw a FIFA futsal court with penalty arcs, center circle and goals."""
    points = _parse_points(df)
    min_x, max_x = df["x"].min(), df["x"].max()
    min_y, max_y = df["y"].min(), df["y"].max()
    court_w = max_x - min_x
    court_h = max_y - min_y
    cx = min_x + court_w / 2
    cy = min_y + court_h / 2
    margin = 2
    lw = 2
    # Get colors
    colors = _get_colors("futsal", color_scheme)
    playing_c = colors["playing"]
    outer_c = colors["outer"]
    line_c = colors["lines"]

    fig, ax = plt.subplots(figsize=(12, 7))
    _setup_axes(ax, min_x, max_x, min_y, max_y, margin, show_axis_values=show_axis_values)

    draw_rectangle(
        ax,
        (min_x - margin, min_y - margin),
        court_w + 2 * margin,
        court_h + 2 * margin,
        edgecolor="none",
        facecolor=outer_c,
        zorder=0,
    )
    draw_rectangle(
        ax,
        (min_x, min_y),
        court_w,
        court_h,
        edgecolor="none",
        facecolor=playing_c,
        zorder=0.5,
    )

    # Boundary
    for p1, p2 in (
        ((min_x, min_y), (min_x, max_y)),
        ((max_x, min_y), (max_x, max_y)),
        ((min_x, min_y), (max_x, min_y)),
        ((min_x, max_y), (max_x, max_y)),
    ):
        draw_line(ax, p1, p2, color=line_c, linewidth=lw, zorder=1)

    # Center line
    draw_line(ax, (cx, min_y), (cx, max_y), color=line_c, linewidth=lw, zorder=1)

    # Center circle (r = 3 m)
    draw_circle(ax, (cx, cy), 3.0, edgecolor=line_c, facecolor="none", linewidth=lw, zorder=1)

    # Center spot
    draw_circle(ax, (cx, cy), 0.10, edgecolor=line_c, facecolor=line_c, linewidth=1, zorder=2)

    # Goal & penalty area constants
    goal_half = 1.5
    penalty_r = 6.0
    penalty_mark = 6.0
    second_penalty = 10.0
    corner_r = 0.25

    for side in ("left", "right"):
        end_x = min_x if side == "left" else max_x
        sign = 1 if side == "left" else -1
        post_bottom = cy - goal_half
        post_top = cy + goal_half

        # Goal (red)
        draw_line(
            ax,
            (end_x, post_bottom),
            (end_x, post_top),
            color="#cc0000",
            linewidth=5,
            zorder=2,
        )
        goal_depth = 0.80
        gd_x = end_x - sign * goal_depth
        draw_line(
            ax, (end_x, post_bottom), (gd_x, post_bottom), color="#cc0000", linewidth=2, zorder=2
        )
        draw_line(ax, (end_x, post_top), (gd_x, post_top), color="#cc0000", linewidth=2, zorder=2)
        draw_line(ax, (gd_x, post_bottom), (gd_x, post_top), color="#cc0000", linewidth=2, zorder=2)

        # Penalty area (quarter circles from each post + connecting line)
        if side == "left":
            t_b = np.linspace(-math.pi / 2, 0, 80)
            t_t = np.linspace(0, math.pi / 2, 80)
        else:
            t_b = np.linspace(-math.pi / 2, -math.pi, 80)
            t_t = np.linspace(math.pi, math.pi / 2, 80)

        arc_bx = end_x + penalty_r * np.cos(t_b)
        arc_by = post_bottom + penalty_r * np.sin(t_b)
        arc_tx = end_x + penalty_r * np.cos(t_t)
        arc_ty = post_top + penalty_r * np.sin(t_t)

        seg_x = end_x + sign * penalty_r
        full_x = np.concatenate([arc_bx, [seg_x, seg_x], arc_tx])
        full_y = np.concatenate([arc_by, [post_bottom, post_top], arc_ty])
        ax.plot(full_x, full_y, color=line_c, linewidth=lw, zorder=1)

        # Penalty mark (6 m)
        pm_x = end_x + sign * penalty_mark
        draw_circle(ax, (pm_x, cy), 0.10, edgecolor=line_c, facecolor=line_c, linewidth=1, zorder=2)

        # Second penalty mark (10 m)
        sp_x = end_x + sign * second_penalty
        draw_circle(ax, (sp_x, cy), 0.10, edgecolor=line_c, facecolor=line_c, linewidth=1, zorder=2)

        # Substitution zone (5 m each side of center line)
        sub_len = 0.80
        for sub_y in (min_y, max_y):
            y_sign = -1 if sub_y == min_y else 1
            for dx in (-5, 5):
                sx = cx + dx
                draw_line(
                    ax,
                    (sx, sub_y),
                    (sx, sub_y + y_sign * sub_len),
                    color=line_c,
                    linewidth=1,
                    zorder=1,
                )

    # Corner arcs
    for corner_x, corner_y in (
        (min_x, min_y),
        (min_x, max_y),
        (max_x, min_y),
        (max_x, max_y),
    ):
        t1 = 0 if corner_x == min_x else 180
        if corner_y == min_y:
            t1_adj = t1
            t2_adj = t1 + 90
        else:
            t1_adj = t1 - 90 if corner_x == min_x else t1
            t2_adj = t1_adj + 90
        if corner_x == min_x and corner_y == min_y:
            t1_adj, t2_adj = 0, 90
        elif corner_x == max_x and corner_y == min_y:
            t1_adj, t2_adj = 90, 180
        elif corner_x == max_x and corner_y == max_y:
            t1_adj, t2_adj = 180, 270
        else:
            t1_adj, t2_adj = 270, 360
        draw_arc(
            ax,
            (corner_x, corner_y),
            corner_r,
            t1_adj,
            t2_adj,
            edgecolor=line_c,
            linewidth=1,
            zorder=1,
        )

    _ref_label(ax, points, court_w, court_h, show=show_reference_points)
    ax.set_title(title, fontsize=11, pad=8)
    return fig, ax


# ---------------------------------------------------------------------------
# Sport Registry — single place to register a new sport
# ---------------------------------------------------------------------------

SPORT_REGISTRY.update(
    {
        "soccer": SportDef(
            label="Soccer Field Visualization",
            model_csv="soccerfield_ref3d.csv",
            title="Soccer field (FIFA 105×68 m)",
            plot_fn=plot_field,
        ),
        "tennis": SportDef(
            label="Tennis Court Visualization",
            model_csv="tenniscourt_ref3d.csv",
            title="Tennis Court (ITF 23.77×10.97 m)",
            plot_fn=plot_court,
        ),
        "basketball": SportDef(
            label="Basketball Court Visualization",
            model_csv="basketballcourt_ref3d.csv",
            title="Basketball (FIBA 28×15 m)",
            plot_fn=plot_basketball_court,
        ),
        "volleyball": SportDef(
            label="Volleyball Court Visualization",
            model_csv="volleyball_ref3d.csv",
            title="Volleyball (FIVB 18×9 m)",
            plot_fn=plot_volleyball_court,
        ),
        "futsal": SportDef(
            label="Futsal Court Visualization",
            model_csv="futsal_ref3d.csv",
            title="Futsal (FIFA 40×20 m)",
            plot_fn=plot_futsal_court,
        ),
        "handball": SportDef(
            label="Handball Court Visualization",
            model_csv="handball_ref3d.csv",
            title="Handball (IHF 40×20 m)",
            plot_fn=plot_handball_court,
        ),
    }
)


def _detect_sport(csv_path: str, df: pd.DataFrame) -> str | None:
    """Detect sport from CSV content (soccer) or filename (all others)."""
    names = set(df["point_name"].astype(str))
    if _SOCCER_FIELD_POINT_NAMES.issubset(names):
        return "soccer"
    base = Path(csv_path).stem.lower()
    for key in SPORT_REGISTRY:
        if key in base:
            return key
    return None


def load_and_plot_markers(
    field_ax,
    csv_path,
    canvas,
    manual_marker_artists_ref,
    frame_markers_ref,
    current_frame_ref,
    selected_markers=None,
):
    """
    Loads data from a CSV file and plots numbered markers with paths.
    All frames are plotted on the same image (hold on).

    Args:
        field_ax: Matplotlib axes of the field
        csv_path: Path to the CSV file with x,y coordinates
        canvas: Matplotlib canvas for updates
        manual_marker_artists_ref: Reference to the list of manual marker artists
        frame_markers_ref: Reference to the dictionary of frame markers
        current_frame_ref: Reference to the list containing the current frame number
        selected_markers: List of marker names to display (None for all)
    """

    # --- BEGIN MODIFICATION: Clear previous markers (CSV and manual) ---
    if field_ax:
        artists_to_remove = []
        for artist in field_ax.get_children():
            if hasattr(artist, "get_zorder"):
                z_order = artist.get_zorder()
                # Z-orders for CSV markers are 50, 51, 52. Manual are >= 100.
                if (z_order >= 50 and z_order < 100) or z_order >= 100:
                    artists_to_remove.append(artist)

        for artist in artists_to_remove:
            artist.remove()

    # Clear manual marker data structures as well, since CSV selection takes precedence
    manual_marker_artists_ref.clear()
    frame_markers_ref.clear()
    current_frame_ref[0] = 0
    # --- END MODIFICATION ---

    # Load CSV
    markers_df = pd.read_csv(csv_path)

    print(f"File loaded: {csv_path}")
    print(f"Number of frames (rows): {len(markers_df)}")

    # Data cleaning - convert empty strings to NaN
    markers_df = markers_df.replace("", np.nan)

    # Identify all coordinate columns (except 'frame')
    cols = markers_df.columns
    marker_names = set()
    for col in cols:
        if col != "frame" and ("_x" in col or "_y" in col):
            marker_names.add(col.split("_")[0])

    marker_names = sorted(marker_names)
    print(f"Markers found: {len(marker_names)}")

    # Store all available markers for the selection dialog
    field_ax._all_marker_names = marker_names

    # Define distinct colors for each marker
    colors = plt.cm.rainbow(np.linspace(0, 1, len(marker_names)))

    # For each marker
    for idx, marker in enumerate(marker_names):
        # Skip if not in selected markers
        if selected_markers is not None and marker not in selected_markers:
            continue

        x_col = f"{marker}_x"
        y_col = f"{marker}_y"

        if x_col in cols and y_col in cols:
            # Filter only valid coordinates
            valid_data = markers_df[[x_col, y_col]].dropna()

            if len(valid_data) > 0:
                # Check if coordinates are within reasonable limits
                valid_mask = (
                    (valid_data[x_col] < 120)
                    & (valid_data[x_col] > -20)
                    & (valid_data[y_col] < 80)
                    & (valid_data[y_col] > -20)
                )

                valid_x = valid_data.loc[valid_mask, x_col].values
                valid_y = valid_data.loc[valid_mask, y_col].values

                if len(valid_x) > 0:
                    # Plot trajectory (line)
                    field_ax.plot(
                        valid_x,
                        valid_y,
                        "-",
                        color=colors[idx],
                        linewidth=1.5,
                        alpha=0.7,
                        zorder=50,
                    )

                    # Plot points
                    field_ax.scatter(
                        valid_x,
                        valid_y,
                        color=colors[idx],
                        s=60,
                        marker="o",
                        edgecolor="black",
                        linewidth=1,
                        alpha=0.8,
                        zorder=51,
                    )

                    # Add label to the last valid point
                    field_ax.text(
                        valid_x[-1] + 0.5,
                        valid_y[-1] + 0.5,
                        # Substituir 'M' por 'p' se o nome do marcador começar com 'M'
                        marker.replace("M", "p") if marker.startswith("M") else marker,
                        fontsize=7,
                        color="black",
                        weight="bold",
                        bbox={
                            "facecolor": colors[idx],
                            "alpha": 0.7,
                            "edgecolor": "black",
                            "boxstyle": "round",
                            "pad": 0.1,
                        },
                        zorder=52,
                    )

                    print(f"Plotted marker {marker} with {len(valid_x)} points")

    # Update canvas once at the end
    canvas.draw()

    if selected_markers:
        print(
            f"Plotting complete - Displaying {len(selected_markers)} of {len(marker_names)} markers"
        )
    else:
        print("Plotting complete - all frames drawn on the same image")


def load_and_plot_scout_events(
    field_ax,
    csv_path,
    canvas,
    manual_marker_artists_ref,
    frame_markers_ref,
    current_frame_ref,
    selected_teams=None,
    selected_players=None,
    selected_actions=None,
):
    """
    Loads scout_vaila CSV data and plots events on the soccer field.

    Args:
        field_ax: Matplotlib axes of the field
        csv_path: Path to the CSV file with scout_vaila data
        canvas: Matplotlib canvas for updates
        manual_marker_artists_ref: Reference to the list of manual marker artists
        frame_markers_ref: Reference to the dictionary of frame markers
        current_frame_ref: Reference to the list containing the current frame number
        selected_teams: List of team names to display (None for all)
        selected_players: List of player numbers to display (None for all)
        selected_actions: List of action names to display (None for all)
    """

    # Clear previous markers (CSV and manual)
    if field_ax:
        artists_to_remove = []
        for artist in field_ax.get_children():
            if hasattr(artist, "get_zorder"):
                z_order = artist.get_zorder()
                # Z-orders for CSV markers are 50, 51, 52. Manual are >= 100.
                if (z_order >= 50 and z_order < 100) or z_order >= 100:
                    artists_to_remove.append(artist)

        for artist in artists_to_remove:
            artist.remove()

    # Clear manual marker data structures
    manual_marker_artists_ref.clear()
    frame_markers_ref.clear()
    current_frame_ref[0] = 0

    # Load CSV
    try:
        events_df = pd.read_csv(csv_path)
        print(f"Scout CSV loaded: {csv_path}")
        print(f"Number of events: {len(events_df)}")

        # Check if this is a scout_vaila CSV by looking for required columns
        required_columns = {
            "timestamp_s",
            "team",
            "player",
            "action",
            "result",
            "pos_x_m",
            "pos_y_m",
        }
        if not required_columns.issubset(events_df.columns):
            print("Warning: This doesn't appear to be a scout_vaila CSV file.")
            print("Expected columns:", required_columns)
            print("Found columns:", set(events_df.columns))
            return

    except Exception as e:
        print(f"Error loading scout CSV: {e}")
        return

    # Data cleaning - convert empty strings to NaN
    events_df = events_df.replace("", np.nan)

    # Apply filters
    filtered_df = events_df.copy()

    if selected_teams:
        filtered_df = filtered_df[filtered_df["team"].isin(selected_teams)]

    if selected_players:
        filtered_df = filtered_df[
            filtered_df["player"].astype(str).isin([str(p) for p in selected_players])
        ]

    if selected_actions:
        filtered_df = filtered_df[filtered_df["action"].isin(selected_actions)]

    if len(filtered_df) == 0:
        print("No events match the selected filters.")
        return

    # Get unique teams and players for color mapping
    teams = filtered_df["team"].unique()
    players = filtered_df["player"].unique()

    # Define colors for teams
    team_colors = {
        "HOME": "#1f77b4",  # blue
        "AWAY": "#d62728",  # red
    }

    # Generate additional colors for custom team names
    color_palette = plt.cm.Set3(np.linspace(0, 1, max(len(teams), 10)))
    for i, team in enumerate(teams):
        if team not in team_colors:
            team_colors[team] = color_palette[i % len(color_palette)]

    # Result colors
    result_colors = {
        "success": "#00FF00",  # bright green
        "fail": "#FF0000",  # bright red
        "neutral": "#FFFF00",  # bright yellow
    }

    # Plot events
    for _, event in filtered_df.iterrows():
        try:
            x = float(event["pos_x_m"])
            y = float(event["pos_y_m"])
            team = str(event["team"])
            player = str(event["player"])
            action = str(event["action"])
            result = str(event["result"]).lower()
            timestamp = float(event["timestamp_s"])

            # Skip invalid coordinates
            if pd.isna(x) or pd.isna(y):
                continue

            # Get colors
            team_color = team_colors.get(team, "#808080")
            result_color = result_colors.get(result, "#808080")

            # Draw player circle
            circle = patches.Circle(
                (x, y),
                radius=0.6,
                facecolor=team_color,
                edgecolor=result_color,
                linewidth=2.0,
                zorder=50,
            )
            field_ax.add_patch(circle)

            # Add player number
            field_ax.text(
                x,
                y,
                player,
                color="white",
                ha="center",
                va="center",
                fontsize=8,
                weight="bold",
                zorder=51,
            )

            # Add action symbol (top-right of player circle)
            action_symbol = "o"  # default symbol
            symbol_color = "#FFD700"  # default color

            # Map common actions to symbols
            action_symbols = {
                "pass": "o",
                "shot": "*",
                "dribble": "D",
                "tackle": "x",
                "interception": "X",
                "header": "^",
                "cross": "+",
                "control": "s",
                "first touch": "P",
                "shield": "h",
                "goalkeeping": "s",
            }

            action_symbol = action_symbols.get(action.lower(), "o")

            # Add symbol
            symbol_x = x + 0.8
            symbol_y = y + 0.8

            if action_symbol in ["+", "x", "X"]:
                # Unfilled markers - use larger size
                field_ax.scatter(
                    [symbol_x],
                    [symbol_y],
                    s=120,
                    c=symbol_color,
                    marker=action_symbol,
                    linewidth=2,
                    edgecolors="black",
                    zorder=52,
                )
            else:
                # Filled markers
                field_ax.scatter(
                    [symbol_x],
                    [symbol_y],
                    s=90,
                    c=symbol_color,
                    edgecolors="black",
                    linewidth=1,
                    marker=action_symbol,
                    zorder=52,
                )

            # Add timestamp label (small, bottom-left of player circle)
            time_str = f"{timestamp:.1f}s"
            field_ax.text(
                x - 0.8,
                y - 0.8,
                time_str,
                color="black",
                fontsize=6,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "black", "alpha": 0.8},
                zorder=53,
            )

        except Exception as e:
            print(f"Error plotting event: {e}")
            continue

    # Update canvas
    canvas.draw()

    print(f"Plotted {len(filtered_df)} scout events")
    print(f"Teams: {list(teams)}")
    print(f"Players: {list(players)}")
    print(f"Actions: {list(filtered_df['action'].unique())}")


def run_soccerfield(
    initial_field_csv: str | None = None,
    *,
    default_field_csv: str | None = None,
    window_title: str = "Sports Field Visualization",
    help_html: str = "sports_fields_courts.html",
):
    """Main function to run the soccerfield.py script with GUI controls.

    Parameters
    ----------
    initial_field_csv
        If set, load this field model on startup (e.g. basketball / volleyball CSV).
    default_field_csv
        Default model loaded when ``initial_field_csv`` is not provided.
        Falls back to ``models/soccerfield_ref3d.csv``.
    window_title
        Tk window title.
    help_html
        Help HTML filename from ``vaila/help`` opened by the Help button.
    """
    print(f"Running script: {os.path.basename(__file__)} — {window_title}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    # Create main Tkinter window
    root = tk.Tk()
    root.title(window_title)
    root.geometry("1200x800")

    # Create frame for buttons
    button_frame = Frame(root)
    button_frame.pack(side=tk.TOP, fill=tk.X)

    # Variables to store current axes and canvas
    current_ax = [None]
    current_canvas = [None]
    show_reference_points = [True]  # Boolean state for reference points visibility
    show_axis_values = [False]  # Boolean state for axis values visibility
    current_field_csv = [None]  # Store the current field CSV path
    current_field_title = ["Sports field"]  # Active field title for UI labels
    current_field_color = ["default"]  # Current appearance scheme
    current_markers_csv = [None]  # Store the current markers CSV path
    current_scout_csv = [None]  # Store the current scout CSV path
    selected_markers = [None]  # Store currently selected markers
    selected_teams = [None]  # Store currently selected teams for scout data
    selected_players = [None]  # Store currently selected players for scout data
    selected_actions = [None]  # Store currently selected actions for scout data

    # Variables for manual marker creation
    manual_marker_mode = [False]  # Whether manual marker mode is active
    current_marker_number = [1]  # Número do marcador atual
    current_frame = [0]  # Frame atual
    frame_markers = {}  # Dicionário para armazenar marcadores por frame: {frame: {marker_num: (x, y)}}
    manual_marker_artists = []  # Lista para armazenar objetos visuais dos marcadores

    def load_field(custom_file=None):
        """Loads and displays the soccer field"""
        try:
            # Get the CSV path - either default or custom
            if custom_file:
                csv_path = custom_file
            else:
                models_dir = os.path.join(os.path.dirname(__file__), "models")
                if default_field_csv and os.path.isfile(default_field_csv):
                    csv_path = default_field_csv
                else:
                    csv_path = os.path.join(models_dir, "soccerfield_ref3d.csv")

            # Store the current CSV path for later redraws
            current_field_csv[0] = csv_path

            # Read CSV file
            df = pd.read_csv(csv_path)
            print(f"Reading field data from {csv_path}")
            print(f"Number of reference points: {len(df)}")

            color_s = current_field_color[0] if current_field_color[0] != "default" else None
            sport = _detect_sport(csv_path, df)

            if sport and sport in SPORT_REGISTRY:
                config = SPORT_REGISTRY[sport]
                fig, ax = config.plot_fn(
                    df,
                    show_reference_points=show_reference_points[0],
                    show_axis_values=show_axis_values[0],
                    color_scheme=color_s,
                )
                current_field_title[0] = config.title
            else:
                fig, ax = plot_simple_field(
                    df,
                    show_reference_points=show_reference_points[0],
                    show_axis_values=show_axis_values[0],
                    title="Sports field",
                    color_scheme=color_s,
                )
                current_field_title[0] = "Sports field"

            # Save current axis for later use
            current_ax[0] = ax

            # Clear existing plot frame, if any
            for widget in plot_frame.winfo_children():
                widget.destroy()

            # Embed plot in Tkinter
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            # Save current canvas for later use
            current_canvas[0] = canvas

            # Add navigation toolbar
            toolbar = NavigationToolbar2Tk(canvas, plot_frame)
            toolbar.update()

            # Clear manual markers when loading a new field
            frame_markers.clear()
            manual_marker_artists.clear()

            # Setup event handlers for manual marker mode
            setup_manual_marker_events(canvas)

            print("Field plotted successfully!")

            # If we had markers loaded before, reload them
            if current_markers_csv[0]:
                load_and_plot_markers(
                    current_ax[0],
                    current_markers_csv[0],
                    current_canvas[0],
                    manual_marker_artists,
                    frame_markers,
                    current_frame,
                    selected_markers[0],
                )
                # Enable the marker selection button
                select_markers_button.config(state=tk.NORMAL)
            elif current_scout_csv[0]:
                load_and_plot_scout_events(
                    current_ax[0],
                    current_scout_csv[0],
                    current_canvas[0],
                    manual_marker_artists,
                    frame_markers,
                    current_frame,
                    selected_teams[0],
                    selected_players[0],
                    selected_actions[0],
                )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load field: {e}")
            import traceback

            traceback.print_exc()

    def change_field_color():
        """Cycle through available color schemes for the current sport."""
        if not current_field_csv[0]:
            return

        base = Path(current_field_csv[0]).stem.lower()
        sport: str | None = None
        for key in SPORT_REGISTRY:
            if key in base:
                sport = key
                break
        if not sport:
            sport = "generic"

        schemes = list(SPORT_COLORS.get(sport, SPORT_COLORS["generic"]).keys())
        current = current_field_color[0]

        try:
            next_idx = (schemes.index(current) + 1) % len(schemes)
        except ValueError:
            next_idx = 0

        current_field_color[0] = schemes[next_idx]
        print(f"Switching {sport} to color scheme: {current_field_color[0]}")
        load_field(current_field_csv[0])

    def load_custom_field():
        """Opens dialog to select a custom field CSV file"""
        csv_path = filedialog.askopenfilename(
            title="Select CSV file with field coordinates",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )

        if csv_path:
            load_field(custom_file=csv_path)

    def toggle_reference_points():
        """Toggle the visibility of reference point numbers on the field"""
        show_reference_points[0] = not show_reference_points[0]

        if show_reference_points[0]:
            ref_points_button.config(text="Hide Reference Points")
        else:
            ref_points_button.config(text="Show Reference Points")

        # Reload the field with the new setting
        if current_field_csv[0]:
            load_field(custom_file=current_field_csv[0])

        # Re-plot markers if any were loaded
        if current_markers_csv[0]:
            load_and_plot_markers(
                current_ax[0],
                current_markers_csv[0],
                current_canvas[0],
                manual_marker_artists,
                frame_markers,
                current_frame,
                selected_markers[0],
            )

    def toggle_axis_values():
        """Toggle the visibility of axis values on the field"""
        show_axis_values[0] = not show_axis_values[0]

        if show_axis_values[0]:
            axis_values_button.config(text="Hide Axis Values")
        else:
            axis_values_button.config(text="Show Axis Values")

        # Reload the field with the new setting
        if current_field_csv[0]:
            load_field(custom_file=current_field_csv[0])

        # Re-plot markers if any were loaded
        if current_markers_csv[0]:
            load_and_plot_markers(
                current_ax[0],
                current_markers_csv[0],
                current_canvas[0],
                manual_marker_artists,
                frame_markers,
                current_frame,
                selected_markers[0],
            )

    def load_markers_csv():
        """Opens dialog to select marker CSV and plot it"""
        if current_ax[0] is None or current_canvas[0] is None:
            print("Please load the field first.")
            return

        # Open dialog to select file
        csv_path = filedialog.askopenfilename(
            title="Select CSV file with marker coordinates",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )

        if not csv_path:
            return

        try:
            print(f"\nStarting file loading: {csv_path}")
            # Check if file exists and can be read
            with open(csv_path) as f:
                first_line = f.readline()
                print(f"First line of file: {first_line}")

            # Store the markers path for potential reloads
            current_markers_csv[0] = csv_path

            # Reset selected markers when loading a new file
            selected_markers[0] = None

            # Use stored canvas
            load_and_plot_markers(
                current_ax[0],
                csv_path,
                current_canvas[0],
                manual_marker_artists,
                frame_markers,
                current_frame,
                selected_markers[0],
            )

            # Enable the marker selection button
            select_markers_button.config(state=tk.NORMAL)

        except Exception as e:
            print(f"Error plotting markers: {str(e)}")
            import traceback

            traceback.print_exc()

    def open_marker_selection_dialog():
        """Opens a dialog to select markers using Listbox (multi-select)"""
        import pandas as pd

        if not current_markers_csv[0] or not os.path.exists(current_markers_csv[0]):
            messagebox.showerror("Error", "No marker CSV loaded. Load a CSV first.")
            return

        markers_df = pd.read_csv(current_markers_csv[0])
        marker_names = sorted(
            {col.split("_")[0] for col in markers_df.columns if "_x" in col or "_y" in col}
        )

        if not marker_names:
            messagebox.showerror("Error", "No markers found in the loaded CSV.")
            return

        select_window = tk.Toplevel(root)
        select_window.title("Select Markers to Display")
        select_window.geometry("300x400")

        frame = Frame(select_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        # Add markers to listbox
        for marker in marker_names:
            listbox.insert(tk.END, marker)

        # Set initial selection (previous selection or all)
        initial_selection = selected_markers[0] if selected_markers[0] else marker_names
        for i, marker in enumerate(marker_names):
            if marker in initial_selection:
                listbox.selection_set(i)

        button_frame = Frame(select_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        def select_all():
            listbox.select_set(0, tk.END)

        def deselect_all():
            listbox.selection_clear(0, tk.END)

        def apply_selection():
            selections = [listbox.get(i) for i in listbox.curselection()]
            if not selections:
                selected_markers[0] = None  # Muda de [] para None
                messagebox.showinfo("Info", "No markers selected. None will be displayed.")
            else:
                selected_markers[0] = selections
                print(f"Markers selected: {selections}")

            # Redraw with selected markers only
            load_and_plot_markers(
                current_ax[0],
                current_markers_csv[0],
                current_canvas[0],
                manual_marker_artists,
                frame_markers,
                current_frame,
                selected_markers[0] if selected_markers[0] else [],
            )
            current_canvas[0].draw()
            select_window.destroy()

        # Buttons
        tk.Button(
            button_frame,
            text="Select All",
            command=select_all,
            bg="#4CAF50",
            fg="white",
        ).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(
            button_frame,
            text="Deselect All",
            command=deselect_all,
            bg="#f44336",
            fg="white",
        ).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(
            button_frame,
            text="Apply",
            command=apply_selection,
            bg="#2196F3",
            fg="white",
        ).pack(side=tk.RIGHT, padx=5, pady=5)

        select_window.transient(root)
        select_window.grab_set()

    def toggle_manual_marker_mode():
        """Toggle the manual marker creation mode"""
        manual_marker_mode[0] = not manual_marker_mode[0]

        if manual_marker_mode[0]:
            manual_marker_button.config(text="Disable Manual Markers")
            print("Manual marker mode enabled. Left-click to add, right-click to delete.")
            print("Hold Shift + left-click to create next marker number.")
            print("Press Ctrl+S to save markers to CSV.")
        else:
            manual_marker_button.config(text="Create Manual Markers")
            print("Manual marker mode disabled.")

    def create_marker(event):
        """Create a marker at the clicked position"""
        if not manual_marker_mode[0] or current_ax[0] is None:
            return

        try:
            # Obtém o eixo atual
            ax = current_ax[0]

            # Usamos a transformação padrão do matplotlib para a coordenada X
            x, y_incorreto = ax.transData.inverted().transform((event.x, event.y))

            # Para a coordenada Y, aplicamos a correção para o espelhamento
            y_min, y_max = ax.get_ylim()
            y = y_min + y_max - y_incorreto
            y += 0.8  # Ajuste para alinhar com a ponta do cursor

            # Verificar se o ponto está dentro dos limites do campo
            if x < -5 or x > 110 or y < -5 or y > 73:
                print(f"Point ({x:.2f}, {y:.2f}) is outside field boundaries.")
                return

            # Verificar se a tecla shift está pressionada para incrementar o número do marcador
            if event.state & 0x1:  # Shift key is pressed
                current_marker_number[0] += 1
                print(f"Creating new marker number: {current_marker_number[0]}")

            # Armazenar o marcador no frame atual
            marker_num = current_marker_number[0]
            current_frame_idx = current_frame[0]

            # Criar entrada para o frame atual se não existir
            if current_frame_idx not in frame_markers:
                frame_markers[current_frame_idx] = {}

            # Adicionar coordenadas para este marcador no frame atual
            frame_markers[current_frame_idx][marker_num] = (x, y)

            # Utilizando draw_circle para garantir centro preciso
            circle = patches.Circle(
                (x, y),
                radius=0.4,
                color=plt.cm.tab10(marker_num % 10),
                edgecolor="black",
                linewidth=1,
                alpha=0.8,
                zorder=100,
            )
            ax.add_patch(circle)

            # Posicionar o texto com tamanho reduzido
            text = ax.text(
                x + 0.5,
                y + 0.5,
                f"p{marker_num}",
                fontsize=7,
                color="black",
                weight="bold",
                bbox={
                    "facecolor": plt.cm.tab10(marker_num % 10),
                    "alpha": 0.7,
                    "edgecolor": "black",
                    "boxstyle": "round",
                    "pad": 0.1,
                },
                zorder=101,
            )

            # Armazenar os objetos para possível exclusão
            manual_marker_artists.append((circle, text, x, y, marker_num, current_frame_idx))

            # Atualizar o canvas
            current_canvas[0].draw()
            print(
                f"Created marker p{marker_num} at frame {current_frame_idx}, position ({x:.2f}, {y:.2f})"
            )

            # Avançar para o próximo frame
            current_frame[0] += 1

        except Exception as e:
            print(f"Error creating marker: {e}")
            import traceback

            traceback.print_exc()

    def delete_marker(event):
        """Delete a marker at the clicked position"""
        if not manual_marker_mode[0] or current_ax[0] is None:
            return

        try:
            # Obtém o eixo atual
            ax = current_ax[0]

            # Mesma lógica da função create_marker
            x, y_incorreto = ax.transData.inverted().transform((event.x, event.y))

            # Corrigir apenas a coordenada Y
            y_min, y_max = ax.get_ylim()
            y = y_min + y_max - y_incorreto

            # Encontrar marcador mais próximo
            closest_idx = -1
            closest_dist = float("inf")

            for i, (_, _, mx, my, _, _) in enumerate(manual_marker_artists):
                dist = np.sqrt((x - mx) ** 2 + (y - my) ** 2)
                if dist < closest_dist and dist < 3:  # Dentro de 3 metros
                    closest_dist = dist
                    closest_idx = i

            if closest_idx >= 0:
                # Remover artists do plot
                circle, text, _, _, marker_num, frame_idx = manual_marker_artists[closest_idx]
                circle.remove()
                text.remove()

                # Remover das listas
                del manual_marker_artists[closest_idx]

                # Remover do dicionário de frames
                if frame_idx in frame_markers and marker_num in frame_markers[frame_idx]:
                    del frame_markers[frame_idx][marker_num]

                # Se o frame ficou vazio, remover o frame
                if not frame_markers[frame_idx]:
                    del frame_markers[frame_idx]

            # Atualizar o canvas
            current_canvas[0].draw()
            print(f"Deleted marker at frame {frame_idx}, position ({x:.2f}, {y:.2f})")

        except Exception as e:
            print(f"Error deleting marker: {e}")
            import traceback

            traceback.print_exc()

    def save_markers_csv(event=None):
        """Save manually created markers to a CSV file automatically"""
        print("Attempting to save markers to CSV...")

        if not manual_marker_mode[0] and not frame_markers:
            print("No markers to save.")
            return

        try:
            # Open file dialog to get the save location - same as PNG saving
            csv_path = filedialog.asksaveasfilename(
                title="Save Markers CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )

            if not csv_path:  # User canceled
                print("Save operation canceled.")
                return

            # Encontrar o número máximo de marcadores
            max_marker_num = 0
            for frame_data in frame_markers.values():
                for marker_num in frame_data:
                    max_marker_num = max(max_marker_num, marker_num)

            # Encontrar o número máximo de frames
            max_frame = max(frame_markers.keys()) if frame_markers else 0

            # Criar dados para o CSV
            data = {"frame": list(range(max_frame + 1))}

            # Criar colunas para todos os marcadores
            for marker_num in range(1, max_marker_num + 1):
                data[f"p{marker_num}_x"] = [None] * (max_frame + 1)
                data[f"p{marker_num}_y"] = [None] * (max_frame + 1)

            # Preencher os dados dos marcadores
            for frame_idx, frame_data in frame_markers.items():
                for marker_num, (x, y) in frame_data.items():
                    data[f"p{marker_num}_x"][frame_idx] = x
                    data[f"p{marker_num}_y"][frame_idx] = y

            # Criar e salvar DataFrame
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)

            # Mensagem clara de confirmação
            print(f"CSV saved successfully to: {csv_path}")

            # Aviso visual (opcional)
            tk.messagebox.showinfo("CSV Saved", f"Markers saved to:\n{csv_path}")

        except Exception as e:
            print(f"Error saving markers: {e}")
            import traceback

            traceback.print_exc()

            # Mostrar erro visual
            tk.messagebox.showerror("Error", f"Failed to save CSV: {str(e)}")

    def setup_manual_marker_events(canvas):
        """Setup event handlers for manual marker creation/deletion"""
        if canvas is None or not hasattr(canvas, "get_tk_widget"):
            return

        canvas_widget = canvas.get_tk_widget()

        # Bind left mouse click (Button-1) to create marker
        canvas_widget.bind("<Button-1>", create_marker)

        # Bind right mouse click (Button-3) to delete marker
        canvas_widget.bind("<Button-3>", delete_marker)

        # Bind Ctrl+S to save markers
        root.bind("<Control-s>", save_markers_csv)

    def clear_all_markers():
        """Clear all markers (manual and loaded from CSV) from the plot."""
        if current_ax[0] is None:
            return

        try:
            # Remove visual artists from the plot
            artists_to_remove = []
            for artist in current_ax[0].get_children():
                if hasattr(artist, "get_zorder"):
                    z_order = artist.get_zorder()
                    # Check for CSV markers (zorder 50-52) or manual markers (zorder >= 100)
                    if (z_order >= 50 and z_order < 100) or z_order >= 100:
                        artists_to_remove.append(artist)

            for artist in artists_to_remove:
                artist.remove()

            # Clear manual marker data structures
            manual_marker_artists.clear()
            frame_markers.clear()
            current_frame[0] = 0  # Reset frame counter for manual markers

            # Reset CSV loaded state
            current_markers_csv[0] = None
            selected_markers[0] = None
            if "select_markers_button" in locals() or "select_markers_button" in globals():
                select_markers_button.config(state=tk.DISABLED)

            # Update the canvas
            if current_canvas[0]:
                current_canvas[0].draw()
                print("All markers (manual and loaded from CSV) cleared.")

        except Exception as e:
            print(f"Error clearing all markers: {e}")
            import traceback

            traceback.print_exc()

    def open_soccerfield_help():
        """Open bundled HTML help in the default browser (no extra Tk window)."""
        html_path = Path(__file__).resolve().parent / "help" / help_html
        if html_path.is_file():
            webbrowser.open_new_tab(html_path.as_uri())
        else:
            messagebox.showinfo(
                "Help",
                f"Help file not found:\n{html_path}\n\nSee vaila/help/sports_fields_courts.html for overview.",
            )

    def load_scout_csv():
        """Opens dialog to select scout_vaila CSV and plot it"""
        if current_ax[0] is None or current_canvas[0] is None:
            print("Please load the field first.")
            return

        # Open dialog to select file
        csv_path = filedialog.askopenfilename(
            title="Select scout_vaila CSV file with events",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )

        if not csv_path:
            return

        try:
            print(f"\nStarting scout CSV loading: {csv_path}")
            # Check if file exists and can be read
            with open(csv_path) as f:
                first_line = f.readline()
                print(f"First line of file: {first_line}")

            # Store the scout path for potential reloads
            current_scout_csv[0] = csv_path

            # Reset selected filters when loading a new file
            selected_teams[0] = None
            selected_players[0] = None
            selected_actions[0] = None

            # Use stored canvas
            load_and_plot_scout_events(
                current_ax[0],
                csv_path,
                current_canvas[0],
                manual_marker_artists,
                frame_markers,
                current_frame,
                selected_teams[0],
                selected_players[0],
                selected_actions[0],
            )

            # Enable the scout filter buttons
            scout_filters_button.config(state=tk.NORMAL)

        except Exception as e:
            print(f"Error plotting scout events: {str(e)}")
            import traceback

            traceback.print_exc()

    def open_scout_filters_dialog():
        """Opens a dialog to select teams, players, and actions for scout data"""
        if not current_scout_csv[0] or not os.path.exists(current_scout_csv[0]):
            messagebox.showerror("Error", "No scout CSV loaded. Load a scout CSV first.")
            return

        try:
            events_df = pd.read_csv(current_scout_csv[0])

            # Get unique values
            teams = sorted(events_df["team"].unique())
            players = sorted(
                events_df["player"].unique(),
                key=lambda x: int(x) if str(x).isdigit() else 0,
            )
            actions = sorted(events_df["action"].unique())

            if not teams or not players or not actions:
                messagebox.showerror("Error", "No valid data found in the scout CSV.")
                return

            select_window = tk.Toplevel(root)
            select_window.title("Select Scout Data Filters")
            select_window.geometry("400x500")

            frame = Frame(select_window)
            frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Teams selection
            tk.Label(frame, text="Teams:", font=("Arial", 10, "bold")).pack(
                anchor=tk.W, pady=(0, 5)
            )
            team_frame = Frame(frame)
            team_frame.pack(fill=tk.X, pady=(0, 10))

            team_scrollbar = tk.Scrollbar(team_frame)
            team_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            team_listbox = tk.Listbox(
                team_frame,
                selectmode=tk.MULTIPLE,
                yscrollcommand=team_scrollbar.set,
                height=4,
            )
            team_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            team_scrollbar.config(command=team_listbox.yview)

            for team in teams:
                team_listbox.insert(tk.END, team)

            # Set initial selection (previous selection or all)
            initial_teams = selected_teams[0] if selected_teams[0] else teams
            for i, team in enumerate(teams):
                if team in initial_teams:
                    team_listbox.selection_set(i)

            # Players selection
            tk.Label(frame, text="Players:", font=("Arial", 10, "bold")).pack(
                anchor=tk.W, pady=(10, 5)
            )
            player_frame = Frame(frame)
            player_frame.pack(fill=tk.X, pady=(0, 10))

            player_scrollbar = tk.Scrollbar(player_frame)
            player_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            player_listbox = tk.Listbox(
                player_frame,
                selectmode=tk.MULTIPLE,
                yscrollcommand=player_scrollbar.set,
                height=6,
            )
            player_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            player_scrollbar.config(command=player_listbox.yview)

            for player in players:
                player_listbox.insert(tk.END, player)

            # Set initial selection
            initial_players = selected_players[0] if selected_players[0] else players
            for i, player in enumerate(players):
                if player in initial_players:
                    player_listbox.selection_set(i)

            # Actions selection
            tk.Label(frame, text="Actions:", font=("Arial", 10, "bold")).pack(
                anchor=tk.W, pady=(10, 5)
            )
            action_frame = Frame(frame)
            action_frame.pack(fill=tk.X, pady=(0, 10))

            action_scrollbar = tk.Scrollbar(action_frame)
            action_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            action_listbox = tk.Listbox(
                action_frame,
                selectmode=tk.MULTIPLE,
                yscrollcommand=action_scrollbar.set,
                height=6,
            )
            action_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            action_scrollbar.config(command=action_listbox.yview)

            for action in actions:
                action_listbox.insert(tk.END, action)

            # Set initial selection
            initial_actions = selected_actions[0] if selected_actions[0] else actions
            for i, action in enumerate(actions):
                if action in initial_actions:
                    action_listbox.selection_set(i)

            button_frame = Frame(select_window)
            button_frame.pack(fill=tk.X, padx=10, pady=10)

            def select_all_teams():
                team_listbox.select_set(0, tk.END)

            def deselect_all_teams():
                team_listbox.selection_clear(0, tk.END)

            def select_all_players():
                player_listbox.select_set(0, tk.END)

            def deselect_all_players():
                player_listbox.selection_clear(0, tk.END)

            def select_all_actions():
                action_listbox.select_set(0, tk.END)

            def deselect_all_actions():
                action_listbox.selection_clear(0, tk.END)

            def apply_filters():
                team_selections = [team_listbox.get(i) for i in team_listbox.curselection()]
                player_selections = [player_listbox.get(i) for i in player_listbox.curselection()]
                action_selections = [action_listbox.get(i) for i in action_listbox.curselection()]

                if not team_selections:
                    selected_teams[0] = None
                else:
                    selected_teams[0] = team_selections

                if not player_selections:
                    selected_players[0] = None
                else:
                    selected_players[0] = player_selections

                if not action_selections:
                    selected_actions[0] = None
                else:
                    selected_actions[0] = action_selections

                # Redraw with selected filters
                load_and_plot_scout_events(
                    current_ax[0],
                    current_scout_csv[0],
                    current_canvas[0],
                    manual_marker_artists,
                    frame_markers,
                    current_frame,
                    selected_teams[0],
                    selected_players[0],
                    selected_actions[0],
                )
                current_canvas[0].draw()
                select_window.destroy()

            # Buttons
            tk.Button(
                button_frame,
                text="Select All Teams",
                command=select_all_teams,
                bg="#4CAF50",
                fg="white",
            ).pack(side=tk.LEFT, padx=2)
            tk.Button(
                button_frame,
                text="Deselect All Teams",
                command=deselect_all_teams,
                bg="#f44336",
                fg="white",
            ).pack(side=tk.LEFT, padx=2)
            tk.Button(
                button_frame,
                text="Select All Players",
                command=select_all_players,
                bg="#4CAF50",
                fg="white",
            ).pack(side=tk.LEFT, padx=2)
            tk.Button(
                button_frame,
                text="Deselect All Players",
                command=deselect_all_players,
                bg="#f44336",
                fg="white",
            ).pack(side=tk.LEFT, padx=2)
            tk.Button(
                button_frame,
                text="Select All Actions",
                command=select_all_actions,
                bg="#4CAF50",
                fg="white",
            ).pack(side=tk.LEFT, padx=2)
            tk.Button(
                button_frame,
                text="Deselect All Actions",
                command=deselect_all_actions,
                bg="#f44336",
                fg="white",
            ).pack(side=tk.LEFT, padx=2)
            tk.Button(
                button_frame,
                text="Apply",
                command=apply_filters,
                bg="#2196F3",
                fg="white",
            ).pack(side=tk.RIGHT, padx=2)

            select_window.transient(root)
            select_window.grab_set()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open filters dialog: {str(e)}")

    def show_heatmap():
        """Generate a KDE heatmap from loaded marker trajectories or scout events."""
        if current_ax[0] is None or current_canvas[0] is None:
            messagebox.showwarning("Warning", "Load the field first.")
            return

        has_markers = current_markers_csv[0] and os.path.exists(current_markers_csv[0])
        has_scout = current_scout_csv[0] and os.path.exists(current_scout_csv[0])

        if not has_markers and not has_scout:
            messagebox.showwarning(
                "No data",
                "Load a Markers CSV or Scout CSV first to generate a heatmap.",
            )
            return

        win = tk.Toplevel(root)
        win.title(f"Heatmap — {current_field_title[0]}")
        win.geometry("950x680")
        win.resizable(True, True)

        ctrl = Frame(win)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ALL = "All"

        # Source selector
        tk.Label(ctrl, text="Source:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        sources = []
        if has_markers:
            sources.append("Markers CSV")
        if has_scout:
            sources.append("Scout CSV")
        source_var = tk.StringVar(value=sources[0])
        src_cb = tk.OptionMenu(ctrl, source_var, *sources)
        src_cb.pack(side=tk.LEFT, padx=4)

        # Marker / player filter
        tk.Label(ctrl, text="Filter:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(8, 0))
        filter_var = tk.StringVar(value=ALL)
        filter_cb = tk.OptionMenu(ctrl, filter_var, ALL)
        filter_cb.pack(side=tk.LEFT, padx=4)

        # Colormap
        tk.Label(ctrl, text="Cmap:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(8, 0))
        cmap_var = tk.StringVar(value="Reds")
        cmap_cb = tk.OptionMenu(
            ctrl,
            cmap_var,
            "Reds",
            "Blues",
            "Greens",
            "Oranges",
            "YlOrRd",
            "viridis",
            "plasma",
            "inferno",
        )
        cmap_cb.pack(side=tk.LEFT, padx=4)

        fig_hm, ax_hm = plt.subplots(figsize=(8.5, 5.5))
        canvas_hm = FigureCanvasTkAgg(fig_hm, master=win)
        canvas_hm.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def refresh_filters(*_args):
            src = source_var.get()
            menu = filter_cb["menu"]
            menu.delete(0, tk.END)
            opts = [ALL]
            if src == "Markers CSV" and has_markers:
                mdf = pd.read_csv(current_markers_csv[0])
                names = sorted(
                    {
                        c.rsplit("_", 1)[0]
                        for c in mdf.columns
                        if c != "frame" and (c.endswith("_x") or c.endswith("_y"))
                    }
                )
                opts += names
            elif src == "Scout CSV" and has_scout:
                sdf = pd.read_csv(current_scout_csv[0])
                if "team" in sdf.columns:
                    opts += sorted(sdf["team"].dropna().unique().tolist())
            for o in opts:
                menu.add_command(label=o, command=lambda v=o: filter_var.set(v))
            filter_var.set(ALL)

        source_var.trace_add("write", refresh_filters)
        refresh_filters()

        def draw_heatmap():
            try:
                ax_hm.clear()

                csv_path_for_field = current_field_csv[0] if current_field_csv[0] else None
                if csv_path_for_field is None:
                    models_dir = os.path.join(os.path.dirname(__file__), "models")
                    csv_path_for_field = os.path.join(models_dir, "soccerfield_ref3d.csv")
                field_df = pd.read_csv(csv_path_for_field)
                plot_field.__wrapped__(field_df, ax=ax_hm) if hasattr(
                    plot_field, "__wrapped__"
                ) else _draw_field_on_ax(ax_hm, field_df)

                src = source_var.get()
                flt = filter_var.get()
                xs, ys = [], []

                if src == "Markers CSV" and has_markers:
                    mdf = pd.read_csv(current_markers_csv[0]).replace("", np.nan)
                    names = sorted(
                        {
                            c.rsplit("_", 1)[0]
                            for c in mdf.columns
                            if c != "frame" and (c.endswith("_x") or c.endswith("_y"))
                        }
                    )
                    for nm in names:
                        if flt != ALL and nm != flt:
                            continue
                        xc, yc = f"{nm}_x", f"{nm}_y"
                        if xc in mdf.columns and yc in mdf.columns:
                            sub = mdf[[xc, yc]].dropna()
                            xs.extend(sub[xc].tolist())
                            ys.extend(sub[yc].tolist())

                elif src == "Scout CSV" and has_scout:
                    sdf = pd.read_csv(current_scout_csv[0]).replace("", np.nan)
                    if flt != ALL and "team" in sdf.columns:
                        sdf = sdf[sdf["team"] == flt]
                    if "pos_x_m" in sdf.columns and "pos_y_m" in sdf.columns:
                        sub = sdf[["pos_x_m", "pos_y_m"]].dropna()
                        xs = sub["pos_x_m"].tolist()
                        ys = sub["pos_y_m"].tolist()

                if len(xs) < 2:
                    ax_hm.set_title("Not enough points for heatmap")
                    canvas_hm.draw()
                    return

                hm_df = pd.DataFrame({"x": xs, "y": ys})
                sns.kdeplot(
                    data=hm_df,
                    x="x",
                    y="y",
                    cmap=cmap_var.get(),
                    fill=True,
                    alpha=0.6,
                    bw_method="scott",
                    thresh=0.05,
                    ax=ax_hm,
                )

                ax_hm.set_xlim(-5, 110)
                ax_hm.set_ylim(-5, 73)
                ax_hm.set_aspect("equal")

                title = "Heatmap"
                if flt != ALL:
                    title += f" — {flt}"
                ax_hm.set_title(title)
                canvas_hm.draw()

            except Exception as exc:
                messagebox.showerror("Error", f"Heatmap failed: {exc}")

        Button(
            ctrl, text="Show", command=draw_heatmap, bg="#4CAF50", fg="white", padx=8, pady=2
        ).pack(side=tk.LEFT, padx=8)

        draw_heatmap()

    def _draw_field_on_ax(ax, field_df, color_scheme=None):
        """Minimal re-draw of the soccer field on a given axes (for heatmap overlay)."""
        points = {}
        for _, row in field_df.iterrows():
            points[row["point_name"]] = (row["x"], row["y"], int(row["point_number"]))

        field_w = points["top_right_corner"][0]
        field_h = points["top_right_corner"][1]

        # Get current colors
        colors = _get_colors("soccer", color_scheme)
        playing_c = colors["playing"]
        outer_c = colors["outer"]
        line_c = colors["lines"]

        ax.set_facecolor(outer_c)
        ax.add_patch(
            patches.Rectangle(
                (0, 0),
                field_w,
                field_h,
                facecolor=playing_c,
                edgecolor=line_c,
                linewidth=2,
                zorder=0,
            )
        )
        # Halfway line
        mid_x = field_w / 2
        ax.plot([mid_x, mid_x], [0, field_h], color=line_c, linewidth=2, zorder=1)
        # Center circle
        ax.add_patch(
            patches.Circle(
                (mid_x, field_h / 2),
                9.15,
                edgecolor=line_c,
                facecolor="none",
                linewidth=2,
                zorder=1,
            )
        )
        # Penalty areas
        for x0, sign in [(0, 1), (field_w, -1)]:
            ax.add_patch(
                patches.Rectangle(
                    (x0 + sign * min(0, -16.5), field_h / 2 - 20.16),
                    16.5,
                    40.32,
                    edgecolor="white",
                    facecolor="none",
                    linewidth=2,
                    zorder=1,
                )
            )
            ax.add_patch(
                patches.Rectangle(
                    (x0 + sign * min(0, -5.5), field_h / 2 - 9.16),
                    5.5,
                    18.32,
                    edgecolor="white",
                    facecolor="none",
                    linewidth=2,
                    zorder=1,
                )
            )
        ax.set_xlim(-5, field_w + 5)
        ax.set_ylim(-5, field_h + 5)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    # Add buttons
    Button(
        button_frame,
        text="Load Default Field",
        command=load_field,
        bg="white",
        fg="black",
        padx=10,
        pady=5,
    ).pack(side=tk.LEFT, padx=5, pady=5)

    Button(
        button_frame,
        text="Load Custom Field",
        command=load_custom_field,
        bg="white",
        fg="black",
        padx=10,
        pady=5,
    ).pack(side=tk.LEFT, padx=5, pady=5)

    Button(
        button_frame,
        text="Surface Color",
        command=change_field_color,
        bg="white",
        fg="black",
        padx=10,
        pady=5,
    ).pack(side=tk.LEFT, padx=5, pady=5)

    Button(
        button_frame,
        text="Load Markers CSV",
        command=load_markers_csv,
        bg="white",
        fg="black",
        padx=10,
        pady=5,
    ).pack(side=tk.LEFT, padx=5, pady=5)

    # Add new button for scout CSV
    Button(
        button_frame,
        text="Load Scout CSV",
        command=load_scout_csv,
        bg="white",
        fg="black",
        padx=10,
        pady=5,
    ).pack(side=tk.LEFT, padx=5, pady=5)

    # Add toggle button for reference points
    ref_points_button = Button(
        button_frame,
        text="Hide Reference Points",
        command=toggle_reference_points,
        bg="white",
        fg="black",
        padx=10,
        pady=5,
    )
    ref_points_button.pack(side=tk.LEFT, padx=5, pady=5)

    # Add toggle button for axis values
    axis_values_button = Button(
        button_frame,
        text="Show Axis Values",
        command=toggle_axis_values,
        bg="white",
        fg="black",
        padx=10,
        pady=5,
    )
    axis_values_button.pack(side=tk.LEFT, padx=5, pady=5)

    # Add marker selection button - initially disabled
    select_markers_button = Button(
        button_frame,
        text="Select Markers",
        command=open_marker_selection_dialog,
        bg="white",
        fg="black",
        padx=10,
        pady=5,
        state=tk.DISABLED,
    )
    select_markers_button.pack(side=tk.LEFT, padx=5, pady=5)

    # Add scout filters button - initially disabled
    scout_filters_button = Button(
        button_frame,
        text="Scout Filters",
        command=open_scout_filters_dialog,
        bg="white",
        fg="black",
        padx=10,
        pady=5,
        state=tk.DISABLED,
    )
    scout_filters_button.pack(side=tk.LEFT, padx=5, pady=5)

    Button(
        button_frame,
        text="Heatmap",
        command=show_heatmap,
        bg="#D84315",
        fg="white",
        padx=10,
        pady=5,
    ).pack(side=tk.LEFT, padx=5, pady=5)

    # Add manual marker mode button
    manual_marker_button = Button(
        button_frame,
        text="Create Manual Markers",
        command=toggle_manual_marker_mode,
        bg="white",
        fg="black",
        padx=10,
        pady=5,
    )
    manual_marker_button.pack(side=tk.LEFT, padx=5, pady=5)

    # Add Clear All button
    Button(
        button_frame,
        text="Clear All Markers",
        command=lambda: clear_all_markers(),
        bg="white",
        fg="black",
        padx=10,
        pady=5,
    ).pack(side=tk.LEFT, padx=5, pady=5)

    # Add Help button
    Button(
        button_frame,
        text="Help",
        command=open_soccerfield_help,
        bg="white",
        fg="black",
        padx=10,
        pady=5,
    ).pack(side=tk.LEFT, padx=5, pady=5)

    # Frame for plotting
    plot_frame = Frame(root)
    plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    # Load field initially
    if initial_field_csv and os.path.isfile(initial_field_csv):
        load_field(custom_file=initial_field_csv)
    else:
        load_field()

    # Start Tkinter loop
    root.mainloop()


def run_drawsportsfields(surface: str) -> None:
    """Unified launcher for all supported sports surfaces.

    Looks up *surface* in ``SPORT_REGISTRY`` and opens the shared Tk GUI
    pre-loaded with the corresponding model CSV.
    """
    key = (surface or "").strip().lower()
    if key not in SPORT_REGISTRY:
        raise ValueError(f"Unknown sports surface: {key!r}. Available: {list(SPORT_REGISTRY)}")
    config = SPORT_REGISTRY[key]
    model = str(Path(__file__).resolve().parent / "models" / config.model_csv)
    run_soccerfield(
        initial_field_csv=model,
        default_field_csv=model,
        window_title=config.label,
        help_html="sports_fields_courts.html",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="vailá — Draw Sports Fields/Courts")
    parser.add_argument(
        "-t",
        "--type",
        dest="surface",
        default="soccer",
        choices=list(SPORT_REGISTRY.keys()),
        help="Surface type to open (default: soccer).",
    )
    parser.add_argument("--field", type=str, help="Path to field model CSV")
    parser.add_argument("--markers", type=str, help="Path to vaila-format markers CSV")
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Generate a heatmap from --markers data (requires --markers)",
    )
    args = parser.parse_args()

    if args.heatmap and args.markers:
        _field_csv = args.field
        if _field_csv is None:
            _models = os.path.join(os.path.dirname(__file__), "models")
            _field_csv = os.path.join(_models, "soccerfield_ref3d.csv")
        _fdf = pd.read_csv(_field_csv)
        _fig, _ax = plot_field(_fdf, show_reference_points=False, show_axis_values=False)
        _mdf = pd.read_csv(args.markers).replace("", np.nan)
        _names = sorted(
            {
                c.rsplit("_", 1)[0]
                for c in _mdf.columns
                if c != "frame" and (c.endswith("_x") or c.endswith("_y"))
            }
        )
        _xs, _ys = [], []
        for _nm in _names:
            _xc, _yc = f"{_nm}_x", f"{_nm}_y"
            if _xc in _mdf.columns and _yc in _mdf.columns:
                _sub = _mdf[[_xc, _yc]].dropna()
                _xs.extend(_sub[_xc].tolist())
                _ys.extend(_sub[_yc].tolist())
        if len(_xs) >= 2:
            sns.kdeplot(
                data=pd.DataFrame({"x": _xs, "y": _ys}),
                x="x",
                y="y",
                cmap="Reds",
                fill=True,
                alpha=0.6,
                bw_method="scott",
                thresh=0.05,
                ax=_ax,
            )
        _ax.set_title("Heatmap")
        plt.tight_layout()
        plt.show()
    elif args.field:
        run_soccerfield(initial_field_csv=args.field, default_field_csv=args.field)
    else:
        run_drawsportsfields(args.surface)
