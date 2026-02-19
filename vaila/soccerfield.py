"""
Project: vailá
Script: soccerfield.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 20 March 2025
Updated: 08 August 2025
Version: 0.0.5

Description:
    This script draws a soccer field based on the coordinates in soccerfield_ref3d.csv.
    It uses matplotlib to create a visual representation with correct dimensions.

Usage:
    Run the script from the command line:
        python soccerfield.py

    Or import the functions to use in other scripts:
        from soccerfield import plot_field

Requirements:
    - Python 3.x
    - pandas
    - matplotlib
    - rich (for enhanced console output)

License:
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import math
import os
import tkinter as tk
from tkinter import Button, Frame, filedialog, messagebox

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from rich import print


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


def plot_field(df, show_reference_points=True, show_axis_values=False):
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

    # Draw extended area (including margin around the field)
    draw_rectangle(
        ax,
        (min_x - margin, min_y - margin),
        field_width + 2 * margin,
        field_height + 2 * margin,
        edgecolor="none",
        facecolor="green",
        zorder=0,
    )

    # Draw the main playing surface with slightly darker green
    draw_rectangle(
        ax,
        (min_x, min_y),  # Use min_x, min_y from DataFrame
        field_width,  # Use calculated field_width
        field_height,  # Use calculated field_height
        edgecolor="none",
        facecolor="forestgreen",
        zorder=0.5,
    )

    # Draw perimeter lines (12cm = 0.12m width)
    # These use named points, should adapt if points are correct in CSV

    # Left sideline
    draw_line(
        ax,
        points["bottom_left_corner"][0:2],
        points["top_left_corner"][0:2],
        color="white",
        linewidth=2,
        zorder=1,
    )

    # Right sideline
    draw_line(
        ax,
        points["bottom_right_corner"][0:2],
        points["top_right_corner"][0:2],
        color="white",
        linewidth=2,
        zorder=1,
    )

    # Bottom goal line
    draw_line(
        ax,
        points["bottom_left_corner"][0:2],
        points["bottom_right_corner"][0:2],
        color="white",
        linewidth=2,
        zorder=1,
    )

    # Top goal line
    draw_line(
        ax,
        points["top_left_corner"][0:2],
        points["top_right_corner"][0:2],
        color="white",
        linewidth=2,
        zorder=1,
    )

    # Center line
    draw_line(
        ax,
        points["midfield_left"][0:2],
        points["midfield_right"][0:2],
        color="white",
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
        edgecolor="white",
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
        edgecolor="white",
        facecolor="white",
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
        edgecolor="white",
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
        edgecolor="white",
        facecolor="none",
        linewidth=2,
        zorder=1,
    )

    # Left penalty spot - fixed small radius
    draw_circle(
        ax,
        points["left_penalty_spot"][0:2],
        0.2,  # Standard penalty spot radius
        edgecolor="white",
        facecolor="white",
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
        edgecolor="white",
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
        edgecolor="white",
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
        edgecolor="white",
        facecolor="none",
        linewidth=2,
        zorder=1,
    )

    # Right penalty spot
    draw_circle(
        ax,
        points["right_penalty_spot"][0:2],
        0.2,  # Standard penalty spot radius
        edgecolor="white",
        facecolor="white",
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
        edgecolor="white",
        linewidth=2,
        zorder=1,
    )

    # Left goal line - thicker than the other lines
    draw_line(
        ax,
        points["left_goal_bottom_post"][0:2],
        points["left_goal_top_post"][0:2],
        color="white",
        linewidth=8,
        zorder=2,
    )

    # Right goal line - thicker than the other lines
    draw_line(
        ax,
        points["right_goal_bottom_post"][0:2],
        points["right_goal_top_post"][0:2],
        color="white",
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


def run_soccerfield():
    """Main function to run the soccerfield.py script with GUI controls"""
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    # Create main Tkinter window
    root = tk.Tk()
    root.title("Soccer Field Visualization")
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
                # Check if models directory exists
                models_dir = os.path.join(os.path.dirname(__file__), "models")
                csv_path = os.path.join(models_dir, "soccerfield_ref3d.csv")

            # Store the current CSV path for later redraws
            current_field_csv[0] = csv_path

            # Read CSV file
            df = pd.read_csv(csv_path)
            print(f"Reading field data from {csv_path}")
            print(f"Number of reference points: {len(df)}")

            # Create figure and embed in Tkinter
            fig, ax = plot_field(
                df,
                show_reference_points=show_reference_points[0],
                show_axis_values=show_axis_values[0],
            )

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

        except Exception as e:
            print(f"Error plotting field: {e}")
            import traceback

            traceback.print_exc()

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

    def show_help_dialog():
        """Show a help dialog with instructions on how to use the application"""
        help_window = tk.Toplevel(root)
        help_window.title("Soccer Field Visualization - Help")
        help_window.geometry("750x650")  # Increased width from 700 to 750

        # Create a frame with scrollbar
        main_frame = Frame(help_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add scrollbar
        scrollbar = tk.Scrollbar(main_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create canvas for scrolling
        canvas = tk.Canvas(main_frame, yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=canvas.yview)

        # Create frame for content
        content_frame = Frame(canvas)
        canvas.create_window((0, 0), window=content_frame, anchor=tk.NW)

        # Add help text content
        tk.Label(content_frame, text="Soccer Field Visualization", font=("Arial", 16, "bold")).pack(
            anchor=tk.W, pady=(0, 10)
        )

        # Introduction
        tk.Label(
            content_frame,
            text="This tool allows you to visualize a soccer field and create or load marker paths.",
            font=("Arial", 10),
            justify=tk.LEFT,
            wraplength=650,
        ).pack(anchor=tk.W, pady=(0, 10))

        # Section: Buttons
        tk.Label(content_frame, text="Button Functions:", font=("Arial", 12, "bold")).pack(
            anchor=tk.W, pady=(10, 5)
        )

        button_help = [
            ("Load Default Field", "Opens the standard soccer field visualization."),
            (
                "Load Custom Field",
                "Allows you to select a custom field definition file (.csv).",
            ),
            ("Load Markers CSV", "Loads and displays marker paths from a CSV file."),
            (
                "Hide/Show Reference Points",
                "Toggles field reference points visibility.",
            ),
            (
                "Select Markers",
                "Choose which markers to display when multiple markers are loaded.",
            ),
            (
                "Create/Disable Manual Markers",
                "Enables or disables manual marker creation mode.",
            ),
            (
                "Clear All Markers",
                "Removes all manually created markers from the field.",
            ),
        ]

        for button, desc in button_help:
            frame = Frame(content_frame)
            frame.pack(fill=tk.X, pady=2, anchor=tk.W)
            tk.Label(
                frame,
                text=f"• {button}: ",
                font=("Arial", 10, "bold"),
                width=30,
                anchor=tk.W,
            ).pack(side=tk.LEFT)  # Increased width from 25 to 30
            tk.Label(frame, text=desc, font=("Arial", 10), justify=tk.LEFT, wraplength=430).pack(
                side=tk.LEFT, fill=tk.X, expand=True
            )

        # Section: Creating markers
        tk.Label(content_frame, text="Creating Markers:", font=("Arial", 12, "bold")).pack(
            anchor=tk.W, pady=(20, 5)
        )

        marker_help = [
            ("1. Enable marker mode by clicking 'Create Manual Markers'"),
            ("2. Left-click on the field to place a marker"),
            ("3. Hold Shift + left-click to create a marker with the next number"),
            ("4. Right-click on a marker to delete it"),
            ("5. Use Ctrl+S to save all markers to a CSV file"),
            ("6. Click 'Clear All Markers' to remove all markers"),
        ]

        for step in marker_help:
            tk.Label(content_frame, text=step, font=("Arial", 10), justify=tk.LEFT).pack(
                anchor=tk.W, pady=2
            )

        # Section: Toolbar
        tk.Label(content_frame, text="Field Navigation Toolbar:", font=("Arial", 12, "bold")).pack(
            anchor=tk.W, pady=(20, 5)
        )

        toolbar_help = [
            ("Home", "Reset the view to the original zoom level"),
            ("Pan", "Click and drag to move around the field"),
            ("Zoom", "Zoom in/out of a rectangular region"),
            ("Save", "Save the current field view as a PNG image"),
        ]

        for tool, desc in toolbar_help:
            frame = Frame(content_frame)
            frame.pack(fill=tk.X, pady=2, anchor=tk.W)
            tk.Label(
                frame,
                text=f"• {tool}: ",
                font=("Arial", 10, "bold"),
                width=15,
                anchor=tk.W,
            ).pack(side=tk.LEFT)  # Increased width from 10 to 15
            tk.Label(frame, text=desc, font=("Arial", 10), justify=tk.LEFT, wraplength=500).pack(
                side=tk.LEFT, fill=tk.X, expand=True
            )

        # Tips and keyboard shortcuts
        tk.Label(content_frame, text="Tips & Shortcuts:", font=("Arial", 12, "bold")).pack(
            anchor=tk.W, pady=(20, 5)
        )

        tips = [
            "• When saving manually created markers (Ctrl+S), the CSV will be saved in the same location and with the same name as your PNG file.",
            "• For precise marker placement, you can zoom in using the navigation toolbar.",
            "• When loading a markers CSV, you can select which markers to display using the 'Select Markers' button.",
            "• The field size follows official FIFA regulations (105m × 68m).",
        ]

        for tip in tips:
            tk.Label(
                content_frame,
                text=tip,
                font=("Arial", 10),
                justify=tk.LEFT,
                wraplength=650,
            ).pack(anchor=tk.W, pady=5)

        # Update canvas scroll region
        content_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox(tk.ALL))

        # Add a Close button at the bottom
        tk.Button(
            help_window,
            text="Close",
            command=help_window.destroy,
            bg="white",
            fg="black",
            padx=20,
            pady=5,
        ).pack(pady=10)

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
        command=lambda: show_help_dialog(),
        bg="white",
        fg="black",
        padx=10,
        pady=5,
    ).pack(side=tk.LEFT, padx=5, pady=5)

    # Frame for plotting
    plot_frame = Frame(root)
    plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    # Load field initially
    load_field()

    # Start Tkinter loop
    root.mainloop()


if __name__ == "__main__":
    run_soccerfield()
