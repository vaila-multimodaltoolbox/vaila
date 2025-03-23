"""
Project: vailá
Script: soccerfield.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 20 March 2025
Updated: 21 March 2025
Version: 0.0.2

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
    This project is licensed under the terms of the MIT License.
"""

import os
from rich import print
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
    arc = patches.Arc(
        center, 2 * radius, 2 * radius, theta1=theta1, theta2=theta2, **kwargs
    )
    ax.add_patch(arc)


def plot_field(df):
    """
    Plots a soccer field using coordinates from the DataFrame.

    Args:
        df: DataFrame with field point coordinates

    Returns:
        fig, ax: Matplotlib figure and axes with the drawn field
    """
    # Define the margin around the field (2 meters)
    margin = 2

    fig, ax = plt.subplots(
        figsize=(10.5 + 0.4, 6.8 + 0.4)
    )  # Increasing slightly to accommodate the margin
    ax.set_xlim(
        -margin - 1, 105 + margin + 1
    )  # Expanding the limits to include the margin
    ax.set_ylim(-margin - 1, 68 + margin + 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # Convert DataFrame to dictionary for easier access
    points = {row["point_name"]: (row["x"], row["y"]) for _, row in df.iterrows()}

    # Draw extended area (including 2m margin around the field)
    draw_rectangle(
        ax,
        (points["canto_inf_esq"][0] - margin, points["canto_inf_esq"][1] - margin),
        105 + 2 * margin,
        68 + 2 * margin,
        edgecolor="none",
        facecolor="green",
        zorder=0,
    )

    # Draw the main playing surface with slightly darker green to differentiate
    draw_rectangle(
        ax,
        points["canto_inf_esq"],
        105,
        68,
        edgecolor="none",
        facecolor="forestgreen",  # Slightly darker green
        zorder=0.5,
    )

    # Draw perimeter lines (12cm = 0.12m width)
    # To maintain the proportion, we will use linewidth=2 to represent the 12cm

    # Left sideline (left sideline)
    draw_line(
        ax,
        points["canto_inf_esq"],
        points["canto_sup_esq"],
        color="white",
        linewidth=2,
        zorder=1,
    )

    # Right sideline (right sideline)
    draw_line(
        ax,
        points["canto_inf_dir"],
        points["canto_sup_dir"],
        color="white",
        linewidth=2,
        zorder=1,
    )

    # Bottom goal line (bottom goal line)
    draw_line(
        ax,
        points["canto_inf_esq"],
        points["canto_inf_dir"],
        color="white",
        linewidth=2,
        zorder=1,
    )

    # Top goal line (top goal line)
    draw_line(
        ax,
        points["canto_sup_esq"],
        points["canto_sup_dir"],
        color="white",
        linewidth=2,
        zorder=1,
    )

    # Center line (center line)
    draw_line(
        ax,
        points["meio_campo_esq"],
        points["meio_campo_dir"],
        color="white",
        linewidth=2,
        zorder=1,
    )

    # Center circle (center circle)
    draw_circle(
        ax,
        points["centro_campo"],
        9.15,
        edgecolor="white",
        facecolor="none",
        linewidth=2,
        zorder=1,
    )

    # Center spot (center spot)
    draw_circle(
        ax,
        points["centro_campo"],
        0.2,
        edgecolor="white",
        facecolor="white",
        linewidth=2,
        zorder=1,
    )

    # Left penalty area (left penalty area)
    draw_rectangle(
        ax,
        points["ga_esq_inf_esq"],
        16.5,
        40.3,
        edgecolor="white",
        facecolor="none",
        linewidth=2,
        zorder=1,
    )

    # Left goal area (left goal area)
    draw_rectangle(
        ax,
        points["pa_esq_inf_esq"],
        5.5,
        18.32,
        edgecolor="white",
        facecolor="none",
        linewidth=2,
        zorder=1,
    )

    # Left penalty spot (left penalty spot)
    draw_circle(
        ax,
        points["penalti_esq"],
        0.2,
        edgecolor="white",
        facecolor="white",
        linewidth=2,
        zorder=1,
    )

    # Left penalty arc - only the arc outside the large area
    draw_arc(
        ax,
        points["penalti_esq"],
        9.15,
        310,
        50,
        edgecolor="white",
        linewidth=2,
        zorder=1,
    )

    # Right penalty area (right penalty area)
    draw_rectangle(
        ax,
        (88.5, 13.85),
        16.5,
        40.3,
        edgecolor="white",
        facecolor="none",
        linewidth=2,
        zorder=1,
    )

    # Right goal area (right goal area)
    draw_rectangle(
        ax,
        (99.5, 24.84),
        5.5,
        18.32,
        edgecolor="white",
        facecolor="none",
        linewidth=2,
        zorder=1,
    )

    # Right penalty spot (right penalty spot)
    draw_circle(
        ax,
        points["penalti_dir"],
        0.2,
        edgecolor="white",
        facecolor="white",
        linewidth=2,
        zorder=1,
    )

    # Right penalty arc (right penalty arc)
    draw_arc(
        ax,
        points["penalti_dir"],
        9.15,
        130,
        230,
        edgecolor="white",
        linewidth=2,
        zorder=1,
    )

    # Left goal line (left goal line) - thicker than the other lines
    draw_line(
        ax,
        points["gol_esq_poste_esq"],
        points["gol_esq_poste_dir"],
        color="white",
        linewidth=8,
        zorder=2,
    )

    # Right goal line (right goal line) - thicker than the other lines
    draw_line(
        ax,
        points["gol_dir_poste_esq"],
        points["gol_dir_poste_dir"],
        color="white",
        linewidth=8,
        zorder=2,
    )

    return fig, ax


def run_soccerfield():
    """Main function to run the soccerfield.py script"""
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    try:
        # Check if models directory exists
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        csv_path = os.path.join(models_dir, "soccerfield_ref3d.csv")

        # Read CSV file
        df = pd.read_csv(csv_path)

        print(f"Reading field data from {csv_path}")
        print(f"Number of reference points: {len(df)}")

        # Plot the field
        fig, ax = plot_field(df)
        plt.title("Soccer Field - Official Dimensions")
        plt.show()

        print("Field plotted successfully!")

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        print("Make sure the file exists in the 'models' directory")
    except Exception as e:
        print(f"Error plotting field: {e}")


if __name__ == "__main__":
    run_soccerfield()
