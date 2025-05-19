"""
================================================================================
Orthonormal Bases Plotting Toolkit
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 2024-07-19
Version: 1.0

Overview:

This Python script provides tools for visualizing orthonormal bases in 3D using both `matplotlib` and `plotly`. It is particularly useful for visualizing biomechanical data such as trunk and pelvis rotations captured by motion capture systems. The toolkit supports dynamic and static visualization with customizable options for color, axis length, and global coordinate systems.

Main Features:

    1. Color Customization:
        - Predefined colors for the axes and markers of the trunk and pelvis, enhancing clarity in biomechanical data visualization.
        - `get_colors()`: Returns preset color schemes for trunk and pelvis axes and markers.

    2. Static Visualization with Matplotlib:
        - `plot_orthonormal_bases_matplotlib`: Visualizes orthonormal bases using `matplotlib` for 3D plotting, supporting multiple time frames and labeled clusters (e.g., trunk, pelvis).
        - `plot_orthonormal_bases_4points_matplotlib`: A similar function for bases created from four points (e.g., anatomical markers) in space, used for biomechanical analysis.

    3. Interactive 3D Visualization with Plotly:
        - `plot_orthonormal_bases_plotly`: Interactive 3D visualization using `plotly` with dynamic frames for viewing time-dependent biomechanical data.
        - `plot_orthonormal_bases_4points_plotly`: A `plotly` version for four-point orthonormal bases visualization, allowing interactive exploration of 3D motion data over time.

    4. Combined Visualization:
        - `plot_orthonormal_bases`: Combines both the `matplotlib` and `plotly` visualizations for side-by-side comparison or exploration of static and dynamic visualizations.
        - Handles both 3-point and 4-point base creation setups for biomechanics.

Key Functions and Their Functionality:

    get_colors():
        Returns predefined color schemes for trunk and pelvis axes and markers, aiding in visual differentiation of body segments.

    plot_orthonormal_bases_matplotlib():
        Generates a 3D plot of orthonormal bases using `matplotlib`. The function plots the axes and markers for the specified bases and allows labeling clusters (e.g., "trunk", "pelvis").

    plot_orthonormal_bases_4points_matplotlib():
        Similar to `plot_orthonormal_bases_matplotlib`, but specifically for bases constructed from four anatomical markers (e.g., RASI, LASI, RPSI, LPSI for the pelvis).

    plot_orthonormal_bases_plotly():
        Creates an interactive 3D plot using `plotly`, with dynamic frames to animate biomechanical motion data over time. Provides detailed axis and marker labeling for trunk and pelvis.

    plot_orthonormal_bases_4points_plotly():
        A `plotly` function for visualizing four-point bases in 3D. Like its matplotlib counterpart, it handles anatomical marker-based data for biomechanics.

    plot_orthonormal_bases():
        A utility function that allows the user to select between 3-point or 4-point bases visualization and combines both `matplotlib` and `plotly` plots. Ideal for comprehensive analysis and comparison.

Usage Notes:

    - This toolkit is designed for use in biomechanical analysis, especially for motion capture data where body segment orientations need to be visualized over time.
    - The `plotly` functions provide interactive 3D plots, making it easy to explore complex datasets frame by frame.
    - The `matplotlib` plots are static but provide a clear and structured way to visualize the global and local coordinate systems.

Changelog for Version 1.0:

    - Initial release with full support for plotting 3-point and 4-point orthonormal bases in both static (matplotlib) and interactive (plotly) 3D environments.
    - Added support for dynamic frame-by-frame visualization using `plotly`.

License:

This script is distributed under the GPL3 License.
================================================================================
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from rich import print
from pathlib import Path


def get_colors():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting plotting...")

    # Cluster 1 (Trunk) colors - inspired by the Jacksonville Jaguars football team
    # Official Jaguars team colors include gold (#fbbb9b), teal (#006778), and navy blue (#223D79)
    trunk_axis_colors = [
        "#fbbb9b",
        "#006778",
        "#223D79",
    ]  # Jaguars-inspired axis colors
    trunk_marker_colors = [
        "#d3bc8d",
        "#006666",
        "#003f5c",
        "#8a7967",
    ]  # More muted variations of Jaguars colors for markers

    # Cluster 2 (Pelvis) colors - inspired by the Jacksonville Jumbo Shrimp baseball team
    # Official Jumbo Shrimp colors include dark orange, khaki, gray, and indigo
    pelvis_axis_colors = [
        "darkorange",
        "darkkhaki",
        "dimgray",
    ]  # Jumbo Shrimp-inspired axis colors
    pelvis_marker_colors = [
        "darkorange",
        "darkkhaki",
        "dimgray",
        "indigo",
    ]  # Marker colors from the Jumbo Shrimp palette

    return (
        trunk_axis_colors,
        trunk_marker_colors,
        pelvis_axis_colors,
        pelvis_marker_colors,
    )


def plot_orthonormal_bases_matplotlib(
    bases_list,
    pm_list,
    points_list,
    labels,
    global_coordinate_system=None,
    title="Orthonormal Bases",
):
    # Get colors for Jacksonville Jaguars and Jumbo Shrimp inspired clusters
    trunk_axis_colors, trunk_marker_colors, pelvis_axis_colors, pelvis_marker_colors = (
        get_colors()
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    # Set the aspect ratio and limits for the plot
    ax.set_box_aspect([2, 2, 4])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 2])

    # If no global coordinate system is provided, use the identity matrix
    if global_coordinate_system is None:
        global_coordinate_system = np.eye(3)

    axis_colors = ["r", "g", "b"]
    vector_length = 0.2

    # Plot global coordinate system axes
    for i, (axis, color) in enumerate(zip(global_coordinate_system, axis_colors)):
        ax.plot(
            [0, axis[0] * vector_length],
            [0, axis[1] * vector_length],
            [0, axis[2] * vector_length],
            color=color,
            linewidth=1,
        )

    # Iterate through bases, points, and labels for plotting
    for i, (bases, pm, points, label) in enumerate(
        zip(bases_list, pm_list, points_list, labels)
    ):
        if "cluster1" in label.lower():
            # Use Jacksonville Jaguars colors for Cluster 1 (Trunk)
            axis_colors = trunk_axis_colors
            marker_colors = trunk_marker_colors
        elif "cluster2" in label.lower():
            # Use Jacksonville Jumbo Shrimp colors for Cluster 2 (Pelvis)
            axis_colors = pelvis_axis_colors
            marker_colors = pelvis_marker_colors
        else:
            # Additional clusters also use Jacksonville Jaguars colors as default
            axis_colors = trunk_axis_colors
            marker_colors = trunk_marker_colors

        # Plot markers for each point in the cluster
        for point_set, color in zip(points, marker_colors):
            ax.scatter(
                point_set[:, 0],
                point_set[:, 1],
                point_set[:, 2],
                color=color,
                s=1,
                marker=".",
            )

        # Plot axes for each frame
        for j in range(0, len(pm), 10):
            origin = pm[j]
            for k, (axis, color) in enumerate(zip(bases[j], axis_colors)):
                ax.plot(
                    [origin[0], origin[0] + axis[0] * vector_length],
                    [origin[1], origin[1] + axis[1] * vector_length],
                    [origin[2], origin[2] + axis[2] * vector_length],
                    color=color,
                )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return fig


def plot_orthonormal_bases_plotly(
    bases_list,
    pm_list,
    points_list,
    labels,
    global_coordinate_system=None,
    title="Orthonormal Bases",
):
    # Get colors inspired by Jacksonville Jaguars and Jumbo Shrimp teams
    trunk_axis_colors, trunk_marker_colors, pelvis_axis_colors, pelvis_marker_colors = (
        get_colors()
    )

    frames = []
    axis_length = 0.2

    if global_coordinate_system is None:
        global_coordinate_system = np.eye(3)

    marker_labels = ["Marker 1", "Marker 2", "Marker 3"]

    fig = make_subplots(
        rows=1, cols=1, specs=[[{"type": "scatter3d"}]], subplot_titles=[title]
    )

    for frame in range(len(pm_list[0])):
        data = []

        for i, (bases, pm, points, label) in enumerate(
            zip(bases_list, pm_list, points_list, labels)
        ):
            if "trunk" in label.lower() or "cluster1" in label.lower():
                # Use Jacksonville Jaguars colors for Cluster 1 (Trunk)
                axis_colors = trunk_axis_colors
                marker_colors = trunk_marker_colors
            elif "pelvis" in label.lower() or "cluster2" in label.lower():
                # Use Jacksonville Jumbo Shrimp colors for Cluster 2 (Pelvis)
                axis_colors = pelvis_axis_colors
                marker_colors = pelvis_marker_colors
            else:
                # Continue to use Jaguars colors for additional clusters (Trunk style)
                axis_colors = trunk_axis_colors
                marker_colors = trunk_marker_colors

            for j, (point_set, color, marker_label) in enumerate(
                zip(points, marker_colors, marker_labels)
            ):
                data.append(
                    go.Scatter3d(
                        x=point_set[: frame + 1, 0],
                        y=point_set[: frame + 1, 1],
                        z=point_set[: frame + 1, 2],
                        mode="lines+markers",
                        line=dict(color=color, width=4),
                        marker=dict(color=color, size=4),
                        name=f"{label} - {marker_label}",
                    )
                )

            origin = pm[frame]
            for k, (axis, color) in enumerate(zip(bases[frame], axis_colors)):
                data.append(
                    go.Scatter3d(
                        x=[origin[0], origin[0] + axis[0] * axis_length],
                        y=[origin[1], origin[1] + axis[1] * axis_length],
                        z=[origin[2], origin[2] + axis[2] * axis_length],
                        mode="lines",
                        line=dict(color=color, width=6),
                        name=f'{label} Axis {["X", "Y", "Z"][k]}',
                    )
                )

        # Global coordinate system (red, green, blue)
        for i, (axis, color) in enumerate(
            zip(global_coordinate_system, ["red", "green", "blue"])
        ):
            data.append(
                go.Scatter3d(
                    x=[0, axis[0] * axis_length],
                    y=[0, axis[1] * axis_length],
                    z=[0, axis[2] * axis_length],
                    mode="lines",
                    line=dict(color=color, width=4),
                    name=f'Global Axis {["X", "Y", "Z"][i]}',
                )
            )

        frames.append(go.Frame(data=data, name=str(frame)))

    fig.update(frames=frames)
    fig.add_traces(frames[0].data)

    sliders = [
        dict(
            steps=[
                dict(
                    method="animate",
                    args=[
                        [str(k)],
                        dict(
                            mode="immediate",
                            frame=dict(duration=100, redraw=True),
                            transition=dict(duration=0),
                        ),
                    ],
                    label=str(k),
                )
                for k in range(len(pm_list[0]))
            ],
            transition=dict(duration=0),
            x=0,
            y=0,
            currentvalue=dict(
                font=dict(size=14), prefix="Frame: ", visible=True, xanchor="center"
            ),
            len=1.0,
        )
    ]

    fig.update_layout(
        sliders=sliders,
        scene=dict(
            xaxis=dict(
                range=[-1, 1],
                autorange=False,
                showgrid=True,
                zeroline=True,
                showline=True,
            ),
            yaxis=dict(
                range=[-1, 1],
                autorange=False,
                showgrid=True,
                zeroline=True,
                showline=True,
            ),
            zaxis=dict(
                range=[0, 2],
                autorange=False,
                showgrid=True,
                zeroline=True,
                showline=True,
            ),
            aspectratio=dict(x=2, y=2, z=4),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.show()


def plot_orthonormal_bases_4points_matplotlib(
    bases_list,
    pm_list,
    points_list,
    labels,
    global_coordinate_system=None,
    title="Orthonormal Bases with 4 Points",
):
    trunk_axis_colors, trunk_marker_colors, pelvis_axis_colors, pelvis_marker_colors = (
        get_colors()
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    ax.set_box_aspect([2, 2, 4])

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 2])

    if global_coordinate_system is None:
        global_coordinate_system = np.eye(3)

    axis_colors = ["r", "g", "b"]
    vector_length = 0.2

    for i, (axis, color) in enumerate(zip(global_coordinate_system, axis_colors)):
        ax.plot(
            [0, axis[0] * vector_length],
            [0, axis[1] * vector_length],
            [0, axis[2] * vector_length],
            color=color,
            linewidth=1,
        )

    for i, (bases, pm, points, label) in enumerate(
        zip(bases_list, pm_list, points_list, labels)
    ):
        if "trunk" in label.lower():
            axis_colors = trunk_axis_colors
            marker_colors = trunk_marker_colors
        elif "pelvis" in label.lower():
            axis_colors = pelvis_axis_colors
            marker_colors = pelvis_marker_colors

        for point_set, color in zip(points, marker_colors):
            ax.scatter(
                point_set[:, 0],
                point_set[:, 1],
                point_set[:, 2],
                color=color,
                s=1,
                marker=".",
            )

        for j in range(0, len(pm), 10):
            origin = pm[j]
            for k, (axis, color) in enumerate(zip(bases[j], axis_colors)):
                ax.plot(
                    [origin[0], origin[0] + axis[0] * vector_length],
                    [origin[1], origin[1] + axis[1] * vector_length],
                    [origin[2], origin[2] + axis[2] * vector_length],
                    color=color,
                )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return fig


def plot_orthonormal_bases_4points_plotly(
    bases_list,
    pm_list,
    points_list,
    labels,
    global_coordinate_system=None,
    title="Orthonormal Bases with 4 Points",
):
    trunk_axis_colors, trunk_marker_colors, pelvis_axis_colors, pelvis_marker_colors = (
        get_colors()
    )

    frames = []
    axis_length = 0.2

    if global_coordinate_system is None:
        global_coordinate_system = np.eye(3)

    marker_labels = ["Marker 1", "Marker 2", "Marker 3", "Marker 4"]

    fig = make_subplots(
        rows=1, cols=1, specs=[[{"type": "scatter3d"}]], subplot_titles=[title]
    )

    for frame in range(len(pm_list[0])):
        data = []

        for i, (bases, pm, points, label) in enumerate(
            zip(bases_list, pm_list, points_list, labels)
        ):
            if "trunk" in label.lower():
                axis_colors = trunk_axis_colors
                marker_colors = trunk_marker_colors
            elif "pelvis" in label.lower():
                axis_colors = pelvis_axis_colors
                marker_colors = pelvis_marker_colors

            for j, (point_set, color, marker_label) in enumerate(
                zip(points, marker_colors, marker_labels)
            ):
                data.append(
                    go.Scatter3d(
                        x=point_set[: frame + 1, 0],
                        y=point_set[: frame + 1, 1],
                        z=point_set[: frame + 1, 2],
                        mode="lines+markers",
                        line=dict(color=color, width=4),
                        marker=dict(color=color, size=1),
                        name=f"{label} - {marker_label}",
                    )
                )

            origin = pm[frame]
            for k, (axis, color) in enumerate(zip(bases[frame], axis_colors)):
                data.append(
                    go.Scatter3d(
                        x=[origin[0], origin[0] + axis[0] * axis_length],
                        y=[origin[1], origin[1] + axis[1] * axis_length],
                        z=[origin[2], origin[2] + axis[2] * axis_length],
                        mode="lines",
                        line=dict(color=color, width=4),
                        name=f'{label} Axis {["X", "Y", "Z"][k]}',
                    )
                )

        for i, (axis, color) in enumerate(
            zip(global_coordinate_system, ["red", "green", "blue"])
        ):
            data.append(
                go.Scatter3d(
                    x=[0, axis[0] * axis_length],
                    y=[0, axis[1] * axis_length],
                    z=[0, axis[2] * axis_length],
                    mode="lines",
                    line=dict(color=color, width=4),
                    name=f'Global Axis {["X", "Y", "Z"][i]}',
                )
            )

        frames.append(go.Frame(data=data, name=str(frame)))

    fig.update(frames=frames)
    fig.add_traces(frames[0].data)

    sliders = [
        dict(
            steps=[
                dict(
                    method="animate",
                    args=[
                        [str(k)],
                        dict(
                            mode="immediate",
                            frame=dict(duration=100, redraw=True),
                            transition=dict(duration=0),
                        ),
                    ],
                    label=str(k),
                )
                for k in range(len(pm_list[0]))
            ],
            transition=dict(duration=0),
            x=0,
            y=0,
            currentvalue=dict(
                font=dict(size=1.52), prefix="Frame: ", visible=True, xanchor="center"
            ),
            len=1.0,
        )
    ]

    fig.update_layout(
        sliders=sliders,
        scene=dict(
            xaxis=dict(
                range=[-1, 1],
                autorange=False,
                showgrid=True,
                zeroline=True,
                showline=True,
            ),
            yaxis=dict(
                range=[-1, 1],
                autorange=False,
                showgrid=True,
                zeroline=True,
                showline=True,
            ),
            zaxis=dict(
                range=[0, 2],
                autorange=False,
                showgrid=True,
                zeroline=True,
                showline=True,
            ),
            aspectratio=dict(x=2, y=2, z=4),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.show()


def plot_orthonormal_bases(
    bases_list,
    pm_list,
    points_list,
    labels,
    global_coordinate_system=None,
    title="Orthonormal Bases",
    four_points=False,
):
    if four_points:
        fig_matplotlib = plot_orthonormal_bases_4points_matplotlib(
            bases_list, pm_list, points_list, labels, global_coordinate_system, title
        )
        plot_orthonormal_bases_4points_plotly(
            bases_list, pm_list, points_list, labels, global_coordinate_system, title
        )
    else:
        fig_matplotlib = plot_orthonormal_bases_matplotlib(
            bases_list, pm_list, points_list, labels, global_coordinate_system, title
        )
        plot_orthonormal_bases_plotly(
            bases_list, pm_list, points_list, labels, global_coordinate_system, title
        )
    return fig_matplotlib