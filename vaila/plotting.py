import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def get_colors():
    trunk_axis_colors = ["red", "green", "blue"]
    trunk_marker_colors = ["darkcyan", "darkmagenta", "darkgoldenrod", "darkblue"]
    pelvis_axis_colors = ["#fbbb9b", "#006778", "#223D79"]
    pelvis_marker_colors = ["darkorange", "darkkhaki", "dimgray", "indigo"]

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
        if "cluster1" in label.lower():
            axis_colors = trunk_axis_colors
            marker_colors = trunk_marker_colors
        elif "cluster2" in label.lower():
            axis_colors = pelvis_axis_colors
            marker_colors = pelvis_marker_colors
        else:
            marker_colors = [
                "black",
                "gray",
                "silver",
                "white",
            ]  # Default generic colors

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


def plot_orthonormal_bases_plotly(
    bases_list,
    pm_list,
    points_list,
    labels,
    global_coordinate_system=None,
    title="Orthonormal Bases",
):
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
            if "cluster1" in label.lower():
                axis_colors = trunk_axis_colors
                marker_colors = trunk_marker_colors
            elif "cluster2" in label.lower():
                axis_colors = pelvis_axis_colors
                marker_colors = pelvis_marker_colors
            else:
                marker_colors = [
                    "black",
                    "gray",
                    "silver",
                    "white",
                ]  # Default generic colors

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
