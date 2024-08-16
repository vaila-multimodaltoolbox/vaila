import os
import numpy as np
from ezc3d import c3d
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import tkinter as tk
from tkinter import (
    filedialog,
    Toplevel,
    Checkbutton,
    BooleanVar,
    Button,
    Scrollbar,
    Canvas,
    Frame,
)


# Function to read marker labels from .c3d file
def get_marker_labels(dat):
    datac3d = c3d(dat)
    marker_labels = datac3d["parameters"]["POINT"]["LABELS"]["value"]
    return marker_labels


# Function to read data of selected markers from .c3d file
def get_selected_marker_data(dat, selected_marker_indices):
    datac3d = c3d(dat)
    point_data = datac3d["data"]["points"] / 1000  # Convert to meters
    marker_labels = datac3d["parameters"]["POINT"]["LABELS"]["value"]
    marker_freq = datac3d["header"]["points"]["frame_rate"]

    # Filter only selected markers
    selected_markers = point_data[:, selected_marker_indices, :]
    markers = selected_markers[:3, :, :].transpose((2, 1, 0))
    num_frames = markers.shape[0]
    num_markers = markers.shape[1]

    return (
        markers,
        [marker_labels[i] for i in selected_marker_indices],
        marker_freq,
        num_frames,
        num_markers,
    )


# Function to create animation of markers using Plotly
def plot_markers_plotly(
    markers, marker_labels, marker_freq, num_frames, num_markers, file_name
):
    global_coordinate_system = np.eye(3)
    axis_length = 0.2  # Define the length of the coordinate system axes

    frames = []
    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"type": "scatter3d"}]],
        subplot_titles=[f"Markers Animation - {file_name}"],
    )

    for frame in range(num_frames):
        data = []
        for marker in range(num_markers):
            x = markers[frame, marker, 0]
            y = markers[frame, marker, 1]
            z = markers[frame, marker, 2]
            data.append(
                go.Scatter3d(
                    x=[x],
                    y=[y],
                    z=[z],
                    mode="markers",
                    marker=dict(size=4),
                    name=marker_labels[marker],
                )
            )

        # Add the global coordinate system
        for axis, color in zip(global_coordinate_system, ["red", "green", "blue"]):
            data.append(
                go.Scatter3d(
                    x=[0, axis[0] * axis_length],
                    y=[0, axis[1] * axis_length],
                    z=[0, axis[2] * axis_length],
                    mode="lines",
                    line=dict(color=color, width=5),
                    showlegend=False,  # Hide legend for global coordinate system
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
                            frame=dict(duration=50, redraw=True),
                            transition=dict(duration=0),
                        ),
                    ],
                    label=str(k),
                )
                for k in range(num_frames)
            ],
            transition=dict(duration=0),
            x=0,
            y=0,
            currentvalue=dict(
                font=dict(size=12), prefix="Frame: ", visible=True, xanchor="center"
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
        # title=f'Markers Kinematics Animation - {file_name}'
    )

    fig.show()


# Function to open file dialog and choose .c3d file
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("C3D files", "*.c3d")])
    return file_path


# Function to show marker selection interface
def select_markers_gui(marker_labels):
    selected_markers = []

    def on_select():
        nonlocal selected_markers
        selected_markers = [i for i, var in enumerate(marker_vars) if var.get()]
        selection_window.quit()  # End the main Tkinter loop
        selection_window.destroy()  # Destroy the marker selection window

    def clear_all():
        for var in marker_vars:
            var.set(False)

    selection_window = Toplevel()
    selection_window.title("Select Markers")
    selection_window.geometry(
        f"{selection_window.winfo_screenwidth()}x{int(selection_window.winfo_screenheight()*0.9)}"
    )

    canvas = Canvas(selection_window)
    scrollbar = Scrollbar(selection_window, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas)

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    marker_vars = [BooleanVar() for _ in marker_labels]

    num_columns = 10  # Number of columns for the labels

    for i, label in enumerate(marker_labels):
        chk = Checkbutton(scrollable_frame, text=label, variable=marker_vars[i])
        chk.grid(row=i // num_columns, column=i % num_columns, sticky="w")

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    btn_frame = Frame(selection_window)
    btn_frame.pack(side="right", padx=10, pady=10, fill="y", anchor="center")

    btn_clear = Button(btn_frame, text="Clear", command=clear_all)
    btn_clear.pack(side="top", pady=5)

    btn_select = Button(btn_frame, text="Confirm", command=on_select)
    btn_select.pack(side="top", pady=5)

    selection_window.mainloop()

    return selected_markers


# Main function to show data from .c3d file
def show_c3d():
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main Tkinter window

        file_path = select_file()
        if file_path:
            marker_labels = get_marker_labels(file_path)
            file_name = os.path.basename(file_path)

            selected_marker_indices = select_markers_gui(marker_labels)

            if selected_marker_indices:
                (
                    markers,
                    selected_marker_labels,
                    marker_freq,
                    num_frames,
                    num_markers,
                ) = get_selected_marker_data(file_path, selected_marker_indices)
                plot_markers_plotly(
                    markers,
                    selected_marker_labels,
                    marker_freq,
                    num_frames,
                    num_markers,
                    file_name,
                )
            else:
                print("No marker was selected.")
        else:
            print("No file was selected.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if "root" in locals():
            try:
                root.update()
                root.destroy()
            except tk.TclError:
                pass


if __name__ == "__main__":
    show_c3d()
