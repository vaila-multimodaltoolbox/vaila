import os
import numpy as np
from ezc3d import c3d
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import tkinter as tk
from tkinter import filedialog, Toplevel, Checkbutton, BooleanVar, Button, Scrollbar, Canvas, Frame

# Function to read marker labels from .c3d file
def get_marker_labels(dat):
    datac3d = c3d(dat)
    marker_labels = datac3d['parameters']['POINT']['LABELS']['value']
    return marker_labels

# Function to read data of selected markers from .c3d file
def get_selected_marker_data(dat, selected_marker_indices):
    datac3d = c3d(dat)
    point_data = datac3d['data']['points'] / 1000  # Convert to meters
    marker_labels = datac3d['parameters']['POINT']['LABELS']['value']
    marker_freq = datac3d['header']['points']['frame_rate']
    
    # Filter only selected markers
    selected_markers = point_data[:, selected_marker_indices, :]
    markers = selected_markers[:3, :, :].transpose((2, 1, 0))
    num_frames = markers.shape[0]
    num_markers = markers.shape[1]
    
    return markers, [marker_labels[i] for i in selected_marker_indices], marker_freq, num_frames, num_markers

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
    selection_window.geometry(f'{selection_window.winfo_screenwidth()}x{int(selection_window.winfo_screenheight()*0.9)}')

    canvas = Canvas(selection_window)
    scrollbar = Scrollbar(selection_window, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    marker_vars = [BooleanVar() for _ in marker_labels]

    num_columns = 10  # Number of columns for the labels

    for i, label in enumerate(marker_labels):
        chk = Checkbutton(scrollable_frame, text=label, variable=marker_vars[i])
        chk.grid(row=i // num_columns, column=i % num_columns, sticky='w')

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    btn_frame = Frame(selection_window)
    btn_frame.pack(side="right", padx=10, pady=10, fill='y', anchor='center')

    btn_clear = Button(btn_frame, text="Clear", command=clear_all)
    btn_clear.pack(side="top", pady=5)

    btn_select = Button(btn_frame, text="Confirm", command=on_select)
    btn_select.pack(side="top", pady=5)

    selection_window.mainloop()

    return selected_markers

# Initialize Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='scatter-plot', style={'height': '90vh'}, config={'scrollZoom': True}),
    dcc.Slider(
        id='frame-slider',
        min=0,
        max=0,
        step=1,
        value=0,
        marks={i: str(i) for i in range(0, 1)},  # Dynamic marks
    ),
    html.Div(id='frame-display', style={'textAlign': 'center', 'marginTop': '10px'}),
    dcc.Store(id='camera-store'),  # Use dcc.Store to store camera state
    dcc.Store(id='annotations-store', data=[]),  # Store to keep track of annotations
    html.Div(id='keypress', style={'display': 'none'}),  # Hidden div to capture keypress
    dcc.Interval(id='interval', interval=1000, n_intervals=0, disabled=True),  # Interval for animation
    html.Button('Play', id='play-button', n_clicks=0),  # Play button
])

# Function to convert camera state to JSON-serializable format
def camera_to_dict(camera):
    if camera is None:
        return None
    return {
        'center': camera['center'],
        'eye': camera['eye'],
        'up': camera['up'],
        'projection': camera['projection'],
    }

# Define the callback to update the plot
@app.callback(
    [Output('scatter-plot', 'figure'), Output('camera-store', 'data'), Output('frame-display', 'children')],
    [Input('frame-slider', 'value'), Input('interval', 'n_intervals')],
    [State('scatter-plot', 'relayoutData')]
)
def update_figure(selected_frame, n_intervals, relayoutData):
    frame = selected_frame + n_intervals  # Use both slider value and interval to update frame
    if frame >= num_frames:  # Reset interval if frame exceeds number of frames
        frame = num_frames - 1

    fig = go.Figure()

    if markers is not None:
        for marker in range(num_markers):
            x = markers[frame, marker, 0]
            y = markers[frame, marker, 1]
            z = markers[frame, marker, 2]
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                marker=dict(size=4),
                name=selected_marker_labels[marker]  # Use selected_marker_labels here
            ))

        # Add the global coordinate system
        for axis, color in zip(global_coordinate_system, ['red', 'green', 'blue']):
            fig.add_trace(go.Scatter3d(
                x=[0, axis[0] * axis_length],
                y=[0, axis[1] * axis_length],
                z=[0, axis[2] * axis_length],
                mode='lines',
                line=dict(color=color, width=5),
                showlegend=False  # Hide legend for global coordinate system
            ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                range=[-1, 1], 
                autorange=False, 
                showgrid=True, 
                zeroline=True, 
                showline=True, 
                #gridcolor='#A7A8A9',  # Add this line
                #zerolinecolor='#A7A8A9'  # Add this line
            ),
            yaxis=dict(
                range=[-1, 1], 
                autorange=False, 
                showgrid=True, 
                zeroline=True, 
                showline=True, 
                #gridcolor='#A7A8A9',  # Add this line
                #zerolinecolor='#A7A8A9'  # Add this line
            ),
            zaxis=dict(
                range=[0, 2], 
                autorange=False, 
                showgrid=True, 
                zeroline=True, 
                showline=True,
                backgroundcolor='#A7A8A9'
                #gridcolor='#A7A8A9',  # Add this line
                #zerolinecolor='#A7A8A9'  # Add this line
            ),
            aspectratio=dict(x=2, y=2, z=4),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title={
            'text': f'{file_name}',
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    # Restore the camera view if available
    if relayoutData and 'scene.camera' in relayoutData:
        fig['layout']['scene']['camera'] = relayoutData['scene.camera']

    # Convert camera to dict for JSON serialization
    camera_state = fig['layout']['scene']['camera'] if 'scene.camera' in fig['layout'] else None

    # Display the current frame number
    frame_display = f'Current frame: {frame}'

    return fig, camera_to_dict(camera_state), frame_display

# Callback to handle keypress events
@app.callback(
    Output('annotations-store', 'data'),
    [Input('frame-slider', 'value')],
    [State('annotations-store', 'data'), State('keypress', 'n_clicks')]
)
def handle_keypress(selected_frame, annotations, n_clicks):
    if not annotations:
        annotations = []

    # This is a placeholder for keypress detection
    # You can implement actual keypress detection using JavaScript and sending events to Dash
    key = None  # Placeholder for actual keypress value

    if key == 'p':
        annotations.append(selected_frame)
        with open(f'{file_name}.txt', 'a') as f:
            f.write(f'{len(annotations)},{selected_frame}\n')
    elif key == 'd' and annotations:
        annotations.pop()
        with open(f'{file_name}.txt', 'w') as f:
            for i, frame in enumerate(annotations):
                f.write(f'{i+1},{frame}\n')

    return annotations

# Callback to handle play button
@app.callback(
    Output('interval', 'disabled'),
    [Input('play-button', 'n_clicks')],
    [State('interval', 'disabled')]
)
def toggle_interval(n_clicks, interval_disabled):
    if n_clicks % 2 == 1:
        return False  # Enable interval
    else:
        return True  # Disable interval

# Main function to show data from .c3d file
def show_c3d():
    global markers, selected_marker_labels, num_markers, num_frames, global_coordinate_system, axis_length, file_name
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main Tkinter window

        file_path = select_file()
        if file_path:
            marker_labels = get_marker_labels(file_path)
            file_name = os.path.basename(file_path)
            
            selected_marker_indices = select_markers_gui(marker_labels)

            if selected_marker_indices:
                markers, selected_marker_labels, marker_freq, num_frames, num_markers = get_selected_marker_data(file_path, selected_marker_indices)
                global_coordinate_system = np.eye(3)
                axis_length = 0.2

                # Update slider max value and marks dynamically
                app.layout.children[1].max = num_frames - 1
                app.layout.children[1].marks = {i: str(i) for i in range(0, num_frames, max(1, num_frames // 10))}

                # Ensure all Tkinter windows are destroyed
                root.update()
                root.destroy()

                # Launch the browser
                import webbrowser
                webbrowser.open('http://127.0.0.1:8050')

                app.run_server(debug=True, use_reloader=False)
            else:
                print("No marker was selected.")
        else:
            print("No file was selected.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'root' in locals():
            try:
                root.update()
                root.destroy()
            except tk.TclError:
                pass

if __name__ == "__main__":
    show_c3d()
