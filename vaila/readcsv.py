"""
readcsv.py

Name: Your Name
Date: 29/07/2024

Description:
Script to visualize data from .csv files using Dash and Plotly,
with header selection interface and frame animation.

Version: 0.1
"""

import os
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import tkinter as tk
from tkinter import filedialog, Toplevel, Checkbutton, BooleanVar, Button, Scrollbar, Canvas, Frame
import threading
import webbrowser
import numpy as np
from flask import request

def determine_header_lines(file_path):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            first_element = line.split(',')[0].strip()
            if first_element.replace('.', '', 1).isdigit():
                return i
    return 0

def headersidx(file_path):
    try:
        header_lines = determine_header_lines(file_path)
        df = pd.read_csv(file_path, header=list(range(header_lines)))

        print("Headers with indices:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i}: {col}")

        print("\nExample of new order:")
        new_order = ['Time']
        for i in range(1, len(df.columns), 3):
            new_order.append(df.columns[i][0])
        print(new_order)

        return new_order

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def reshapedata(file_path, new_order, save_directory):
    try:
        header_lines = determine_header_lines(file_path)
        df = pd.read_csv(file_path, skiprows=header_lines, header=None)
        actual_header = pd.read_csv(file_path, nrows=header_lines, header=None).values
        new_order_indices = [0]

        for header in new_order[1:]:
            base_idx = [i for i, col in enumerate(actual_header[0]) if col == header][0]
            new_order_indices.extend([base_idx, base_idx + 1, base_idx + 2])

        df_reordered = df.iloc[:, new_order_indices]
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        new_file_path = os.path.join(save_directory, f"{base_name}_reord.csv")
        
        new_header = []
        for header in new_order:
            if header == 'Time':
                new_header.append(header)
            else:
                new_header.extend([header + '_x', header + '_y', header + '_z'])

        df_reordered.to_csv(new_file_path, index=False, header=new_header)
        print(f"Reordered data saved to {new_file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Function to read header labels from .csv file
def get_csv_headers(file_path):
    df = pd.read_csv(file_path)
    return list(df.columns)

# Function to open file dialog and choose .csv file
def select_file():
    file_path = filedialog.askopenfilename(title="Pick file to select headers", filetypes=[("CSV files", "*.csv")])
    return file_path

# Function to show header selection interface
def select_headers_gui(headers):
    selected_headers = []

    def on_select():
        nonlocal selected_headers
        selected_headers = [header for header, var in zip(headers, header_vars) if var.get()]
        if len(selected_headers) % 3 != 0:
            tk.messagebox.showerror("Error", "Please select columns in multiples of 3 (X, Y, Z for each marker).")
            selected_headers.clear()
        else:
            selection_window.quit()  # End the main Tkinter loop
            selection_window.destroy()  # Destroy the selection window

    def select_all():
        for var in header_vars:
            var.set(True)

    def unselect_all():
        for var in header_vars:
            var.set(False)

    selection_window = Toplevel()
    selection_window.title("Select Headers")
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

    header_vars = [BooleanVar() for _ in headers]

    num_columns = 6  # Number of columns for the labels

    for i, label in enumerate(headers):
        chk = Checkbutton(scrollable_frame, text=label, variable=header_vars[i])
        chk.grid(row=i // num_columns, column=i % num_columns, sticky='w')

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    btn_frame = Frame(selection_window)
    btn_frame.pack(side="right", padx=10, pady=10, fill='y', anchor='center')

    btn_select_all = Button(btn_frame, text="Select All", command=select_all)
    btn_select_all.pack(side="top", pady=5)

    btn_unselect_all = Button(btn_frame, text="Unselect All", command=unselect_all)
    btn_unselect_all.pack(side="top", pady=5)

    btn_select = Button(btn_frame, text="Confirm", command=on_select)
    btn_select.pack(side="top", pady=5)

    selection_window.mainloop()

    return selected_headers

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

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
    dcc.Interval(id='interval', interval=50, n_intervals=0, disabled=True),  # Interval for animation
    html.Div([
        html.Button('Play', id='play-button', n_clicks=0, style={'marginRight': '10px'}),  # Play button with margin
        html.Button('Stop Server', id='stop-button', n_clicks=0)  # Stop Server button
    ], style={'textAlign': 'center'}),  # Center the buttons
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
    frame = selected_frame % len(markers[0]['x'])  # Ensure the frame index is within bounds

    fig = go.Figure()

    if 'markers' in globals():
        for marker in markers:
            fig.add_trace(go.Scatter3d(
                x=[marker['x'][frame]],
                y=[marker['y'][frame]],
                z=[marker['z'][frame]],
                mode='markers',
                marker=dict(size=4),
                name=marker['name']
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
                backgroundcolor='#A7A8A9'
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

# Callback to update slider position during playback
@app.callback(
    Output('frame-slider', 'value'),
    [Input('interval', 'n_intervals')],
    [State('frame-slider', 'value')]
)
def update_slider(n_intervals, current_value):
    new_value = (current_value + 1) % len(markers[0]['x'])
    return new_value

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

# Callback to handle stop button
@app.callback(
    Output('stop-button', 'n_clicks'),
    [Input('stop-button', 'n_clicks')]
)
def stop_server(n_clicks):
    if n_clicks > 0:
        # Stop the Dash server by closing the Flask server
        func = request.environ.get('werkzeug.server.shutdown')
        if func is not None:
            func()
    return n_clicks

# Function to run the Dash app in a separate thread
def run_dash():
    app.run_server(debug=True, use_reloader=False)

# Main function to show data from .csv file
def show_csv():
    global markers, selected_headers, file_name, global_coordinate_system, axis_length
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main Tkinter window

        file_path = select_file()
        if file_path:
            headers = get_csv_headers(file_path)
            file_name = os.path.basename(file_path)
            
            selected_headers = select_headers_gui(headers[1:])  # Skip the first column

            if selected_headers:
                df = pd.read_csv(file_path, usecols=selected_headers)
                # Verify if all selected headers exist in the DataFrame
                missing_headers = [header for header in selected_headers if header not in headers]
                if missing_headers:
                    print(f"Error: The following selected headers were not found in the CSV file: {missing_headers}")
                else:
                    print("All selected headers are present in the CSV file.")
                
                markers = []
                for i in range(0, len(selected_headers), 3):
                    if i+2 < len(selected_headers):
                        # Check if values are in millimeters
                        values_x = df[selected_headers[i]].values
                        values_y = df[selected_headers[i+1]].values
                        values_z = df[selected_headers[i+2]].values
                        
                        if np.mean(np.abs(values_x)) > 100 or np.mean(np.abs(values_y)) > 100 or np.mean(np.abs(values_z)) > 100:
                            factor = 0.001  # Conversion factor from mm to meters
                        else:
                            factor = 1  # No conversion needed

                        marker = {
                            'x': values_x * factor,  # Convert to meters if needed
                            'y': values_y * factor,  # Convert to meters if needed
                            'z': values_z * factor,  # Convert to meters if needed
                            'name': selected_headers[i].rsplit('_', 1)[0]  # Group by marker name before last '_'
                        }
                        markers.append(marker)

                global_coordinate_system = np.eye(3)
                axis_length = 0.2

                # Update slider max value and marks dynamically
                app.layout.children[1].max = len(df) - 1
                app.layout.children[1].marks = {i: str(i) for i in range(0, len(df), max(1, len(df) // 10))}

                # Ensure all Tkinter windows are destroyed
                root.update()
                root.destroy()

                # Launch the browser
                threading.Thread(target=run_dash).start()
                webbrowser.open('http://127.0.0.1:8050')

            else:
                print("No headers were selected.")
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
    show_csv()
