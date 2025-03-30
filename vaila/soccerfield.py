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

import os
from rich import print
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tkinter as tk
from tkinter import filedialog, Button, Frame, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import math


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


def plot_field(df, show_reference_points=True):
    """
    Plots a soccer field using coordinates from the DataFrame.

    Args:
        df: DataFrame with field point coordinates
        show_reference_points: Whether to show reference point numbers on the field

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
    points = {row["point_name"]: (row["x"], row["y"], row["point_number"]) for _, row in df.iterrows()}

    # Draw extended area (including 2m margin around the field)
    draw_rectangle(
        ax,
        (points["bottom_left_corner"][0] - margin, points["bottom_left_corner"][1] - margin),
        105 + 2 * margin,
        68 + 2 * margin,
        edgecolor="none",
        facecolor="green",
        zorder=0,
    )

    # Draw the main playing surface with slightly darker green to differentiate
    draw_rectangle(
        ax,
        points["bottom_left_corner"][0:2],
        105,
        68,
        edgecolor="none",
        facecolor="forestgreen",  # Slightly darker green
        zorder=0.5,
    )

    # Draw perimeter lines (12cm = 0.12m width)
    # To maintain the proportion, we will use linewidth=2 to represent the 12cm

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

    # Center circle
    draw_circle(
        ax,
        points["center_field"][0:2],
        9.15,
        edgecolor="white",
        facecolor="none",
        linewidth=2,
        zorder=1,
    )

    # Center spot
    draw_circle(
        ax,
        points["center_field"][0:2],
        0.2,
        edgecolor="white",
        facecolor="white",
        linewidth=2,
        zorder=1,
    )

    # Left penalty area
    draw_rectangle(
        ax,
        points["left_penalty_area_bottom_left"][0:2],
        16.5,
        40.3,
        edgecolor="white",
        facecolor="none",
        linewidth=2,
        zorder=1,
    )

    # Left goal area
    draw_rectangle(
        ax,
        points["left_goal_area_bottom_left"][0:2],
        5.5,
        18.32,
        edgecolor="white",
        facecolor="none",
        linewidth=2,
        zorder=1,
    )

    # Left penalty spot
    draw_circle(
        ax,
        points["left_penalty_spot"][0:2],
        0.2,
        edgecolor="white",
        facecolor="white",
        linewidth=2,
        zorder=1,
    )

    # Left penalty arc - desenha APENAS a parte FORA da grande área
    # Para o arco esquerdo, usamos os limites reais para garantir que nada da meia lua
    # com x < 16.5 seja desenhado (dentro da grande área)
    
    # Calculando os ângulos exatos onde a circunferência cruza a linha x=16.5
    left_pen_x = points["left_penalty_spot"][0]  # 11.0
    left_pen_y = points["left_penalty_spot"][1]  # 34.0
    radius = 9.15
    
    # Encontrar o ângulo exato onde x=16.5 na circunferência
    # A equação da circunferência é: (x-x0)² + (y-y0)² = r²
    # Para x=16.5: (16.5-11.0)² + (y-34.0)² = 9.15²
    # Resolvendo para y: (y-34.0)² = 9.15² - (16.5-11.0)²
    # Ângulo theta = atan2(y-y0, x-x0)
    
    dx = 16.5 - left_pen_x  # 5.5
    dy = math.sqrt(radius**2 - dx**2)  # Distância vertical do centro ao ponto de interseção
    
    # Dois pontos de interseção: (16.5, 34.0+dy) e (16.5, 34.0-dy)
    top_y = left_pen_y + dy
    bottom_y = left_pen_y - dy
    
    # Calculando os ângulos exatos (em graus)
    top_angle = math.degrees(math.atan2(top_y - left_pen_y, 16.5 - left_pen_x))
    bottom_angle = math.degrees(math.atan2(bottom_y - left_pen_y, 16.5 - left_pen_x))
    
    # Normalizando os ângulos para o intervalo 0-360
    if top_angle < 0:
        top_angle += 360
    if bottom_angle < 0:
        bottom_angle += 360
    
    draw_arc(
        ax,
        points["left_penalty_spot"][0:2],
        9.15,
        theta1=bottom_angle,  # Ângulo exato inferior
        theta2=top_angle,     # Ângulo exato superior
        edgecolor="white",
        linewidth=2,
        zorder=1,
    )

    # Right penalty area
    draw_rectangle(
        ax,
        points["right_penalty_area_top_left"][0:2],
        16.5,
        40.3,
        edgecolor="white",
        facecolor="none",
        linewidth=2,
        zorder=1,
    )

    # Right goal area
    draw_rectangle(
        ax,
        points["right_goal_area_top_left"][0:2],
        5.5,
        18.32,
        edgecolor="white",
        facecolor="none",
        linewidth=2,
        zorder=1,
    )

    # Right penalty spot
    draw_circle(
        ax,
        points["right_penalty_spot"][0:2],
        0.2,
        edgecolor="white",
        facecolor="white",
        linewidth=2,
        zorder=1,
    )

    # Right penalty arc - desenha APENAS a parte FORA da grande área
    # Para o arco direito, usamos os limites reais para garantir que nada da meia lua
    # com x > 88.5 seja desenhado (dentro da grande área)
    
    # Calculando os ângulos exatos onde a circunferência cruza a linha x=88.5
    right_pen_x = points["right_penalty_spot"][0]  # 94.0
    right_pen_y = points["right_penalty_spot"][1]  # 34.0
    
    # Encontrar o ângulo exato onde x=88.5 na circunferência
    dx = 88.5 - right_pen_x  # -5.5
    dy = math.sqrt(radius**2 - dx**2)  # Distância vertical do centro ao ponto de interseção
    
    # Dois pontos de interseção: (88.5, 34.0+dy) e (88.5, 34.0-dy)
    right_top_y = right_pen_y + dy
    right_bottom_y = right_pen_y - dy
    
    # Calculando os ângulos exatos (em graus)
    right_top_angle = math.degrees(math.atan2(right_top_y - right_pen_y, 88.5 - right_pen_x))
    right_bottom_angle = math.degrees(math.atan2(right_bottom_y - right_pen_y, 88.5 - right_pen_x))
    
    # Normalizando os ângulos para o intervalo 0-360
    if right_top_angle < 0:
        right_top_angle += 360
    if right_bottom_angle < 0:
        right_bottom_angle += 360
    
    draw_arc(
        ax,
        points["right_penalty_spot"][0:2],
        9.15,
        theta1=right_top_angle,    # Ângulo exato superior 
        theta2=right_bottom_angle, # Ângulo exato inferior
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
        for name, (x, y, num) in points.items():
            ax.text(x + 0.5, y + 0.5, str(num), 
                    color='black', fontsize=8, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                    zorder=10)
    
    return fig, ax


def load_and_plot_markers(field_ax, csv_path, canvas, selected_markers=None):
    """
    Loads data from a CSV file and plots numbered markers with paths.
    All frames are plotted on the same image (hold on).
    
    Args:
        field_ax: Matplotlib axes of the field
        csv_path: Path to the CSV file with x,y coordinates
        canvas: Matplotlib canvas for updates
        selected_markers: List of marker names to display (None for all)
    """
    # Load CSV
    markers_df = pd.read_csv(csv_path)
    
    print(f"File loaded: {csv_path}")
    print(f"Number of frames (rows): {len(markers_df)}")
    
    # Data cleaning - convert empty strings to NaN
    markers_df = markers_df.replace('', np.nan)
    
    # Clear previous markers - modificar para preservar pontos de referência (zorder=10)
    for artist in field_ax.get_children():
        if hasattr(artist, 'get_zorder') and artist.get_zorder() > 10 and artist.get_zorder() < 100:
            artist.remove()
    
    # Identify all coordinate columns (except 'frame')
    cols = markers_df.columns
    marker_names = set()
    for col in cols:
        if col != 'frame' and ('_x' in col or '_y' in col):
            marker_names.add(col.split('_')[0])
    
    marker_names = sorted(list(marker_names))
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
                valid_mask = (valid_data[x_col] < 120) & (valid_data[x_col] > -20) & \
                             (valid_data[y_col] < 80) & (valid_data[y_col] > -20)
                
                valid_x = valid_data.loc[valid_mask, x_col].values
                valid_y = valid_data.loc[valid_mask, y_col].values
                
                if len(valid_x) > 0:
                    # Plot trajectory (line)
                    field_ax.plot(valid_x, valid_y, '-', color=colors[idx], 
                                 linewidth=1.5, alpha=0.7, zorder=50)
                    
                    # Plot points
                    field_ax.scatter(valid_x, valid_y, color=colors[idx], 
                                    s=60, marker='o', edgecolor='black',
                                    linewidth=1, alpha=0.8, zorder=51)
                    
                    # Add label to the last valid point
                    field_ax.text(valid_x[-1]+0.5, valid_y[-1]+0.5, 
                                 # Substituir 'M' por 'p' se o nome do marcador começar com 'M'
                                 marker.replace('M', 'p') if marker.startswith('M') else marker,
                                 fontsize=7, color='black', weight='bold',
                                 bbox=dict(facecolor=colors[idx], alpha=0.7, 
                                          edgecolor='black', boxstyle='round', pad=0.1),
                                 zorder=52)
                    
                    print(f"Plotted marker {marker} with {len(valid_x)} points")
    
    # Update canvas once at the end
    canvas.draw()
    
    if selected_markers:
        print(f"Plotting complete - Displaying {len(selected_markers)} of {len(marker_names)} markers")
    else:
        print("Plotting complete - all frames drawn on the same image")


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
    current_field_csv = [None]  # Store the current field CSV path
    current_markers_csv = [None]  # Store the current markers CSV path
    selected_markers = [None]  # Store currently selected markers
    
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
            fig, ax = plot_field(df, show_reference_points=show_reference_points[0])
            
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
                load_and_plot_markers(current_ax[0], current_markers_csv[0], 
                                     current_canvas[0], selected_markers[0])
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
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
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
            load_and_plot_markers(current_ax[0], current_markers_csv[0], current_canvas[0], selected_markers[0])
    
    def load_markers_csv():
        """Opens dialog to select marker CSV and plot it"""
        if current_ax[0] is None or current_canvas[0] is None:
            print("Please load the field first.")
            return
            
        # Open dialog to select file
        csv_path = filedialog.askopenfilename(
            title="Select CSV file with marker coordinates",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not csv_path:
            return
            
        try:
            print(f"\nStarting file loading: {csv_path}")
            # Check if file exists and can be read
            with open(csv_path, 'r') as f:
                first_line = f.readline()
                print(f"First line of file: {first_line}")
            
            # Store the markers path for potential reloads
            current_markers_csv[0] = csv_path
            
            # Reset selected markers when loading a new file
            selected_markers[0] = None
            
            # Use stored canvas
            load_and_plot_markers(current_ax[0], csv_path, current_canvas[0])
            
            # Enable the marker selection button
            select_markers_button.config(state=tk.NORMAL)
            
        except Exception as e:
            print(f"Error plotting markers: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def open_marker_selection_dialog():
        """Opens a dialog to select which markers to display"""
        if not hasattr(current_ax[0], '_all_marker_names') or not current_ax[0]._all_marker_names:
            print("No markers available to select.")
            return
            
        # Create a new top-level window
        select_window = tk.Toplevel(root)
        select_window.title("Select Markers to Display")
        select_window.geometry("300x400")
        
        # Add a frame with scrollbar for many markers
        frame = Frame(select_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Canvas for scrolling
        canvas = tk.Canvas(frame, yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=canvas.yview)
        
        # Frame inside canvas for checkboxes
        checkbox_frame = Frame(canvas)
        canvas.create_window((0, 0), window=checkbox_frame, anchor=tk.NW)
        
        # Dictionary to store checkbox variables
        checkbox_vars = {}
        
        # Create a label at the top
        tk.Label(checkbox_frame, text="Select markers to display:", 
                font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Create checkboxes for all markers
        for i, marker in enumerate(current_ax[0]._all_marker_names):
            var = tk.IntVar(value=1)  # Default checked
            checkbox_vars[marker] = var
            
            tk.Checkbutton(checkbox_frame, text=f"Marker {marker}", 
                          variable=var).grid(row=i+1, column=0, sticky=tk.W)
        
        # Update canvas scroll region
        checkbox_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox(tk.ALL))
        
        # Buttons frame
        button_frame = Frame(select_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def select_all():
            for var in checkbox_vars.values():
                var.set(1)
                
        def deselect_all():
            for var in checkbox_vars.values():
                var.set(0)
        
        def apply_selection():
            # Get list of selected markers
            selected = [marker for marker, var in checkbox_vars.items() if var.get() == 1]
            
            if not selected:
                print("Warning: No markers selected, showing all markers.")
                selected_markers[0] = None
            else:
                selected_markers[0] = selected
                
            # Replot with selected markers
            if current_markers_csv[0]:
                load_and_plot_markers(current_ax[0], current_markers_csv[0], 
                                     current_canvas[0], selected_markers[0])
            
            select_window.destroy()
        
        # Add buttons
        tk.Button(button_frame, text="Select All", command=select_all,
                 bg="white", fg="black", padx=5, pady=2).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Deselect All", command=deselect_all,
                 bg="white", fg="black", padx=5, pady=2).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Apply", command=apply_selection,
                 bg="white", fg="black", padx=20, pady=2).pack(side=tk.RIGHT, padx=5)
    
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
            circle = patches.Circle((x, y), radius=0.4, 
                                  color=plt.cm.tab10(marker_num % 10),
                                  edgecolor='black', linewidth=1, 
                                  alpha=0.8, zorder=100)
            ax.add_patch(circle)
            
            # Posicionar o texto com tamanho reduzido
            text = ax.text(x + 0.5, y + 0.5, f"p{marker_num}", 
                          fontsize=7, color='black', weight='bold',
                          bbox=dict(facecolor=plt.cm.tab10(marker_num % 10), alpha=0.7, 
                                   edgecolor='black', boxstyle='round', pad=0.1),
                          zorder=101)
            
            # Armazenar os objetos para possível exclusão
            manual_marker_artists.append((circle, text, x, y, marker_num, current_frame_idx))
            
            # Atualizar o canvas
            current_canvas[0].draw()
            print(f"Created marker p{marker_num} at frame {current_frame_idx}, position ({x:.2f}, {y:.2f})")
            
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
            closest_dist = float('inf')
            
            for i, (_, _, mx, my, _, _) in enumerate(manual_marker_artists):
                dist = np.sqrt((x - mx)**2 + (y - my)**2)
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
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if not csv_path:  # User canceled
                print("Save operation canceled.")
                return
            
            # Encontrar o número máximo de marcadores
            max_marker_num = 0
            for frame_data in frame_markers.values():
                for marker_num in frame_data.keys():
                    max_marker_num = max(max_marker_num, marker_num)
            
            # Encontrar o número máximo de frames
            max_frame = max(frame_markers.keys()) if frame_markers else 0
            
            # Criar dados para o CSV
            data = {'frame': list(range(max_frame + 1))}
            
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
        if canvas is None or not hasattr(canvas, 'get_tk_widget'):
            return
            
        canvas_widget = canvas.get_tk_widget()
        
        # Bind left mouse click (Button-1) to create marker
        canvas_widget.bind("<Button-1>", create_marker)
        
        # Bind right mouse click (Button-3) to delete marker
        canvas_widget.bind("<Button-3>", delete_marker)
        
        # Bind Ctrl+S to save markers
        root.bind("<Control-s>", save_markers_csv)
    
    def clear_all_markers():
        """Clear all manually created markers"""
        if current_ax[0] is None:
            return
            
        try:
            # Remove all marker artists from the plot
            for circle, text, _, _, _, _ in manual_marker_artists:
                circle.remove()
                text.remove()
            
            # Clear the lists and dictionaries
            manual_marker_artists.clear()
            frame_markers.clear()
            
            # Reset frame counter
            current_frame[0] = 0
            
            # Update the canvas
            current_canvas[0].draw()
            print("All markers cleared")
            
        except Exception as e:
            print(f"Error clearing markers: {e}")
            import traceback
            traceback.print_exc()
    
    # Add buttons
    Button(button_frame, text="Load Default Field", command=load_field, 
           bg="white", fg="black", padx=10, pady=5).pack(side=tk.LEFT, padx=5, pady=5)
    
    Button(button_frame, text="Load Custom Field", command=load_custom_field, 
           bg="white", fg="black", padx=10, pady=5).pack(side=tk.LEFT, padx=5, pady=5)
    
    Button(button_frame, text="Load Markers CSV", command=load_markers_csv,
           bg="white", fg="black", padx=10, pady=5).pack(side=tk.LEFT, padx=5, pady=5)
    
    # Add toggle button for reference points
    ref_points_button = Button(button_frame, text="Hide Reference Points", command=toggle_reference_points,
                              bg="white", fg="black", padx=10, pady=5)
    ref_points_button.pack(side=tk.LEFT, padx=5, pady=5)
    
    # Add marker selection button - initially disabled
    select_markers_button = Button(button_frame, text="Select Markers", command=open_marker_selection_dialog,
                                  bg="white", fg="black", padx=10, pady=5, state=tk.DISABLED)
    select_markers_button.pack(side=tk.LEFT, padx=5, pady=5)
    
    # Add manual marker mode button
    manual_marker_button = Button(button_frame, text="Create Manual Markers", command=toggle_manual_marker_mode,
                                 bg="white", fg="black", padx=10, pady=5)
    manual_marker_button.pack(side=tk.LEFT, padx=5, pady=5)
    
    # Add Clear All button
    Button(button_frame, text="Clear All Markers", command=lambda: clear_all_markers(),
           bg="white", fg="black", padx=10, pady=5).pack(side=tk.LEFT, padx=5, pady=5)
    
    # Frame for plotting
    plot_frame = Frame(root)
    plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    
    # Load field initially
    load_field()
    
    # Start Tkinter loop
    root.mainloop()


if __name__ == "__main__":
    run_soccerfield()
