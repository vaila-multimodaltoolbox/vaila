"""
===============================================================================
cube2d_kinematics.py
===============================================================================
Author: Prof. Dr. Paulo Roberto Pereira Santiago
Date: 2025-01-16
Version: 0.0.1
Python Version: 3.12.8

Description:
This module provides functionality for analyzing 2D kinematics data from cube-based movement assessments.

Key Features:
- Processes CSV files containing 2D position data (x,y coordinates)
- Calculates kinematic metrics including:
  - Total distance traveled
  - Average speed
  - Time spent stationary
  - Total movement time
- Divides movement space into 9 quadrants for spatial analysis
- Generates visualizations:
  - Movement pathway plots with quadrant overlays
  - Color-coded speed profiles
- Handles batch processing of multiple files
- Outputs results to organized directories with timestamps
- Provides GUI interface for:
  - File/directory selection
  - Parameter input
  - Results viewing

Dependencies:
- NumPy: Numerical computations and array operations
- Pandas: Data manipulation and analysis
- Matplotlib: Data visualization and plotting
- Tkinter: Graphical user interface
- SciPy: Signal processing functions

The module is designed for research and clinical applications in movement science,
particularly for analyzing confined space movement patterns like those seen in
agility or balance tests using a 9-square grid layout.

How to use:

1. Run the module using the run_cube2d_kinematics() function.
2. Select the data directory containing the CSV files.
3. Select the quadrants file (optional, default is a 9-square grid).
4. Select the output directory for the results.
5. Enter the sampling frequency (fs) for all files.
6. The module will process all CSV files in the selected directory,
   generate pathway plots with quadrants, and save the results to the output directory.

File need to be in the following format:
The input CSV files should contain two columns:
- Column 1: X coordinates (medio-lateral position in meters)
- Column 2: Y coordinates (antero-posterior position in meters)

Example format:
x,y
0.1,0.2
0.15,0.25
0.2,0.3
...

The data should be sampled at a consistent frequency (fs) which will be specified
during processing. The coordinates should represent positions in meters relative
to a defined origin point (typically the center of the movement area).

Example files can be found in the tests/Cube2d_kinematics directory.
conda activate vaila
python vaila/cube2d_kinematics.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from tkinter import (
    Tk,
    filedialog,
    messagebox,
    StringVar,
    Label,
    Entry,
    Button,
    simpledialog,
)
from datetime import datetime
from scipy.signal import butter, filtfilt
from matplotlib.collections import LineCollection

# Define the default quadrants using numpy arrays
quadrants = np.array(
    [
        [1, 0.5, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.0],
        [2, 0.0, 0.5, 0.0, 1.0, 0.5, 1.0, 0.5, 0.5],
        [3, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5],
        [4, 1.0, 0.5, 1.0, 1.0, 1.5, 1.0, 1.5, 0.5],
        [5, 1.0, 0.0, 1.0, 0.5, 1.5, 0.5, 1.5, 0.0],
        [6, 1.0, -0.5, 1.0, 0.0, 1.5, 0.0, 1.5, -0.5],
        [7, 0.5, -0.5, 0.5, 0.0, 1.0, 0.0, 1.0, -0.5],
        [8, 0.0, -0.5, 0.0, 0.0, 0.5, 0.0, 0.5, -0.5],
        [9, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0],
    ]
)

# Define column names for better understanding
column_names = [
    "quadrant",
    "vertex1_x",
    "vertex1_y",
    "vertex2_x",
    "vertex2_y",
    "vertex3_x",
    "vertex3_y",
    "vertex4_x",
    "vertex4_y",
]


def load_quadrants(file_path=None):
    """
    Load quadrants from a file or return default quadrants as a pandas DataFrame.
    """
    # Convert default quadrants to DataFrame
    default_df = pd.DataFrame(quadrants, columns=column_names)

    if file_path and file_path.endswith(".txt"):
        try:
            # Read the txt file into a DataFrame and convert to numeric values
            loaded_df = pd.read_csv(file_path)
            # Convert all columns except 'quadrant' to float
            for col in loaded_df.columns:
                if col != "quadrant":
                    loaded_df[col] = pd.to_numeric(loaded_df[col], errors="coerce")
            return loaded_df
        except Exception as e:
            print(f"Error reading quadrants file: {e}. Using default quadrants.")
            return default_df
    return default_df


def load_data(input_file):
    """
    Loads the input file, processes the columns for X and Y coordinates,
    and computes their mean if there are multiple X and Y columns.

    Args:
        input_file (str): Path to the input CSV file.

    Returns:
        tuple: Tuple containing arrays for X and Y coordinates.
    """
    # Load the data, skipping the header
    data = np.genfromtxt(input_file, delimiter=",", skip_header=1)

    # Exclude the first column (time/frame)
    data = data[:, 1:]

    # Separate X and Y columns (odd columns for X, even columns for Y)
    x = data[:, ::2]  # Columns at indices 1, 3, 5, ...
    y = data[:, 1::2]  # Columns at indices 2, 4, 6, ...

    # Compute the mean along the rows if there are multiple columns
    x_mean = np.mean(x, axis=1) if x.shape[1] > 1 else x.flatten()
    y_mean = np.mean(y, axis=1) if y.shape[1] > 1 else y.flatten()

    return x_mean, y_mean


def butter_lowpass_filter(data, cutoff, fs, order=4, padding=True):
    """
    Applies a Butterworth low-pass filter to the input data with optional padding.

    Parameters:
    - data: array-like
        The input signal to be filtered.
    - cutoff: float
        The cutoff frequency for the low-pass filter.
    - fs: float
        The sampling frequency of the signal.
    - order: int, default=4
        The order of the Butterworth filter.
    - padding: bool, default=True
        Whether to pad the signal to mitigate edge effects.

    Returns:
    - filtered_data: array-like
        The filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    data = np.asarray(data)

    if padding:
        data_len = len(data)
        max_padlen = data_len - 1
        padlen = min(int(fs), max_padlen, 15)

        if data_len <= padlen:
            raise ValueError(
                f"The length of the input data ({data_len}) must be greater than the padding length ({padlen})."
            )

        # Apply reflection padding
        padded_data = np.pad(data, pad_width=(padlen, padlen), mode="reflect")
        filtered_padded_data = filtfilt(b, a, padded_data, padlen=0)
        filtered_data = filtered_padded_data[padlen:-padlen]
    else:
        filtered_data = filtfilt(b, a, data, padlen=0)

    return filtered_data


def calculate_distance(x, y):
    distance = np.insert(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2), 0, 0)
    return distance


def calculate_speed(distance, fs):
    return distance * fs


def plot_pathway_with_quadrants(x, y, quadrants_df, time_vector):
    """Plot pathway with time-based color gradient and quadrants."""

    # Plot the quadrants and pathway
    fig, ax = plt.subplots(figsize=(8, 8))

    # Para cada quadrante no DataFrame
    for _, quad in quadrants_df.iterrows():
        # Extrair os vértices
        vertices_x = [
            quad["vertex1_x"],
            quad["vertex2_x"],
            quad["vertex3_x"],
            quad["vertex4_x"],
            quad["vertex1_x"],
        ]
        vertices_y = [
            quad["vertex1_y"],
            quad["vertex2_y"],
            quad["vertex3_y"],
            quad["vertex4_y"],
            quad["vertex1_y"],
        ]

        # Desenhar o quadrante
        ax.plot(vertices_x, vertices_y, color="gray", linewidth=2)

        # Calcular o centro do quadrante para o número
        center_x = np.mean(vertices_x[:-1])  # Exclui o último ponto que é repetido
        center_y = np.mean(vertices_y[:-1])

        # Ajustar a posição do número do quadrante 1
        if int(quad["quadrant"]) == 1:
            center_y -= (
                0.25  # Ajusta a posição do texto para baixo apenas para o quadrante 1
            )

        # Adicionar o número do quadrante
        ax.text(
            center_x,
            center_y,
            str(int(quad["quadrant"])),
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="circle", facecolor="white"),
        )

    # Criar o gradiente de cores para o caminho
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap="plasma", norm=norm)
    lc.set_array(np.linspace(0, 1, len(time_vector)))
    lc.set_linewidth(2)
    line = ax.add_collection(lc)

    # Plot first point as a green dot
    ax.scatter(x[0], y[0], color="green", s=100, zorder=5)

    # Plot last point as a red dot
    ax.scatter(x[-1], y[-1], color="red", s=100, zorder=5)

    # Adicionar barra de cores
    cbar = plt.colorbar(line, ax=ax, orientation="vertical")
    cbar.set_label("Time (s)")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([f"{time_vector[0]:.2f}", f"{time_vector[-1]:.2f}"])

    # Configurações do gráfico
    ax.set_title("CUBE 2D Pathway with Time-Based Color Gradient", fontsize=14)
    ax.set_xlabel("X - Medio-lateral (m)")
    ax.set_ylabel("Y - Antero-posterior (m)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_aspect("equal", "box")

    return fig


def process_all_files(file_paths, quadrants, output_dir):
    # Solicitar fs uma vez no início
    fs = None
    while fs is None:
        try:
            fs_value = input("Enter the sampling frequency (fs) for all files: ")
            fs = float(fs_value)
            if fs <= 0:
                raise ValueError("Sampling frequency must be positive")
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter a valid positive number.")

    for file_path in file_paths:
        process_file(file_path, quadrants, output_dir, fs)


def process_file(file_path, quadrants_df, output_dir, fs, base_name):
            # Process the file with the selected fs
    x, y = load_data(file_path)
    x = butter_lowpass_filter(x, 6, fs)
    y = butter_lowpass_filter(y, 6, fs)

    # Calculate metrics
    distance = calculate_distance(x, y)
    speed = calculate_speed(distance, fs)
    total_distance = np.sum(distance)
    avg_speed = np.mean(speed)
    time_stationary = np.sum(speed < 0.05) / fs  # Time below 0.05 m/s
    total_time = len(x) / fs  # Total time in seconds

    # Create a time vector
    time_vector = np.linspace(0, (len(x) - 1) / fs, len(x))

    # Plot and save pathway with quadrants
    plt.figure(figsize=(8, 8))
    plot_pathway_with_quadrants(x, y, quadrants_df, time_vector)
    plt.savefig(os.path.join(output_dir, f"{base_name}_cube2d_result.png"))
    plt.close()

    # Save metrics to text file
    with open(os.path.join(output_dir, f"{base_name}_cube2d_result.txt"), "w") as f:
        f.write(f"Total distance: {total_distance:.2f} m\n")
        f.write(f"Average speed: {avg_speed:.2f} m/s\n")
        f.write(f"Time stationary: {time_stationary:.2f} s\n")
        f.write(f"Total time: {total_time:.2f} s\n")


def run_cube2d_kinematics():
    # Inicializar a interface Tkinter
    root = Tk()
    root.withdraw()  # Ocultar a janela principal

    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    print("Starting CUBE 2D Kinematics analysis...")

    # Solicitar o diretório de dados
    data_dir = filedialog.askdirectory(title="Select the Data Directory")
    if not data_dir:
        print("No data directory selected. Exiting.")
        return

    # Solicitar o arquivo de quadrantes
    quadrants_file = filedialog.askopenfilename(
        title="Select the Quadrants File", filetypes=[("Text files", "*.txt")]
    )
    if not quadrants_file:
        print("No quadrants file selected. Exiting.")
        return

    # Solicitar o diretório de saída
    output_dir = filedialog.askdirectory(title="Select the Output Directory")
    if not output_dir:
        print("No output directory selected. Exiting.")
        return

    # Solicitar fs uma vez no início usando um diálogo
    fs = None
    while fs is None:
        try:
            fs_value = simpledialog.askstring(
                "Input", "Enter the sampling frequency (fs) for all files:"
            )
            if fs_value is None:
                print("No sampling frequency entered. Exiting.")
                return
            fs = float(fs_value)
            if fs <= 0:
                raise ValueError("Sampling frequency must be positive")
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter a valid positive number.")

    # Carregar quadrantes
    quadrants_df = load_quadrants(quadrants_file)

    # Criar diretório base com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(output_dir, f"vaila_cube2d_{timestamp}")
    os.makedirs(base_output_dir, exist_ok=True)

    # Processar todos os arquivos no diretório de dados
    files_processed = 0
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(data_dir, file_name)
            # Remover a extensão .csv do nome do arquivo
            base_name = os.path.splitext(file_name)[0]

            # Criar subdiretório para cada arquivo com timestamp
            file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_output_dir = os.path.join(
                base_output_dir, f"{base_name}_cube2d_{file_timestamp}"
            )
            os.makedirs(file_output_dir, exist_ok=True)

            # Processar o arquivo com os novos nomes para os arquivos de resultado
            process_file(file_path, quadrants_df, file_output_dir, fs, base_name)
            files_processed += 1

    # Mostrar mensagem de conclusão
    messagebox.showinfo(
        "Processing Complete",
        f"Analysis completed successfully!\n\n"
        f"Files processed: {files_processed}\n"
        f"Output directory: {base_output_dir}",
    )


if __name__ == "__main__":
    run_cube2d_kinematics()
