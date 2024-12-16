"""
===============================================================================
heatmap_pathway_plot.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Date: 13 December 2024
Version: 2.0.0
Python Version: 3.11.11

Description:
------------
This script processes movement data of animals in an open field test, performing 
kinematic analyses including total distance traveled, average speed, and time 
spent in specific zones.

Key Features:
-------------
- Reads movement data from .csv files (X and Y positions over time).
- Calculates the total distance traveled.
- Calculates the average speed of the animal.
- Analyzes the time spent in different zones of a 60x60 cm open field, divided into 
  3x3 grid cells of 20x20 cm each.
- Generates visualizations including heatmaps and pathways of the animal's movement.
- Saves results and figures in an organized directory structure.

Dependencies:
-------------
- Python 3.x
- numpy
- matplotlib
- seaborn
- scipy
- tkinter

Usage:
------
- Run the script, select the directory containing .csv files with movement data.
- The .csv files should contain columns:
  - time(s), position_x(m), position_y(m).
- Results, including figures and a text summary, will be saved in a timestamped 
  directory with subdirectories for each processed file.

Example:
--------
$ python heatmap_pathway_plot.py

Notes:
------
- Ensure input .csv files are correctly formatted with positions in meters.
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk, filedialog, simpledialog, messagebox
from datetime import datetime
from pathlib import Path


def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


def calculate_zone_occupancy(x, y):
    zones = {
        (i, j): 0 for i in range(3) for j in range(3)
    }  # Inicializar contagem das zonas
    total_points = len(x)

    # Contar os pontos em cada zona
    for i in range(len(x)):
        if 0 <= x[i] <= 0.6 and 0 <= y[i] <= 0.6:
            zone_x = min(int(x[i] // 0.2), 2)
            zone_y = min(int(y[i] // 0.2), 2)
            zones[(zone_x, zone_y)] += 1

    # Calcular os percentuais
    raw_percentages = {
        zone: (count / total_points) * 100 for zone, count in zones.items()
    }
    total_percentage = sum(raw_percentages.values())

    # Ajustar o último percentual para fechar 100%
    last_zone = list(raw_percentages.keys())[-1]  # Última zona
    raw_percentages[last_zone] += 100 - total_percentage  # Ajuste no último percentual

    return zones, raw_percentages


def calculate_center_and_border_occupancy(x, y):
    total_points = len(x)
    points_in_center = sum(
        1 for i in range(len(x)) if 0.1 <= x[i] <= 0.5 and 0.1 <= y[i] <= 0.5
    )
    points_in_border = total_points - points_in_center
    return {
        "points_in_center": points_in_center,
        "percentage_in_center": (points_in_center / total_points) * 100,
        "points_in_border": points_in_border,
        "percentage_in_border": (points_in_border / total_points) * 100,
    }


def calculate_zone_occupancy(x, y):
    """
    Calcula a quantidade de pontos e porcentagens em cada zona 3x3.

    Args:
        x (array-like): Coordenadas X.
        y (array-like): Coordenadas Y.

    Returns:
        zones (dict): Contagem de pontos por zona.
        percentages (dict): Porcentagens de pontos por zona.
    """
    zones = {(i, j): 0 for i in range(3) for j in range(3)}  # Inicializar zonas
    total_points = len(x)

    # Contar pontos em cada zona
    for i in range(len(x)):
        if 0 <= x[i] <= 0.6 and 0 <= y[i] <= 0.6:
            zone_x = int(x[i] // 0.2)  # Dividir em faixas de 0.2 m (3 zonas no X)
            zone_y = 2 - int(
                y[i] // 0.2
            )  # Inverter o eixo Y para corresponder ao gráfico
            zones[(zone_x, zone_y)] += 1

    # Calcular as porcentagens
    percentages = {zone: (count / total_points) * 100 for zone, count in zones.items()}
    return zones, percentages


def plot_pathway(x, y, output_dir, base_name):
    """
    Plota o caminho (pathway) do movimento do animal no open field.

    Args:
        x (array-like): Coordenadas X.
        y (array-like): Coordenadas Y.
        output_dir (str): Diretório de saída.
        base_name (str): Nome base do arquivo.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Criar a figura do caminho
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, color="blue", linewidth=1.5, alpha=0.7, label="Pathway")
    ax.scatter(
        x[0], y[0], color="green", s=50, label="Start", zorder=5
    )  # Ponto inicial
    ax.scatter(x[-1], y[-1], color="red", s=50, label="End", zorder=5)  # Ponto final

    # Adicionar grid e limites
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Position X (m)")
    ax.set_ylabel("Position Y (m)")
    ax.set_title("Pathway of Animal Movement")
    # ax.legend()

    # Adicionar linhas de grade das zonas (3x3)
    for i in range(1, 3):
        ax.axvline(i * 0.2, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(i * 0.2, color="black", linestyle="--", linewidth=0.8)

    # Salvar o gráfico
    output_file_path = os.path.join(output_dir, f"{base_name}_pathway.png")
    plt.savefig(output_file_path, bbox_inches="tight")
    plt.close()
    print(f"Pathway plot saved at: {output_file_path}")


def plot_heatmap(x, y, output_dir, base_name, zone_percentages):
    """
    Plota o heatmap corrigido com as zonas e porcentagens corretas.

    Args:
        x (array-like): Coordenadas X.
        y (array-like): Coordenadas Y.
        output_dir (str): Diretório de saída.
        base_name (str): Nome base do arquivo.
        zone_percentages (dict): Percentagens calculadas para cada zona.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Criar o heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.kdeplot(
        x=x, y=y, cmap="coolwarm", fill=True, levels=100, bw_adjust=1.5, thresh=0, ax=ax
    )
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Position X (m)")
    ax.set_ylabel("Position Y (m)")
    ax.set_title("Heatmap with Zone Grid")

    # Adicionar linhas de grade
    for i in range(1, 3):
        ax.axvline(i * 0.2, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(i * 0.2, color="black", linestyle="--", linewidth=0.8)

    # Mapear as zonas corretamente (de baixo para cima)
    zone_mapping = {
        1: (0, 2),
        2: (1, 2),
        3: (2, 2),
        4: (0, 1),
        5: (1, 1),
        6: (2, 1),
        7: (0, 0),
        8: (1, 0),
        9: (2, 0),
    }

    # Adicionar labels das zonas e porcentagens
    for zone_id, (zone_x, zone_y) in zone_mapping.items():
        center_x = (zone_x + 0.5) * 0.2
        center_y = (zone_y + 0.5) * 0.2
        percentage = zone_percentages.get((zone_x, zone_y), 0)
        ax.text(
            center_x,
            center_y,
            f"Z{zone_id}\n{percentage:.1f}%",
            color="black",
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
        )

    # Salvar o heatmap
    output_file_path = os.path.join(output_dir, f"{base_name}_heatmap.png")
    plt.savefig(output_file_path, bbox_inches="tight")
    plt.close()
    print(f"Heatmap plot saved at: {output_file_path}")


def plot_center_and_border_heatmap(x, y, output_dir, base_name):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.kdeplot(
        x=x, y=y, cmap="coolwarm", fill=True, levels=100, bw_adjust=1.5, thresh=0, ax=ax
    )
    rect = plt.Rectangle(
        (0.1, 0.1), 0.4, 0.4, linewidth=2, edgecolor="black", facecolor="none"
    )
    ax.add_patch(rect)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Position X (m)")
    ax.set_ylabel("Position Y (m)")
    ax.set_title("Heatmap with Central and Border Areas")
    output_file_path = os.path.join(
        output_dir, f"{base_name}_center_border_heatmap.png"
    )
    plt.savefig(output_file_path, bbox_inches="tight")
    plt.close()
    print(f"Central and border heatmap saved at: {output_file_path}")
    return output_file_path


def save_results_to_csv(results, center_border_results, output_dir, base_name):
    try:
        combined_file_path = os.path.join(output_dir, f"{base_name}_summary_zones.csv")
        header = ["zone_id", "points", "percentage"]
        data = []
        for (zone_x, zone_y), count in results["zone_counts"].items():
            zone_id = (zone_x * 3 + zone_y) + 1
            data.append([zone_id, count, results["zone_percentages"][(zone_x, zone_y)]])
        data.append(
            [
                "Center",
                center_border_results["points_in_center"],
                center_border_results["percentage_in_center"],
            ]
        )
        data.append(
            [
                "Border",
                center_border_results["points_in_border"],
                center_border_results["percentage_in_border"],
            ]
        )
        with open(combined_file_path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for row in data:
                f.write(",".join(map(str, row)) + "\n")
        print(f"Results saved to: {combined_file_path}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        raise


def save_position_data(time_vector, x, y, distance, speed, output_dir, base_name):
    try:
        position_file_path = os.path.join(output_dir, f"{base_name}_position_data.csv")
        with open(position_file_path, "w", encoding="utf-8") as f:
            f.write("time_s,x_m,y_m,distance_m,speed_m/s\n")
            for i in range(len(time_vector)):
                f.write(
                    f"{time_vector[i]:.6f},{x[i]:.6f},{y[i]:.6f},{distance[i]:.6f},{speed[i]:.6f}\n"
                )
        print(f"Position data saved to: {position_file_path}")
    except Exception as e:
        print(f"Error saving position data to CSV: {e}")
        raise


def process_open_field_data(input_file, main_output_dir, fs):
    try:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_dir = os.path.join(main_output_dir, base_name)
        os.makedirs(output_dir, exist_ok=True)
        data = np.loadtxt(input_file, delimiter=",", skiprows=1, usecols=(1, 2))
        x, y = data[:, 0], data[:, 1]
        time_vector = np.linspace(0, len(x) / fs, len(x))
        distance = np.insert(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2), 0, 0)
        speed = np.insert(distance[1:] / (1 / fs), 0, 0)
        zones, zone_percentages = calculate_zone_occupancy(x, y)
        center_border_results = calculate_center_and_border_occupancy(x, y)
        results = {"zone_counts": zones, "zone_percentages": zone_percentages}
        save_results_to_csv(results, center_border_results, output_dir, base_name)
        save_position_data(time_vector, x, y, distance, speed, output_dir, base_name)
        plot_center_and_border_heatmap(x, y, output_dir, base_name)
        plot_heatmap(x, y, output_dir, base_name, zone_percentages)
        plot_pathway(x, y, output_dir, base_name)  # Adicionar a plotagem do pathway
        plot_center_and_border_heatmap(x, y, output_dir, base_name)
        print(f"Processing of file {input_file} completed successfully.")
    except Exception as e:
        print(f"An error occurred while processing {input_file}: {e}")
        raise


def process_all_files_in_directory(target_dir, fs):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(target_dir, f"openfield_results_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)
    csv_files = [
        os.path.join(target_dir, f)
        for f in os.listdir(target_dir)
        if f.endswith(".csv")
    ]
    for input_file in csv_files:
        print(f"Processing file: {input_file}")
        process_open_field_data(input_file, main_output_dir, fs)
    print("All files have been processed successfully.")


def run_animal_open_field():
    print("Running open field analysis...")
    root = Tk()
    root.withdraw()
    target_dir = filedialog.askdirectory(
        title="Select the directory containing .csv files of open field data"
    )
    if not target_dir:
        messagebox.showwarning("Warning", "No directory selected.")
        return
    fs = simpledialog.askfloat(
        "Sampling Frequency", "Enter the sampling frequency (Hz):", minvalue=0.1
    )
    if not fs:
        messagebox.showwarning("Warning", "Sampling frequency not provided.")
        return
    process_all_files_in_directory(target_dir, fs)
    root.destroy()
    messagebox.showinfo(
        "Success", "All .csv files have been processed and results saved."
    )


if __name__ == "__main__":
    run_animal_open_field()
