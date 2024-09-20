"""
================================================================================
Cluster Data Processing Tool
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 2024-08-29
Version: 1.3

Description:
------------
This Python script processes kinematic data from CSV files to calculate and visualize 
cluster-based rotations. It applies filtering techniques, computes Euler angles, and allows 
the user to choose whether to display the results as figures. The user can analyze multiple 
files in batch mode and optionally compare results with anatomical data.

Main Features:
--------------
1. Process CSV files containing cluster-based data.
2. Apply filtering to the data before performing calculations.
3. Compute and visualize Euler angles for clusters.
4. Optionally compare with anatomical Euler angle data.
5. Allow the user to save results and optionally display figures after processing.

Changelog:
----------
Version 1.3 - 2024-08-29:
    - Added option to show or hide figures after processing.
    - Improved user input handling for file selection and configuration.
    - Enhanced error handling during file processing.

License:
--------
This script is licensed under the MIT License.

================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tkinter import messagebox, filedialog, Tk
from vaila.filtering import apply_filter
from vaila.rotation import createortbase, calcmatrot, rotmat2euler
from vaila.plotting import plot_orthonormal_bases
from vaila.readcsv import get_csv_headers, select_headers_gui
from vaila.dialogsuser_cluster import get_user_inputs
from PIL import Image
from rich import print


def save_results_to_csv(
    base_dir, time, cluster1_euler_angles, cluster2_euler_angles, file_name
):
    results = {
        "time": time,
        "cluster1_euler_x": cluster1_euler_angles[:, 0],
        "cluster1_euler_y": cluster1_euler_angles[:, 1],
        "cluster1_euler_z": cluster1_euler_angles[:, 2],
        "cluster2_euler_x": cluster2_euler_angles[:, 0],
        "cluster2_euler_y": cluster2_euler_angles[:, 1],
        "cluster2_euler_z": cluster2_euler_angles[:, 2],
    }
    df = pd.DataFrame(results)
    base_name = os.path.splitext(file_name)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file_path = os.path.join(base_dir, f"{base_name}_cluster_{timestamp}.csv")
    df.to_csv(result_file_path, index=False)


def read_anatomical_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        cluster1_median = (
            data[["cluster1_euler_x", "cluster1_euler_y", "cluster1_euler_z"]]
            .median()
            .values
        )
        cluster2_median = (
            data[["cluster2_euler_x", "cluster2_euler_y", "cluster2_euler_z"]]
            .median()
            .values
        )
        return {"cluster1": cluster1_median, "cluster2": cluster2_median}
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def analyze_cluster_data():
    print("Starting analyze_cluster_data...")

    # Configuração inicial com janela do Tkinter
    root = Tk()
    root.withdraw()  # Esconde a janela principal

    # Seleção do diretório de arquivos CSV
    selected_path = filedialog.askdirectory(title="Select Directory with CSV Files")
    if not selected_path:
        messagebox.showerror("No Directory Selected", "No directory selected. Exiting.")
        root.destroy()
        return

    print(f"Selected path: {selected_path}")

    # Pergunta se o usuário deseja utilizar dados anatômicos
    use_anatomical = messagebox.askyesno(
        "Use Anatomical Angles", "Do you want to analyze with anatomical angle data?"
    )

    # Exibe a imagem de configuração dos clusters
    image_path = os.path.join("vaila", "images", "cluster_config.png")
    image = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.title(
        "Memorize the cluster configuration (A/B/C/D) of the trunk (Cluster 1) and pelvis (Cluster 2)"
    )
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    # Coleta de entrada do usuário para as configurações dos clusters
    print("Calling get_user_inputs...")
    user_input = get_user_inputs()
    sample_rate = user_input.get("sample_rate")
    cluster1_config = user_input.get("cluster1_config")
    cluster2_config = user_input.get("cluster2_config")

    # Verifica se as configurações são válidas
    configurations = ["A", "B", "C", "D"]
    if cluster1_config not in configurations or cluster2_config not in configurations:
        messagebox.showerror(
            "Invalid Input",
            "Invalid input for configuration. Please enter 'A', 'B', 'C', or 'D'.",
        )
        root.destroy()
        return

    # Solicitação do arquivo para seleção de cabeçalhos
    selected_file = filedialog.askopenfilename(
        title="Pick file to select headers", filetypes=[("CSV files", "*.csv")]
    )
    if not selected_file:
        messagebox.showerror("Error", "No file selected to choose headers.")
        root.destroy()
        return

    print(f"Selected file for headers: {selected_file}")

    headers = get_csv_headers(selected_file)
    selected_headers = select_headers_gui(headers)  # Seleção dos cabeçalhos

    print(f"Selected headers: {selected_headers}")

    filter_method = "butterworth"
    file_names = sorted([f for f in os.listdir(selected_path) if f.endswith(".csv")])

    # Seleção do diretório para salvar os resultados
    save_directory = filedialog.askdirectory(title="Choose Directory to Save Results")
    if not save_directory:
        messagebox.showerror(
            "No Directory Selected",
            "No directory selected for saving results. Exiting.",
        )
        root.destroy()
        return

    # Criação dos diretórios base para salvar figuras e dados processados
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir_figures = os.path.join(save_directory, f"Cluster_{current_time}/figures")
    base_dir_processed_data = os.path.join(
        save_directory, f"Cluster_{current_time}/processed_data"
    )
    os.makedirs(base_dir_figures, exist_ok=True)
    os.makedirs(base_dir_processed_data, exist_ok=True)

    print(f"Base directories created: {base_dir_figures}, {base_dir_processed_data}")

    anatomical_data = {}
    if use_anatomical:
        anatomical_file_path = filedialog.askopenfilename(
            title="Select Anatomical Euler Angles CSV File",
            filetypes=[("CSV files", "*.csv")],
        )
        if not anatomical_file_path:
            messagebox.showerror(
                "No File Selected", "No anatomical data file selected. Exiting."
            )
            root.destroy()
            return

        anat_data = read_anatomical_csv(anatomical_file_path)
        if anat_data:
            anatomical_data = anat_data
        else:
            messagebox.showerror(
                "Error Reading File", "Failed to read anatomical data file. Exiting."
            )
            root.destroy()
            return

    print(f"Anatomical data: {anatomical_data}")
    matplotlib_figs = []  # Lista para armazenar as figuras matplotlib

    for idx, file_name in enumerate(file_names):
        print(f"Processing file: {file_name}")
        file_path = os.path.join(selected_path, file_name)
        data = pd.read_csv(file_path, usecols=selected_headers).values

        # Criação do vetor de tempo
        time = np.linspace(0, len(data) / sample_rate, len(data))
        data = np.insert(data, 0, time, axis=1)

        # Aplicação do filtro nos dados
        dataf = apply_filter(data[:, 1:], sample_rate, method=filter_method)

        # Extração dos pontos com base nas entradas do usuário
        points = [dataf[:, i : i + 3] for i in range(0, 18, 3)]
        cluster1_p1, cluster1_p2, cluster1_p3, cluster2_p1, cluster2_p2, cluster2_p3 = (
            points
        )

        # Criação das bases ortonormais para os clusters
        cluster1_base, orig_cluster1 = createortbase(
            cluster1_p1, cluster1_p2, cluster1_p3, cluster1_config
        )
        cluster2_base, orig_cluster2 = createortbase(
            cluster2_p1, cluster2_p2, cluster2_p3, cluster2_config
        )

        # Plotando as bases ortonormais
        fig_matplotlib = plot_orthonormal_bases(
            bases_list=[cluster1_base, cluster2_base],
            pm_list=[orig_cluster1, orig_cluster2],
            points_list=[
                [cluster1_p1, cluster1_p2, cluster1_p3],
                [cluster2_p1, cluster2_p2, cluster2_p3],
            ],
            labels=["Cluster1", "Cluster2"],
            title=f"Cluster Bases - {file_name}",
        )

        matplotlib_figs.append(fig_matplotlib)

        # Cálculo das matrizes de rotação e ângulos de Euler
        cluster1_rotmat = calcmatrot(cluster1_base, np.eye(3))
        cluster2_rotmat = calcmatrot(cluster2_base, np.eye(3))
        cluster1_euler_angles = rotmat2euler(cluster1_rotmat)
        cluster2_euler_angles = rotmat2euler(cluster2_rotmat)

        # Salvando os resultados em CSV
        save_results_to_csv(
            base_dir_processed_data,
            time,
            cluster1_euler_angles,
            cluster2_euler_angles,
            file_name,
        )

        # Pergunta ao usuário se ele quer visualizar as figuras
    show_figures = messagebox.askyesno(
        "Show Figures", "Do you want to display all figures after processing?"
    )

    if show_figures:
        for fig in matplotlib_figs:
            fig.show()

    root.destroy()


if __name__ == "__main__":
    analyze_cluster_data()
