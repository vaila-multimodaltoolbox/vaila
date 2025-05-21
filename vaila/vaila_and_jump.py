"""
===============================================================================
vaila_and_jump.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 24 Oct 2024
Update Date: 01 May 2025
Version: 0.0.3
Python Version: 3.12.9

Description:
------------
This script processes jump data from multiple .csv files in a specified directory,
performing biomechanical calculations based on either the time of flight or the
jump height. The results are saved in a new output directory with a timestamp
for each processed file.

Features:
---------
- Supports two calculation modes:
  - Based on time of flight (calculates jump height)
  - Based on measured jump height (uses the height directly)
- Calculates various metrics for each jump:
  - Force
  - Liftoff Force (Thrust)
  - Velocity (takeoff)
  - Potential Energy
  - Kinetic Energy
  - Average Power (if contact time is provided)
  - Relative Power (W/kg)
  - Jump Performance Index (JPI)
  - Total time (if time of flight and contact time are available)
- Processes all .csv files in the selected directory.
- Saves the results in a new output directory named "vaila_verticaljump_<timestamp>".

Dependencies:
-------------
- Python 3.x
- pandas
- math
- tkinter

Usage:
------
- Run the script, select the target directory containing .csv files, and specify
  whether the data is based on time of flight or jump height.
- The script will process each .csv file, performing calculations and saving
  the results in a new directory.

Example:
--------
$ python vaila_and_jump.py

Input File Format:
-----------------
The input CSV files should have the following format:

1. Time-of-flight based format:
   mass(kg), time_of_flight(s), contact_time(s)[optional]
   
   Example:
   ```
   mass_kg,time_of_flight_s,contact_time_s
   75.0,0.45,0.22
   80.2,0.42,0.25
   65.5,0.48,0.20
   ```

2. Jump-height based format:
   mass(kg), height(m), contact_time(s)[optional]
   
   Example:
   ```
   mass_kg,heigth_m,contact_time_s
   75.0,0.25,0.22
   80.2,0.22,0.25
   65.5,0.28,0.20
   ```

Output:
-------
The script generates a CSV file with the following columns:
- height_m: Jump height in meters
- liftoff_force_N: Liftoff force in Newtons (if contact time is provided)
- velocity_m/s: Takeoff velocity in meters per second
- potential_energy_J: Potential energy in Joules
- kinetic_energy_J: Kinetic energy in Joules
- average_power_W: Average power in Watts (if contact time is provided)
- relative_power_W/kg: Power relative to body mass (if contact time is provided)
- jump_performance_index: Jump Performance Index (if both time of flight and contact time are available)
- total_time_s: Total time in seconds (if both time of flight and contact time are available)

Notes:
------
- The .csv files are expected to have the following columns in order:
  - Column 0: mass
  - Column 1: time of flight or jump height, depending on the selected mode
  - Column 2 (optional): contact time
- Ensure that all necessary libraries are installed.
===============================================================================
"""

import os
import math
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog, messagebox, simpledialog
from datetime import datetime
from pathlib import Path
from rich import print
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


def calculate_force(mass, gravity=9.81):
    """
    Calculate the weight force based on the mass and gravity.

    Args:
        mass (float): The mass of the jumper in kilograms.
        gravity (float, optional): The gravitational acceleration. Default is 9.81 m/s^2.

    Returns:
        float: The weight force in Newtons.
    """
    return mass * gravity


def calculate_jump_height(time_of_flight, gravity=9.81):
    """
    Calculate the jump height from the time of flight.

    Args:
        time_of_flight (float): The time of flight in seconds.
        gravity (float, optional): The gravitational acceleration. Default is 9.81 m/s^2.

    Returns:
        float: The jump height in meters.
    """
    return (gravity * time_of_flight**2) / 8


def calculate_power(force, height, contact_time):
    """
    Calculate the power output during the jump.

    Args:
        force (float): The force exerted during the jump in Newtons.
        height (float): The jump height in meters.
        contact_time (float): The contact time in seconds.

    Returns:
        float: The power output in Watts.
    """
    work = force * height
    return work / contact_time


def calculate_velocity(height, gravity=9.81):
    """
    Calculate the takeoff velocity needed to reach the given height.

    Args:
        height (float): The jump height in meters.
        gravity (float, optional): The gravitational acceleration. Default is 9.81 m/s^2.

    Returns:
        float: The takeoff velocity in meters per second.
    """
    return math.sqrt(2 * gravity * height)


def calculate_kinetic_energy(mass, velocity):
    """
    Calculate the kinetic energy based on mass and velocity.

    Args:
        mass (float): The mass of the jumper in kilograms.
        velocity (float): The velocity in meters per second.

    Returns:
        float: The kinetic energy in Joules.
    """
    return 0.5 * mass * velocity**2


def calculate_potential_energy(mass, height, gravity=9.81):
    """
    Calculate the potential energy based on mass and height.

    Args:
        mass (float): The mass of the jumper in kilograms.
        height (float): The height in meters.
        gravity (float, optional): The gravitational acceleration. Default is 9.81 m/s^2.

    Returns:
        float: The potential energy in Joules.
    """
    return mass * gravity * height


def calculate_average_power(potential_energy, contact_time):
    """
    Calculate the average power output during the contact phase.

    Args:
        potential_energy (float): The potential energy in Joules.
        contact_time (float): The contact time in seconds.

    Returns:
        float: The average power in Watts.
    """
    return potential_energy / contact_time


def calculate_liftoff_force(mass, velocity, contact_time, gravity=9.81):
    """
    Calculate the total liftoff force (thrust) required during the contact phase.

    Args:
        mass (float): The mass of the jumper in kilograms.
        velocity (float): The takeoff velocity in m/s.
        contact_time (float): The contact time in seconds.
        gravity (float, optional): The gravitational acceleration. Default is 9.81 m/s^2.

    Returns:
        float: The liftoff force in Newtons.
    """
    # Calculate the weight force
    weight_force = mass * gravity

    # Calculate the force needed to accelerate to the takeoff velocity
    acceleration_force = (mass * velocity) / contact_time

    # Total liftoff force
    liftoff_force = weight_force + acceleration_force

    return liftoff_force


def calculate_time_of_flight(height, gravity=9.81):
    """
    Calculate the time of flight from the jump height.
    
    Args:
        height (float): The jump height in meters.
        gravity (float, optional): The gravitational acceleration. Default is 9.81 m/s^2.
        
    Returns:
        float: The time of flight in seconds.
    """
    return math.sqrt(8 * height / gravity)

def calculate_baseline(data, n_frames=10):
    """
    Calcula a linha de base usando os primeiros n frames.
    
    Args:
        data (pd.DataFrame): DataFrame com os dados do MediaPipe
        n_frames (int): Número de frames para calcular a média da base
    
    Returns:
        tuple: (baseline_feet, baseline_cg)
    """
    # Calcula média dos primeiros n frames para os pés
    right_foot_baseline = data['right_foot_index_y'].iloc[:n_frames].mean()
    left_foot_baseline = data['left_foot_index_y'].iloc[:n_frames].mean()
    feet_baseline = (right_foot_baseline + left_foot_baseline) / 2
    
    # Calcula média dos primeiros n frames para o CG
    cg_y_baseline = data['cg_y'].iloc[:n_frames].mean()
    
    return feet_baseline, cg_y_baseline

def identify_jump_phases(data, feet_baseline, cg_baseline, fps):
    """
    Identifica as diferentes fases do salto.
    
    Args:
        data (pd.DataFrame): DataFrame com os dados
        feet_baseline (float): Linha de base dos pés
        cg_baseline (float): Linha de base do CG
        fps (int): Frames por segundo
    
    Returns:
        dict: Dicionário com informações das fases do salto
    """
    # Converter índices de frame para tempo
    time = data.index / fps
    
    # Encontrar fase de propulsão (quando CG começa a subir)
    propulsion_start = data[data['cg_y'] > cg_baseline].index[0]
    
    # Encontrar início da fase aérea (quando os pés deixam o solo)
    takeoff_idx = data[data['right_foot_index_y'] < feet_baseline].index[0]
    
    # Encontrar ponto mais alto (menor valor de CG_y já que y cresce para baixo)
    max_height_idx = data['cg_y'].idxmin()
    
    # Encontrar aterrissagem (quando os pés voltam ao solo)
    landing_idx = data[data.index > max_height_idx][
        data[data.index > max_height_idx]['right_foot_index_y'] >= feet_baseline
    ].index[0]
    
    # Calcular tempo de fase aérea
    flight_time = (landing_idx - takeoff_idx) / fps
    
    # Calcular altura máxima em relação à linha de base
    max_height = abs(data['cg_y'].min() - cg_baseline)
    
    return {
        'propulsion_start_frame': propulsion_start,
        'takeoff_frame': takeoff_idx,
        'max_height_frame': max_height_idx,
        'landing_frame': landing_idx,
        'flight_time_s': flight_time,
        'max_height_m': max_height,
        'propulsion_time_s': (takeoff_idx - propulsion_start) / fps,
        'ascent_time_s': (max_height_idx - takeoff_idx) / fps,
        'descent_time_s': (landing_idx - max_height_idx) / fps
    }

def generate_jump_plots(data, results, output_dir, base_name):
    """
    Generate time series plots for CG and feet position data.
    
    Args:
        data (pd.DataFrame): DataFrame with processed data
        results (dict): Dictionary with jump metrics
        output_dir (str): Directory to save the plots
        base_name (str): Base name for the output files
    
    Returns:
        list: Paths to the generated plot files
    """
    plot_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Create time series plot of CG and feet positions
    plt.figure(figsize=(12, 8))
    
    # Add frame numbers on x-axis
    frames = data.index.values if 'frame_index' not in data.columns else data['frame_index']
    time_seconds = frames / results['fps']
    
    # Plot CG position
    plt.plot(time_seconds, data['cg_y_m'], 'b-', linewidth=2, label='Center of Gravity (Y)')
    
    # Plot feet position if available
    if 'left_foot_index_y_m' in data.columns:
        plt.plot(time_seconds, data['left_foot_index_y_m'], 'g-', 
                 linewidth=1.5, label='Left Foot')
    if 'right_foot_index_y_m' in data.columns:
        plt.plot(time_seconds, data['right_foot_index_y_m'], 'r-', 
                 linewidth=1.5, label='Right Foot')
    
    # Mark important frames
    takeoff_time = time_seconds[results['takeoff_frame']]
    max_height_time = time_seconds[results['max_height_frame']]
    landing_time = time_seconds[results['landing_frame']]
    
    # Mark takeoff, max height, and landing points
    plt.axvline(x=takeoff_time, color='g', linestyle='--', label='Takeoff')
    plt.axvline(x=max_height_time, color='r', linestyle='--', label='Max Height')
    plt.axvline(x=landing_time, color='k', linestyle='--', label='Landing')
    
    # Add baseline
    if 'cg_y_m_baseline' in data.columns:
        baseline = data['cg_y_m_baseline'].iloc[0]
        plt.axhline(y=baseline, color='gray', linestyle='-', label='CG Baseline')
    
    # Add labels and title
    plt.xlabel('Time (seconds)')
    plt.ylabel('Position (meters)')
    plt.title('Jump Analysis - CG and Feet Positions')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save plot
    ts_plot_path = os.path.join(output_dir, f"{base_name}_time_series_{timestamp}.png")
    plt.savefig(ts_plot_path, dpi=300, bbox_inches='tight')
    plot_files.append(ts_plot_path)
    plt.close()
    
    # 2. Create a jump phases visualization
    plt.figure(figsize=(12, 6))
    
    # Create flight phase rectangle
    flight_start = takeoff_time
    flight_end = landing_time
    flight_duration = flight_end - flight_start
    
    # Plot rectangles for each phase
    plt.axvspan(flight_start, flight_end, alpha=0.2, color='skyblue', label='Flight Phase')
    plt.axvspan(flight_start, max_height_time, alpha=0.2, color='lightgreen', label='Ascent')
    plt.axvspan(max_height_time, flight_end, alpha=0.2, color='salmon', label='Descent')
    
    # Plot CG path
    plt.plot(time_seconds, data['cg_y_m'], 'b-', linewidth=2.5, label='CG Path')
    
    # Mark max height
    max_height_value = data['cg_y_m'].iloc[results['max_height_frame']]
    plt.scatter([max_height_time], [max_height_value], color='red', s=100, 
                marker='o', label=f'Max Height: {results["height_m"]:.3f}m')
    
    # Add annotations
    plt.annotate(f"Flight Time: {results['flight_time_s']:.3f}s",
                xy=(flight_start + flight_duration/2, max_height_value - 0.05),
                xytext=(flight_start + flight_duration/2, max_height_value - 0.1),
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
    
    plt.annotate(f"Velocity: {results['velocity_m/s']:.2f} m/s",
                xy=(flight_start, max_height_value - 0.15),
                xytext=(flight_start, max_height_value - 0.15),
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
    
    # Add labels and title
    plt.xlabel('Time (seconds)')
    plt.ylabel('Position (meters)')
    plt.title('Jump Phases Analysis')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save plot
    phases_plot_path = os.path.join(output_dir, f"{base_name}_jump_phases_{timestamp}.png")
    plt.savefig(phases_plot_path, dpi=300, bbox_inches='tight')
    plot_files.append(phases_plot_path)
    plt.close()
    
    return plot_files

def generate_html_report(data, results, plot_files, output_dir, base_name):
    """
    Generate an HTML report with jump metrics and plots.
    
    Args:
        data (pd.DataFrame): DataFrame with processed data
        results (dict): Dictionary with jump metrics
        plot_files (list): List of paths to plot files
        output_dir (str): Directory to save the report
        base_name (str): Base name for the output file
    
    Returns:
        str: Path to the generated HTML report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"{base_name}_report_{timestamp}.html")
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Jump Analysis Report - {base_name}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #2980b9;
                margin-top: 30px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .img-container {{
                text-align: center;
                margin: 30px 0;
            }}
            .img-container img {{
                max-width: 100%;
                height: auto;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .footer {{
                margin-top: 50px;
                border-top: 1px solid #ddd;
                padding-top: 20px;
                color: #7f8c8d;
                font-size: 0.9em;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <h1>Jump Analysis Report</h1>
        <p><strong>Subject:</strong> {base_name}</p>
        <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Jump Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Unit</th>
            </tr>
            <tr>
                <td>Mass</td>
                <td>{results["mass_kg"]}</td>
                <td>kg</td>
            </tr>
            <tr>
                <td>Jump Height</td>
                <td>{results["height_m"]}</td>
                <td>m</td>
            </tr>
            <tr>
                <td>Flight Time</td>
                <td>{results["flight_time_s"]}</td>
                <td>s</td>
            </tr>
            <tr>
                <td>Takeoff Velocity</td>
                <td>{results["velocity_m/s"]}</td>
                <td>m/s</td>
            </tr>
            <tr>
                <td>Potential Energy</td>
                <td>{results["potential_energy_J"]}</td>
                <td>J</td>
            </tr>
            <tr>
                <td>Kinetic Energy</td>
                <td>{results["kinetic_energy_J"]}</td>
                <td>J</td>
            </tr>
        </table>
        
        <h2>Jump Phase Frames</h2>
        <table>
            <tr>
                <th>Phase</th>
                <th>Frame</th>
                <th>Time (s)</th>
            </tr>
            <tr>
                <td>Takeoff</td>
                <td>{results["takeoff_frame"]}</td>
                <td>{results["takeoff_frame"] / results["fps"]:.3f}</td>
            </tr>
            <tr>
                <td>Maximum Height</td>
                <td>{results["max_height_frame"]}</td>
                <td>{results["max_height_frame"] / results["fps"]:.3f}</td>
            </tr>
            <tr>
                <td>Landing</td>
                <td>{results["landing_frame"]}</td>
                <td>{results["landing_frame"] / results["fps"]:.3f}</td>
            </tr>
        </table>
        
        <h2>Jump Analysis Visualizations</h2>
    """
    
    # Add images to the report
    for plot_file in plot_files:
        plot_filename = os.path.basename(plot_file)
        # Use relative paths in the HTML
        html_content += f"""
        <div class="img-container">
            <img src="{plot_filename}" alt="Jump analysis plot">
            <p><em>{plot_filename}</em></p>
        </div>
        """
    
    # Close the HTML content
    html_content += """
        <div class="footer">
            <p>Generated by VAILA - Vertical Jump Analysis Tool</p>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return report_path

def process_mediapipe_data(input_file, output_dir):
    """
    Process MediaPipe data and generate visualizations and report.
    """
    try:
        data = pd.read_csv(input_file)
        results = {}
        
        # Solicitar massa e FPS
        root = Tk()
        root.withdraw()
        
        mass = simpledialog.askfloat(
            "Mass Input", 
            "Enter the subject's mass (kg):",
            minvalue=20.0, maxvalue=200.0
        )
        
        if mass is None:
            print(f"Processing cancelled for {input_file} - no mass provided.")
            return
        
        fps = simpledialog.askinteger(
            "FPS Input", 
            "Enter the video FPS (frames per second):",
            minvalue=1, maxvalue=240
        )
        
        if fps is None:
            fps = 30
            print(f"Using default FPS value: {fps}")
        
        # Definir um fator de conversão para metros
        comprimento_shank_real = simpledialog.askfloat(
            "Scale Factor",
            "Enter the approximate shank length in meters (e.g., 0.4):",
            minvalue=0.1, maxvalue=1.0
        )
        if comprimento_shank_real is None:
            comprimento_shank_real = 0.4  # Valor padrão
        
        # Calcular o fator de conversão
        fator_conversao = calc_fator_convert_mediapipe(data, comprimento_shank_real)
        
        # Encontrar altura máxima da imagem para inverter o eixo Y
        # Pode ser estimada como 1.0 para dados normalizados
        altura_imagem = 1.0
        
        # Inverter todas as coordenadas Y para o sistema natural (Y cresce para cima)
        y_columns = [col for col in data.columns if col.endswith('_y')]
        new_data = pd.DataFrame()
        for col in y_columns:
            new_data[f"{col}_natural"] = altura_imagem - data[col]
        
        # Depois, adicione todas as colunas de uma vez
        data = pd.concat([data, new_data], axis=1)
        
        # Processamento para calcular o CG
        cg_x_m_list = []
        cg_y_m_list = []
        
        # Processamos por chunks para calcular o CG
        chunk_size = 50
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i+chunk_size]
            for _, row in chunk.iterrows():
                temp_df = pd.DataFrame([row])
                # Calcular CG e armazenar os resultados
                cg_x, cg_y = calcula_cg_frame(temp_df, fator_conversao)
                # Corrigir o problema de Series vs scalar
                cg_x_m_list.append(float(cg_x.iloc[0]) if hasattr(cg_x, 'iloc') else float(cg_x))
                cg_y_m_list.append(float(cg_y.iloc[0]) if hasattr(cg_y, 'iloc') else float(cg_y))
        
        # Adicionar as colunas de CG em metros ao DataFrame
        data['cg_x_m'] = cg_x_m_list
        data['cg_y_m'] = cg_y_m_list
        
        # Calcular baseline dos pés e CG em metros
        n_baseline_frames = 10

        # Usar pés se disponíveis
        has_left_foot = 'left_foot_index_y_m' in data.columns
        has_right_foot = 'right_foot_index_y_m' in data.columns

        if has_left_foot and has_right_foot:
            feet_y_values = (data['left_foot_index_y_m'] + data['right_foot_index_y_m']) / 2
        elif has_left_foot:
            feet_y_values = data['left_foot_index_y_m']
        elif has_right_foot:
            feet_y_values = data['right_foot_index_y_m']
        else:
            # Fallback para tornozelos
            if 'left_ankle_y_m' in data.columns and 'right_ankle_y_m' in data.columns:
                feet_y_values = (data['left_ankle_y_m'] + data['right_ankle_y_m']) / 2
            elif 'left_ankle_y_m' in data.columns:
                feet_y_values = data['left_ankle_y_m']
            elif 'right_ankle_y_m' in data.columns:
                feet_y_values = data['right_ankle_y_m']
            else:
                feet_y_values = data['cg_y_m'] * 0.8  # Estimativa

        feet_baseline = feet_y_values.iloc[:n_baseline_frames].mean()
        cg_y_m_baseline = data['cg_y_m'].iloc[:n_baseline_frames].mean()

        # Preparar dicionários para novas colunas
        new_columns = {}

        # Converter coordenadas x
        x_columns = [col for col in data.columns if col.endswith('_x')]
        for col in x_columns:
            base_name = col[:-2]  # Remove _x
            new_columns[f"{base_name}_x_m"] = data[col] * fator_conversao

        # Converter coordenadas y_natural
        y_natural_columns = [col for col in data.columns if col.endswith('_y_natural')]
        for col in y_natural_columns:
            base_name = col[:-9]  # Remove _y_natural
            new_columns[f"{base_name}_y_m"] = data[col] * fator_conversao

        # Converter coordenadas z
        z_columns = [col for col in data.columns if col.endswith('_z') and '_z_m' not in col]
        for col in z_columns:
            base_name = col[:-2]  # Remove _z
            new_columns[f"{base_name}_z_m"] = data[col] * fator_conversao

        # Adicionar outras colunas necessárias
        new_columns['mass_kg'] = mass
        new_columns['fps'] = fps
        new_columns['feet_baseline'] = feet_baseline
        new_columns['cg_y_m_baseline'] = cg_y_m_baseline

        # Adicionar todas as novas colunas de uma vez
        data = pd.concat([data, pd.DataFrame(new_columns)], axis=1)
        
        # Calcular altura do salto usando a componente vertical (cg_y_m)
        # Com sistema natural, valores MAIORES de cg_y_m representam posições mais altas
        jump_height = data['cg_y_m'].max() - cg_y_m_baseline
        
        # Corrigir a identificação dos frames importantes
        # O ponto mais alto do salto é quando o CG está no valor MÍNIMO (não máximo)
        max_height_frame = data['cg_y_m'].idxmin()  # Frame no ponto mais alto (vale no gráfico)

        # Procurar o takeoff: movimento para baixo antes do ponto mais alto
        pre_max_mask = data.index < max_height_frame
        # Buscar pelo frame onde o CG começa a diminuir significativamente abaixo da baseline
        pre_takeoff_mask = pre_max_mask & (data['cg_y_m'] < (cg_y_m_baseline - 0.03))
        if pre_takeoff_mask.any():
            takeoff_frame = data.index[pre_takeoff_mask].min()
        else:
            # Estimativa: alguns frames antes do ponto mais alto
            takeoff_frame = max(0, max_height_frame - 10)

        # Procurar o landing: movimento para cima após o ponto mais alto
        post_max_mask = data.index > max_height_frame
        # Buscar pelo frame onde o CG retorna para perto da baseline
        post_landing_mask = post_max_mask & (data['cg_y_m'] > (cg_y_m_baseline - 0.03))
        if post_landing_mask.any():
            landing_frame = data.index[post_landing_mask].min()
        else:
            # Estimativa: alguns frames após o ponto mais alto
            landing_frame = min(len(data) - 1, max_height_frame + 10)

        # Calcular o tempo de voo
        flight_time = (landing_frame - takeoff_frame) / fps

        # Calcular métricas do salto
        velocity = calculate_velocity(jump_height)
        potential_energy = calculate_potential_energy(mass, jump_height)
        kinetic_energy = calculate_kinetic_energy(mass, velocity)
        
        # Preparar resultados
        results = {
            "height_m": round(jump_height, 3),
            "mass_kg": mass,
            "fps": fps,
            "flight_time_s": round(flight_time, 3),
            "velocity_m/s": round(velocity, 3),
            "potential_energy_J": round(potential_energy, 3),
            "kinetic_energy_J": round(kinetic_energy, 3),
            "max_height_frame": max_height_frame,
            "takeoff_frame": takeoff_frame,
            "landing_frame": landing_frame,
            "conversion_factor": round(fator_conversao, 6)
        }
        
        # Salvar resultados em DataFrame
        results_df = pd.DataFrame([results])
        
        # Gerar nome base para arquivos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Salvar dados completos
        output_data_file = os.path.join(output_dir, f"{base_name}_processed_{timestamp}.csv")
        data.to_csv(output_data_file, index=False)
        
        # Salvar métricas do salto
        output_metrics_file = os.path.join(output_dir, f"{base_name}_jump_metrics_{timestamp}.csv")
        results_df.to_csv(output_metrics_file, index=False)
        
        # Gerar gráficos
        plot_files = generate_jump_plots(data, results, output_dir, base_name)
        
        # Gerar gráfico diagnóstico
        diagnostic_plot = generate_diagnostic_plot(
            data, feet_baseline, cg_y_m_baseline, 
            takeoff_frame, landing_frame, max_height_frame,
            fps, output_dir, base_name
        )
        
        # Adicionar o gráfico diagnóstico à lista de plot_files
        plot_files.append(diagnostic_plot)
        
        # Gerar relatório HTML
        report_path = generate_html_report(data, results, plot_files, output_dir, base_name)
        
        print(f"Jump metrics saved at: {output_metrics_file}")
        print(f"Complete data saved at: {output_data_file}")
        print(f"Jump analysis plots saved in: {output_dir}")
        print(f"HTML report generated: {report_path}")
        
        # Imprimir informações de diagnóstico
        print(f"Diagnostic info:")
        print(f"  Max height frame (vale): {max_height_frame} (time: {max_height_frame/fps:.3f}s)")
        print(f"  Takeoff frame: {takeoff_frame} (time: {takeoff_frame/fps:.3f}s)")
        print(f"  Landing frame: {landing_frame} (time: {landing_frame/fps:.3f}s)")
        print(f"  Flight time: {flight_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_file} (MediaPipe): {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_all_mediapipe_files(target_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(target_dir, f"vaila_mediapipejump_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    csv_files = [
        os.path.join(target_dir, f)
        for f in os.listdir(target_dir)
        if f.endswith(".csv")
    ]

    for input_file in csv_files:
        print(f"Processing MediaPipe file: {input_file}")
        process_mediapipe_data(input_file, output_dir)
    print("Todos os arquivos MediaPipe foram processados com sucesso.")

def process_jump_data(input_file, output_dir, use_time_of_flight):
    """
    Process the jump data from an input file and save results to the output directory.
    """
    try:
        # Read data from the input file
        data = pd.read_csv(input_file)
        results = []
        
        # Auto-detect column names
        columns = list(data.columns)
        
        # Look for appropriate columns by name pattern
        mass_col = next((col for col in columns if 'mass' in col.lower()), columns[0])
        
        if use_time_of_flight:
            value_col = next((col for col in columns if 'time' in col.lower() and 'contact' not in col.lower()), columns[1])
        else:
            value_col = next((col for col in columns if 'height' in col.lower() or 'heigth' in col.lower()), columns[1])
            
        contact_col = next((col for col in columns if 'contact' in col.lower()), None)
        
        print(f"Using columns: Mass={mass_col}, {'Time' if use_time_of_flight else 'Height'}={value_col}, Contact={contact_col}")
        
        for index, row in data.iterrows():
            # Get values using detected column names
            mass = row[mass_col]
            second_value = row[value_col]
            contact_time = row[contact_col] if contact_col else None
            
            # Determine height based on whether we are using time_of_flight or height directly
            if use_time_of_flight:
                # Calculate height based on time_of_flight
                time_of_flight = second_value
                height = calculate_jump_height(time_of_flight)
            else:
                # Use height directly and calculate time_of_flight from it
                height = second_value
                time_of_flight = calculate_time_of_flight(height)

            # Perform calculations
            velocity = calculate_velocity(height)
            potential_energy = calculate_potential_energy(mass, height)
            kinetic_energy = calculate_kinetic_energy(mass, velocity)
            liftoff_force = (
                calculate_liftoff_force(mass, velocity, contact_time)
                if contact_time
                else None
            )
            average_power = (
                calculate_average_power(potential_energy, contact_time)
                if contact_time
                else None
            )
            relative_power = (average_power / mass) if average_power else None
            jump_performance_index = (
                (average_power * time_of_flight)
                if average_power and time_of_flight
                else None
            )
            total_time = (
                time_of_flight + contact_time
                if time_of_flight and contact_time
                else None
            )

            # Append the results for each row
            results.append(
                {
                    "height_m": round(height, 3),
                    "liftoff_force_N": (
                        round(liftoff_force, 3) if liftoff_force else None
                    ),
                    "velocity_m/s": round(velocity, 3),
                    "potential_energy_J": round(potential_energy, 3),
                    "kinetic_energy_J": round(kinetic_energy, 3),
                    "average_power_W": (
                        round(average_power, 3) if average_power else None
                    ),
                    "relative_power_W/kg": (
                        round(relative_power, 3) if relative_power else None
                    ),
                    "jump_performance_index": (
                        round(jump_performance_index, 3)
                        if jump_performance_index
                        else None
                    ),
                    "total_time_s": round(total_time, 3) if total_time else None,
                }
            )

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Generate output file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file_path = os.path.join(
            output_dir, f"{base_name}_vjump_{timestamp}.csv"
        )
        results_df.to_csv(output_file_path, index=False)

        print(f"Results saved successfully at: {output_file_path}")
    except Exception as e:
        print(f"An error occurred while processing {input_file}: {str(e)}")

def calc_fator_convert_mediapipe(df, knee='right_knee', ankle='right_ankle', comprimento_shank_real=0.4):
    # frame 0
    rkx, rky = df[f"{knee}_x"].iloc[0], df[f"{knee}_y"].iloc[0]
    rax, ray = df[f"{ankle}_x"].iloc[0], df[f"{ankle}_y"].iloc[0]
    comprimento_norm = np.sqrt((rkx-rax)**2 + (rky-ray)**2)
    fator = comprimento_shank_real / comprimento_norm
    return fator

def calc_fator_convert_mediapipe(df, comprimento_shank_real):
    rkx, rky = df["right_knee_x"].iloc[0], df["right_knee_y"].iloc[0]
    rax, ray = df["right_ankle_x"].iloc[0], df["right_ankle_y"].iloc[0]
    comprimento_norm = np.sqrt((rkx-rax)**2 + (rky-ray)**2)
    fator = comprimento_shank_real / comprimento_norm
    return fator

def ponto_medio(df, p1, p2):
    return (df[f"{p1}_x"] + df[f"{p2}_x"]) / 2, (df[f"{p1}_y"] + df[f"{p2}_y"]) / 2

def calcula_cg_frame(df, fator):
    # Localizações dos CGs em proporção ao segmento
    localiz = {
        "head": 0.5, "trunk": 0.5, "upperarm": 0.436, "forearm": 0.430, "hand": 0.506,
        "thigh": 0.433, "shank": 0.433, "foot": 0.5
    }
    # Massas relativas
    massas = {
        "head": 0.081, "trunk": 0.497, "upperarm": 0.028, "forearm": 0.016, "hand": 0.006,
        "thigh": 0.100, "shank": 0.047, "foot": 0.014
    }
    # Cabeça: ponto médio entre eyes e shoulders
    head_prox_x, head_prox_y = ponto_medio(df, "left_eye", "right_eye")
    head_dist_x, head_dist_y = ponto_medio(df, "left_shoulder", "right_shoulder")
    cg_head_x = head_prox_x + localiz["head"] * (head_dist_x - head_prox_x)
    cg_head_y = head_prox_y + localiz["head"] * (head_dist_y - head_prox_y)
    # Tronco: ponto médio ombros e quadril
    tronco_prox_x, tronco_prox_y = ponto_medio(df, "left_shoulder", "right_shoulder")
    tronco_dist_x, tronco_dist_y = ponto_medio(df, "left_hip", "right_hip")
    cg_trunk_x = tronco_prox_x + localiz["trunk"] * (tronco_dist_x - tronco_prox_x)
    cg_trunk_y = tronco_prox_y + localiz["trunk"] * (tronco_dist_y - tronco_prox_y)
    # Segmentos membro superior esquerdo
    cg_upperarm_l_x = df["left_shoulder_x"] + localiz["upperarm"] * (df["left_elbow_x"] - df["left_shoulder_x"])
    cg_upperarm_l_y = df["left_shoulder_y"] + localiz["upperarm"] * (df["left_elbow_y"] - df["left_shoulder_y"])
    cg_forearm_l_x = df["left_elbow_x"] + localiz["forearm"] * (df["left_wrist_x"] - df["left_elbow_x"])
    cg_forearm_l_y = df["left_elbow_y"] + localiz["forearm"] * (df["left_wrist_y"] - df["left_elbow_y"])
    cg_hand_l_x = df["left_wrist_x"] + localiz["hand"] * (ponto_medio(df, "left_pinky", "left_index")[0] - df["left_wrist_x"])
    cg_hand_l_y = df["left_wrist_y"] + localiz["hand"] * (ponto_medio(df, "left_pinky", "left_index")[1] - df["left_wrist_y"])
    # Segmentos membro superior direito
    cg_upperarm_r_x = df["right_shoulder_x"] + localiz["upperarm"] * (df["right_elbow_x"] - df["right_shoulder_x"])
    cg_upperarm_r_y = df["right_shoulder_y"] + localiz["upperarm"] * (df["right_elbow_y"] - df["right_shoulder_y"])
    cg_forearm_r_x = df["right_elbow_x"] + localiz["forearm"] * (df["right_wrist_x"] - df["right_elbow_x"])
    cg_forearm_r_y = df["right_elbow_y"] + localiz["forearm"] * (df["right_wrist_y"] - df["right_elbow_y"])
    cg_hand_r_x = df["right_wrist_x"] + localiz["hand"] * (ponto_medio(df, "right_pinky", "right_index")[0] - df["right_wrist_x"])
    cg_hand_r_y = df["right_wrist_y"] + localiz["hand"] * (ponto_medio(df, "right_pinky", "right_index")[1] - df["right_wrist_y"])
    # Coxa, perna, pé esquerdo
    cg_thigh_l_x = df["left_hip_x"] + localiz["thigh"] * (df["left_knee_x"] - df["left_hip_x"])
    cg_thigh_l_y = df["left_hip_y"] + localiz["thigh"] * (df["left_knee_y"] - df["left_hip_y"])
    cg_shank_l_x = df["left_knee_x"] + localiz["shank"] * (df["left_ankle_x"] - df["left_knee_x"])
    cg_shank_l_y = df["left_knee_y"] + localiz["shank"] * (df["left_ankle_y"] - df["left_knee_y"])
    cg_foot_l_x = df["left_heel_x"] + localiz["foot"] * (df["left_foot_index_x"] - df["left_heel_x"])
    cg_foot_l_y = df["left_heel_y"] + localiz["foot"] * (df["left_foot_index_y"] - df["left_heel_y"])
    # Coxa, perna, pé direito
    cg_thigh_r_x = df["right_hip_x"] + localiz["thigh"] * (df["right_knee_x"] - df["right_hip_x"])
    cg_thigh_r_y = df["right_hip_y"] + localiz["thigh"] * (df["right_knee_y"] - df["right_hip_y"])
    cg_shank_r_x = df["right_knee_x"] + localiz["shank"] * (df["right_ankle_x"] - df["right_knee_x"])
    cg_shank_r_y = df["right_knee_y"] + localiz["shank"] * (df["right_ankle_y"] - df["right_knee_y"])
    cg_foot_r_x = df["right_heel_x"] + localiz["foot"] * (df["right_foot_index_x"] - df["right_heel_x"])
    cg_foot_r_y = df["right_heel_y"] + localiz["foot"] * (df["right_foot_index_y"] - df["right_heel_y"])
    # Vetores das coordenadas dos CGs dos segmentos
    cg_x = (massas["head"] * cg_head_x + massas["trunk"] * cg_trunk_x +
            massas["upperarm"] * (cg_upperarm_l_x + cg_upperarm_r_x) +
            massas["forearm"] * (cg_forearm_l_x + cg_forearm_r_x) +
            massas["hand"] * (cg_hand_l_x + cg_hand_r_x) +
            massas["thigh"] * (cg_thigh_l_x + cg_thigh_r_x) +
            massas["shank"] * (cg_shank_l_x + cg_shank_r_x) +
            massas["foot"] * (cg_foot_l_x + cg_foot_r_x)
           ) / 1.0

    cg_y = (massas["head"] * cg_head_y + massas["trunk"] * cg_trunk_y +
            massas["upperarm"] * (cg_upperarm_l_y + cg_upperarm_r_y) +
            massas["forearm"] * (cg_forearm_l_y + cg_forearm_r_y) +
            massas["hand"] * (cg_hand_l_y + cg_hand_r_y) +
            massas["thigh"] * (cg_thigh_l_y + cg_thigh_r_y) +
            massas["shank"] * (cg_shank_l_y + cg_shank_r_y) +
            massas["foot"] * (cg_foot_l_y + cg_foot_r_y)
           ) / 1.0
    # Aplica fator de conversão para metros
    cg_x_m = cg_x * fator
    cg_y_m = cg_y * fator
    return cg_x_m, cg_y_m

def altura_salto_mediapipe(df, comprimento_shank_real):
    fator = calc_fator_convert_mediapipe(df, comprimento_shank_real)
    cg_x, cg_y = calcula_cg_frame(df, fator)
    # Atenção: y cresce para baixo, então menor valor é o ponto mais alto!
    altura_salto = abs(cg_y.iloc[0] - cg_y.min())
    return altura_salto, cg_x, cg_y

def process_all_files_in_directory(target_dir, use_time_of_flight):
    """
    Process all .csv files in the specified directory and save the results in a new output directory.

    Args:
        target_dir (str): The path to the target directory containing .csv files.
        use_time_of_flight (bool): Whether to use time of flight data.
    """
    # Generate the output directory with the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(target_dir, f"vaila_verticaljump_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # List all .csv files in the target directory
    csv_files = [
        os.path.join(target_dir, f)
        for f in os.listdir(target_dir)
        if f.endswith(".csv")
    ]

    # Process each .csv file
    for input_file in csv_files:
        print(f"Processing file: {input_file}")
        process_jump_data(input_file, output_dir, use_time_of_flight)

    print("All files have been processed successfully.")


def vaila_and_jump():
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent.resolve()}")
    root = Tk()
    root.withdraw()
    target_dir = filedialog.askdirectory(title="Select the target directory containing .csv files")
    if not target_dir:
        messagebox.showwarning("Warning", "No target directory selected.")
        return

    # Agora com opção 3!
    data_type = simpledialog.askinteger(
        "Select Data Type",
        "Select the type of data in your CSV files:\n\n"
        "1. Time of Flight Data\n"
        "2. Jump Height Data\n"
        "3. MediaPipe Ankle Data",
        minvalue=1, maxvalue=3
    )
    if data_type is None:
        messagebox.showwarning("Warning", "No data type selected. Exiting.")
        return

    if data_type == 1 or data_type == 2:
        use_time_of_flight = (data_type == 1)
        process_all_files_in_directory(target_dir, use_time_of_flight)
    elif data_type == 3:
        process_all_mediapipe_files(target_dir)
    root.destroy()
    msg = "All CSV files have been processed and results saved.\n\n"
    if data_type == 1:
        msg += f"Input data type: Time of Flight\n"
    elif data_type == 2:
        msg += f"Input data type: Jump Height\n"
    else:
        msg += f"Input data type: MediaPipe Ankle\n"
    msg += f"Output directory: {os.path.join(target_dir, 'vaila_verticaljump_*')}"
    messagebox.showinfo("Success", msg)


if __name__ == "__main__":
    vaila_and_jump()

def generate_diagnostic_plot(data, feet_baseline, cg_baseline, takeoff_frame, landing_frame, max_height_frame, fps, output_dir, base_name):
    """
    Generate a detailed diagnostic plot showing all CG components and jump phases.
    
    Args:
        data (pd.DataFrame): DataFrame with processed data
        feet_baseline (float): Baseline value for feet position
        cg_baseline (float): Baseline value for CG position
        takeoff_frame (int): Frame where takeoff is detected
        landing_frame (int): Frame where landing is detected
        max_height_frame (int): Frame where maximum height is reached
        fps (int): Frames per second
        output_dir (str): Directory to save the plot
        base_name (str): Base name for the output file
    
    Returns:
        str: Path to the generated plot file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, ax = plt.figure(figsize=(14, 10)), plt.subplot(111)
    
    # Convert frames to time
    time = data.index / fps
    
    # Plot CG trajectory
    ax.plot(time, data['cg_y_m'], 'b-', linewidth=2, label='CG_Y (m)')
    
    # Plot feet positions
    if 'left_foot_index_y_m' in data.columns:
        ax.plot(time, data['left_foot_index_y_m'], 'g-', linewidth=1, alpha=0.7, label='Left Foot Y (m)')
    if 'right_foot_index_y_m' in data.columns:
        ax.plot(time, data['right_foot_index_y_m'], 'r-', linewidth=1, alpha=0.7, label='Right Foot Y (m)')
    
    # Plot baselines
    ax.axhline(y=cg_baseline, color='blue', linestyle='--', alpha=0.7, label=f'CG Baseline: {cg_baseline:.3f}m')
    ax.axhline(y=feet_baseline, color='green', linestyle='--', alpha=0.7, label=f'Feet Baseline: {feet_baseline:.3f}m')
    
    # Mark takeoff and landing points
    takeoff_time = takeoff_frame / fps
    landing_time = landing_frame / fps
    max_height_time = max_height_frame / fps
    
    ax.axvline(x=takeoff_time, color='purple', linestyle='-', label=f'Takeoff (Frame {takeoff_frame}, {takeoff_time:.3f}s)')
    ax.axvline(x=landing_time, color='orange', linestyle='-', label=f'Landing (Frame {landing_frame}, {landing_time:.3f}s)')
    ax.axvline(x=max_height_time, color='red', linestyle='-', label=f'Max Height (Frame {max_height_frame}, {max_height_time:.3f}s)')
    
    # Shade the flight phase
    ax.axvspan(takeoff_time, landing_time, alpha=0.2, color='yellow', label=f'Flight Phase: {landing_time-takeoff_time:.3f}s')
    
    # Add frame numbers at regular intervals
    frames_to_show = np.arange(0, len(data), 10)
    for frame in frames_to_show:
        ax.text(frame/fps, data['cg_y_m'].min() - 0.02, f"{frame}", fontsize=8, ha='center')
    
    # Add annotations for key details
    flight_time = (landing_frame - takeoff_frame) / fps
    ax.annotate(f"Flight Time: {flight_time:.3f}s ({landing_frame-takeoff_frame} frames)", 
                xy=(0.5, 0.05), xycoords='figure fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
                ha='center')
    
    # Set labels and title
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Position (meters)')
    ax.set_title('Jump Analysis - Diagnostic Plot')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{base_name}_diagnostic_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path
