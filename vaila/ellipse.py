"""
Módulo: ellipse.py
Descrição: Fornece funções para calcular e plotar elipses de confiança para dados do Centro de Pressão (CoP).
             Utiliza Análise de Componentes Principais (PCA) para determinar a orientação e o tamanho da elipse.
             O módulo também inclui funcionalidades de plotagem para visualizar o caminho do CoP com a elipse de confiança.
             
Autor: Prof. Dr. Paulo R. P. Santiago
Versão: 1.0
Data: 2024-09-12

Histórico de Alterações:
- Versão 1.0 (2024-09-12):
  - Implementação inicial do cálculo da elipse usando PCA.
  - Adicionadas funções de plotagem para visualizar os caminhos do CoP com elipses de confiança.
  - Integrado mapeamento de cores para representar a progressão do tempo no gráfico do caminho do CoP.
"""


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_ellipse_pca(data, confidence=0.95):
    """Calculates the ellipse using PCA with a specified confidence level."""
    pca = PCA(n_components=2)
    pca.fit(data)

    # Eigenvalues and eigenvectors
    eigvals = np.sqrt(pca.explained_variance_)
    eigvecs = pca.components_

    # Scale factor for confidence level
    chi2_val = np.sqrt(2) * np.sqrt(np.log(1 / (1 - confidence)))
    scaled_eigvals = eigvals * chi2_val

    # Ellipse parameters
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse = np.array(
        [scaled_eigvals[0] * np.cos(theta), scaled_eigvals[1] * np.sin(theta)]
    )
    ellipse_rot = np.dot(eigvecs.T, ellipse)  # Adjustment for rotated ellipse

    # Area and angle of the ellipse
    area = np.pi * scaled_eigvals[0] * scaled_eigvals[1]
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0]) * 180 / np.pi

    # Calculate ellipse bounds
    ellipse_x = ellipse_rot[0, :] + pca.mean_[0]
    ellipse_y = ellipse_rot[1, :] + pca.mean_[1]
    x_bounds = [min(ellipse_x), max(ellipse_x)]
    y_bounds = [min(ellipse_y), max(ellipse_y)]

    # Return the ellipse data as a tuple of all necessary elements
    ellipse_data = (ellipse_x, ellipse_y, eigvecs, scaled_eigvals, pca.mean_)

    return area, angle, x_bounds + y_bounds, ellipse_data

def plot_cop_pathway_with_ellipse(cop_x, cop_y, area, angle, ellipse_data, title, output_path):
    """Plots the CoP pathway along with the 95% confidence ellipse and saves the figure."""

    # Unpack ellipse data
    ellipse_x, ellipse_y = ellipse_data[0], ellipse_data[1]
    eigvecs, scaled_eigvals, pca_mean = ellipse_data[2], ellipse_data[3], ellipse_data[4]

    # Create colormap for CoP path
    cmap = LinearSegmentedColormap.from_list("CoP_path", ["blue", "green", "yellow", "red"])

    # Plot CoP pathway with color segments
    plt.figure(figsize=(10, 8))
    for i in range(len(cop_x) - 1):
        plt.plot(cop_x[i:i + 2], cop_y[i:i + 2], color=cmap(i / len(cop_x)), linewidth=2)

    # Plot start and end points
    plt.plot(cop_x[0], cop_y[0], color="gray", marker=".", markersize=17, label="Start")
    plt.plot(cop_x[-1], cop_y[-1], color="black", marker=".", markersize=17, label="End")

    # Plot the ellipse
    plt.plot(ellipse_x, ellipse_y, color='gray', linestyle='--', linewidth=2)

    # Plot major and minor axes of the ellipse
    major_axis_start = pca_mean - eigvecs[0] * scaled_eigvals[0]
    major_axis_end = pca_mean + eigvecs[0] * scaled_eigvals[0]
    plt.plot(
        [major_axis_start[0], major_axis_end[0]],
        [major_axis_start[1], major_axis_end[1]],
        color='gray', linestyle='--', linewidth=1,
    )

    minor_axis_start = pca_mean - eigvecs[1] * scaled_eigvals[1]
    minor_axis_end = pca_mean + eigvecs[1] * scaled_eigvals[1]
    plt.plot(
        [minor_axis_start[0], minor_axis_end[0]],
        [minor_axis_start[1], minor_axis_end[1]],
        color='gray', linestyle='--', linewidth=1,
    )

    # Add legend for Start and End points
    plt.legend()

    # Calculate margins to expand the xlim and ylim
    x_margin = 0.02 * (np.max(ellipse_x) - np.min(ellipse_x))
    y_margin = 0.02 * (np.max(ellipse_y) - np.min(ellipse_y))

    # Adjust xlim and ylim based on ellipse bounds and add margin
    plt.xlim(np.min(ellipse_x) - x_margin, np.max(ellipse_x) + x_margin)
    plt.ylim(np.min(ellipse_y) - y_margin, np.max(ellipse_y) + y_margin)

    plt.xlabel("Medio-Lateral (cm)")
    plt.ylabel("Antero-Posterior (cm)")
    plt.grid(True, linestyle=':', color='lightgray')
    plt.gca().set_aspect("equal", adjustable="box")

    # Add colorbar for time progression
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(cop_x)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Time Progression [%]', rotation=270, labelpad=15)

    # Set the title of the plot
    plt.title(f'{title}\n95% Ellipse (Area: {area:.2f} cm², Angle: {angle:.2f}°)', fontsize=12)

    # Save the figure
    plt.savefig(f"{output_path}.png")
    plt.savefig(f"{output_path}.svg")
    plt.close()  # Close the plot to free memory and prevent overlapping in subsequent plots

