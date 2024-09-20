"""
Module: ellipse.py
Description: This module provides functions to compute and visualize confidence ellipses for Center of Pressure (CoP) data, 
             which is often used in postural control studies to assess balance and stability. The confidence ellipse represents 
             the region where the majority of data points are expected to fall, based on a specified confidence level.

             The module includes:
             - `plot_ellipse_pca`: Calculates the parameters of a confidence ellipse using Principal Component Analysis (PCA). 
               PCA is employed to identify the principal axes of the data distribution, allowing for the computation of the ellipse 
               that best fits the CoP data within a given confidence level. The function returns the ellipse's area, orientation, 
               and boundary coordinates.
             
             - `plot_cop_pathway_with_ellipse`: Visualizes the CoP pathway along with the calculated confidence ellipse. 
               This function plots the CoP movement over time, adds a color gradient to indicate time progression, and overlays 
               the computed confidence ellipse to provide insight into the range and directionality of CoP variations. It also 
               marks the starting and ending points of the CoP pathway and illustrates the major and minor axes of the ellipse.

Inputs:
- `plot_ellipse_pca(data, confidence=0.95)`:
  - `data` (numpy array): A 2D array where each row represents a time point, and columns represent CoP coordinates (typically medio-lateral and antero-posterior).
  - `confidence` (float): The confidence level for the ellipse, default is 95%.

- `plot_cop_pathway_with_ellipse(cop_x, cop_y, area, angle, ellipse_data, title, output_path)`:
  - `cop_x` (numpy array): X-coordinates of the CoP data.
  - `cop_y` (numpy array): Y-coordinates of the CoP data.
  - `area` (float): The calculated area of the confidence ellipse.
  - `angle` (float): The orientation angle of the ellipse in degrees.
  - `ellipse_data` (tuple): Contains the ellipse boundary coordinates, eigenvectors, scaled eigenvalues, and the mean center.
  - `title` (str): The title for the plot.
  - `output_path` (str): The file path where the plot should be saved.

Usage:
- Import the functions from `ellipse.py` in your main program or script:

Example usage within your program:

```python
from ellipse import plot_ellipse_pca, plot_cop_pathway_with_ellipse
# Assuming `cop_x` and `cop_y` are arrays containing the CoP data
cop_data = np.column_stack((cop_x, cop_y))  # Combine x and y into a single array

# Calculate the confidence ellipse using PCA
area, angle, bounds, ellipse_data = plot_ellipse_pca(cop_data, confidence=0.95)

# Plot the CoP pathway with the calculated confidence ellipse
plot_cop_pathway_with_ellipse(
    cop_x=cop_x,
    cop_y=cop_y,
    area=area,
    angle=angle,
    ellipse_data=ellipse_data,
    title="CoP Pathway with 95% Confidence Ellipse",
    output_path="output/cop_analysis"
)
```

    The above example demonstrates how to compute the confidence ellipse for a given CoP dataset and plot the CoP pathway with the ellipse, providing visual insights into the subject's postural control behavior.

Author: Prof. Dr. Paulo R. P. Santiago Version: 1.0 Date: 2024-09-12

Changelog:

    Version 1.0 (2024-09-12):
        Initial implementation of confidence ellipse calculation using PCA.
        Added plotting functions for CoP pathways with confidence ellipses.
        Integrated color mapping to visually represent time progression along the CoP path.
        Included visual indicators for the start and end of the CoP trajectory and the major and minor axes of the ellipse.

References:

    GitHub Repository: Code Descriptors Postural Control. https://github.com/Jythen/code_descriptors_postural_control
    Further reading on the use of PCA for analyzing CoP data in postural control studies: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8623280

## Key Additions:

1. **Inputs**: Detailed the required inputs for each function, specifying the expected data types and default values.
2. **Usage Example**: Provided a practical example demonstrating how to use the functions within the context of your program. This shows how to calculate and visualize a confidence ellipse for CoP data.

"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


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


def plot_cop_pathway_with_ellipse(
    cop_x, cop_y, area, angle, ellipse_data, title, output_path
):
    """Plots the CoP pathway along with the 95% confidence ellipse and saves the figure."""

    # Unpack ellipse data
    ellipse_x, ellipse_y = ellipse_data[0], ellipse_data[1]
    eigvecs, scaled_eigvals, pca_mean = (
        ellipse_data[2],
        ellipse_data[3],
        ellipse_data[4],
    )

    # Create colormap for CoP path
    cmap = LinearSegmentedColormap.from_list(
        "CoP_path", ["blue", "green", "yellow", "red"]
    )

    # Plot CoP pathway with color segments
    # Plot CoP pathway with cross points using a loop
    plt.figure(figsize=(10, 8))
    for i in range(len(cop_x)):
        plt.plot(
            cop_x[i],
            cop_y[i],
            color=cmap(i / len(cop_x)),
            marker=".",
            markersize=4,
            linestyle="None",
        )
    # Plot start and end points
    plt.plot(cop_x[0], cop_y[0], color="gray", marker=".", markersize=17, label="Start")
    plt.plot(
        cop_x[-1], cop_y[-1], color="black", marker=".", markersize=17, label="End"
    )

    # Plot the ellipse
    plt.plot(ellipse_x, ellipse_y, color="gray", linestyle="--", linewidth=2)

    # Plot major and minor axes of the ellipse
    major_axis_start = pca_mean - eigvecs[0] * scaled_eigvals[0]
    major_axis_end = pca_mean + eigvecs[0] * scaled_eigvals[0]
    plt.plot(
        [major_axis_start[0], major_axis_end[0]],
        [major_axis_start[1], major_axis_end[1]],
        color="gray",
        linestyle="--",
        linewidth=1,
    )

    minor_axis_start = pca_mean - eigvecs[1] * scaled_eigvals[1]
    minor_axis_end = pca_mean + eigvecs[1] * scaled_eigvals[1]
    plt.plot(
        [minor_axis_start[0], minor_axis_end[0]],
        [minor_axis_start[1], minor_axis_end[1]],
        color="gray",
        linestyle="--",
        linewidth=1,
    )

    # Add legend for Start and End points
    plt.legend()

    # Calculate margins to expand the xlim and ylim
    x_margin = 0.02 * (
        np.max([np.max(ellipse_x), np.max(cop_x)])
        - np.min([np.min(ellipse_x), np.min(cop_x)])
    )
    y_margin = 0.02 * (
        np.max([np.max(ellipse_y), np.max(cop_y)])
        - np.min([np.min(ellipse_y), np.min(cop_y)])
    )

    # Adjust xlim and ylim based on ellipse bounds and add margin
    plt.xlim(
        min(np.min(ellipse_x), np.min(cop_x)) - x_margin,
        max(np.max(ellipse_x), np.max(cop_x)) + x_margin,
    )
    plt.ylim(
        min(np.min(ellipse_y), np.min(cop_y)) - y_margin,
        max(np.max(ellipse_y), np.max(cop_y)) + y_margin,
    )

    plt.xlabel("Medio-Lateral (cm)")
    plt.ylabel("Antero-Posterior (cm)")
    plt.grid(True, linestyle=":", color="lightgray")
    plt.gca().set_aspect("equal", adjustable="box")

    # Add colorbar for time progression
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(cop_x)))
    sm.set_array([])
    cbar = plt.colorbar(
        sm, ax=plt.gca(), orientation="vertical", fraction=0.046, pad=0.04
    )
    cbar.set_label("Time Progression [%]", rotation=270, labelpad=15)

    # Set the title of the plot
    plt.title(
        f"{title}\n95% Ellipse (Area: {area:.2f} cm², Angle: {angle:.2f}°)", fontsize=12
    )

    # Save the figure
    plt.savefig(f"{output_path}.png")
    plt.savefig(f"{output_path}.svg")
    plt.close()  # Close the plot to free memory and prevent overlapping in subsequent plots
