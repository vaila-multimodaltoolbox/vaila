"""
Project: vailá Multimodal Toolbox
Script: mocap_analysis.py - Mocap Analysis

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2024
Update Date: 27 July 2025
Version: 0.5.1

Description:
This script performs batch processing of videos for 3D pose estimation using
Mocap data. It processes videos from a specified input directory,
overlays pose landmarks on each video frame, and exports both normalized and
pixel-based landmark coordinates to CSV files.

The user can configure key MediaPipe parameters via a graphical interface,
including detection confidence, tracking confidence, model complexity, and
whether to enable segmentation and smooth segmentation.

Features:
- Added temporal filtering to smooth landmark movements.
- Added estimation of occluded landmarks based on anatomical constraints.
- Added log file with video metadata and processing information.

Usage:
- Run the script to open a graphical interface for selecting the input directory
  containing video files (.csv), the output directory, and for
  specifying the Mocap configuration parameters.
- The script processes each video, generating an output video with overlaid pose
  landmarks, and CSV files containing both normalized and pixel-based landmark
  coordinates in original video dimensions.

Requirements:
- Python 3.12.11
- Tkinter (usually included with Python installations)
- Pandas (for coordinate conversion: `pip install pandas`)

Output:
The following files are generated for each processed video:
1. Processed Video (`*_mp.mp4`):
   The video with the 2D pose landmarks overlaid on the original frames.
2. Normalized Landmark CSV (`*_mp_norm.csv`):
   A CSV file containing the landmark coordinates normalized to a scale between 0 and 1
   for each frame. These coordinates represent the relative positions of landmarks in the video.
3. Pixel Landmark CSV (`*_mp_pixel.csv`):
   A CSV file containing the landmark coordinates in pixel format. The x and y coordinates
   are scaled to the video's resolution, representing the exact pixel positions of the landmarks.
4. Original Coordinates CSV (`*_mp_original.csv`):
   If resize was used, coordinates converted back to original video dimensions.
5. Log File (`log_info.txt`):
   A log file containing video metadata and processing information.

License:
    This project is licensed under the terms of GNU General Public License v3.0.
"""

import os
from rich import print
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tkinter import messagebox, filedialog, simpledialog, Tk
from vaila.data_processing import read_mocap_csv
from vaila.filtering import apply_filter
from vaila.rotation import createortbase_4points, calcmatrot, rotmat2euler
from vaila.plotting import plot_orthonormal_bases
from vaila.readcsv import get_csv_headers, select_headers_gui


def save_results_to_csv(
    base_dir, time, trunk_euler_angles, pelvis_euler_angles, file_name
):
    results = {
        "time": time,
        "trunk_euler_x": trunk_euler_angles[:, 0],
        "trunk_euler_y": trunk_euler_angles[:, 1],
        "trunk_euler_z": trunk_euler_angles[:, 2],
        "pelvis_euler_x": pelvis_euler_angles[:, 0],
        "pelvis_euler_y": pelvis_euler_angles[:, 1],
        "pelvis_euler_z": pelvis_euler_angles[:, 2],
    }
    df = pd.DataFrame(results)
    base_name = os.path.splitext(file_name)[0]
    result_file_path = os.path.join(base_dir, f"{base_name}_results.csv")
    df.to_csv(result_file_path, index=False)


def read_anatomical_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        trunk_median = (
            data[["trunk_euler_x", "trunk_euler_y", "trunk_euler_z"]].median().values
        )
        pelvis_median = (
            data[["pelvis_euler_x", "pelvis_euler_y", "pelvis_euler_z"]].median().values
        )
        return {"trunk": trunk_median, "pelvis": pelvis_median}
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def analyze_mocap_fullbody_data():
    # B_r1_c3
    print("B_r1_c3")
    # Print the directory and name of the script being executed
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent.resolve()}")
    print()

    root = Tk()
    root.withdraw()  # Hide the main Tkinter window

    directory_path = filedialog.askdirectory(title="Select Directory with CSV Files")
    if not directory_path:
        messagebox.showerror("No Directory Selected", "No directory selected. Exiting.")
        root.destroy()
        return

    sample_rate = simpledialog.askfloat(
        "Input", "Please enter the sample rate:", minvalue=0.0
    )
    if sample_rate is None:
        messagebox.showerror("No Sample Rate", "No sample rate entered. Exiting.")
        root.destroy()
        return

    use_anatomical = messagebox.askyesno(
        "Use Anatomical Angles", "Do you want to analyze with anatomical angle data?"
    )

    filter_method = "butterworth"

    file_names = sorted([f for f in os.listdir(directory_path) if f.endswith(".csv")])

    save_directory = filedialog.askdirectory(title="Choose Directory to Save Results")
    if not save_directory:
        messagebox.showerror(
            "No Directory Selected",
            "No directory selected for saving results. Exiting.",
        )
        root.destroy()
        return

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir_figures = os.path.join(save_directory, f"MocapFull_{current_time}/figures")
    base_dir_processed_data = os.path.join(
        save_directory, f"MocapFull_{current_time}/processed_data"
    )
    os.makedirs(base_dir_figures, exist_ok=True)
    os.makedirs(base_dir_processed_data, exist_ok=True)

    anatomical_data = {}
    if use_anatomical:
        anatomical_dir = filedialog.askdirectory(
            title="Select Directory with Anatomical CSV Files"
        )
        if not anatomical_dir:
            messagebox.showerror(
                "No Directory Selected",
                "No directory selected for anatomical files. Exiting.",
            )
            root.destroy()
            return

        anatomical_file_names = sorted(
            [f for f in os.listdir(anatomical_dir) if f.endswith(".csv")]
        )
        if len(anatomical_file_names) != len(file_names):
            print(
                "Warning: Number of anatomical files does not match the number of data files."
            )
        for idx, anatomical_file in enumerate(anatomical_file_names):
            anatomical_file_path = os.path.join(anatomical_dir, anatomical_file)
            anat_data = read_anatomical_csv(anatomical_file_path)
            if anat_data:
                anatomical_data[file_names[idx]] = anat_data

    # Request the input file to select headers from
    selected_file = filedialog.askopenfilename(
        title="Pick file to select headers", filetypes=[("CSV files", "*.csv")]
    )
    if not selected_file:
        messagebox.showerror("Error", "No file selected to choose headers.")
        root.destroy()
        return

    headers = get_csv_headers(selected_file)
    messagebox.showinfo(
        "Select Headers",
        "Please select 24 headers in the following order:\n"
        "Trunk: STRN (x, y, z), CLAV (x, y, z), C7 (x, y, z), T10 (x, y, z)\n"
        "Pelvis: RASI (x, y, z), LASI (x, y, z), RPSI (x, y, z), LPSI (x, y, z)",
    )
    selected_headers = select_headers_gui(headers)
    if len(selected_headers) != 24:
        messagebox.showerror("Error", "Please select exactly 24 headers.")
        root.destroy()
        return

    trunk_headers = selected_headers[:12]
    pelvis_headers = selected_headers[12:]

    matplotlib_figs = []  # List to store matplotlib figures
    for idx, file_name in enumerate(file_names):
        file_path = os.path.join(directory_path, file_name)
        data = read_mocap_csv(file_path)
        time = np.linspace(0, len(data) / sample_rate, len(data))

        # Check for the presence of a time column and add it if not present
        if not any(col.lower() == "time" for col in data.columns):
            data.insert(0, "Time", time)

        # Filter data based on required columns
        if not set(trunk_headers).issubset(data.columns):
            print(f"File {file_name} does not contain all required trunk columns.")
            continue
        if not set(pelvis_headers).issubset(data.columns):
            print(f"File {file_name} does not contain all required pelvis columns.")
            continue

        trunk_points = data[trunk_headers].values
        pelvis_points = data[pelvis_headers].values

        trunk_pointsf = apply_filter(trunk_points, sample_rate, method=filter_method)
        pelvis_pointsf = apply_filter(pelvis_points, sample_rate, method=filter_method)

        trunk_p1 = trunk_pointsf[:, [0, 1, 2]]
        trunk_p2 = trunk_pointsf[:, [3, 4, 5]]
        trunk_p3 = trunk_pointsf[:, [6, 7, 8]]
        trunk_p4 = trunk_pointsf[:, [9, 10, 11]]

        pelvis_p1 = pelvis_pointsf[:, [0, 1, 2]]
        pelvis_p2 = pelvis_pointsf[:, [3, 4, 5]]
        pelvis_p3 = pelvis_pointsf[:, [6, 7, 8]]
        pelvis_p4 = pelvis_pointsf[:, [9, 10, 11]]

        blab = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        trunk_base, orig_trunk = createortbase_4points(
            trunk_p1, trunk_p2, trunk_p3, trunk_p4, configuration="x"
        )
        pelvis_base, orig_pelvis = createortbase_4points(
            pelvis_p1, pelvis_p2, pelvis_p3, pelvis_p4, configuration="z"
        )

        fig_matplotlib = plot_orthonormal_bases(
            [trunk_base, pelvis_base],
            [orig_trunk, orig_pelvis],
            [
                [trunk_p1, trunk_p2, trunk_p3, trunk_p4],
                [pelvis_p1, pelvis_p2, pelvis_p3, pelvis_p4],
            ],
            ["Trunk", "Pelvis"],
            title=f"Mocap Bases - {file_name}",
            global_coordinate_system=None,
            four_points=True,
        )

        matplotlib_figs.append(fig_matplotlib)

        trunk_rotmat = calcmatrot(trunk_base, blab)
        pelvis_rotmat = calcmatrot(pelvis_base, blab)

        trunk_euler_angles = rotmat2euler(trunk_rotmat)
        pelvis_euler_angles = rotmat2euler(pelvis_rotmat)

        if use_anatomical and file_name in anatomical_data:
            trunk_euler_angles_anat = anatomical_data[file_name]["trunk"]
            pelvis_euler_angles_anat = anatomical_data[file_name]["pelvis"]
            trunk_euler_angles = trunk_euler_angles - trunk_euler_angles_anat
            pelvis_euler_angles = pelvis_euler_angles - pelvis_euler_angles_anat

        max_val = np.max([trunk_euler_angles, pelvis_euler_angles])
        min_val = np.min([trunk_euler_angles, pelvis_euler_angles])

        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        axes[0].plot(
            time,
            trunk_euler_angles[:, 0],
            label="Trunk X [Ext. backward (+) Flex. forward (-)]",
            color="red",
        )
        axes[0].plot(
            time,
            pelvis_euler_angles[:, 0],
            label="Pelvis X [Ext. backward (+) Flex. forward (-)]",
            linestyle="--",
            color="red",
        )
        if use_anatomical and file_name in anatomical_data:
            axes[0].axhline(
                trunk_euler_angles_anat[0],
                color="gray",
                linestyle="-",
                label="Anatomical Trunk X",
            )
            axes[0].axhline(
                pelvis_euler_angles_anat[0],
                color="gray",
                linestyle="--",
                label="Anatomical Pelvis X",
            )
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("X: Ext(+)/Flex(-) (degrees)")
        axes[0].legend()
        axes[0].set_title(f"Euler Angles - {file_name} (X-axis)")

        axes[1].plot(
            time,
            trunk_euler_angles[:, 1],
            label="Trunk Y [Side Bending Right (+) Side Bending Left (-)]",
            color="green",
        )
        axes[1].plot(
            time,
            pelvis_euler_angles[:, 1],
            label="Pelvis Y [Side Bending Right (+) Side Bending Left (-)]",
            linestyle="--",
            color="green",
        )
        if use_anatomical and file_name in anatomical_data:
            axes[1].axhline(
                trunk_euler_angles_anat[1],
                color="gray",
                linestyle="-",
                label="Anatomical Trunk Y",
            )
            axes[1].axhline(
                pelvis_euler_angles_anat[1],
                color="gray",
                linestyle="--",
                label="Anatomical Pelvis Y",
            )
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Y: Side Bending R(+)/L(-) (degrees)")
        axes[1].legend()
        axes[1].set_title(f"Euler Angles - {file_name} (Y-axis)")

        axes[2].plot(
            time,
            trunk_euler_angles[:, 2],
            label="Trunk Z [Axial Rot. Right (+) Axial Rot. Left (-)]",
            color="blue",
        )
        axes[2].plot(
            time,
            pelvis_euler_angles[:, 2],
            label="Pelvis Z [Axial Rot. Right (+) Axial Rot. Left (-)]",
            linestyle="--",
            color="blue",
        )
        if use_anatomical and file_name in anatomical_data:
            axes[2].axhline(
                trunk_euler_angles_anat[2],
                color="gray",
                linestyle="-",
                label="Anatomical Trunk Z",
            )
            axes[2].axhline(
                pelvis_euler_angles_anat[2],
                color="gray",
                linestyle="--",
                label="Anatomical Pelvis Z",
            )
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Z: Axial Rot. R(+)/L(-) (degrees)")
        axes[2].legend()
        axes[2].set_title(f"Euler Angles - {file_name} (Z-axis)")

        margin = 0.05 * (max_val - min_val)
        axes[0].set_ylim([min_val - margin, max_val + margin])
        axes[1].set_ylim([min_val - margin, max_val + margin])
        axes[2].set_ylim([min_val - margin, max_val + margin])

        plt.tight_layout()

        base_name = os.path.splitext(file_name)[0]
        fig_path = os.path.join(base_dir_figures, f"{base_name}_figure.png")
        plt.savefig(fig_path)

        save_results_to_csv(
            base_dir_processed_data,
            time,
            trunk_euler_angles,
            pelvis_euler_angles,
            base_name,
        )

    for fig in matplotlib_figs:
        fig.show()

    root.destroy()


if __name__ == "__main__":
    analyze_mocap_fullbody_data()
