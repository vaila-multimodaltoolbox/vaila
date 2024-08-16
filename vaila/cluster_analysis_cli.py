import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from vaila.data_processing import read_cluster_csv
from vaila.filtering import apply_filter
from vaila.rotation import createortbase, calcmatrot, rotmat2euler
from vaila.plotting import plot_orthonormal_bases


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


def analyze_cluster_data(directory_path, use_anatomical):
    sample_rate = float(input("Please enter the sample rate: "))

    print(
        """
         (A)               (B)

        * P2            * P3
        / |               | \\
    P3 *  |               |  * P2
        \\ |              | /
        * P1              * P1


         (C)               (D)

    P3 *-----* P2          * P2
        \\   /            /   \\
        * P1          P3 *-----* P1

    """
    )

    trunk_config = (
        input("Please enter the configuration for the trunk (A/B/C/D): ")
        .strip()
        .upper()
    )
    pelvis_config = (
        input("Please enter the configuration for the pelvis (A/B/C/D): ")
        .strip()
        .upper()
    )

    if trunk_config not in ["A", "B", "C", "D"] or pelvis_config not in [
        "A",
        "B",
        "C",
        "D",
    ]:
        print("Invalid input for configuration. Please enter 'A', 'B', 'C', or 'D'.")
        return

    filter_method = "butterworth"
    file_names = sorted([f for f in os.listdir(directory_path) if f.endswith(".csv")])

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir_figures = f"results/cluster/figure/Cluster_{current_time}"
    base_dir_processed_data = f"results/cluster/processed_data/Cluster_{current_time}"
    os.makedirs(base_dir_figures, exist_ok=True)
    os.makedirs(base_dir_processed_data, exist_ok=True)

    anatomical_data = {}
    if use_anatomical:
        anatomical_dir = "./data/cluster_csv/anatomical_position"
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

    matplotlib_figs = []  # List to store matplotlib figures
    for idx, file_name in enumerate(file_names):
        file_path = os.path.join(directory_path, file_name)
        data = read_cluster_csv(file_path)

        # Create time vector
        time = np.linspace(0, len(data) / sample_rate, len(data))
        # Insert time column in the data in the first column
        data = np.insert(data, 0, time, axis=1)
        # Remove time column if present in the data
        data = data[:, 1:]
        # Apply filter to the data
        dataf = apply_filter(data, sample_rate, method=filter_method)
        # Extract the points for the trunk and pelvis
        trunk_p1 = dataf[:, [0, 1, 2]]
        trunk_p2 = dataf[:, [3, 4, 5]]
        trunk_p3 = dataf[:, [6, 7, 8]]
        pelvis_p1 = dataf[:, [9, 10, 11]]
        pelvis_p2 = dataf[:, [12, 13, 14]]
        pelvis_p3 = dataf[:, [15, 16, 17]]

        blab = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        trunk_base, orig_trunk = createortbase(
            trunk_p1, trunk_p2, trunk_p3, trunk_config
        )
        pelvis_base, orig_pelvis = createortbase(
            pelvis_p1, pelvis_p2, pelvis_p3, pelvis_config
        )

        fig_matplotlib = plot_orthonormal_bases(
            [trunk_base, pelvis_base],
            [orig_trunk, orig_pelvis],
            [[trunk_p1, trunk_p2, trunk_p3], [pelvis_p1, pelvis_p2, pelvis_p3]],
            ["Trunk", "Pelvis"],
            title=f"Cluster Bases - {file_name}",
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


if __name__ == "__main__":
    use_anatomical = (
        input("Do you want to analyze with anatomical position data? (y/n): ")
        .strip()
        .lower()
        == "y"
    )
    directory_path = "./data/cluster_csv/"
    analyze_cluster_data(directory_path, use_anatomical)
