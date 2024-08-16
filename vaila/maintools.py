import numpy as np
import ezc3d
import os
import pandas as pd
from typing import Dict, Optional
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib.patches as mpatches
from scipy import signal


def get_kinematics_c3d(file_path):
    """
    Extracts and returns kinematics data (x, y, z coordinates) for all point labels
    from a given C3D file.

    Parameters:
    file_path (str): The path to the C3D file.

    Returns:
    dict: A dictionary containing x, y, z coordinates for each point label in the C3D file.
    """
    try:
        # Load the C3D file using ezc3d
        c3d = ezc3d.c3d(file_path)

        # Initialize a dictionary to store x, y, z coordinates for each point label
        all_points = {
            label: {"x": [], "y": [], "z": []}
            for label in c3d["parameters"]["POINT"]["LABELS"]["value"]
        }

        # Get the number of frames
        num_frames = c3d["data"]["points"].shape[2]

        # Iterate through frames in the C3D file
        for frame_idx in range(num_frames):
            # Get the coordinates for the current frame
            frame = c3d["data"]["points"][:, :, frame_idx]
            # Iterate through each point label in the current frame
            for idx, label in enumerate(all_points.keys()):
                # Append x, y, z coordinates of the current point label to the dictionary
                all_points[label]["x"].append(frame[0, idx])
                all_points[label]["y"].append(frame[1, idx])
                all_points[label]["z"].append(frame[2, idx])

        # Return the dictionary containing all point data
        all_points = {key.strip(): value for key, value in all_points.items()}
        return all_points

    except FileNotFoundError:
        print(f"C3D file not found: {file_path}")
        return {}

    except Exception as e:
        print(f"Error processing C3D file: {file_path}. Error: {e}")
        return {}


def get_kinematic_framerate(file_path: str) -> int:
    """
    Retrieves the kinematic framerate from a given C3D file.

    Parameters:
    file_path (str): The path to the C3D file.

    Returns:
    int: The kinematic framerate of the C3D file.

    Raises:
    FileNotFoundError: If the specified file does not exist.
    ValueError: If the provided file is not a C3D file.
    Exception: For general exceptions while reading the C3D file.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The specified file does not exist: {file_path}")

    _, ext = os.path.splitext(file_path)
    if ext.lower() != ".c3d":
        raise ValueError("The provided file is not a C3D file")

    try:
        # Load the C3D file using ezc3d
        c3d = ezc3d.c3d(file_path)
        # Retrieve and return the kinematic framerate
        return int(c3d["parameters"]["POINT"]["RATE"]["value"][0])
    except Exception as e:
        raise Exception(f"An error occurred while reading the C3D file: {e}")


def butter_lowpass(freq, data, fc=6, order=4):
    w = fc / (freq / 2)  # Normalize the frequency
    b, a = signal.butter(order, w, btype="low")
    filt_signal = signal.filtfilt(b, a, data)
    return filt_signal


def get_joint_df(
    kinematic_dict: Dict[str, Dict[str, list]], joint: Optional[str] = None
) -> pd.DataFrame:
    if not isinstance(kinematic_dict, dict):
        raise TypeError("kinematic_dict must be a dictionary")

    if joint is None:
        raise KeyError("Please specify a joint.")

    if joint not in kinematic_dict:
        raise KeyError(f"The specified joint '{joint}' is not found in kinematic_dict")

    return pd.DataFrame(kinematic_dict[joint])


def timenormalize_data(signal, T1=None, T2=None, n_el=101):
    T1 = T1 if T1 is not None else 0
    T2 = T2 if T2 is not None else len(signal)

    x_base_norm = np.linspace(1, T2 - T1 + 1, n_el)
    x_base_current = np.arange(1, T2 - T1 + 1)
    nM = signal.shape[1]
    signal_timenormalized = np.empty((n_el, nM))
    signal_timenormalized[:] = np.nan
    for i in range(nM):
        y_current = signal.iloc[T1:T2, i]
        if np.sum(np.isnan(y_current.astype("float").values)):
            y_current = np.zeros(len(y_current), 1)
        spline = interpolate.InterpolatedUnivariateSpline(x_base_current, y_current)
        signal_timenormalized[:, i] = spline(x_base_norm)
    return signal_timenormalized


def calculate_coupling_angle(joint1_array: np.ndarray, joint2_array: np.ndarray):
    if len(joint1_array) != len(joint2_array) or len(joint1_array) == 0:
        raise ValueError("Input arrays must be of equal non-zero length.")

    array_joint1 = np.diff(joint1_array, axis=0)
    array_joint2 = np.diff(joint2_array, axis=0)

    vm_ab = np.hypot(array_joint1, array_joint2)
    cosang_ab = np.divide(array_joint1, vm_ab, where=vm_ab != 0)
    sinang_ab = np.divide(array_joint2, vm_ab, where=vm_ab != 0)
    coupangle = np.degrees(np.arctan2(cosang_ab, sinang_ab))

    coupangle[coupangle < 0] += 360

    CtgVar_vc_DG = np.select(
        condlist=[
            (coupangle >= 0) & (coupangle < 22.5),
            (coupangle >= 22.5) & (coupangle < 67.5),
            (coupangle >= 67.5) & (coupangle < 112.5),
            (coupangle >= 112.5) & (coupangle < 157.5),
            (coupangle >= 157.5) & (coupangle < 202.5),
            (coupangle >= 202.5) & (coupangle < 247.5),
            (coupangle >= 247.5) & (coupangle < 292.5),
            (coupangle >= 292.5) & (coupangle < 337.5),
            (coupangle >= 337.5) & (coupangle < 360),
        ],
        choicelist=[1, 2, 3, 4, 1, 2, 3, 4, 1],
        default=0,
    )

    group_phase = [
        round((np.count_nonzero(CtgVar_vc_DG == i) / len(CtgVar_vc_DG)) * 100, 2)
        for i in range(1, 5)
    ]

    return group_phase, coupangle


def create_coupling_angle_figure(
    group_percent,
    coupangle,
    array_joint1,
    array_joint2,
    joint1_name="Joint1",
    joint2_name="Joint2",
    axis_title="X-Axis",
    size=15,
):
    letter_size = size - 5
    mark_size = size / 2
    alpha_value = 0.5
    gray_colors = ["0.1", "0.6", "0.3", "0.8"]

    plt.close("all")
    fig, ax = plt.subplots(3, figsize=(size, size / 1.5))
    plt.subplots_adjust(hspace=0.35)

    ax[0].set_title(
        f"Joint Angles | {joint1_name} - {joint2_name} | Axis: {axis_title}",
        size=letter_size,
        weight="bold",
    )
    ax[0].plot(
        array_joint1,
        marker="o",
        linestyle="-",
        color="b",
        markersize=mark_size,
        alpha=alpha_value,
        label=joint1_name,
    )
    ax[0].plot(
        array_joint2,
        marker="o",
        linestyle="-",
        color="r",
        markersize=mark_size,
        alpha=alpha_value,
        label=joint2_name,
    )
    ax[0].legend(loc="best", fontsize=letter_size, frameon=False)
    ax[0].set_ylabel("Joint Angle (°)", fontsize=letter_size)
    ax[0].set_xlim(0, 100)
    ax[0].set_xlabel("Cycle (%)", fontsize=letter_size)

    ax[1].set_title(
        f"Coupling Angle | {joint1_name} - {joint2_name} | Axis: {axis_title}",
        size=letter_size,
        weight="bold",
    )
    ax[1].plot(
        coupangle,
        color="k",
        marker="o",
        markersize=mark_size,
        linestyle=":",
        alpha=alpha_value,
        label="Coupling Angle",
    )
    ax[1].legend(loc="best", fontsize=letter_size, frameon=False)
    ax[1].set_ylabel("Coupling Angle (°)", fontsize=letter_size)
    ax[1].set_xlim(0, 100)
    ax[1].set_xlabel("Cycle (%)", fontsize=letter_size)
    ax[1].set_title(
        f"Coupling Angle | {joint1_name} - {joint2_name} | Axis: {axis_title}",
        size=letter_size,
        weight="bold",
    )
    ax[1].axhline(22.50, color="#55555B", linestyle="dotted", linewidth=0.5)
    ax[1].axhline(67.50, color="#55555B", linestyle="dotted", linewidth=0.5)
    ax[1].axhline(112.5, color="#55555B", linestyle="dotted", linewidth=0.5)
    ax[1].axhline(157.5, color="#55555B", linestyle="dotted", linewidth=0.5)
    ax[1].axhline(202.5, color="#55555B", linestyle="dotted", linewidth=0.5)
    ax[1].axhline(247.5, color="#55555B", linestyle="dotted", linewidth=0.5)
    ax[1].axhline(292.5, color="#55555B", linestyle="dotted", linewidth=0.5)
    ax[1].axhline(337.5, color="#55555B", linestyle="dotted", linewidth=0.5)
    ax[1].axhline(360, color="#55555B", linestyle="dotted", linewidth=0.5)
    ax[1].tick_params(axis="y", labelsize=letter_size)
    ax[1].tick_params(axis="x", labelsize=letter_size)
    ax2 = ax[1].twinx()
    ax2.set_yticks(
        [
            22.5 - 11.25,
            67.5 - 22.5,
            112.5 - 22.5,
            157.5 - 22.5,
            202.5 - 22.5,
            247.5 - 22.5,
            292.5 - 22.5,
            337.5 - 22.5,
            360,
        ],
        [
            f"{joint1_name}",
            "In-Phase",
            f"{joint2_name}",
            "Anti-Phase",
            f"{joint1_name}",
            "In-Phase",
            f"{joint2_name}",
            "Anti-Phase",
            f"{joint1_name}",
        ],
        weight="bold",
    )

    labels = ["Anti-Phase", "In-Phase", f"{joint1_name} Phase", f"{joint2_name} Phase"]
    ax[2].set_title(
        f"Categorization of Coordination Patterns | {joint1_name} - {joint2_name} | Axis: {axis_title}",
        size=letter_size,
        weight="bold",
    )
    ax[2].set_ylabel("Percentage (%)", fontsize=letter_size)
    # bars = ax[2].bar(labels, group_percent, color=gray_colors, alpha=0.7)

    patches = [
        mpatches.Patch(color=color, label=f"{label}: {perc:.0f}%")
        for color, label, perc in zip(gray_colors, labels, group_percent)
    ]

    ax[2].legend(handles=patches, loc="best", fontsize=letter_size, frameon=False)

    return fig, ax
