# import os
# import sys
import numpy as np
import pandas as pd


def determine_header_lines(file_path):
    """
    Determines the number of header lines in a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    int: The number of header lines.
    """
    with open(file_path) as f:
        for i, line in enumerate(f):
            first_element = line.split(",")[0].strip()
            if first_element.replace(".", "", 1).isdigit():
                return i
    return 0


def determine_header_lines_mocap(file_path):
    """
    Determines the number of header lines in a CSV file by checking for the presence of certain expected headers.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    int: The number of header lines (0, 1, or 2).
    """
    with open(file_path) as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if "Time" in line or "LFHD_X" in line or "CLAV_X" in line:
                return i
    return 0


def read_cluster_csv(file_path):
    """
    Reads a cluster CSV file, detecting if it has 1, 2, or 3 header lines, and converts units from mm to meters.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    np.ndarray: The data from the CSV file as a NumPy array.
    """
    try:
        header_lines = determine_header_lines(file_path)
        # Use np.loadtxt instead of np.genfromtxt to avoid nan values
        data = np.loadtxt(file_path, delimiter=",", skiprows=header_lines)
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def read_mocap_csv(file_path):
    """
    Reads a mocap CSV file, detecting if it has 1, 2, or 3 header lines, and converts units from mm to meters.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The data from the CSV file as a pandas DataFrame with units converted from mm to meters.
    """
    try:
        header_lines = determine_header_lines_mocap(file_path)
        data = pd.read_csv(file_path, header=header_lines)
        data.iloc[:, :] = data.iloc[:, :].astype(float)
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
