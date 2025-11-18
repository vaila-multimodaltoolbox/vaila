"""
Project: vailá Multimodal Toolbox
Script: filtering.py - Filtering

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2025
Update Date: 29 July 2025
Version: 0.0.1

Description:
    Filtering.

Usage:
    Run the script from the command line:
        python filtering.py

Requirements:
    - Python 3.x
    - Scipy
    - Numpy

License:
    This project is licensed under the terms of GNU General Public License v3.0.

Change History:
    - v0.0.1: First version
"""

import os

import numpy as np
from rich import print
from scipy.signal import butter, filtfilt, firwin


def apply_filter(
    data, sample_rate, method="butterworth", cutoff=5, order=4, numtaps=255, padlen=128
):
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    """Apply a specified filter to the data with padding."""
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist

    if method == "butterworth":
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        padded_data = np.pad(data, ((padlen, padlen), (0, 0)), mode="reflect")
        filtered_data = filtfilt(b, a, padded_data, axis=0)
        filtered_data = filtered_data[padlen:-padlen, :]

    elif method == "fir":
        fir_coeff = firwin(numtaps, normal_cutoff, window="blackman")
        padded_data = np.pad(data, ((padlen, padlen), (0, 0)), mode="reflect")
        filtered_data = filtfilt(fir_coeff, 1.0, padded_data, axis=0)
        filtered_data = filtered_data[padlen:-padlen, :]

    else:
        raise ValueError("Method must be 'butterworth' or 'fir'")

    return filtered_data
