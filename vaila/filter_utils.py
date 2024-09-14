"""
Module: filter_utils.py
Description: This module provides a unified and flexible Butterworth filter function for low-pass and band-pass filtering of signals. The function supports edge effect mitigation through optional signal padding and uses second-order sections (SOS) for improved numerical stability.

Author: Prof. Dr. Paulo R. P. Santiago
Version: 1.1
Date: 2024-09-12

Changelog:
- Version 1.1 (2024-09-12):
  - Modified `butter_filter` to handle multidimensional data.
  - Adjusted padding length dynamically based on data length.
  - Fixed issues causing errors when data length is less than padding length.

Usage Example:
- Low-pass filter:
  `filtered_data_low = butter_filter(data, fs=1000, filter_type='low', cutoff=10, order=4)`

- Band-pass filter:
  `filtered_data_band = butter_filter(data, fs=1000, filter_type='band', lowcut=5, highcut=15, order=4)`
"""

from scipy.signal import butter, sosfiltfilt
import numpy as np


def butter_filter(
    data,
    fs,
    filter_type="low",
    cutoff=None,
    lowcut=None,
    highcut=None,
    order=4,
    padding=True,
):
    """
    Applies a Butterworth filter (low-pass or band-pass) to the input data.

    Parameters:
    - data: array-like
        The input signal to be filtered. Can be 1D or multidimensional. Filtering is applied along the first axis.
    - fs: float
        The sampling frequency of the signal.
    - filter_type: str, default='low'
        The type of filter to apply: 'low' for low-pass or 'band' for band-pass.
    - cutoff: float, optional
        The cutoff frequency for a low-pass filter.
    - lowcut: float, optional
        The lower cutoff frequency for a band-pass filter.
    - highcut: float, optional
        The upper cutoff frequency for a band-pass filter.
    - order: int, default=4
        The order of the Butterworth filter.
    - padding: bool, default=True
        Whether to pad the signal to mitigate edge effects.

    Returns:
    - filtered_data: array-like
        The filtered signal.
    """
    # Check filter type and set parameters
    nyq = 0.5 * fs  # Nyquist frequency
    if filter_type == "low":
        if cutoff is None:
            raise ValueError("Cutoff frequency must be provided for low-pass filter.")
        normal_cutoff = cutoff / nyq
        sos = butter(order, normal_cutoff, btype="low", analog=False, output="sos")
    elif filter_type == "band":
        if lowcut is None or highcut is None:
            raise ValueError(
                "Lowcut and highcut frequencies must be provided for band-pass filter."
            )
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype="band", analog=False, output="sos")
    else:
        raise ValueError(
            "Unsupported filter type. Use 'low' for low-pass or 'band' for band-pass."
        )

    data = np.asarray(data)
    axis = 0  # Filtering along the first axis (rows)

    # Apply padding if needed to handle edge effects
    if padding:
        data_len = data.shape[axis]
        # Ensure padding length is suitable for data length
        max_padlen = data_len - 1
        padlen = min(int(fs), max_padlen, 15)

        if data_len <= padlen:
            raise ValueError(
                f"The length of the input data ({data_len}) must be greater than the padding length ({padlen})."
            )

        # Pad the data along the specified axis
        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (padlen, padlen)
        padded_data = np.pad(data, pad_width=pad_width, mode="reflect")
        filtered_padded_data = sosfiltfilt(sos, padded_data, axis=axis, padlen=0)
        # Remove padding
        idx = [slice(None)] * data.ndim
        idx[axis] = slice(padlen, -padlen)
        filtered_data = filtered_padded_data[tuple(idx)]
    else:
        filtered_data = sosfiltfilt(sos, data, axis=axis, padlen=0)

    return filtered_data
