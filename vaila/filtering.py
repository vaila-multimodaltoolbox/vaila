import numpy as np
from scipy.signal import butter, filtfilt, firwin


def apply_filter(
    data, sample_rate, method="butterworth", cutoff=5, order=4, numtaps=255, padlen=128
):
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
