"""
Module: spectral_features.py
Description: Provides functions to calculate spectral features from power spectral density (PSD) data,
including total power, power frequency percentiles, power mode, spectral moments, centroid frequency,
frequency dispersion, energy content in specific frequency bands, and frequency quotient.

Author: Prof. Dr. Paulo R. P. Santiago
Version: 1.1
Date: 2024-11-13

Changelog:
- Version 1.1 (2024-11-13):
  - Added robust handling for empty frequency ranges.
  - Adjusted frequency range dynamically when out of bounds.
- Version 1.0 (2024-09-12):
  - Initial release with functions to compute various spectral features from PSD data.

Usage:
- Import the module and use the functions to compute spectral features:
  from spectral_features import *
  total_power_ml = total_power(freqs_ml, psd_ml)
  power_freq_50_ml = power_frequency_50(freqs_ml, psd_ml)
  # etc.
"""

import numpy as np


def adjust_frequency_range(freqs, fmin, fmax):
    """
    Adjusts the frequency range to ensure it fits within the bounds of available frequencies.
    """
    if fmin < freqs.min():
        fmin = freqs.min()
    if fmax > freqs.max():
        fmax = freqs.max()
    return fmin, fmax


def total_power(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the total power within the specified frequency range."""
    fmin, fmax = adjust_frequency_range(freqs, fmin, fmax)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))
    selected_powers = psd[idx]
    if len(selected_powers) == 0:
        return np.nan
    return np.sum(selected_powers)


def power_frequency_50(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the frequency at which 50% of the total power is reached."""
    fmin, fmax = adjust_frequency_range(freqs, fmin, fmax)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))
    selected_freqs = freqs[idx]
    selected_powers = psd[idx]

    if len(selected_powers) == 0:
        print(f"No frequencies in range: fmin={fmin}, fmax={fmax}")
        return np.nan

    cum_power = np.cumsum(selected_powers)
    total_power = cum_power[-1]

    if total_power == 0:
        return np.nan

    freq_50_idx = np.where(cum_power >= total_power * 0.5)[0]
    if freq_50_idx.size == 0:
        return np.nan
    return selected_freqs[freq_50_idx[0]]


def power_frequency_95(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the frequency at which 95% of the total power is reached."""
    fmin, fmax = adjust_frequency_range(freqs, fmin, fmax)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))
    selected_freqs = freqs[idx]
    selected_powers = psd[idx]

    if len(selected_powers) == 0:
        print(f"No frequencies in range: fmin={fmin}, fmax={fmax}")
        return np.nan

    cum_power = np.cumsum(selected_powers)
    total_power = cum_power[-1]

    if total_power == 0:
        return np.nan

    freq_95_idx = np.where(cum_power >= total_power * 0.95)[0]
    if freq_95_idx.size == 0:
        return np.nan
    return selected_freqs[freq_95_idx[0]]


def power_mode(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the frequency with the maximum power within the specified range."""
    fmin, fmax = adjust_frequency_range(freqs, fmin, fmax)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))
    if idx[0].size == 0:
        return np.nan
    selected_freqs = freqs[idx]
    selected_powers = psd[idx]
    mode_idx = np.argmax(selected_powers)
    return selected_freqs[mode_idx]


def spectral_moment(freqs, psd, moment=1, fmin=0.15, fmax=5):
    """Calculates the spectral moment of the given order within the specified frequency range."""
    fmin, fmax = adjust_frequency_range(freqs, fmin, fmax)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))
    selected_freqs = freqs[idx]
    selected_powers = psd[idx]
    if len(selected_powers) == 0:
        return np.nan
    return np.sum((selected_freqs**moment) * selected_powers)


def centroid_frequency(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the centroid frequency."""
    m0 = spectral_moment(freqs, psd, moment=0, fmin=fmin, fmax=fmax)
    m2 = spectral_moment(freqs, psd, moment=2, fmin=fmin, fmax=fmax)
    if m0 == 0:
        return np.nan
    return np.sqrt(m2 / m0)


def frequency_dispersion(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the frequency dispersion."""
    m0 = spectral_moment(freqs, psd, moment=0, fmin=fmin, fmax=fmax)
    m1 = spectral_moment(freqs, psd, moment=1, fmin=fmin, fmax=fmax)
    m2 = spectral_moment(freqs, psd, moment=2, fmin=fmin, fmax=fmax)
    if m0 * m2 == 0:
        return np.nan
    return np.sqrt(1 - (m1**2) / (m0 * m2))


def energy_content(freqs, psd, f_low, f_high, fmin=0.15, fmax=5):
    """Calculates the energy content between f_low and f_high Hz."""
    fmin, fmax = adjust_frequency_range(freqs, fmin, fmax)
    idx = np.where((freqs >= f_low) & (freqs <= f_high) & (freqs >= fmin) & (freqs <= fmax))
    selected_powers = psd[idx]
    if len(selected_powers) == 0:
        return np.nan
    return np.sum(selected_powers)


def energy_content_below_0_5(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the energy content below 0.5 Hz."""
    return energy_content(freqs, psd, f_low=0, f_high=0.5, fmin=fmin, fmax=fmax)


def energy_content_0_5_2(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the energy content between 0.5 Hz and 2 Hz."""
    return energy_content(freqs, psd, f_low=0.5, f_high=2, fmin=fmin, fmax=fmax)


def energy_content_above_2(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the energy content above 2 Hz."""
    return energy_content(freqs, psd, f_low=2, f_high=fmax, fmin=fmin, fmax=fmax)


def frequency_quotient(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the frequency quotient."""
    power_below_2 = energy_content(freqs, psd, f_low=0, f_high=2, fmin=fmin, fmax=fmax)
    power_above_2 = energy_content(freqs, psd, f_low=2, f_high=fmax, fmin=fmin, fmax=fmax)
    if power_below_2 == 0:
        return np.nan
    return power_above_2 / power_below_2
