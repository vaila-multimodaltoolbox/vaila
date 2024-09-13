"""
Module: spectral_features.py
Description: Provides functions to calculate spectral features from power spectral density (PSD) data,
including total power, power frequency percentiles, power mode, spectral moments, centroid frequency,
frequency dispersion, energy content in specific frequency bands, and frequency quotient.

Author: Prof. Dr. Paulo R. P. Santiago
Version: 1.0
Date: 2024-09-12

Changelog:
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

def total_power(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the total power within the specified frequency range."""
    idx = np.where((freqs >= fmin) & (freqs <= fmax))
    selected_powers = psd[idx]
    feature = np.sum(selected_powers)
    return feature

def power_frequency_50(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the frequency at which 50% of the total power is reached."""
    idx = np.where((freqs >= fmin) & (freqs <= fmax))
    selected_freqs = freqs[idx]
    selected_powers = psd[idx]
    cum_power = np.cumsum(selected_powers)
    total_power = cum_power[-1]
    if total_power == 0:
        return np.nan
    freq_50_idx = np.where(cum_power >= total_power * 0.5)[0]
    if freq_50_idx.size == 0:
        return np.nan
    freq_50 = selected_freqs[freq_50_idx[0]]
    return freq_50

def power_frequency_95(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the frequency at which 95% of the total power is reached."""
    idx = np.where((freqs >= fmin) & (freqs <= fmax))
    selected_freqs = freqs[idx]
    selected_powers = psd[idx]
    cum_power = np.cumsum(selected_powers)
    total_power = cum_power[-1]
    if total_power == 0:
        return np.nan
    freq_95_idx = np.where(cum_power >= total_power * 0.95)[0]
    if freq_95_idx.size == 0:
        return np.nan
    freq_95 = selected_freqs[freq_95_idx[0]]
    return freq_95

def power_mode(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the frequency with the maximum power within the specified range."""
    idx = np.where((freqs >= fmin) & (freqs <= fmax))
    if idx[0].size == 0:
        return np.nan
    selected_freqs = freqs[idx]
    selected_powers = psd[idx]
    mode_idx = np.argmax(selected_powers)
    freq_mode = selected_freqs[mode_idx]
    return freq_mode

def spectral_moment(freqs, psd, moment=1, fmin=0.15, fmax=5):
    """Calculates the spectral moment of the given order within the specified frequency range."""
    idx = np.where((freqs >= fmin) & (freqs <= fmax))
    selected_freqs = freqs[idx]
    selected_powers = psd[idx]
    feature = np.sum((selected_freqs ** moment) * selected_powers)
    return feature

def centroid_frequency(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the centroid frequency."""
    m0 = spectral_moment(freqs, psd, moment=0, fmin=fmin, fmax=fmax)
    m2 = spectral_moment(freqs, psd, moment=2, fmin=fmin, fmax=fmax)
    if m0 == 0:
        return np.nan
    feature = np.sqrt(m2 / m0)
    return feature

def frequency_dispersion(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the frequency dispersion."""
    m0 = spectral_moment(freqs, psd, moment=0, fmin=fmin, fmax=fmax)
    m1 = spectral_moment(freqs, psd, moment=1, fmin=fmin, fmax=fmax)
    m2 = spectral_moment(freqs, psd, moment=2, fmin=fmin, fmax=fmax)
    if m0 * m2 == 0:
        return np.nan
    feature = np.sqrt(1 - (m1 ** 2) / (m0 * m2))
    return feature

def energy_content(freqs, psd, f_low, f_high, fmin=0.15, fmax=5):
    """Calculates the energy content between f_low and f_high Hz."""
    idx = np.where((freqs >= f_low) & (freqs <= f_high) & (freqs >= fmin) & (freqs <= fmax))
    selected_powers = psd[idx]
    feature = np.sum(selected_powers)
    return feature

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
    feature = power_above_2 / power_below_2
    return feature

