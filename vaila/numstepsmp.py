"""
numstepsmp.py

Description:
    Opens a dialog to select a CSV file of foot coordinates
    and calculates the number of steps based on foot position using
    MediaPipe data. Includes Butterworth filtering and advanced metrics using
    multiple markers (ankle, heel, toe).

Author:
    Paulo Roberto Pereira Santiago

Created:
    14 May 2025
Updated:
    16 May 2025

Usage:
    python numstepsmp.py

Dependencies:
    - pandas
    - numpy
    - scipy
    - tkinter (GUI for file selection)
    - matplotlib (optional, for visualization)
"""

import datetime
import glob
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter


def butterworth_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Applies a Butterworth low-pass filter to the signal.

    Parameters:
    - data: 1D array of values
    - cutoff: cutoff frequency in Hz
    - fs: sampling frequency in Hz
    - order: filter order

    Returns:
    - filtered array
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


def filter_signals(data: np.ndarray, window_length: int = 11, polyorder: int = 2) -> np.ndarray:
    """
    Applies a Savitzky-Golay filter to smooth the data and reduce noise.

    Parameters:
    - data: array of data to be filtered
    - window_length: smoothing window size (must be odd)
    - polyorder: polynomial order for fitting

    Returns:
    - filtered array
    """
    if window_length % 2 == 0:
        window_length += 1  # Ensures window size is odd

    if len(data) <= window_length:
        # Can't apply the filter if data is too short
        return data

    try:
        filtered = savgol_filter(data, window_length, polyorder)
        return filtered
    except Exception as e:
        print(f"Error applying filter: {e}")
        # In case of error, return original data
        return data


def calculate_feet_metrics(
    df: pd.DataFrame, fs: float = 30.0, cutoff: float = 3.0, order: int = 4
) -> dict[str, np.ndarray]:
    """
    Calculates various metrics between feet that can indicate steps.
    Applies Butterworth filter to smooth the data.

    Parameters:
    - df: DataFrame with foot coordinates
    - fs: sampling frequency (frames per second)
    - cutoff: cutoff frequency for the filter (Hz)
    - order: Butterworth filter order

    Returns:
    - Dictionary with different calculated metrics
    """
    # Extract basic coordinates
    left_x = df["left_foot_index_x"].astype(float).values
    left_y = df["left_foot_index_y"].astype(float).values
    left_z = df["left_foot_index_z"].astype(float).values
    right_x = df["right_foot_index_x"].astype(float).values
    right_y = df["right_foot_index_y"].astype(float).values
    right_z = df["right_foot_index_z"].astype(float).values

    # Extract heel coordinates for improved foot strike detection
    left_heel_z = df["left_heel_z"].astype(float).values
    right_heel_z = df["right_heel_z"].astype(float).values

    # Calculate hip midpoint for Z reference
    hip_z = (df["left_hip_z"].astype(float) + df["right_hip_z"].astype(float)) / 2.0

    # Print diagnostic info
    separation = np.abs(left_x - right_x)
    print("Horizontal separation diagnostics:")
    print(f"  - Min: {separation.min():.6f}, Max: {separation.max():.6f}")
    print(f"  - Mean: {separation.mean():.6f}, Std: {separation.std():.6f}")

    # Calculate Euclidean distance between feet
    euclidean_distance = np.sqrt((left_x - right_x) ** 2 + (left_y - right_y) ** 2)

    # Calculate horizontal separation (as in original algorithm)
    horizontal_separation = separation

    # Calculate height difference between feet (may indicate which foot is in the air)
    vertical_difference = left_y - right_y

    # Calculate horizontal velocity of each foot (position derivative)
    left_velocity = np.gradient(left_x)
    right_velocity = np.gradient(right_x)

    # Absolute velocity
    abs_left_velocity = np.abs(left_velocity)
    abs_right_velocity = np.abs(right_velocity)

    # Combined speed
    combined_speed = (abs_left_velocity + abs_right_velocity) / 2

    # Inverted speed (to detect minima as peaks)
    inverted_speed = -combined_speed

    # NEW METRICS: MEAN Y AND Z FOR ANKLE, HEEL AND INDEX MARKERS
    # LEFT foot
    left_y_markers = df[["left_ankle_y", "left_heel_y", "left_foot_index_y"]].astype(float).values
    left_z_markers = df[["left_ankle_z", "left_heel_z", "left_foot_index_z"]].astype(float).values
    mean_y_left = np.mean(left_y_markers, axis=1)
    mean_z_left = np.mean(left_z_markers, axis=1) - hip_z.values

    # RIGHT foot
    right_y_markers = (
        df[["right_ankle_y", "right_heel_y", "right_foot_index_y"]].astype(float).values
    )
    right_z_markers = (
        df[["right_ankle_z", "right_heel_z", "right_foot_index_z"]].astype(float).values
    )
    mean_y_right = np.mean(right_y_markers, axis=1)
    mean_z_right = np.mean(right_z_markers, axis=1) - hip_z.values

    # Difference between Y means (useful for step detection)
    mean_y_diff = mean_y_left - mean_y_right

    # NEW METRIC: Mean of heel and index only (excluding ankle) for more precise foot strike
    # LEFT foot
    left_heel_index_y = df[["left_heel_y", "left_foot_index_y"]].astype(float).values
    mean_heel_index_y_left = np.mean(left_heel_index_y, axis=1)

    # RIGHT foot
    right_heel_index_y = df[["right_heel_y", "right_foot_index_y"]].astype(float).values
    mean_heel_index_y_right = np.mean(right_heel_index_y, axis=1)

    # Difference between heel+index Y means (for more precise foot strike detection)
    heel_index_y_diff = mean_heel_index_y_left - mean_heel_index_y_right

    # NEW METRIC: Heel Z velocity (useful for foot strike detection)
    # Higher negative velocity indicates heel moving down, local minimum indicates heel strike
    left_heel_z_velocity = np.gradient(left_heel_z)
    right_heel_z_velocity = np.gradient(right_heel_z)

    # Calculate heel Z depth relative to hip
    left_heel_z_depth = left_heel_z - hip_z.values
    right_heel_z_depth = right_heel_z - hip_z.values

    # Apply Butterworth filter to all important metrics
    euclidean_distance = butterworth_filter(euclidean_distance, cutoff, fs, order)
    horizontal_separation = butterworth_filter(horizontal_separation, cutoff, fs, order)
    vertical_difference = butterworth_filter(vertical_difference, cutoff, fs, order)
    mean_y_left = butterworth_filter(mean_y_left, cutoff, fs, order)
    mean_y_right = butterworth_filter(mean_y_right, cutoff, fs, order)
    mean_y_diff = butterworth_filter(mean_y_diff, cutoff, fs, order)
    mean_z_left = butterworth_filter(mean_z_left, cutoff, fs, order)
    mean_z_right = butterworth_filter(mean_z_right, cutoff, fs, order)
    combined_speed = butterworth_filter(combined_speed, cutoff, fs, order)
    inverted_speed = butterworth_filter(inverted_speed, cutoff, fs, order)
    mean_heel_index_y_left = butterworth_filter(mean_heel_index_y_left, cutoff, fs, order)
    mean_heel_index_y_right = butterworth_filter(mean_heel_index_y_right, cutoff, fs, order)
    heel_index_y_diff = butterworth_filter(heel_index_y_diff, cutoff, fs, order)
    left_heel_z_depth = butterworth_filter(left_heel_z_depth, cutoff, fs, order)
    right_heel_z_depth = butterworth_filter(right_heel_z_depth, cutoff, fs, order)
    left_heel_z_velocity = butterworth_filter(left_heel_z_velocity, cutoff, fs, order)
    right_heel_z_velocity = butterworth_filter(right_heel_z_velocity, cutoff, fs, order)

    return {
        "euclidean_distance": euclidean_distance,
        "horizontal_separation": horizontal_separation,
        "vertical_difference": vertical_difference,
        "left_velocity": left_velocity,
        "right_velocity": right_velocity,
        "abs_left_velocity": abs_left_velocity,
        "abs_right_velocity": abs_right_velocity,
        "combined_speed": combined_speed,
        "inverted_speed": inverted_speed,
        "mean_y_left": mean_y_left,
        "mean_y_right": mean_y_right,
        "mean_y_diff": mean_y_diff,
        "mean_z_left": mean_z_left,
        "mean_z_right": mean_z_right,
        "hip_mid_z": hip_z.values,
        "mean_heel_index_y_left": mean_heel_index_y_left,
        "mean_heel_index_y_right": mean_heel_index_y_right,
        "heel_index_y_diff": heel_index_y_diff,
        "left_heel_z_depth": left_heel_z_depth,
        "right_heel_z_depth": right_heel_z_depth,
        "left_heel_z_velocity": left_heel_z_velocity,
        "right_heel_z_velocity": right_heel_z_velocity,
    }


def count_steps_original(
    df: pd.DataFrame, peak_distance: int = 10, fix_double_count: bool = True
) -> int:
    """
    Original step counting implementation (for reference).

    Parameters:
    - df: DataFrame with coordinates
    - peak_distance: minimum distance between peaks
    - fix_double_count: if True, corrects double counting by dividing by 2
    """
    left_x = df["left_foot_index_x"].astype(float).values
    right_x = df["right_foot_index_x"].astype(float).values
    separation = np.abs(left_x - right_x)

    # Apply filter to reduce noise
    separation = filter_signals(separation)

    # Adjust prominence to detect only main peaks
    min_val, max_val = separation.min(), separation.max()
    range_val = max_val - min_val
    prominence = range_val * 0.15  # Value adjusted for 60 FPS

    peaks, _ = find_peaks(separation, distance=peak_distance, prominence=prominence)

    num_peaks = len(peaks)
    if fix_double_count and num_peaks > 7:  # Suspect double counting if more than 7 steps
        return num_peaks // 2
    return num_peaks


def count_steps_basic(df: pd.DataFrame, peak_distance: int = 10, sensitivity: float = 0.15) -> int:
    """
    Basic method using only horizontal separation with explicit parameters.

    Parameters:
    - df: DataFrame with coordinates
    - peak_distance: minimum distance between peaks
    - sensitivity: detection sensitivity (0.05-0.3, lower = more sensitive)
    """
    left_x = df["left_foot_index_x"].astype(float).values
    right_x = df["right_foot_index_x"].astype(float).values
    separation = np.abs(left_x - right_x)

    # Find min/max values for parameter adjustment
    min_val, max_val = separation.min(), separation.max()
    range_val = max_val - min_val

    # Use adaptive values for height and prominence
    height = min_val + range_val * sensitivity
    prominence = range_val * (sensitivity / 3)  # Adjusted proportion

    print("Parameters for find_peaks:")
    print(f"  - Distance: {peak_distance}")
    print(f"  - Height: {height:.6f}")
    print(f"  - Prominence: {prominence:.6f}")

    peaks, _ = find_peaks(separation, distance=peak_distance, prominence=prominence, height=height)

    print(f"Peaks found: {len(peaks)} -> {peaks}")
    return len(peaks)


def count_steps_velocity(
    df: pd.DataFrame,
    peak_distance: int = 10,
    sensitivity: float = 0.1,
    fix_double_count: bool = True,
) -> int:
    """
    Velocity-based method - detects moments when the foot stops.

    Parameters:
    - df: DataFrame with coordinates
    - sensitivity: controls detection sensitivity
    - fix_double_count: if True, corrects double counting
    """
    # Central derivative
    v_left = np.gradient(df["left_foot_index_x"].astype(float).values)
    v_right = np.gradient(df["right_foot_index_x"].astype(float).values)

    # Combined velocity
    speed = (np.abs(v_left) + np.abs(v_right)) / 2

    # Invert to find minima as "peaks"
    inv_speed = -speed

    # Apply filter to reduce noise
    inv_speed = filter_signals(inv_speed)

    # Find adaptive parameters
    min_val, max_val = inv_speed.min(), inv_speed.max()
    range_val = max_val - min_val
    prominence = range_val * sensitivity

    peaks, _ = find_peaks(inv_speed, distance=peak_distance, prominence=prominence)

    num_peaks = len(peaks)
    print(f"Foot-strike candidates: {num_peaks}")

    # Correct double counting if necessary
    if fix_double_count and num_peaks > 10:  # Suspect double counting
        return num_peaks // 2
    return num_peaks


def count_steps_sliding_window(
    df: pd.DataFrame,
    window_size: int = 30,
    threshold_factor: float = 0.5,
    fix_double_count: bool = True,
) -> int:
    """
    Sliding window method - counts a step per window if there's a foot stop.

    Parameters:
    - window_size: window size in frames
    - threshold_factor: factor to define the velocity threshold (0-1)
    - fix_double_count: if True, corrects double counting
    """
    # Calculate velocity
    v_left = np.gradient(df["left_foot_index_x"].astype(float).values)
    v_right = np.gradient(df["right_foot_index_x"].astype(float).values)
    speed = (np.abs(v_left) + np.abs(v_right)) / 2

    # Apply filter to reduce noise
    speed = filter_signals(speed)

    # Find adaptive threshold
    velocity_threshold = speed.mean() * threshold_factor

    steps = 0
    step_positions = []

    # Use a more robust detection approach for 60 FPS
    # Reduce window overlap
    step_window = window_size // 2  # Step between consecutive windows

    for start in range(0, len(speed), step_window):
        win = speed[start : start + window_size]
        if len(win) < window_size // 2:  # Window too small at the end
            break

        # Identify the minimum index in the window
        idx = np.argmin(win)

        # Only count if this minimum is below the velocity threshold
        if win[idx] < velocity_threshold:
            steps += 1
            step_positions.append(start + idx)

    # Correct double counting if necessary
    if fix_double_count and steps > 10:  # Suspect double counting
        steps = steps // 2

    print(f"Steps (sliding window): {steps}")
    return steps


def count_steps_mean_y(
    metrics: dict[str, np.ndarray], peak_distance: int = 10, sensitivity: float = 0.15
) -> int:
    """
    Step counting method based on the average of the markers of each foot in Y.
    This method can be more robust because it uses the average of three markers (ankle, heel, foot_index).

    Parameters:
    - metrics: dictionary with calculated metrics (result of calculate_feet_metrics)
    - peak_distance: minimum distance between peaks
    - sensitivity: detection sensitivity (lower = more sensitive)

    Returns:
    - number of detected steps
    """
    # Use the mean Y difference between the two feet
    # This metric tends to oscillate around zero, with positive and negative peaks
    # representing alternating steps
    mean_y_diff = metrics["mean_y_diff"]

    # Process positive extremes (left foot advanced) and negative extremes (right foot advanced)
    pos_peaks, _ = find_peaks(
        mean_y_diff,
        distance=peak_distance,
        prominence=np.abs(mean_y_diff).max() * sensitivity,
    )

    neg_peaks, _ = find_peaks(
        -mean_y_diff,
        distance=peak_distance,
        prominence=np.abs(mean_y_diff).max() * sensitivity,
    )

    # The total number of steps is the sum of positive and negative peaks
    total_peaks = len(pos_peaks) + len(neg_peaks)

    print(
        f"Steps using mean_y_diff: {total_peaks} (positive peaks: {len(pos_peaks)}, negative: {len(neg_peaks)})"
    )
    return total_peaks


def count_steps_z_depth(
    metrics: dict[str, np.ndarray], peak_distance: int = 10, sensitivity: float = 0.2
) -> int:
    """
    Step counting method based on the Z depth of the feet relative to the hip.
    Observes moments when one foot is higher than the other in relation to the Z plane.

    Parameters:
    - metrics: dictionary with calculated metrics
    - peak_distance: minimum distance between peaks
    - sensitivity: detection sensitivity (lower = more sensitive)

    Returns:
    - number of detected steps
    """
    # Use the Z difference between the two feet
    z_diff = metrics["mean_z_left"] - metrics["mean_z_right"]

    # Find positive peaks (left foot higher) and negative peaks (right foot higher)
    pos_peaks, _ = find_peaks(
        z_diff, distance=peak_distance, prominence=np.abs(z_diff).max() * sensitivity
    )

    neg_peaks, _ = find_peaks(
        -z_diff, distance=peak_distance, prominence=np.abs(z_diff).max() * sensitivity
    )

    # Total steps is the sum of peaks
    total_peaks = len(pos_peaks) + len(neg_peaks)

    print(
        f"Steps using z_depth: {total_peaks} (positive peaks: {len(pos_peaks)}, negative: {len(neg_peaks)})"
    )
    return total_peaks


def detect_foot_strikes_heel_z(
    metrics: dict[str, np.ndarray], peak_distance: int = 20
) -> dict[str, Any]:
    """
    Detects foot strike events based on heel Z-depth and velocity.

    This method uses:
    1. Heel Z-depth (vertical position relative to hip) to detect when the heel is lowest
    2. Heel Z-velocity (vertical movement speed) to precisely identify the moment of contact
    3. Y position as additional verification for foot alternation
    4. Combined data to determine the exact frame of each foot strike

    Parameters:
    - metrics: dictionary with calculated metrics
    - peak_distance: minimum distance between peaks (frames)

    Returns:
    - dictionary with foot strike information
    """
    # Extract relevant signals
    left_heel_z = metrics["left_heel_z_depth"]  # Z-depth of left heel
    right_heel_z = metrics["right_heel_z_depth"]  # Z-depth of right heel

    # Velocity signals (used to detect when the heel stops moving)
    left_vel_z = metrics["left_heel_z_velocity"]
    right_vel_z = metrics["right_heel_z_velocity"]

    # Position signals for step alternation detection
    y_diff = metrics["heel_index_y_diff"]

    # Detect crossover points in Y (when step alternation happens)
    # This helps determine which foot is active for each step
    y_signal = np.sign(y_diff)
    y_signal[y_signal == 0] = 1
    crossovers = np.where(np.diff(y_signal) != 0)[0] + 1

    # Initialize strike detection
    all_strikes = []
    first_side = "left" if y_signal[0] > 0 else "right"

    # Add the first strike as reference (at the first frame)
    all_strikes.append((0, first_side))

    # Current side (alternates between left and right)
    current_side = first_side

    # For each position crossover, find the precise foot strike frame
    for i, cross_idx in enumerate(crossovers):
        # Switch the active foot side
        current_side = "right" if current_side == "left" else "left"

        # Select the active heel depth and velocity signals
        active_z = left_heel_z if current_side == "left" else right_heel_z
        active_vel = left_vel_z if current_side == "left" else right_vel_z

        # Define a window to search for the actual foot strike
        # Start near the crossover and look ahead for the heel contact
        search_window_size = peak_distance // 2
        window_start = max(0, cross_idx - 2)  # Start slightly before crossover
        window_end = min(len(active_z) - 1, cross_idx + search_window_size)

        # Create a scoring system for each frame in the search window
        scores = np.zeros(window_end - window_start + 1)

        for idx, frame in enumerate(range(window_start, window_end + 1)):
            if frame >= len(active_z):
                continue

            # Score 1: Z-depth - lower Z values (closer to ground) get better scores
            # Normalize Z-depth within window
            window_z = active_z[window_start : window_end + 1]
            z_min, z_max = np.min(window_z), np.max(window_z)
            z_range = z_max - z_min

            # If there's variation in Z within the window
            if z_range > 0:
                # Normalize to get 0-1, where 0 is lowest (best)
                z_score = (active_z[frame] - z_min) / z_range
            else:
                z_score = 0.5  # Neutral if no variation

            # Score 2: Z-velocity - a heel strike happens when velocity crosses from negative to near zero
            # Calculate velocity trend (deceleration is a strong indicator of foot strike)
            if frame > 0 and frame < len(active_vel):
                # Ideal foot strike is when velocity changes from negative (downward) to zero
                # Normalize velocity score - best when velocity approaches zero from below
                vel = active_vel[frame]
                prev_vel = active_vel[frame - 1] if frame > 0 else vel

                # Calculate velocity score (0 is best, indicates deceleration to stop)
                if prev_vel < 0 and abs(vel) < abs(prev_vel):
                    # Perfect case: decelerating from negative velocity
                    vel_score = 0.0
                elif vel == 0:
                    # Stopped
                    vel_score = 0.1
                elif abs(vel) < 0.01:
                    # Almost stopped
                    vel_score = 0.2
                else:
                    # Not ideal
                    vel_score = 0.5 + abs(vel) / (abs(active_vel).max() + 0.001)
            else:
                vel_score = 0.5  # Neutral

            # Score 3: Proximity to crossover point
            # (foot strikes usually happen shortly after the feet cross in Y-position)
            proximity_score = abs(frame - cross_idx) / search_window_size

            # Combined score (lower is better)
            # The weights prioritize Z-depth (50%), velocity (30%), and proximity (20%)
            scores[idx] = 0.5 * z_score + 0.3 * vel_score + 0.2 * proximity_score

        # Find the frame with the best (lowest) score
        best_idx = np.argmin(scores)
        strike_frame = window_start + best_idx

        # Add to the list of foot strikes
        all_strikes.append((int(strike_frame), current_side))

    # Handle edge cases
    if len(crossovers) == 0:
        last_frame = 0
        last_side = first_side
    else:
        last_frame = all_strikes[-1][0]
        last_side = all_strikes[-1][1]

    # Count strikes per side
    left_strikes = sum(1 for _, side in all_strikes if side == "left")
    right_strikes = sum(1 for _, side in all_strikes if side == "right")

    # Remove incomplete cycle (if last strike is the same side as first)
    if all_strikes and all_strikes[-1][1] == first_side:
        if last_side == "left":
            left_strikes -= 1
        else:
            right_strikes -= 1

    return {
        "first_strike": {"frame": all_strikes[0][0], "side": first_side},
        "last_strike": {"frame": last_frame, "side": last_side},
        "all_strikes": all_strikes,
        "left_strikes": left_strikes,
        "right_strikes": right_strikes,
        "total_strikes": left_strikes + right_strikes,
    }


def detect_foot_strikes(metrics: dict[str, np.ndarray], delay_frames: int = 10) -> dict[str, Any]:
    """
    Compatibility function that calls the heel Z-based foot strike detection.
    Maintained for backwards compatibility with existing code.

    Parameters:
    - metrics: dictionary with calculated metrics
    - delay_frames: frame delay to compensate for early detection (not used)

    Returns:
    - result from heel Z-based foot strike detection
    """
    # Call the new heel Z-based method
    return detect_foot_strikes_heel_z(metrics)


def count_steps(
    df: pd.DataFrame,
    peak_distance: int | None = None,
    height_threshold: float = 0.5,
    visualize: bool = False,
    target_steps: int | None = None,
    output_dir: str = ".",
    fps: int = 30,
    fix_double_count: bool = True,
    cutoff_freq: float = 3.0,
) -> dict[str, Any]:
    """
    Counts the number of steps from the DataFrame with greater precision.
    Tries multiple methods and returns the most reasonable result.

    Parameters:
    - df: DataFrame containing foot coordinates
    - peak_distance: minimum number of frames between consecutive peaks
    - height_threshold: threshold for peak height (relative to max value)
    - visualize: if True, generates a visualization chart
    - target_steps: if provided, tries to approximate detection to this number (optional)
    - output_dir: directory to save visualizations
    - fps: frames per second of original video (default: 30)
    - fix_double_count: if True, tries to correct double counts
    - cutoff_freq: cutoff frequency for Butterworth filter (Hz)

    Returns:
    - Dict with results and metadata
    """
    # Calculate feet metrics with Butterworth filtering
    metrics = calculate_feet_metrics(df, fs=fps, cutoff=cutoff_freq)

    # Estimate optimal parameters based on FPS
    if peak_distance is None:
        # For 60 FPS, we need larger distance between peaks
        if fps >= 50:  # High frame rate video
            peak_distance = max(10, min(int(fps / 3), 30))
        else:  # Normal frame rate
            peak_distance = max(5, min(int(fps / 3), 20))

    print(f"\n=== Step counting using different methods (FPS: {fps}) ===")
    print(f"Minimum distance between peaks: {peak_distance} frames")

    # Method 1: Original with correction for high frame rate
    steps_original = count_steps_original(df, peak_distance, fix_double_count)
    print(f"Original method: {steps_original} steps")

    # Method 2: Basic with adapted parameters
    # This method seems most robust for 6 steps
    steps_basic = count_steps_basic(df, peak_distance, sensitivity=0.15)
    print(f"Basic method: {steps_basic} steps")

    # Method 3: Velocity-based
    steps_velocity = count_steps_velocity(
        df, peak_distance, sensitivity=0.10, fix_double_count=fix_double_count
    )
    print(f"Velocity method: {steps_velocity} steps")

    # Method 4: Sliding window adjusted for FPS
    window_size = int(peak_distance * 2.5)  # Larger window for 60fps
    steps_window = count_steps_sliding_window(df, window_size, fix_double_count=fix_double_count)
    print(f"Window method: {steps_window} steps")

    # Method 5: New method based on marker Y average
    steps_mean_y = count_steps_mean_y(metrics, peak_distance, sensitivity=0.15)
    print(f"Mean Y method: {steps_mean_y} steps")

    # Method 6: Based on Z depth
    steps_z_depth = count_steps_z_depth(metrics, peak_distance, sensitivity=0.2)
    print(f"Z depth method: {steps_z_depth} steps")

    # Method 7: Foot strike detection based on heel Z-depth and velocity
    foot_strikes = detect_foot_strikes_heel_z(metrics, peak_distance)
    steps_heel_z = foot_strikes["total_strikes"]
    print(
        f"Heel Z method: {steps_heel_z} steps (L:{foot_strikes['left_strikes']}, R:{foot_strikes['right_strikes']})"
    )
    print(
        f"First foot strike: frame {foot_strikes['first_strike']['frame']} - side {foot_strikes['first_strike']['side']}"
    )
    print(
        f"Last foot strike: frame {foot_strikes['last_strike']['frame']} - side {foot_strikes['last_strike']['side']}"
    )

    # Store results by method
    methods = {
        "original": steps_original,
        "basic": steps_basic,
        "velocity": steps_velocity,
        "window": steps_window,
        "mean_y": steps_mean_y,
        "z_depth": steps_z_depth,
        "heel_z": steps_heel_z,
    }

    # If we have a target step count, select the closest method
    if target_steps is not None:
        closest_method = min(methods.keys(), key=lambda x: abs(methods[x] - target_steps))
        steps = methods[closest_method]
        print(
            f"\nMethod closest to expected ({target_steps} steps): {closest_method} with {steps} steps"
        )
    else:
        # Assign weights to methods - new methods get higher weights
        weights = {
            "original": 0.05,
            "basic": 0.15,
            "velocity": 0.1,
            "window": 0.05,
            "mean_y": 0.15,
            "z_depth": 0.15,
            "heel_z": 0.35,  # Heel Z method gets highest weight
        }
        weighted_sum = sum(methods[m] * weights[m] for m in methods)
        weighted_avg = weighted_sum / sum(weights.values())

        # Check for suspicious values
        suspected_double_count = weighted_avg > 9

        # If the average is much higher than 6 (expected), suspect double counting
        if fix_double_count and suspected_double_count:
            print("Possible double counting detected. Adjusting...")
            # Use the heel Z method, which is more reliable
            steps = steps_heel_z
        else:
            # Round to nearest integer
            steps = round(weighted_avg)

        print(f"\nWeighted average of methods: {weighted_avg:.2f} → {steps} steps")

    # Alert for possible incorrect count
    expected_range = range(5, 8)  # Expected range for 6 steps (error margin)
    if steps not in expected_range:
        print(f"\nWARNING: Detected step count ({steps}) is outside expected range (5-7).")
        print("Consider manually specifying expected step count.")

    # Optional visualization
    viz_path = None
    if visualize:
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_path = os.path.join(output_dir, f"step_detection_{timestamp}.png")

            plt.figure(figsize=(15, 18))

            # Plot 1: Horizontal separation
            plt.subplot(9, 1, 1)
            separation = metrics["horizontal_separation"]
            plt.plot(separation)

            # Detect peaks for visualization with more sensitive parameters
            min_val, max_val = separation.min(), separation.max()
            range_val = max_val - min_val
            prominence = range_val * 0.05  # Reduced to detect more peaks

            peaks_sep, _ = find_peaks(separation, distance=peak_distance, prominence=prominence)

            plt.plot(peaks_sep, separation[peaks_sep], "rx")
            plt.title(f"Horizontal Separation (Original Method: {steps_original} steps)")

            # Plot 2: Feet velocity
            plt.subplot(9, 1, 2)
            plt.plot(metrics["abs_left_velocity"], "g-", label="Left Foot")
            plt.plot(metrics["abs_right_velocity"], "b-", label="Right Foot")
            plt.title(f"Absolute Feet Velocity (Velocity Method: {steps_velocity} steps)")
            plt.legend()

            # Plot 3: Inverted velocity (velocity minima = foot strike)
            plt.subplot(9, 1, 3)
            inv_speed = metrics["inverted_speed"]

            # Detect peaks for visualization
            min_val, max_val = inv_speed.min(), inv_speed.max()
            range_val = max_val - min_val
            prominence = range_val * 0.1

            peaks_vel, _ = find_peaks(inv_speed, distance=peak_distance, prominence=prominence)

            plt.plot(inv_speed)
            plt.plot(peaks_vel, inv_speed[peaks_vel], "rx")
            plt.title("Inverted Velocity (Minima = Steps)")

            # Plot 4: Mean Y (new method)
            plt.subplot(9, 1, 4)
            mean_y_diff = metrics["mean_y_diff"]
            plt.plot(mean_y_diff)

            # Detect positive and negative peaks
            pos_peaks, _ = find_peaks(
                mean_y_diff,
                distance=peak_distance,
                prominence=np.abs(mean_y_diff).max() * 0.15,
            )

            neg_peaks, _ = find_peaks(
                -mean_y_diff,
                distance=peak_distance,
                prominence=np.abs(mean_y_diff).max() * 0.15,
            )

            plt.plot(pos_peaks, mean_y_diff[pos_peaks], "rx", label="Positive peaks")
            plt.plot(neg_peaks, mean_y_diff[neg_peaks], "bx", label="Negative peaks")
            plt.title(f"Mean Y Difference (Mean Y Method: {steps_mean_y} steps)")
            plt.legend()

            # Plot 5: Z Difference (new method)
            plt.subplot(9, 1, 5)
            z_diff = metrics["mean_z_left"] - metrics["mean_z_right"]
            plt.plot(z_diff)

            # Detect positive and negative peaks
            pos_peaks_z, _ = find_peaks(
                z_diff, distance=peak_distance, prominence=np.abs(z_diff).max() * 0.2
            )

            neg_peaks_z, _ = find_peaks(
                -z_diff, distance=peak_distance, prominence=np.abs(z_diff).max() * 0.2
            )

            plt.plot(pos_peaks_z, z_diff[pos_peaks_z], "rx", label="Positive peaks")
            plt.plot(neg_peaks_z, z_diff[neg_peaks_z], "bx", label="Negative peaks")
            plt.title(f"Z Depth Difference (Z Depth Method: {steps_z_depth} steps)")
            plt.legend()

            # Plot 6: Heel Z Depth
            plt.subplot(9, 1, 6)
            plt.plot(metrics["left_heel_z_depth"], "g-", label="Left Heel")
            plt.plot(metrics["right_heel_z_depth"], "b-", label="Right Heel")

            # Mark the foot strikes on heel Z depths
            for i, (frame, side) in enumerate(foot_strikes["all_strikes"]):
                if 0 <= frame < len(metrics["left_heel_z_depth"]):
                    if side == "left":
                        plt.plot(
                            frame,
                            metrics["left_heel_z_depth"][frame],
                            "go",
                            markersize=8,
                        )
                    else:
                        plt.plot(
                            frame,
                            metrics["right_heel_z_depth"][frame],
                            "bo",
                            markersize=8,
                        )

            plt.title(f"Heel Z Depth (Heel Z Method: {steps_heel_z} steps)")
            plt.legend()

            # Plot 7: Heel Z Velocity
            plt.subplot(9, 1, 7)
            plt.plot(metrics["left_heel_z_velocity"], "g-", label="Left Heel")
            plt.plot(metrics["right_heel_z_velocity"], "b-", label="Right Heel")
            plt.axhline(y=0, color="r", linestyle="--")

            # Mark foot strikes on velocity
            for i, (frame, side) in enumerate(foot_strikes["all_strikes"]):
                if side == "left" and 0 <= frame < len(metrics["left_heel_z_velocity"]):
                    plt.plot(
                        frame,
                        metrics["left_heel_z_velocity"][frame],
                        "go",
                        markersize=8,
                    )
                elif side == "right" and 0 <= frame < len(metrics["right_heel_z_velocity"]):
                    plt.plot(
                        frame,
                        metrics["right_heel_z_velocity"][frame],
                        "bo",
                        markersize=8,
                    )

            plt.title("Heel Z Velocity (vertical velocity at strike moments)")
            plt.legend()

            # Plot 8: Combined View - Heel Z and Y Position
            plt.subplot(9, 1, 8)
            heel_index_y_diff = metrics["heel_index_y_diff"]
            plt.plot(heel_index_y_diff, "k-", label="Y Position")

            # Normalize Z depths for combined visualization
            left_z_norm = (
                metrics["left_heel_z_depth"]
                / np.max(np.abs(metrics["left_heel_z_depth"]))
                * np.max(np.abs(heel_index_y_diff))
                * 0.8
            )
            right_z_norm = (
                metrics["right_heel_z_depth"]
                / np.max(np.abs(metrics["right_heel_z_depth"]))
                * np.max(np.abs(heel_index_y_diff))
                * 0.8
            )

            plt.plot(left_z_norm, "g--", alpha=0.7, label="Left Z (norm)")
            plt.plot(right_z_norm, "b--", alpha=0.7, label="Right Z (norm)")
            plt.axhline(y=0, color="r", linestyle="--")

            # Mark the foot strikes
            for i, (frame, side) in enumerate(foot_strikes["all_strikes"]):
                if 0 <= frame < len(heel_index_y_diff):
                    color = "g" if side == "left" else "b"
                    marker = (
                        "o"
                        if i < len(foot_strikes["all_strikes"]) - 1
                        or side != foot_strikes["first_strike"]["side"]
                        else "x"
                    )
                    plt.plot(
                        frame,
                        heel_index_y_diff[frame],
                        marker=marker,
                        color=color,
                        markersize=10,
                    )

                    # Draw vertical lines to aid visualization
                    plt.axvline(x=frame, color=color, linestyle="--", alpha=0.3)

            plt.plot([], [], "go", label="Left foot strike")
            plt.plot([], [], "bo", label="Right foot strike")
            plt.plot([], [], "rx", label="Last strike (ignored)")
            plt.title(f"Combined Position and Z Depth ({steps_heel_z} steps)")
            plt.legend()

            # Plot 9: Sliding window
            plt.subplot(9, 1, 9)
            plt.plot(metrics["combined_speed"])

            # Visualize windows
            window_size = peak_distance * 2
            for start in range(0, len(metrics["combined_speed"]), window_size // 2):
                plt.axvline(x=start, color="r", linestyle="--", alpha=0.3)

            plt.title(
                f"Combined Speed - Windows of {window_size} frames (Window Method: {steps_window} steps)"
            )

            plt.tight_layout()
            plt.savefig(viz_path)
            print(f"\nVisualization saved as: '{viz_path}'")
            plt.close()
        except Exception as e:
            print(f"Error generating visualization: {e}")
            # Try to print diagnostic info
            if "heel_index_y_diff" in locals():
                print(f"Length of heel_index_y_diff: {len(heel_index_y_diff)}")
            import traceback

            traceback.print_exc()

    # Return dictionary with all results
    return {
        "steps": steps,
        "methods": methods,
        "foot_strikes": foot_strikes,
        "visualization_path": viz_path,
        "metrics": {
            "peak_distance": peak_distance,
            "window_size": window_size,
            "fps": fps,
            "cutoff_freq": cutoff_freq,
        },
    }


def export_results(
    results: dict[str, Any], csv_path: str, output_path: str | None = None
) -> str:
    """
    Exports the analysis results to a CSV or TXT file.

    Parameters:
    - results: dictionary with analysis results
    - csv_path: path to the original CSV file
    - output_path: path to save the result (optional)

    Returns:
    - Path to the results file
    """
    if output_path is None:
        # Create file name based on the original
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{base_name}_results_{timestamp}.txt"

    try:
        with open(output_path, "w") as f:
            f.write(f"==== Step Analysis - {datetime.datetime.now()} ====\n\n")
            f.write(f"Analyzed file: {csv_path}\n")
            f.write(f"Total number of steps: {results['steps']}\n\n")

            # Add information about foot strikes
            if "foot_strikes" in results:
                foot_strikes = results["foot_strikes"]
                f.write("=== Foot Strike Information ===\n")
                f.write(
                    f"First foot strike: frame {foot_strikes['first_strike']['frame']} - side {foot_strikes['first_strike']['side']}\n"
                )
                f.write(
                    f"Last foot strike: frame {foot_strikes['last_strike']['frame']} - side {foot_strikes['last_strike']['side']}\n"
                )
                f.write(f"Left side strikes: {foot_strikes['left_strikes']}\n")
                f.write(f"Right side strikes: {foot_strikes['right_strikes']}\n")
                f.write(f"Total strikes (complete cycle): {foot_strikes['total_strikes']}\n\n")

                f.write("List of all foot strikes:\n")
                for i, (frame, side) in enumerate(foot_strikes["all_strikes"]):
                    # Mark the last strike if it's disregarded (same side as the first)
                    note = (
                        " (ignored - incomplete cycle)"
                        if (
                            i == len(foot_strikes["all_strikes"]) - 1
                            and side == foot_strikes["first_strike"]["side"]
                        )
                        else ""
                    )
                    f.write(f"  {i + 1}. Frame {frame}: {side}{note}\n")
                f.write("\n")

            f.write("Results by method:\n")
            for method, steps in results["methods"].items():
                f.write(f"  - {method} method: {steps} steps\n")

            if results["visualization_path"]:
                f.write(f"\nVisualization saved at: {results['visualization_path']}\n")

            f.write("\nAnalysis parameters:\n")
            for key, value in results["metrics"].items():
                f.write(f"  - {key}: {value}\n")

        print(f"Results exported to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error exporting results: {e}")
        return ""


def run_numsteps(
    file_path=None,
    visualize=True,
    target_steps=None,
    output_dir=".",
    fps=30,
    cutoff_freq=3.0,
):
    """
    Main function for programmatic execution of the step detection algorithm.

    Parameters:
    - file_path: path to CSV file (if None, opens selection dialog)
    - visualize: if True, generates visualizations
    - target_steps: expected number of steps (optional)
    - output_dir: directory for output files
    - fps: frames per second of the original video
    - cutoff_freq: cutoff frequency for the Butterworth filter (Hz)

    Returns:
    - Number of detected steps and dictionary with detailed results
    """
    # If no file is specified, open dialog
    if file_path is None:
        # Initialize Tk and hide the main window
        root = tk.Tk()
        root.withdraw()

        # Open dialog for CSV selection
        file_path = filedialog.askopenfilename(
            title="Select the CSV coordinate file",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not file_path:
            print("No file selected. Aborting.")
            return None, {}

    try:
        # Try to read CSV with auto-detection of delimiter (comma or tab)
        try:
            df = pd.read_csv(file_path, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(file_path)

        # Check required columns for the new metrics
        required_cols = [
            "left_foot_index_x",
            "left_foot_index_y",
            "left_foot_index_z",
            "right_foot_index_x",
            "right_foot_index_y",
            "right_foot_index_z",
            "left_ankle_x",
            "left_ankle_y",
            "left_ankle_z",
            "right_ankle_x",
            "right_ankle_y",
            "right_ankle_z",
            "left_heel_x",
            "left_heel_y",
            "left_heel_z",
            "right_heel_x",
            "right_heel_y",
            "right_heel_z",
            "left_hip_z",
            "right_hip_z",
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            error_msg = f"Required columns not found: {', '.join(missing_cols)}"
            print(error_msg)
            messagebox.showerror("Data error", error_msg)
            return None, {}

        # Count steps with the improved algorithm
        results = count_steps(
            df,
            visualize=visualize,
            target_steps=target_steps,
            output_dir=output_dir,
            fps=fps,
            cutoff_freq=cutoff_freq,
        )
        steps = results["steps"]
        print(f"\nNumber of detected steps: {steps}")

        # Information about foot strikes
        foot_strikes = results.get("foot_strikes", {})
        if foot_strikes:
            print("\nFoot strike details:")
            print(
                f"First foot strike: frame {foot_strikes['first_strike']['frame']} - side {foot_strikes['first_strike']['side']}"
            )
            print(
                f"Last foot strike: frame {foot_strikes['last_strike']['frame']} - side {foot_strikes['last_strike']['side']}"
            )
            print(
                f"Strikes by side: Left={foot_strikes['left_strikes']}, Right={foot_strikes['right_strikes']}"
            )

            # Display the complete sequence of foot strikes
            print("\nComplete foot strike sequence:")
            for i, (frame, side) in enumerate(foot_strikes["all_strikes"]):
                # Mark the last strike if it's disregarded (same side as the first)
                note = (
                    " (ignored - incomplete cycle)"
                    if (
                        i == len(foot_strikes["all_strikes"]) - 1
                        and side == foot_strikes["first_strike"]["side"]
                    )
                    else ""
                )
                print(f"  {i + 1}. Frame {frame}: {side}{note}")

        return steps, results

    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        print(error_msg)
        messagebox.showerror("Error", error_msg)
        return None, {}


def extract_gait_features(df, foot_strikes, participant=None, trial=None):
    """
    Extracts spatial, temporal, and kinematic features for each step/block.
    Uses the frames of foot strikes to segment the steps.
    Returns a list of dicts, one per step.
    """
    features_list = []
    strikes = foot_strikes.get("all_strikes", [])
    for i in range(len(strikes) - 1):
        start, side_start = strikes[i]
        end, side_end = strikes[i + 1]
        block = df.iloc[start : end + 1]
        features = {
            "participant": participant,
            "trial": trial,
            "step_block": i + 1,
            "side": side_start,
            "start_frame": start,
            "end_frame": end,
            "duration_frames": end - start,
            # Spatial means
            "left_heel_x_mean": block["left_heel_x"].mean(),
            "left_heel_y_mean": block["left_heel_y"].mean(),
            "left_foot_index_x_mean": block["left_foot_index_x"].mean(),
            "left_foot_index_y_mean": block["left_foot_index_y"].mean(),
            "right_heel_x_mean": block["right_heel_x"].mean(),
            "right_heel_y_mean": block["right_heel_y"].mean(),
            "right_foot_index_x_mean": block["right_foot_index_x"].mean(),
            "right_foot_index_y_mean": block["right_foot_index_y"].mean(),
            # Variance
            "left_heel_x_var": block["left_heel_x"].var(),
            "left_heel_y_var": block["left_heel_y"].var(),
            "left_foot_index_x_var": block["left_foot_index_x"].var(),
            "left_foot_index_y_var": block["left_foot_index_y"].var(),
            "right_heel_x_var": block["right_heel_x"].var(),
            "right_heel_y_var": block["right_heel_y"].var(),
            "right_foot_index_x_var": block["right_foot_index_x"].var(),
            "right_foot_index_y_var": block["right_foot_index_y"].var(),
            # Range
            "left_heel_x_range": block["left_heel_x"].max() - block["left_heel_x"].min(),
            "left_heel_y_range": block["left_heel_y"].max() - block["left_heel_y"].min(),
            "right_heel_x_range": block["right_heel_x"].max() - block["right_heel_x"].min(),
            "right_heel_y_range": block["right_heel_y"].max() - block["right_heel_y"].min(),
            "left_foot_index_x_range": block["left_foot_index_x"].max()
            - block["left_foot_index_x"].min(),
            "left_foot_index_y_range": block["left_foot_index_y"].max()
            - block["left_foot_index_y"].min(),
            "right_foot_index_x_range": block["right_foot_index_x"].max()
            - block["right_foot_index_x"].min(),
            "right_foot_index_y_range": block["right_foot_index_y"].max()
            - block["right_foot_index_y"].min(),
            # Speed (mean diff per frame)
            "left_heel_x_speed": np.mean(np.diff(block["left_heel_x"])),
            "left_heel_y_speed": np.mean(np.diff(block["left_heel_y"])),
            "left_foot_index_x_speed": np.mean(np.diff(block["left_foot_index_x"])),
            "left_foot_index_y_speed": np.mean(np.diff(block["left_foot_index_y"])),
            "right_heel_x_speed": np.mean(np.diff(block["right_heel_x"])),
            "right_heel_y_speed": np.mean(np.diff(block["right_heel_y"])),
            "right_foot_index_x_speed": np.mean(np.diff(block["right_foot_index_x"])),
            "right_foot_index_y_speed": np.mean(np.diff(block["right_foot_index_y"])),
            # Step length (Euclidean distance between heels)
            "left_step_length": np.mean(
                np.sqrt(
                    (block["left_heel_x"] - block["right_heel_x"]) ** 2
                    + (block["left_heel_y"] - block["right_heel_y"]) ** 2
                )
            ),
            "right_step_length": np.mean(
                np.sqrt(
                    (block["right_heel_x"] - block["left_heel_x"]) ** 2
                    + (block["right_heel_y"] - block["left_heel_y"]) ** 2
                )
            ),
        }
        features_list.append(features)
    return features_list


def run_numsteps_gui():
    # Initialize Tk and hide the main window
    root = tk.Tk()
    root.withdraw()

    try:
        # Open dialog for directory selection
        input_dir = filedialog.askdirectory(title="Select Input Directory with CSV Files")
        if not input_dir:
            print("No directory selected. Aborting.")
            return

        # Option to specify expected number of steps
        target_steps = None
        use_target = messagebox.askyesno(
            "Configuration", "Do you know the expected number of steps?"
        )
        if use_target:
            target_steps = simpledialog.askinteger(
                "Number of steps",
                "Enter the expected number of steps:",
                minvalue=1,
                maxvalue=100,
            )

        # Option for frame rate
        fps = 30  # Default value
        custom_fps = messagebox.askyesno(
            "Configuration", "Do you want to specify the frame rate (FPS)?"
        )
        if custom_fps:
            fps = simpledialog.askinteger(
                "Frame rate",
                "Enter the video frame rate (FPS):",
                minvalue=10,
                maxvalue=240,
            )

        # Option for Butterworth filter cutoff frequency
        cutoff_freq = 3.0  # Default value
        custom_cutoff = messagebox.askyesno(
            "Configuration", "Do you want to specify the filter cutoff frequency (Hz)?"
        )
        if custom_cutoff:
            cutoff_freq = simpledialog.askfloat(
                "Cutoff frequency",
                "Enter the filter cutoff frequency (Hz):",
                minvalue=0.5,
                maxvalue=10.0,
            )

        # Option for visualization
        visualize = messagebox.askyesno("Visualization", "Generate visualization graphs?")

        # Create main output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        main_output_dir = os.path.join(input_dir, f"vailagait_{timestamp}")
        os.makedirs(main_output_dir, exist_ok=True)

        # List to collect all features for batch CSV
        all_features = []

        # Process each CSV file in the selected directory
        csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
        total_files = len(csv_files)
        print(f"Found {total_files} CSV files to process")

        for i, file in enumerate(csv_files):
            # Get the base name for creating subdirectory
            base_name = os.path.splitext(os.path.basename(file))[0]
            print(f"\nProcessing file {i + 1}/{total_files}: {base_name}")

            # Create subdirectory for this file
            file_output_dir = os.path.join(main_output_dir, base_name)
            os.makedirs(file_output_dir, exist_ok=True)

            # Process the file with output going to the subdirectory
            steps, results = run_numsteps(
                file, visualize, target_steps, file_output_dir, fps, cutoff_freq
            )

            if steps is not None:
                # Save results text file
                results_file = os.path.join(file_output_dir, f"{base_name}_results.txt")
                output_path = export_results(results, file, results_file)
                print(f"Results saved at: {output_path}")

                # Extract features for this file
                if "foot_strikes" in results:
                    df = pd.read_csv(file)
                    features = extract_gait_features(
                        df,
                        results["foot_strikes"],
                        participant=base_name,
                        trial=base_name,
                    )

                    # Save features for this file
                    features_file = os.path.join(file_output_dir, f"{base_name}_features.csv")
                    pd.DataFrame(features).to_csv(features_file, index=False)
                    print(f"Features saved at: {features_file}")

                    # Add to collection for batch output
                    all_features.extend(features)

                # Move visualization if it exists to the subdirectory
                if results["visualization_path"] and os.path.exists(results["visualization_path"]):
                    viz_filename = os.path.basename(results["visualization_path"])
                    new_viz_path = os.path.join(file_output_dir, viz_filename)
                    if results["visualization_path"] != new_viz_path:  # Avoid unnecessary copy
                        try:
                            import shutil

                            shutil.copy2(results["visualization_path"], new_viz_path)
                            print(f"Visualization saved at: {new_viz_path}")
                        except Exception as e:
                            print(f"Error moving visualization: {e}")

        # Save all features to a combined CSV in the main directory
        if all_features:
            all_features_file = os.path.join(main_output_dir, "all_features.csv")
            pd.DataFrame(all_features).to_csv(all_features_file, index=False)
            print(f"\nAll features combined and saved at: {all_features_file}")

        messagebox.showinfo(
            "Analysis completed",
            f"Processed {total_files} files.\nResults saved in:\n{main_output_dir}",
        )

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        import traceback

        traceback.print_exc()
        messagebox.showerror("Error", error_msg)


if __name__ == "__main__":
    run_numsteps_gui()
