import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import csv
from os.path import basename
from datetime import datetime
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from rich import print
from ydata_profiling import ProfileReport

print(
    r"""
 _          ____  _        ____       __  __   | Biomechanics and Motor Control Laboratory
| |        |  _ \(_)      /  __|     |  \/  |  | Developed by: Paulo R. P. Santiago
| |    __ _| |_) |_  ___ |  /    ___ | \  / |  | paulosantiago@usp.br
| |   / _' |  _ <| |/ _ \| |    / _ \| |\/| |  | University of Sao Paulo
| |__' (_| | |_) | | (_) |  \__' (_) | |  | |  | https://orcid.org/0000-0002-9460-8847
|____|\__'_|____/|_|\___/ \____|\___/|_|  |_|  | Date: 05 Jun 2024
"""
)


def read_csv_skip_header(filename, usecols):
    try:
        col_indices = list(map(int, usecols.split(",")))
        loaddata = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=col_indices)

        # Read the CSV file using pandas, parsing the timestamp column
        df = pd.read_csv(filename, usecols=[0], nrows=1)
        timestamp = df.iat[0, 0]

        return loaddata, timestamp

    except Exception as e:
        print(f"Error reading the CSV file: {str(e)}")
        return None  # It might be better to let this exception propagate rather than returning None.


def create_main_output_directory(filename):
    today_date = datetime.now().strftime("%Y%m%d")  # Formato AAAAMMDD
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    main_output_dir = os.path.join(
        os.getcwd(),
        f"Results_force_cube_analysis_{today_date}",
        f"Results_{base_filename}",
    )
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)
    return main_output_dir


def butterworthfilt(data, cutoff=59, Fs=1000):
    # Padding
    pad_length = 100
    padded_data = np.pad(data, (pad_length, pad_length), "edge")

    b, a = butter(4, cutoff / (Fs / 2), "low")

    # Aplicação do filtro
    filtered_padded = filtfilt(b, a, padded_data)

    # Remoção do padding
    filtered_data = filtered_padded[pad_length:-pad_length]
    datafiltered = filtered_data
    return datafiltered


def makefig1(data, output_dir, filename):
    """
    Creates an interactive plot for selecting data points.
    Space + Left Click to select, Right Click to remove last point, press 'Enter' to finish.

    Parameters:
        data (array-like): The dataset to plot for interaction.
        output_dir (str): Directory to save the figure.
        filename (str): Base name for the output file.

    Returns:
        list of int: The indices of selected points.
    """
    fig1, ax1 = plt.subplots()
    ax1.plot(data)
    ax1.set_title(
        "Space + Left Click to select, Right Click to remove, press 'Enter' to finish"
    )
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Force Value")
    ax1.grid(True)

    points = []
    space_held = False  # Variable to track if Space is held down

    def on_key_press(event):
        nonlocal space_held
        if event.key == " ":
            space_held = True
        elif event.key == "enter":
            plt.close(fig1)  # Closes the figure on pressing 'Enter'

    def on_key_release(event):
        nonlocal space_held
        if event.key == " ":
            space_held = False

    def onclick(event):
        if (
            space_held and event.button == 1
        ):  # Check if Space is held and left mouse is clicked
            x_value = event.xdata
            if x_value is not None:
                points.append((x_value, event.ydata))
                ax1.axvline(x=x_value, color="red", linestyle="--")
                fig1.canvas.draw()
        elif event.button == 3:  # Right Click to remove the last point
            if points:
                points.pop()
                ax1.cla()
                ax1.plot(data)
                ax1.grid(True)
                for point in points:
                    ax1.axvline(x=point[0], color="red", linestyle="--")
                fig1.canvas.draw()

    fig1.canvas.mpl_connect("button_press_event", onclick)
    fig1.canvas.mpl_connect("key_press_event", on_key_press)
    fig1.canvas.mpl_connect("key_release_event", on_key_release)

    # Save the figure as a PNG file with _raw suffix inside the correct directory
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    figure1_filename = os.path.join(output_dir, base_filename + "_raw.png")
    plt.savefig(figure1_filename)
    plt.show(block=True)  # Block is True to keep the figure open

    if len(points) < 2:
        print("At least two points required.")
        sys.exit(1)

    indices = [int(point[0]) for point in points]
    return indices


def calculate_median(data, start, end, window=5):
    # Garantindo que os índices não ultrapassem as fronteiras dos dados
    start_region = max(0, start - window)
    end_region = min(
        len(data), end + window + 1
    )  # +1 para incluir o índice 'end+window'

    # Selecionando os dados ao redor de 'start' e 'end'
    start_data = data[
        start_region : start + window + 1
    ]  # +1 para incluir o índice 'start+window'
    end_data = data[end - window : end_region]

    # Calculando a mediana de cada trecho
    start_median = np.median(start_data)
    end_median = np.median(end_data)

    return start_median, end_median


def find_active_indices(section_data, start_median, end_median, threshold):
    # Encontrar o índice inicial de atividade
    for i, value in enumerate(section_data):
        if abs(value - start_median) >= threshold:
            start_active = i
            break
    else:
        start_active = 0

    # Encontrar o índice final de atividade
    for i, value in enumerate(reversed(section_data)):
        if abs(value - end_median) >= threshold:
            end_active = len(section_data) - 1 - i
            break
    else:
        end_active = len(section_data) - 1

    return start_active, end_active


## FIGURE 2
def makefig2(data, indices, threshold):
    active_ranges = []
    # Create a Figure 2 for all subplots
    num_segments = len(indices) - 1
    fig2, ax2 = plt.subplots(3, 3)  # 3x3 subplot matrix
    ax2 = ax2.flatten()  # Flatten the array to simplify indexing

    for i in range(min(num_segments, 9)):  # Limit to 9 segments for the 3x3 matrix
        start = int(indices[i])  # Ensure index is integer
        end = int(indices[i + 1])  # Ensure index is integer

        # Validate indices within data boundaries
        if end > len(data):
            end = len(data)
        if start >= len(data):
            break  # Exit the loop if start is not within data length

        section_data = data[start:end]
        start_median, end_median = calculate_median(data, start, end)
        start_active, end_active = find_active_indices(
            section_data, start_median, end_median, threshold
        )

        adjusted_start = start + start_active
        adjusted_end = start + end_active

        # Plot data and markers
        ax2[i].plot(np.arange(start, end), section_data)
        ax2[i].axvline(x=adjusted_start, color="green", label="Start")
        ax2[i].axvline(x=adjusted_end, color="red", label="End")
        ax2[i].set_title(f"{i+1}: index {start} to {end}")
        ax2[i].set_ylabel("Force (N)")
        ax2[i].set_xlabel("Time (s)")
        ax2[i].grid(True)
        ax2[i].legend(
            fontsize="xx-small",
            loc="upper right",
            handlelength=1,
            handletextpad=0.2,
            borderaxespad=0.2,
            framealpha=0.6,
        )
        plt.tight_layout()

        active_ranges.append([adjusted_start, adjusted_end])

    plt.show()

    return active_ranges


def build_headers():
    # Define CSV headers
    headers = [
        "FileName",
        "TimeStamp",
        "Trial",
        "BW_kg",
        "SideFoot_RL",
        "Dominance_RL",
        "Quality",
        "Num_Samples",
        "Index_40ms",
        "Index_100ms",
        "Index_ITransient",
        "Index_VIP",
        "Index_Max",
        "Test_Duration_s",
        "CumSum_Times_s",
        "Contact_Time_s",
        "Time_40ms_s",
        "Time_100ms_s",
        "Time_ITransient_s",
        "Time_VIP_s",
        "Time_Peak_VMax_s",
        "VPeak_40ms_BW",
        "VPeak_100ms_BW",
        "Peak_VITransient_BW",
        "Peak_VIP_BW",
        "Peak_VMax_BW",
        "Total_Imp_BW.s",
        "Imp_40ms_BW.s",
        "Imp_100ms_BW.s",
        "Imp_ITransient_BW.s",
        "Imp_Brake_VMax_BW.s",
        "Imp_Propulsion_BW.s",
        "RFD_40ms_BW.s-1",
        "RFD_100ms_BW.s-1",
        "RFD_ITransient_BW.s-1",
        "RFD_Brake_VMax_BW.s-1",
        "RFD_Propulsion_BW.s-1",
        "Simple_stiffness_constant",
        "High_stiffness",
        "Low_stiffness",
        "Transition_time",
        "Average_loading_rate",
    ]

    return headers


def calculate_loading_rates(vgrf, time_data):
    """Calculates the vertical loading rate based on the Vertical Impact Peak (VIP)."""

    # Function to find VIP index within the first 40% of the signal
    def find_vip_index(vgrf, time_data):
        forty_percent_index = int(
            0.4 * len(vgrf)
        )  # Limiting the search to the first 40%
        # peaks, _ = find_peaks(vgrf[:forty_percent_index], prominence=1, width=20, distance=150)
        peaks, _ = find_peaks(vgrf[:forty_percent_index])
        if peaks.size > 0:
            vip_index = peaks[
                np.argmax(vgrf[peaks])
            ]  # VIP is the highest peak within the first 40%
        else:
            vip_index = np.argmax(
                vgrf[:forty_percent_index]
            )  # If no peaks, take the highest point
        return vip_index

    vip_index = find_vip_index(vgrf, time_data)
    vip_time = time_data[vip_index]
    vip_value = vgrf[vip_index]

    # Calculate the 20% and 80% time points relative to the start of the data to the VIP time
    start_time = 0.2 * vip_time
    end_time = 0.8 * vip_time

    start_index_poi20_80 = np.argmin(np.abs(time_data - start_time))
    end_index_poi20_80 = np.argmin(np.abs(time_data - end_time))

    print("\n")
    print(
        "VIP Index:",
        vip_index,
        "Start index:",
        start_index_poi20_80,
        "End index:",
        end_index_poi20_80,
    )
    print("Start time:", start_time, "End time:", end_time)
    print("VIP Value:", vip_value)

    vgrf_poi20_80 = vgrf[start_index_poi20_80:end_index_poi20_80]
    time_poi20_80 = time_data[start_index_poi20_80:end_index_poi20_80]

    if start_index_poi20_80 < end_index_poi20_80:
        valr = np.mean(np.diff(vgrf_poi20_80) / np.diff(time_poi20_80))
    else:
        valr = np.nan  # Handle cases where the calculated range is not valid

    return (
        valr,
        vip_value,
        vip_index,
        vgrf_poi20_80,
        time_poi20_80,
        start_index_poi20_80,
        end_index_poi20_80,
    )


def logistic_ogive(t, kl, kh, tT, m):
    """Models a logistic curve for stiffness adjustment."""
    return kl + (kh - kl) / (1 + np.exp(-m * (t - tT)))


def fit_stiffness_models(vgrf_poi20_80, time_poi20_80):
    """Calculates stiffness constant and fits a logistic ogive model to the ground reaction force data."""
    # Simple model: constant kc, assuming it's 20% of the maximum VGRF in the selected window
    kc = 0.2 * np.max(vgrf_poi20_80)

    # Fit complex model using logistic ogive
    try:
        # popt contains the optimized parameters [kl, kh, tT, m]
        popt, _ = curve_fit(
            logistic_ogive,
            time_poi20_80,
            vgrf_poi20_80,
            bounds=(0, [np.inf, np.inf, np.max(time_poi20_80), np.inf]),
        )
        kl, kh, tT, m = popt
    except RuntimeError as e:
        print("Curve fitting failed:", e)
        kl, kh, tT, m = (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )  # Default to NaN if fitting fails

    return kc, kh, kl, tT, m


def calculate_cube_values(signal, Fs):
    """
    Calculate key biomechanical metrics from a signal based on fixed time intervals.

    Parameters:
    signal (array): The input biomechanical signal.
    Fs (float): Sampling frequency of the signal.

    Returns:
    tuple: A tuple containing calculated metrics including times, peaks, areas, and rate of force development (RFD).

    * Reference to t40ms = 0.04  # 40 ms and t100ms = 0.1  # 100 ms
    Koga H, Nakamae A, Shima Y, Iwasa J, Myklebust G, Engebretsen L, Bahr R, Krosshaug T.
    Mechanisms for noncontact anterior cruciate ligament injuries: knee joint kinematics in 10 injury situations from female team handball and basketball.
    Am J Sports Med. 2010 Nov;38(11):2218-25.
    https://doi.org/10.1177/0363546510373570

    * Reference to Impact Transient
    Lieberman, D., Venkadesan, M., Werbel, W. et al.
    Foot strike patterns and collision forces in habitually barefoot versus shod runners. Nature 463, 531–535 (2010).
    https://doi.org/10.1038/nature08723
    """
    num_samples = len(signal)
    if num_samples < int(0.1 * Fs):
        raise ValueError("Signal is too short for analysis.")

    time_interval = np.arange(num_samples) / Fs
    total_time = time_interval[-1]

    t40ms = 0.04  # 40 ms
    t100ms = 0.1  # 100 ms
    index_40ms = int(t40ms * Fs)
    index_100ms = int(t100ms * Fs)
    index_20ms = int(0.02 * Fs)  # 20 ms index

    vpeak_40ms = signal[index_40ms]
    area_until_40ms = np.trapz(signal[:index_40ms], time_interval[:index_40ms])
    rfd_40ms = vpeak_40ms / t40ms

    vpeak_100ms = signal[index_100ms]
    area_until_100ms = np.trapz(signal[:index_100ms], time_interval[:index_100ms])
    rfd_100ms = vpeak_100ms / t100ms

    # Search for peakmax after 100 ms
    (
        valr,
        vip_value,
        vip_index,
        vgrf_poi20_80,
        time_poi20_80,
        start_index_poi20_80,
        end_index_poi20_80,
    ) = calculate_loading_rates(signal, time_interval)
    index_impact_transient = end_index_poi20_80
    vpeak_impact_transient = vip_value
    time_impact_transient = time_poi20_80[-1]

    index_poi = vip_index
    vpeak_poi = vip_value
    time_poi = time_interval[vip_index]

    area_impact_transient = np.trapz(
        signal[:index_impact_transient], time_interval[:index_impact_transient]
    )
    rfd_impact_transient = (
        vpeak_impact_transient / time_impact_transient
        if time_impact_transient != 0
        else np.nan
    )

    peaks_peakmax, _ = find_peaks(signal[index_poi:])
    if peaks_peakmax.size > 0:
        index_peakmax = (
            index_poi + peaks_peakmax[np.argmax(signal[index_poi + peaks_peakmax])]
        )
    else:
        index_peakmax = index_poi + np.argmax(signal[index_poi:])

    vpeakmax = signal[index_peakmax]
    time_peakmax = time_interval[index_peakmax]
    area_peakmax = np.trapz(signal[:index_peakmax], time_interval[:index_peakmax])
    rfd_peakmax = (
        vpeakmax / time_peakmax if time_peakmax and time_peakmax != 0 else float("inf")
    )

    total_area = np.trapz(signal, time_interval)
    area_propulsion = np.trapz(
        signal[index_peakmax:],
        time_interval[index_peakmax:] - time_interval[index_peakmax],
    )
    rfd_propulsion = (
        (signal[-1] - vpeakmax) / (total_time - time_peakmax)
        if time_peakmax and (total_time - time_peakmax) != 0
        else float("inf")
    )

    return (
        num_samples,
        index_40ms,
        index_100ms,
        index_impact_transient,
        index_peakmax,
        total_time,
        time_interval,
        t40ms,
        t100ms,
        time_impact_transient,
        time_peakmax,
        vpeak_40ms,
        vpeak_100ms,
        vpeak_impact_transient,
        vpeakmax,
        total_area,
        area_until_40ms,
        area_until_100ms,
        area_impact_transient,
        area_peakmax,
        area_propulsion,
        rfd_40ms,
        rfd_100ms,
        rfd_impact_transient,
        rfd_peakmax,
        rfd_propulsion,
        index_poi,
        vpeak_poi,
        time_poi,
    )


## FIGURE 3
def makefig3(
    databw,
    active_ranges,
    body_weight_kg,
    output_dir,
    filename,
    sidefoot,
    dominance,
    quality,
    simple_filename,
    timestamp,
    indices,
    Fs,
):
    fig3, ax3 = plt.subplots(3, 3, figsize=(30, 30))
    ax3 = ax3.flatten()  # Flatten the array to simplify indexing

    # Get the filename without extension
    filename_without_extension = os.path.splitext(os.path.basename(filename))[0]

    # Define the CSV result file name using the new directory path
    result_csv_filename = os.path.join(
        output_dir, filename_without_extension + "_result.csv"
    )

    # Define the plot PNG file name using the new directory path
    result_plot_filename = os.path.join(
        output_dir, filename_without_extension + "_result.png"
    )

    # Now also define the index file name using the new directory path
    indices_filename = os.path.join(
        output_dir, filename_without_extension + "_index.txt"
    )

    active_ranges_np = np.array(active_ranges)
    # Extrair o primeiro elemento
    first_element = active_ranges_np[0, 0]

    # Extrair os últimos elementos de cada sublista
    last_elements = active_ranges_np[:, 1]

    # Subtrair o primeiro elemento de cada último elemento
    differences = last_elements - first_element

    # Calcular a soma acumulada das diferenças
    test_duration = differences * (1 / Fs)

    # Save indices to a file
    with open(indices_filename, "w") as f:
        for index in indices:
            f.write(f"{index}\n")

    # Initialize all_time to sum up all total_time values
    cumulative_time = 0

    # Open the CSV file for writing
    result_array = []

    with open(result_csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        headers = build_headers()
        writer.writerow(headers)  # Write headers to the CSV file

        for i, (start, end) in enumerate(active_ranges[:9]):
            active_segment_data = databw[start : end + 1]
            # Unpack results from the calculate_cube_values function
            (
                num_samples,
                index_40ms,
                index_100ms,
                index_impact_transient,
                index_peakmax,
                total_time,
                time_interval,
                t40ms,
                t100ms,
                time_impact_transient,
                time_peakmax,
                vpeak_40ms,
                vpeak_100ms,
                vpeak_impact_transient,
                vpeakmax,
                total_area,
                area_until_40ms,
                area_until_100ms,
                area_impact_transient,
                area_peakmax,
                area_propulsion,
                rfd_40ms,
                rfd_100ms,
                rfd_impact_transient,
                rfd_peakmax,
                rfd_propulsion,
                index_poi,
                vpeak_poi,
                time_poi,
            ) = calculate_cube_values(active_segment_data, Fs)

            # Accumulate total_time
            cumulative_time += total_time

            time_data = time_interval
            vgrf = active_segment_data

            kc, kh, kl, tT, m = fit_stiffness_models(vgrf, time_data)
            (
                valr,
                vip_value,
                vip_index,
                vgrf_poi20_80,
                time_poi20_80,
                start_index_poi20_80,
                end_index_poi20_80,
            ) = calculate_loading_rates(
                active_segment_data, np.arange(len(active_segment_data)) / Fs
            )
            index_impact_transient = end_index_poi20_80
            time_impact_transient = time_poi20_80[-1]
            vpeak_impact_transient = active_segment_data[end_index_poi20_80]

            # Plotting the segment data
            ax3[i].plot(time_interval, active_segment_data, linewidth=3)
            # Convertendo start e end de índices para segundos
            start_seconds = start / Fs
            end_seconds = end / Fs
            # Formatação do título com tempo em segundos
            ax3[i].set_title(f"{i+1}: time {start_seconds:.3f} to {end_seconds:.3f} s")
            ax3[i].set_xlabel("Time (s)")
            ax3[i].set_ylabel("Force (BW)")
            # Ajustando o limite x do gráfico em x
            max_time_interval = np.max(time_interval)
            if max_time_interval > 0.65:
                ax3[i].set_xlim(0, max_time_interval)
            else:
                ax3[i].set_xlim(0, 0.65)

            # Ajustando o limite y do gráfico em y
            max_active_segment_data = np.max(active_segment_data)
            if max_active_segment_data > 3.7:
                ax3[i].set_ylim(0, max_active_segment_data + 0.1)
            else:
                ax3[i].set_ylim(0, 3.7)

            # Get current ticks and add custom tick for 40ms
            current_ticks = np.append(
                ax3[i].get_xticks(), 0.04
            )  # Include 0.04s in the existing ticks
            current_ticks = np.unique(
                np.sort(current_ticks)
            )  # Sort and remove any duplicates
            ax3[i].set_xticks(current_ticks)  # Set the ticks to include 0.04s

            # Optional: you can format ticks to show more clearly or with specific precision
            ax3[i].get_xaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:.2f}")
            )

            # Preenchendo a área sob a curva
            ax3[i].fill_between(
                time_interval, active_segment_data, color="gray", alpha=0.2
            )

            # Create an array of x values from 0 to impact transient for filling - IMPACT TRANSIENT
            x_fill_impact_transient = np.linspace(0, time_impact_transient, 100)
            # Create corresponding y values for the line
            y_fill_impact_transient = (
                vpeak_impact_transient / time_impact_transient
            ) * x_fill_impact_transient
            # Fill the area below the line in red with transparency
            ax3[i].fill_between(
                x_fill_impact_transient,
                0,
                y_fill_impact_transient,
                color="red",
                alpha=0.3,
            )

            # Create an array of x values from 0 to peakmax for filling - BRAKE
            x_fill_peakmax = np.linspace(0, time_peakmax, 100)
            # Create corresponding y values for the line
            y_fill_peakmax = (vpeakmax / time_peakmax) * x_fill_peakmax
            # Fill the area below the line in magenta with transparency
            ax3[i].fill_between(
                x_fill_peakmax, 0, y_fill_peakmax, color="orange", alpha=0.3
            )

            # Preenchimento para o peakmax to end - PROPULSION
            x_fill_end = np.linspace(time_peakmax, total_time, 100)
            y_start = vpeakmax  # Valor de Y no ponto peakmax to end - PROPULSION
            y_fill_end = np.linspace(
                y_start, 0, 100
            )  # Decresce linearmente de vpeakmax até 0

            # Fill the area below the line in another color (e.g., green) with transparency
            ax3[i].fill_between(x_fill_end, 0, y_fill_end, color="green", alpha=0.3)

            # Highlight the peaks on the plot
            ax3[i].plot(
                time_impact_transient,
                vpeak_impact_transient,
                "r*",
                markersize=12,
                label=f"Transient: {time_impact_transient:.3f}s:{vpeak_impact_transient:.3f}BW",
            )
            ax3[i].plot(
                time_peakmax,
                vpeakmax,
                "kv",
                markersize=8,
                label=f"Peak Max: {time_peakmax:.3f}s:{vpeakmax:.3f}BW",
            )
            ax3[i].axvline(
                x=t40ms,
                color="red",
                label=f"Peak in 40ms: {vpeak_40ms:.3f}BW",
                linestyle=":",
                linewidth=1,
            )
            ax3[i].axvline(
                x=t100ms,
                color="olive",
                label=f"Peak in 100ms: {vpeak_100ms:.3f}BW",
                linestyle=":",
                linewidth=1,
            )
            ax3[i].plot(
                time_poi,
                vpeak_poi,
                "rv",
                markersize=8,
                label=f"VIP_Transient: {time_peakmax:.3f}s:{vpeakmax:.3f}BW",
            )

            # Linhas para RFD
            # Plotando linhas inclinadas para RFD Impact Transient
            ax3[i].plot(
                [0, time_impact_transient],
                [active_segment_data[0], vpeak_impact_transient],
                "r--",
                linewidth=2.5,
                label=f"RFD Impact Transient: {rfd_impact_transient:.3f}BW\u00B7s\u207B\u00B9",
            )
            # Se rfd_peakmax tem o mesmo intervalo de tempo que rfd_impact_transient
            ax3[i].plot(
                [0, time_peakmax],
                [active_segment_data[0], vpeakmax],
                color="orange",
                linestyle="--",
                linewidth=2.5,
                label=f"RFD Brake: {rfd_peakmax:.3f}BW\u00B7s\u207B\u00B9",
            )
            # Para RFD Propulsion, que começa n o pico máximo e vai até o fim do tempo total
            ax3[i].plot(
                [time_peakmax, total_time],
                [vpeakmax, active_segment_data[-1]],
                "g--",
                linewidth=2.5,
                label=f"RFD Propulsion: {rfd_propulsion:.3f}BW\u00B7s\u207B\u00B9",
            )
            ax3[i].plot(
                [],
                [],
                "s",
                color="red",
                alpha=0.3,
                label=f"Impulse Impact Transient: {area_impact_transient:.3f}BW\u00B7s",
            )
            ax3[i].plot(
                [],
                [],
                "s",
                color="orange",
                alpha=0.3,
                label=f"Impulse Brake: {area_peakmax:.3f}BW\u00B7s",
            )
            ax3[i].plot(
                [],
                [],
                "s",
                color="green",
                alpha=0.3,
                label=f"Impulse Propulsion: {area_propulsion:.3f}BW\u00B7s",
            )
            ax3[i].plot(
                [],
                [],
                "s",
                color="gray",
                alpha=0.2,
                label=f"Total Impulse: {total_area:.3f}BW\u00B7s",
            )
            ax3[i].plot(
                [], [], " ", label=f"Contact Time: {total_time:.3f}s"
            )  # Invisible line for Contact Time
            # Add legend with small font size
            ax3[i].legend(
                fontsize="small",
                loc="upper right",
                handlelength=1,
                handletextpad=0.2,
                borderaxespad=0.2,
                framealpha=0.6,
            )
            plt.tight_layout()

            result = [
                f"{simple_filename}",
                f"{timestamp}",
                f"{i+1}",
                f"{body_weight_kg:.3f}",
                f"{sidefoot}",
                f"{dominance}",
                f"{quality}",
                f"{num_samples}",
                f"{index_40ms}",
                f"{index_100ms}",
                f"{index_impact_transient}",
                f"{index_poi}",
                f"{index_peakmax}",
                f"{test_duration[i]:.3f}",
                f"{cumulative_time:.3f}",
                f"{total_time:.3f}",
                f"{t40ms:.3f}",
                f"{t100ms:.3f}",
                f"{time_impact_transient:.3f}",
                f"{time_poi:.3f}",
                f"{time_peakmax:.3f}",
                f"{vpeak_40ms:.3f}",
                f"{vpeak_100ms:.3f}",
                f"{vpeak_impact_transient:.3f}",
                f"{vpeak_poi:.3f}",
                f"{vpeakmax:.3f}",
                f"{total_area:.3f}",
                f"{area_until_40ms:.3f}",
                f"{area_until_100ms:.3f}",
                f"{area_impact_transient:.3f}",
                f"{area_peakmax:.3f}",
                f"{area_propulsion:.3f}",
                f"{rfd_40ms:.3f}",
                f"{rfd_100ms:.3f}",
                f"{rfd_impact_transient:.3f}",
                f"{rfd_peakmax:.3f}",
                f"{rfd_propulsion:.3f}",
                f"{kc:.3f}",
                f"{kh:.3f}",
                f"{kl:.3f}",
                f"{tT:.3f}",
                f"{valr:.3f}",
            ]
            writer.writerow(result)

            matresults = np.asarray(
                [
                    simple_filename,
                    timestamp,
                    i + 1,
                    body_weight_kg,
                    sidefoot,
                    dominance,
                    quality,
                    num_samples,
                    index_40ms,
                    index_100ms,
                    index_impact_transient,
                    index_poi,
                    index_peakmax,
                    test_duration[i],
                    cumulative_time,
                    total_time,
                    t40ms,
                    t100ms,
                    time_impact_transient,
                    time_poi,
                    time_peakmax,
                    vpeak_40ms,
                    vpeak_100ms,
                    vpeak_impact_transient,
                    vpeak_poi,
                    vpeakmax,
                    total_area,
                    area_until_40ms,
                    area_until_100ms,
                    area_impact_transient,
                    area_peakmax,
                    area_propulsion,
                    rfd_40ms,
                    rfd_100ms,
                    rfd_impact_transient,
                    rfd_peakmax,
                    rfd_propulsion,
                    kc,
                    kh,
                    kl,
                    tT,
                    valr,
                ],
                dtype=object,
            )  # Note que mudamos para 'dtype=object' para acomodar strings e floats.
            result_array.append(matresults)

        plt.savefig(result_plot_filename)
        plt.show()

    return result_array


def run_statistics(data2stats, filename, output_dir, timestamp):
    result_stat_filename = os.path.join(
        output_dir, os.path.splitext(os.path.basename(filename))[0] + "_stats.csv"
    )
    result_profile_filename = os.path.join(
        output_dir, os.path.splitext(os.path.basename(filename))[0] + "_profile.html"
    )

    df = pd.DataFrame(data2stats, columns=build_headers())
    df.drop(columns=["FileName", "TimeStamp"], inplace=True)

    # Temporariamente mapeia 'R' e 'L' para 0 e 1 para descrição estatística
    temp_df = df.copy()
    temp_df["SideFoot_RL"] = temp_df["SideFoot_RL"].map({"R": 0, "L": 1}).astype(float)
    temp_df["Dominance_RL"] = (
        temp_df["Dominance_RL"].map({"R": 0, "L": 1}).astype(float)
    )

    # Calculando estatísticas básicas apenas com dados numéricos temporários
    results_stats = temp_df.describe().round(3)
    results_stats.loc["cv"] = (temp_df.std() / temp_df.mean() * 100).round(2)

    # Renomear o índice para "Stats"
    results_stats.index.name = "Stats"

    # Adicionando filename e timestamp como novas colunas no início do dataframe de estatísticas
    results_stats.insert(0, "Timestamp", timestamp)
    results_stats.insert(0, "Filename", os.path.basename(filename))

    # Salvando o dataframe estendido com as novas colunas
    results_stats.to_csv(result_stat_filename, index=True)

    # Revertendo as colunas para categórico para uso no ProfileReport
    df["SideFoot_RL"] = df["SideFoot_RL"].astype("category")
    df["Dominance_RL"] = df["Dominance_RL"].astype("category")

    # Gerando perfil de dados usando as categorias
    profile = ProfileReport(df, title="Profiling Report")
    profile.to_file(result_profile_filename)

    return results_stats, profile


if __name__ == "__main__":
    Fs = 1000  # Sampling Frequency

    if len(sys.argv) < 2:  # Check if filename is provided
        print(
            "Usage: python script.py <filename> [sidefoot] [dominance] [quality] [threshold] [columns] [index_file] [bw_kg]"
        )
        sys.exit(1)  # Exit if filename is not provided

    filename = sys.argv[1]  # Filename is required
    simple_filename = basename(filename)  # Use basename to get just the file name
    sidefoot = sys.argv[2] if len(sys.argv) > 2 else "R"
    dominance = sys.argv[3] if len(sys.argv) > 3 else "R"
    quality = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    threshold = float(sys.argv[5]) if len(sys.argv) > 5 else 0.025
    columns = sys.argv[6] if len(sys.argv) > 6 else "12,13,14"
    index_file = sys.argv[7] if len(sys.argv) > 7 else None
    body_weight_kg = float(sys.argv[8]) if len(sys.argv) > 8 else None

    print(f"Filename: {filename}")
    print(f"Sidefoot: {sidefoot}")
    print(f"Dominance: {dominance}")
    print(f"Quality: {quality}")
    print(f"Threshold: {threshold}")
    print(f"Columns: {columns}")

    # Directly try to load and process the data to see detailed errors
    data, timestamp = read_csv_skip_header(filename, columns)
    if data is None:
        print("No data returned from file reading function.")
        sys.exit(1)

    selected_data = data[:, -1] * -1 if data.ndim > 1 else data * -1
    if not body_weight_kg:
        body_weight_newton = np.median(selected_data[10:110])
        body_weight_kg = body_weight_newton / 9.81

    databw_norm = selected_data / (body_weight_kg * 9.81)

    main_output_dir = create_main_output_directory(filename)

    if index_file:
        indices = np.loadtxt(index_file).astype(
            int
        )  # Load and convert indices from file if provided
    else:
        indices = makefig1(
            databw_norm, main_output_dir, filename
        )  # Pass output_dir and filename to makefig1

    active_ranges = makefig2(databw_norm, indices, threshold)
    databw = butterworthfilt(databw_norm, 59, Fs)
    results = makefig3(
        databw,
        active_ranges,
        body_weight_kg,
        main_output_dir,
        filename,
        sidefoot,
        dominance,
        quality,
        simple_filename,
        timestamp,
        indices,
        Fs,
    )
    result_stats, result_profile = run_statistics(
        results, filename, main_output_dir, timestamp
    )
