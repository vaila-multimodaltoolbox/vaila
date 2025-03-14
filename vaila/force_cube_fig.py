"""
================================================================================
Force Platform Data Analysis Toolkit - force_cube_fig.py
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago Ligia
Date: 9 September 2024
Version: 0.5
Python Version: 3.11

Description:
------------
This script processes biomechanical data from force platforms, analyzing
the vertical ground reaction force (VGRF) to compute key metrics, including:
- Peak forces, time intervals, impulse, rate of force development (RFD),
  and stiffness parameters.
The results are visualized through interactive plots and saved to CSV files for
further analysis. The script allows batch processing of multiple files and provides
descriptive statistics for all analyzed data.

Key Functionalities:
---------------------
1. Data Selection:
   - Allows the user to select input CSV files containing biomechanical data.
   - Prompts the user to specify output directories.
   - Prompts the user for input parameters (sampling frequency, thresholds, etc.).
2. Data Processing:
   - Normalizes data, applies Butterworth filters, and computes key biomechanical metrics.
   - Computes metrics such as peak force, impulse, and rate of force development.
3. Visualization:
   - Generates and saves plots for force-time curves with relevant markers and highlighted regions.
4. Statistical Analysis:
   - Provides descriptive statistics and optional profiling reports using pandas and ydata_profiling.
5. Batch Processing:
   - Processes all CSV files in the selected source directory.

Input:
------
- CSV Files:
   Each CSV file should contain biomechanical data, specifically force data recorded
   from a force platform. The file must include a column for vertical ground reaction force (VGRF).
   Example format:
   Sample, Force (N)
   0, 50.25
   1, 51.60
   2, 49.80
   ...

- User Input:
   The user will input various parameters through a graphical interface, including:
   - Sidefoot (R/L)
   - Dominance (R/L)
   - Quality (integer)
   - Threshold for activity detection
   - Sampling frequency (Fs in Hz)
   - Whether to generate a profiling report.

Output:
-------
- CSV Files:
   A CSV file for each input file, containing results for key metrics such as:
   * Peak force at 40 ms and 100 ms
   * Impulse over different time intervals
   * Rate of force development (RFD)
   * Stiffness parameters
   * Total contact time and time to reach peak forces

- Plot Files:
   PNG and SVG plots of force-time curves, highlighting important events such as peak forces,
   rate of force development, and impact transient.

- Statistical Report:
   A summary CSV file containing descriptive statistics for the processed files.
   Optionally, a profiling report in HTML format for each file.

How to Run:
-----------
1. Ensure that all dependencies are installed, including numpy, pandas, matplotlib, scipy,
   ydata_profiling, rich, and tkinter. You can install these using pip.
2. Run the script from the terminal:
   python force_cube_fig.py
3. A graphical interface will guide you to:
   - Select the source directory containing CSV files.
   - Select the output directory where results will be saved.
   - Specify input parameters such as sampling frequency, sidefoot, and dominance.
4. The script will process all files in the selected directory and generate CSV results
   and plots for each file.

License:
--------
This script is licensed under the GNU General Public License v3.0 (GPLv3).
You may redistribute and modify the script under these terms. See the LICENSE file
or visit https://www.gnu.org/licenses/gpl-3.0.html for more details.

Disclaimer:
-----------
This script is provided "as is" without warranty of any kind. It is intended for
academic and research purposes only. The author is not liable for any damage or
loss resulting from its use.

Changelog:
----------
- 2024-09-09: Initial release with core biomechanical analysis functions.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import csv
from datetime import datetime
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from rich import print
from ydata_profiling import ProfileReport
from tkinter import (
    Tk,
    Button,
    filedialog,
    Toplevel,
    Checkbutton,
    BooleanVar,
    Canvas,
    Scrollbar,
    Frame,
    messagebox,
    simpledialog,
)


# Print the directory and name of the script being executed
print(f"vailá - Running script: {os.path.basename(__file__)}")
print(f"vailá - Script directory: {os.path.dirname(os.path.abspath(__file__))}")


def select_source_directory():
    """
    Opens a directory dialog to select the source directory containing CSV files.
    """
    root = Tk()
    root.withdraw()  # Hides the main Tkinter window
    source_dir = filedialog.askdirectory(title="Select Source Directory")
    root.destroy()
    return source_dir


def select_output_directory():
    """
    Opens a dialog to select the output directory for saving results.
    """
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    output_dir = filedialog.askdirectory(title="Select Output Directory for Results")
    root.destroy()
    return output_dir


def select_body_weight(data):
    """
    Allows the user to interactively select the range for calculating body_weight_newton.
    If the majority of data is negative, it multiplies the data by -1 for visualization.
    """

    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title(
        "select the range for body weight calculation (2 clicks). Hold Space + Left Click to mark, Right Click to remove, 'Enter' to confirm."
    )
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Force Value")
    ax.grid(True)

    points = []
    space_held = False  # Variable to track if Space is held down

    def on_key_press(event):
        nonlocal space_held
        if event.key == " ":
            space_held = True
        elif event.key == "enter":
            plt.close(fig)  # Closes the figure when 'Enter' is pressed

    def on_key_release(event):
        nonlocal space_held
        if event.key == " ":
            space_held = False

    def onclick(event):
        if space_held and event.button == 1:  # Left mouse button with space held down
            x_value = event.xdata
            if x_value is not None:
                points.append(int(x_value))
                ax.axvline(x=x_value, color="black", linestyle="--")
                fig.canvas.draw()
        elif event.button == 3:  # Right mouse button to remove the last point
            if points:
                points.pop()
                ax.cla()
                ax.plot(data)
                ax.grid(True)
                for point in points:
                    ax.axvline(x=point, color="black", linestyle="--")
                fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)

    plt.show(block=True)

    if len(points) != 2:
        print("Please select exactly two points for body weight.")
        sys.exit(1)

    start_index, end_index = sorted(points)
    body_weight_newton = np.median(data[start_index:end_index])

    print(f"Selected body weight range: {start_index} to {end_index}")
    print(f"Calculated Body Weight (in Newton): {body_weight_newton}")
    plt.close(fig)

    return body_weight_newton


def process_file(
    file_path,
    selected_column,
    sidefoot,
    dominance,
    quality,
    threshold,
    output_dir,
    Fs=1000,
    generate_profile="No",
):
    """
    Processes a single file for the selected column using the provided parameters.
    """

    # Attempt to load the file and extract the selected column with various encoding and delimiter options
    def load_csv_file(file_path):
        encodings = ["utf-8", "ISO-8859-1", "latin1", "windows-1252"]
        delimiters = [",", ";", "\t"]

        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
                    return df
                except UnicodeDecodeError:
                    print(
                        f"Failed to read {file_path} with encoding {encoding} and delimiter '{delimiter}', trying next..."
                    )
                except Exception as e:
                    print(f"An error occurred: {e}")

        # If all attempts fail, raise an error
        raise UnicodeDecodeError(
            f"Failed to read the file {file_path} with the provided encodings and delimiters."
        )

    # Load the file using the new load_csv_file function
    df = load_csv_file(file_path)

    # Extract the selected column data
    data = df[selected_column].to_numpy()

    # Check if data mean is negative
    force_negative = np.mean(data)
    if force_negative < 0:
        print("Fz force is negative. Inverting the sign of the data.")
        data = data * -1  # Invert the sign if the majority is negative

    # Interactive body weight selection
    body_weight_newton = select_body_weight(data)

    # Determine body weight from the data if not provided
    body_weight_kg = body_weight_newton / 9.81
    databw_norm = data / (body_weight_kg * 9.81)

    # Create output directory for the file
    main_output_dir = create_main_output_directory(output_dir, file_path)

    # Prompt user to select indices for the analysis
    indices = makefig1(databw_norm, main_output_dir, file_path)

    # Calculate active ranges
    active_ranges = makefig2(databw_norm, indices, threshold)

    # Apply the Butterworth filter to the normalized data
    databw = butterworthfilt(databw_norm, cutoff=59, Fs=Fs)

    # Perform the main analysis and generate figures
    results = makefig3(
        databw,
        active_ranges,
        body_weight_kg,
        main_output_dir,
        file_path,
        sidefoot,
        dominance,
        quality,
        os.path.basename(file_path),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        indices,
        Fs,
    )

    # Run statistics on the results
    result_stats, result_profile = run_statistics(
        results,
        file_path,
        main_output_dir,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        generate_profile,
    )

    print(f"Processing completed for file: {file_path}")
    return results, result_stats, result_profile


def batch_process_directory(
    source_dir,
    selected_column,
    output_dir,
    sidefoot,
    dominance,
    quality,
    threshold,
    fs,
    generate_profile,
):
    # Correct function definition with all required parameters
    files = sorted([f for f in os.listdir(source_dir) if f.endswith(".csv")])
    for file_name in files:
        file_path = os.path.join(source_dir, file_name)
        process_file(
            file_path,
            selected_column,
            sidefoot,
            dominance,
            quality,
            threshold,
            output_dir,
            Fs=fs,
            generate_profile=generate_profile,
        )


def select_headers_and_load_data(file_path):
    """
    Displays a GUI to select the desired headers with Select All and Unselect All options
    and loads the corresponding data for the selected headers.
    """

    def get_csv_headers(file_path):
        """
        Reads the headers from a CSV file with fallback for different encodings and delimiters.
        """
        encodings = [
            "utf-8",
            "ISO-8859-1",
            "latin1",
            "windows-1252",
        ]  # Common encodings for CSV files
        delimiters = [",", ";", "\t"]  # Common delimiters (comma, semicolon, tab)

        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
                    return (
                        list(df.columns),
                        df,
                    )  # Return the headers and DataFrame if successful
                except UnicodeDecodeError:
                    print(
                        f"Failed to read {file_path} with encoding {encoding} and delimiter '{delimiter}', trying next..."
                    )
                except Exception as e:
                    print(f"An error occurred: {e}")

        # If all encoding and delimiter attempts fail, raise an error
        raise UnicodeDecodeError(
            f"Failed to read the file {file_path} with the provided encodings and delimiters."
        )

    headers, df = get_csv_headers(file_path)
    selected_headers = []

    def on_select():
        nonlocal selected_headers
        selected_headers = [
            header for header, var in zip(headers, header_vars) if var.get()
        ]
        selection_window.quit()  # Ends the main Tkinter loop
        selection_window.destroy()  # Closes the selection window

    def select_all():
        for var in header_vars:
            var.set(True)

    def unselect_all():
        for var in header_vars:
            var.set(False)

    selection_window = Toplevel()
    selection_window.title(f"Select Headers for {os.path.basename(file_path)}")
    selection_window.geometry(
        f"{selection_window.winfo_screenwidth()}x{int(selection_window.winfo_screenheight()*0.9)}"
    )

    canvas = Canvas(selection_window)
    scrollbar = Scrollbar(selection_window, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas)

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    header_vars = [BooleanVar() for _ in headers]

    num_columns = 7  # Number of columns for header labels

    for i, label in enumerate(headers):
        chk = Checkbutton(scrollable_frame, text=label, variable=header_vars[i])
        chk.grid(row=i // num_columns, column=i % num_columns, sticky="w")

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    btn_frame = Frame(selection_window)
    btn_frame.pack(side="right", padx=10, pady=10, fill="y", anchor="center")

    btn_select_all = Button(btn_frame, text="Select All", command=select_all)
    btn_select_all.pack(side="top", pady=5)

    btn_unselect_all = Button(btn_frame, text="Unselect All", command=unselect_all)
    btn_unselect_all.pack(side="top", pady=5)

    btn_select = Button(btn_frame, text="Confirm", command=on_select)
    btn_select.pack(side="top", pady=5)

    selection_window.mainloop()

    if not selected_headers:
        messagebox.showinfo("Info", "No headers were selected.")
        return None, None

    # Load the data for the selected headers
    selected_data = df[selected_headers]

    return selected_headers, selected_data


def create_main_output_directory(output_dir, filename):
    """
    Creates a main output directory for results based on the output directory chosen by the user,
    the current date, and the input filename.
    """
    today_date = datetime.now().strftime("%Y%m%d")  # Format YYYYMMDD
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    main_output_dir = os.path.join(
        output_dir,  # Use the output directory chosen by the user
        f"Results_force_cube_analysis_{today_date}",
        f"Results_{base_filename}",
    )

    # Verifica se o diretório de saída já existe, e se não, o cria
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)

    # Print para verificar o diretório de saída
    print(f"Directory for saving results: {main_output_dir}")

    return main_output_dir


def prompt_user_input(file_name):
    """
    Prompts the user for input parameters and displays the file name being processed.
    Ensures case-insensitive input and standardizes it properly.
    """
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window

    # Display the current file being processed in the input dialogs
    sidefoot = simpledialog.askstring(
        "Input", f"Enter Sidefoot (R or L) for {file_name}:", initialvalue="R"
    )
    sidefoot = (
        sidefoot.upper() if sidefoot and sidefoot.upper() in ["R", "L"] else "R"
    )  # Ensure uppercase R or L

    dominance = simpledialog.askstring(
        "Input", f"Enter Dominance (R or L) for {file_name}:", initialvalue="R"
    )
    dominance = (
        dominance.upper() if dominance and dominance.upper() in ["R", "L"] else "R"
    )  # Ensure uppercase R or L

    quality = simpledialog.askinteger(
        "Input", f"Enter Quality (integer) for {file_name}:", initialvalue=5
    )
    threshold = simpledialog.askfloat(
        "Input", f"Enter Threshold (float) for {file_name}:", initialvalue=0.025
    )
    fs = simpledialog.askfloat(
        "Input", f"Enter Sampling Frequency (Fs) for {file_name}:", initialvalue=1000.0
    )

    generate_profile = simpledialog.askstring(
        "Input",
        f"Generate Profiling Report? (Yes or No) for {file_name}:",
        initialvalue="No",
    )
    # Convert 'y' or 'Y' to 'Yes' and 'n' or 'N' to 'No'
    if generate_profile:
        generate_profile = (
            generate_profile.strip().lower()
        )  # Make input case-insensitive
        if generate_profile in ["y", "yes"]:
            generate_profile = "Yes"
        elif generate_profile in ["n", "no"]:
            generate_profile = "No"
        else:
            generate_profile = "No"  # Default to 'No' if not recognized

    root.destroy()
    return sidefoot, dominance, quality, threshold, fs, generate_profile


def butterworthfilt(data, cutoff=59, Fs=1000):
    """
    Applies a Butterworth filter to the data.
    """
    pad_length = 100
    padded_data = np.pad(data, (pad_length, pad_length), "edge")
    b, a = butter(4, cutoff / (Fs / 2), "low")
    filtered_padded = filtfilt(b, a, padded_data)
    filtered_data = filtered_padded[pad_length:-pad_length]
    return filtered_data


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

    time_interval = np.linspace(0, (num_samples - 1) / Fs, num_samples)

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
        "Select intervals of interest for Force Fz: Hold Space + Left Click to mark, Right Click to remove, 'Enter' to confirm"
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

    # Exibir e fechar a figura fora do loop
    plt.show(block=False)  # Mostra a figura sem bloquear o restante do código
    plt.pause(1)  # Espera por 1 segundo
    plt.close(fig2)  # Fecha a figura

    return active_ranges


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

    # Define the plot PNG and SVG file names using the new directory path
    result_plot_filename = os.path.join(
        output_dir, filename_without_extension + "_result"
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
                label=f"RFD Impact Transient: {rfd_impact_transient:.3f}BW\u00b7s\u207b\u00b9",
            )
            # Se rfd_peakmax tem o mesmo intervalo de tempo que rfd_impact_transient
            ax3[i].plot(
                [0, time_peakmax],
                [active_segment_data[0], vpeakmax],
                color="orange",
                linestyle="--",
                linewidth=2.5,
                label=f"RFD Brake: {rfd_peakmax:.3f}BW\u00b7s\u207b\u00b9",
            )
            # Para RFD Propulsion, que começa n o pico máximo e vai até o fim do tempo total
            ax3[i].plot(
                [time_peakmax, total_time],
                [vpeakmax, active_segment_data[-1]],
                "g--",
                linewidth=2.5,
                label=f"RFD Propulsion: {rfd_propulsion:.3f}BW\u00b7s\u207b\u00b9",
            )
            ax3[i].plot(
                [],
                [],
                "s",
                color="red",
                alpha=0.3,
                label=f"Impulse Impact Transient: {area_impact_transient:.3f}BW\u00b7s",
            )
            ax3[i].plot(
                [],
                [],
                "s",
                color="orange",
                alpha=0.3,
                label=f"Impulse Brake: {area_peakmax:.3f}BW\u00b7s",
            )
            ax3[i].plot(
                [],
                [],
                "s",
                color="green",
                alpha=0.3,
                label=f"Impulse Propulsion: {area_propulsion:.3f}BW\u00b7s",
            )
            ax3[i].plot(
                [],
                [],
                "s",
                color="gray",
                alpha=0.2,
                label=f"Total Impulse: {total_area:.3f}BW\u00b7s",
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

        # Salva a figura em PNG e SVG
        plt.savefig(result_plot_filename + ".png", format="png", dpi=300)
        plt.savefig(result_plot_filename + ".svg", format="svg")

    # Exibir e fechar a figura fora do loop
    plt.show(block=False)  # Mostra a figura sem bloquear o restante do código
    plt.pause(1)  # Espera por 1 segundo
    plt.close(fig3)  # Fecha a figura

    return result_array


## FIGURE 4
def makefig4(
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
    fig4, ax3 = plt.subplots(3, 3, figsize=(30, 30))
    ax4 = ax3.flatten()  # Flatten the array to simplify indexing

    # Get the filename without extension
    filename_without_extension = os.path.splitext(os.path.basename(filename))[0]

    # Define the CSV result file name using the new directory path
    result_csv_filename = os.path.join(
        output_dir, filename_without_extension + "_result.csv"
    )

    # Define the plot PNG and SVG file names using the new directory path
    result_plot_filename_png = os.path.join(
        output_dir, filename_without_extension + "_result.png"
    )
    result_plot_filename_svg = os.path.join(
        output_dir, filename_without_extension + "_result.svg"
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
            ax4[i].plot(time_interval, active_segment_data, linewidth=3)
            # Convertendo start e end de índices para segundos
            start_seconds = start / Fs
            end_seconds = end / Fs
            # Formatação do título com tempo em segundos
            ax4[i].set_title(f"{i+1}: time {start_seconds:.3f} to {end_seconds:.3f} s")
            ax4[i].set_xlabel("Time (s)")
            ax4[i].set_ylabel("Force (BW)")
            # Ajustando o limite x do gráfico em x
            max_time_interval = np.max(time_interval)
            if max_time_interval > 0.65:
                ax4[i].set_xlim(0, max_time_interval)
            else:
                ax4[i].set_xlim(0, 0.65)

            # Ajustando o limite y do gráfico em y
            max_active_segment_data = np.max(active_segment_data)
            if max_active_segment_data > 3.7:
                ax4[i].set_ylim(0, max_active_segment_data + 0.1)
            else:
                ax4[i].set_ylim(0, 3.7)

            # Get current ticks and add custom tick for 40ms
            current_ticks = np.append(
                ax4[i].get_xticks(), 0.04
            )  # Include 0.04s in the existing ticks
            current_ticks = np.unique(
                np.sort(current_ticks)
            )  # Sort and remove any duplicates
            ax4[i].set_xticks(current_ticks)  # Set the ticks to include 0.04s

            # Optional: you can format ticks to show more clearly or with specific precision
            ax4[i].get_xaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:.2f}")
            )

            # Preenchendo a área sob a curva
            ax4[i].fill_between(
                time_interval, active_segment_data, color="gray", alpha=0.2
            )

            # Create an array of x values from 0 to impact transient for filling - IMPACT TRANSIENT
            x_fill_impact_transient = np.linspace(0, time_impact_transient, 100)
            # Create corresponding y values for the line
            y_fill_impact_transient = (
                vpeak_impact_transient / time_impact_transient
            ) * x_fill_impact_transient
            # Fill the area below the line in red with transparency
            ax4[i].fill_between(
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
            ax4[i].fill_between(
                x_fill_peakmax, 0, y_fill_peakmax, color="orange", alpha=0.3
            )

            # Preenchimento para o peakmax to end - PROPULSION
            x_fill_end = np.linspace(time_peakmax, total_time, 100)
            y_start = vpeakmax  # Valor de Y no ponto peakmax to end - PROPULSION
            y_fill_end = np.linspace(
                y_start, 0, 100
            )  # Decresce linearmente de vpeakmax até 0

            # Fill the area below the line in another color (e.g., green) with transparency
            ax4[i].fill_between(x_fill_end, 0, y_fill_end, color="green", alpha=0.3)

            # Highlight the peaks on the plot
            ax4[i].plot(
                time_impact_transient,
                vpeak_impact_transient,
                "r*",
                markersize=12,
                label=f"Transient: {time_impact_transient:.3f}s:{vpeak_impact_transient:.3f}BW",
            )
            ax4[i].plot(
                time_peakmax,
                vpeakmax,
                "kv",
                markersize=8,
                label=f"Peak Max: {time_peakmax:.3f}s:{vpeakmax:.3f}BW",
            )
            ax4[i].axvline(
                x=t40ms,
                color="red",
                label=f"Peak in 40ms: {vpeak_40ms:.3f}BW",
                linestyle=":",
                linewidth=1,
            )
            ax4[i].axvline(
                x=t100ms,
                color="olive",
                label=f"Peak in 100ms: {vpeak_100ms:.3f}BW",
                linestyle=":",
                linewidth=1,
            )
            ax4[i].plot(
                time_poi,
                vpeak_poi,
                "rv",
                markersize=8,
                label=f"VIP_Transient: {time_peakmax:.3f}s:{vpeakmax:.3f}BW",
            )

            # Linhas para RFD
            # Plotando linhas inclinadas para RFD Impact Transient
            ax4[i].plot(
                [0, time_impact_transient],
                [active_segment_data[0], vpeak_impact_transient],
                "r--",
                linewidth=2.5,
                label=f"RFD Impact Transient: {rfd_impact_transient:.3f}BW\u00b7s\u207b\u00b9",
            )
            # Se rfd_peakmax tem o mesmo intervalo de tempo que rfd_impact_transient
            ax4[i].plot(
                [0, time_peakmax],
                [active_segment_data[0], vpeakmax],
                color="orange",
                linestyle="--",
                linewidth=2.5,
                label=f"RFD Brake: {rfd_peakmax:.3f}BW\u00b7s\u207b\u00b9",
            )
            # Para RFD Propulsion, que começa n o pico máximo e vai até o fim do tempo total
            ax4[i].plot(
                [time_peakmax, total_time],
                [vpeakmax, active_segment_data[-1]],
                "g--",
                linewidth=2.5,
                label=f"RFD Propulsion: {rfd_propulsion:.3f}BW\u00b7s\u207b\u00b9",
            )
            ax4[i].plot(
                [],
                [],
                "s",
                color="red",
                alpha=0.3,
                label=f"Impulse Impact Transient: {area_impact_transient:.3f}BW\u00b7s",
            )
            ax4[i].plot(
                [],
                [],
                "s",
                color="orange",
                alpha=0.3,
                label=f"Impulse Brake: {area_peakmax:.3f}BW\u00b7s",
            )
            ax4[i].plot(
                [],
                [],
                "s",
                color="green",
                alpha=0.3,
                label=f"Impulse Propulsion: {area_propulsion:.3f}BW\u00b7s",
            )
            ax4[i].plot(
                [],
                [],
                "s",
                color="gray",
                alpha=0.2,
                label=f"Total Impulse: {total_area:.3f}BW\u00b7s",
            )
            ax4[i].plot(
                [], [], " ", label=f"Contact Time: {total_time:.3f}s"
            )  # Invisible line for Contact Time
            # Add legend with small font size
            ax4[i].legend(
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
            )  # Note with changes to dytpe=object to accomodate strings and floats.
            result_array.append(matresults)

        # Save the figure in PNG and SVG formats
        plt.savefig(result_plot_filename_png, format="png", dpi=300)
        plt.savefig(result_plot_filename_svg, format="svg")
        plt.show(block=False)  # Show the figure without blocking the rest of the code.
        plt.pause(1)  # Wait for 1 second before closing the figure
        plt.close(fig4)  # close the figure

        return result_array


def run_statistics(data2stats, filename, output_dir, timestamp, generate_profile):
    result_stat_filename = os.path.join(
        output_dir, os.path.splitext(os.path.basename(filename))[0] + "_stats.csv"
    )
    result_profile_filename = os.path.join(
        output_dir, os.path.splitext(os.path.basename(filename))[0] + "_profile.html"
    )

    df = pd.DataFrame(data2stats, columns=build_headers())
    df.drop(columns=["FileName", "TimeStamp"], inplace=True)

    # Temporarily map 'R' and 'L' to 0 and 1 for statistical description
    temp_df = df.copy()
    temp_df["SideFoot_RL"] = temp_df["SideFoot_RL"].map({"R": 0, "L": 1}).astype(float)
    temp_df["Dominance_RL"] = (
        temp_df["Dominance_RL"].map({"R": 0, "L": 1}).astype(float)
    )

    # Calculating basic statistics with temporary numerical data
    results_stats = temp_df.describe().round(3)
    results_stats.loc["cv"] = (temp_df.std() / temp_df.mean() * 100).round(2)

    # Rename the index to "Stats"
    results_stats.index.name = "Stats"

    # Add filename and timestamp as new columns at the beginning of the statistics dataframe
    results_stats.insert(0, "Timestamp", timestamp)
    results_stats.insert(0, "Filename", os.path.basename(filename))

    # Save the extended dataframe with the new columns
    results_stats.to_csv(result_stat_filename, index=True)

    # Revert the columns to categorical for use in ProfileReport
    df["SideFoot_RL"] = df["SideFoot_RL"].astype("category")
    df["Dominance_RL"] = df["Dominance_RL"].astype("category")

    # Generate profiling report if the user opted for it
    if generate_profile.lower() == "yes":
        profile = ProfileReport(df, title="Profiling Report")
        profile.to_file(result_profile_filename)

    return results_stats, None if generate_profile.lower() != "yes" else profile


def main():
    """
    Main function to handle batch processing of CSV files.
    """
    source_dir = select_source_directory()
    if not source_dir:
        messagebox.showerror("Error", "No source directory selected.")
        return

    # Get the first file for header selection
    first_file_path = sorted([f for f in os.listdir(source_dir) if f.endswith(".csv")])[
        0
    ]
    first_file_full_path = os.path.join(source_dir, first_file_path)

    selected_headers, selected_data = select_headers_and_load_data(first_file_full_path)
    if not selected_headers or selected_data is None:
        print("No headers or data selected.")
        return

    # Ask the user to select the column to analyze
    selected_column = selected_headers[
        0
    ]  # Example: Assume first column is selected for analysis

    # Prompt for output directory
    output_dir = select_output_directory()
    if not output_dir:
        messagebox.showerror("Error", "No output directory selected.")
        return

    # Verifica se o diretório de saída existe, e se não, cria
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ask for user input once, including Fs
    # Get the first file's name for input prompts
    first_file_name = os.path.basename(first_file_full_path)

    # Ask for user input once, including Fs, and pass the file name
    sidefoot, dominance, quality, threshold, fs, generate_profile = prompt_user_input(
        first_file_name
    )

    # Batch process all files in the source directory
    batch_process_directory(
        source_dir,
        selected_column,
        output_dir,
        sidefoot,
        dominance,
        quality,
        threshold,
        fs,
        generate_profile,
    )


if __name__ == "__main__":
    # Print the directory and name of the script being executed
    print(f"vailá running script : {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    # Run the main function
    main()
