import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from tkinter import (
    Tk,
    Toplevel,
    Canvas,
    Scrollbar,
    Frame,
    Button,
    Checkbutton,
    BooleanVar,
    messagebox,
    filedialog,
    simpledialog,
)


def read_csv_full(filename):
    """Reads the full CSV file."""
    try:
        data = pd.read_csv(filename, delimiter=",")
        # multiply -1 all values in the first column
        data.iloc[:, 0] = data.iloc[:, 0]

        return data
    except Exception as e:
        raise Exception(f"Error reading the CSV file: {str(e)}")


def butterworth_filter(data, cutoff, fs, order=4):
    """Applies a Butterworth filter to the data."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data, axis=0)

    return y


def select_two_columns(file_path):
    """Displays a GUI to select two columns for 2D analysis."""

    def get_csv_headers(file_path):
        """Reads the headers from a CSV file."""
        df = pd.read_csv(file_path)
        return list(df.columns), df

    headers, df = get_csv_headers(file_path)
    selected_headers = []

    def on_select():
        nonlocal selected_headers
        selected_headers = [
            header for header, var in zip(headers, header_vars) if var.get()
        ]
        selection_window.quit()
        selection_window.destroy()

    def select_all():
        for var in header_vars:
            var.set(True)

    def unselect_all():
        for var in header_vars:
            var.set(False)

    selection_window = Toplevel()
    selection_window.title("Select Two Headers")
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
    btn_frame.pack(side="right", padx=10, pady=10, anchor="center")

    Button(btn_frame, text="Select All", command=select_all).pack(side="top", pady=5)
    Button(btn_frame, text="Unselect All", command=unselect_all).pack(
        side="top", pady=5
    )
    Button(btn_frame, text="Confirm", command=on_select).pack(side="top", pady=5)

    selection_window.mainloop()

    if len(selected_headers) != 2:
        messagebox.showinfo("Info", "Please select exactly two headers for analysis.")
        return None, None

    selected_data = df[selected_headers]
    return selected_headers, selected_data


def analyze_data_2d(
    data, output_dir, file_name, fs, plate_width, plate_height, timestamp
):
    """Analyzes selected 2D data and saves results."""
    # Filter data
    data = butterworth_filter(data, cutoff=10, fs=fs)

    cop_x = data[:, 0]
    cop_y = data[:, 1]

    # Create vector time
    time = np.arange(0, len(cop_x) / fs, 1 / fs)

    # Create figure
    plt.figure()

    # Plot displacement X
    plt.subplot(2, 1, 1)
    plt.plot(time, cop_x, "-", color="black")
    plt.xlabel("Time [s]")
    plt.ylabel("X-axis ML [cm]")
    plt.title("Medio-Lateral (ML) Displacement")
    plt.grid(True)

    # Plot displacement Y
    plt.subplot(2, 1, 2)
    plt.plot(time, cop_y, "-", color="black")
    plt.xlabel("Time [s]")
    plt.ylabel("Y-axis AP [cm]")
    plt.title("Antero-Posterior (AP) Displacement")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_name}_timedisp_{timestamp}.png"))
    plt.savefig(os.path.join(output_dir, f"{file_name}_timedisp_{timestamp}.svg"))

    # Sets the confidence level
    confidence = 0.95  # Exemple for 95%

    # Plot pathway of CoP and ellipse 95% confidence
    plt.figure()
    plt.plot(cop_x, cop_y, "-", color="black", label="CoP Pathway")
    plt.plot(cop_x[0], cop_y[0], "g.", markersize=17)  # Primeiro ponto em verde
    plt.plot(cop_x[-1], cop_y[-1], "r.", markersize=17)  # Último ponto em vermelho

    plt.xlabel("Medio-Lateral Displacement (cm)")
    plt.ylabel("Antero-Posterior Displacement (cm)")
    plt.xlim(-plate_width / 2, plate_width / 2)
    plt.ylim(-plate_height / 2, plate_height / 2)
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")

    # Ellipse e PCA
    area, angle = plot_ellipse_pca(data, confidence)
    plt.title(
        f"CoP {confidence*100:.1f}% Confidence Ellipse (Area: {area:.2f} cm^2, Angle: {angle:.2f}$^\\circ$)"
    )

    # Save figures of CoP pathway and ellipse
    plt.savefig(os.path.join(output_dir, f"{file_name}analysis_{timestamp}.png"))
    plt.savefig(os.path.join(output_dir, f"{file_name}_cop_analysis_{timestamp}.svg"))


def plot_ellipse_pca(data, confidence=0.95):
    """Calculates and plots the ellipse using PCA with a specified confidence level."""
    pca = PCA(n_components=2)
    pca.fit(data)

    # Eigenvalues e eigenvectors
    eigvals = np.sqrt(pca.explained_variance_)
    eigvecs = pca.components_

    # Scale factor for confidence level
    chi2_val = np.sqrt(2) * np.sqrt(np.log(1 / (1 - confidence)))
    scaled_eigvals = eigvals * chi2_val

    # Ellipse parameters
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse = np.array(
        [scaled_eigvals[0] * np.cos(theta), scaled_eigvals[1] * np.sin(theta)]
    )
    ellipse_rot = np.dot(eigvecs.T, ellipse)  # Adjustement for rotated ellipse

    # Area and angle of the ellipse
    area = np.pi * scaled_eigvals[0] * scaled_eigvals[1]
    angle = (
        np.arctan2(eigvecs[1, 0], eigvecs[0, 0]) * 180 / np.pi
    )  # Adjustement for rotated ellipse

    plt.figure()
    # Plot of data and ellipse
    plt.plot(data[:, 0], data[:, 1], "-", color="black", label="CoP Pathway")
    plt.plot(
        ellipse_rot[0, :] + pca.mean_[0],
        ellipse_rot[1, :] + pca.mean_[1],
        "r--",
        linewidth=2,
    )

    # Plot of major and minor axis
    major_axis_start = pca.mean_
    major_axis_end = pca.mean_ + eigvecs[0] * scaled_eigvals[0]
    plt.plot(
        [major_axis_start[0], major_axis_end[0]],
        [major_axis_start[1], major_axis_end[1]],
        "b-",
        linewidth=1,
    )

    minor_axis_start = pca.mean_
    minor_axis_end = pca.mean_ + eigvecs[1] * scaled_eigvals[1]
    plt.plot(
        [minor_axis_start[0], minor_axis_end[0]],
        [minor_axis_start[1], minor_axis_end[1]],
        "b-",
        linewidth=1,
    )

    return area, angle


def main():
    """Function to run the CoP analysis"""
    root = Tk()
    root.withdraw()  # Hides the main Tkinter window

    # Request input and output directories
    input_dir = filedialog.askdirectory(title="Select Input Directory")
    if not input_dir:
        print("No input directory selected.")
        return

    output_dir = filedialog.askdirectory(title="Select Output Directory")
    if not output_dir:
        print("No output directory selected.")
        return

    # Request sampling frequency and plate dimensions
    fs = simpledialog.askfloat(
        "Signal Frequency",
        "Enter the sampling frequency (Fs) in Hz:",
        initialvalue=100.0,
    )
    if not fs:
        print("No valid frequency provided.")
        return

    plate_width = simpledialog.askfloat(
        "Force Plate Width",
        "Enter the width of the force plate in cm:",
        initialvalue=46.4,
    )
    plate_height = simpledialog.askfloat(
        "Force Plate Height",
        "Enter the height of the force plate in cm:",
        initialvalue=50.75,
    )

    if not plate_width or not plate_height:
        print("Invalid force plate dimensions provided.")
        return

    # Select sample file
    sample_file_path = filedialog.askopenfilename(
        title="Select a Sample CSV File", filetypes=[("CSV files", "*.csv")]
    )
    if not sample_file_path:
        print("No sample file selected.")
        return

    # Select two headers for 2D analysis
    selected_headers, _ = select_two_columns(sample_file_path)
    if not selected_headers:
        print("No valid headers selected.")
        return

    # Create timestamp for output directory
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Create main output directory using timestamp
    main_output_dir = os.path.join(output_dir, f"vaila_cop_balance_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)

    # Process each CSV file in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_dir, file_name)
            data = read_csv_full(file_path)[selected_headers].to_numpy()

            # Create output directory for current file
            file_output_dir = os.path.join(main_output_dir, file_name)
            os.makedirs(file_output_dir, exist_ok=True)

            # Analyze and save results
            analyze_data_2d(
                data,
                file_output_dir,
                file_name,
                fs,
                plate_width,
                plate_height,
                timestamp,
            )
    plt.show()
    print("Analysis complete.")


if __name__ == "__main__":
    main()
