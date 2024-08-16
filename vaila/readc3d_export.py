import os
import dask.dataframe as dd
import pandas as pd
from ezc3d import c3d
from datetime import datetime
from tkinter import Tk, filedialog, messagebox
from tqdm import tqdm


def importc3d(dat):
    datac3d = c3d(dat)
    print(f"\nProcessing file: {dat}")
    print(f'Number of markers = {datac3d["parameters"]["POINT"]["USED"]["value"][0]}')
    point_data = datac3d["data"]["points"]
    points_residuals = datac3d["data"]["meta_points"]["residuals"]
    analogs = datac3d["data"]["analogs"]
    marker_labels = datac3d["parameters"]["POINT"]["LABELS"]["value"]
    analog_labels = datac3d["parameters"]["ANALOG"]["LABELS"]["value"]
    markers = point_data[0:3, :, :].T.reshape(-1, len(marker_labels) * 3)
    marker_freq = datac3d["header"]["points"]["frame_rate"]
    analog_freq = datac3d["header"]["analogs"]["frame_rate"]

    # Print summary information
    num_analog_channels = datac3d["parameters"]["ANALOG"]["USED"]["value"][0]
    print(f"Number of marker labels = {len(marker_labels)}")
    print(f"Number of analog channels = {num_analog_channels}")
    print(f"Marker frequency = {marker_freq} Hz")
    print(f"Analog frequency = {analog_freq} Hz")

    return (
        markers,
        marker_labels,
        marker_freq,
        analogs,
        points_residuals,
        analog_labels,
        analog_freq,
    )


def save_info_file(
    marker_labels, marker_freq, analog_labels, analog_freq, dir_name, file_name
):
    print(f"Saving info file for {file_name}")
    info_file_path = os.path.join(dir_name, f"{file_name}.info")
    with open(info_file_path, "w") as f:
        f.write(f"Marker frequency: {marker_freq} Hz\n")
        f.write(f"Analog frequency: {analog_freq} Hz\n\n")
        f.write("Marker labels:\n")
        for label in marker_labels:
            f.write(f"{label}\n")
        f.write("\nAnalog labels:\n")
        for label in analog_labels:
            f.write(f"{label}\n")


def save_to_files(
    markers,
    marker_labels,
    marker_freq,
    analogs,
    points_residuals,
    analog_labels,
    analog_freq,
    file_name,
    output_dir,
    save_excel,
):
    print(f"Saving data to files for {file_name}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = os.path.join(
        output_dir, "multimodal_c3d_to_csv", f"{file_name}_{timestamp}"
    )
    os.makedirs(dir_name, exist_ok=True)

    # Prepare marker columns
    marker_columns = [
        f"{label}_{axis}" for label in marker_labels for axis in ["X", "Y", "Z"]
    ]
    markers_df = pd.DataFrame(markers, columns=marker_columns)

    # Add time column to markers_df
    num_marker_frames = markers_df.shape[0]
    marker_time_column = pd.Series(
        [f"{i / marker_freq:.3f}" for i in range(num_marker_frames)], name="Time"
    )
    markers_df.insert(0, "Time", marker_time_column)

    # Convert to Dask DataFrame
    markers_ddf = dd.from_pandas(markers_df, npartitions=10)

    # Prepare analog data
    analogs = analogs.squeeze(
        axis=0
    ).T  # Remove the extra dimension and transpose to have time in rows
    analogs_df = pd.DataFrame(analogs, columns=analog_labels)

    # Add time column to analogs_df
    num_analog_frames = analogs_df.shape[0]
    analog_time_column = pd.Series(
        [f"{i / analog_freq:.3f}" for i in range(num_analog_frames)], name="Time"
    )
    analogs_df.insert(0, "Time", analog_time_column)

    # Convert to Dask DataFrame
    analogs_ddf = dd.from_pandas(analogs_df, npartitions=10)

    # Prepare points residuals
    points_residuals_df = pd.DataFrame(points_residuals.squeeze(axis=0).T)

    # Add time column to points_residuals_df
    num_points_frames = points_residuals_df.shape[0]
    points_time_column = pd.Series(
        [f"{i / marker_freq:.3f}" for i in range(num_points_frames)], name="Time"
    )
    points_residuals_df.insert(0, "Time", points_time_column)

    # Convert to Dask DataFrame
    points_residuals_ddf = dd.from_pandas(points_residuals_df, npartitions=10)

    # Save to CSV
    markers_ddf.to_csv(
        os.path.join(dir_name, f"{file_name}_markers.csv"),
        index=False,
        single_file=True,
    )
    analogs_ddf.to_csv(
        os.path.join(dir_name, f"{file_name}_analogs.csv"),
        index=False,
        single_file=True,
    )
    points_residuals_ddf.to_csv(
        os.path.join(dir_name, f"{file_name}_points_residuals.csv"),
        index=False,
        single_file=True,
    )

    if save_excel:
        # Save to Excel (using pandas)
        print("Saving to Excel. This process can take a long time...")
        with pd.ExcelWriter(
            os.path.join(dir_name, f"{file_name}.xlsx"), engine="openpyxl"
        ) as writer:
            markers_ddf.compute().to_excel(
                writer, sheet_name="Markers", index=False, float_format="%.7f"
            )
            analogs_ddf.compute().to_excel(
                writer, sheet_name="Analogs", index=False, float_format="%.7f"
            )
            points_residuals_ddf.compute().to_excel(
                writer, sheet_name="Points Residuals", index=False, float_format="%.7f"
            )

    # Save info file
    save_info_file(
        marker_labels, marker_freq, analog_labels, analog_freq, dir_name, file_name
    )

    print(f"Files for {file_name} saved successfully!")


def convert_c3d_to_csv():
    root = Tk()
    root.withdraw()  # Hide the root window

    # Perguntar ao usuário se deseja salvar como Excel
    save_excel = messagebox.askyesno(
        "Save as Excel",
        "Do you want to save the data as Excel files? This process can be very slow.",
    )
    print(f"Debug: save_excel = {save_excel}")

    # Pedir diretórios de entrada e saída
    input_directory = filedialog.askdirectory(title="Select Input Directory")
    print(f"Debug: input_directory = {input_directory}")

    output_directory = filedialog.askdirectory(title="Select Output Directory")
    print(f"Debug: output_directory = {output_directory}")

    if input_directory and output_directory:
        # Find all .c3d files in the input directory and sort them
        c3d_files = sorted(
            [f for f in os.listdir(input_directory) if f.endswith(".c3d")]
        )

        # Process each .c3d file
        for c3d_file in tqdm(c3d_files, desc="Processing C3D files"):
            try:
                file_path = os.path.join(input_directory, c3d_file)
                (
                    markers,
                    marker_labels,
                    marker_freq,
                    analogs,
                    points_residuals,
                    analog_labels,
                    analog_freq,
                ) = importc3d(file_path)
                file_name = os.path.splitext(c3d_file)[0]
                save_to_files(
                    markers,
                    marker_labels,
                    marker_freq,
                    analogs,
                    points_residuals,
                    analog_labels,
                    analog_freq,
                    file_name,
                    output_directory,
                    save_excel,
                )
            except Exception as e:
                print(f"Error processing {c3d_file}: {e}")

        print("All files have been processed and saved successfully!")
        messagebox.showinfo(
            "Information", "C3D files conversion completed successfully!"
        )
    else:
        print("Input or output directory not selected.")
        messagebox.showwarning("Warning", "Input or output directory not selected.")


if __name__ == "__main__":
    convert_c3d_to_csv()
