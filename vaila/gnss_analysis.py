"""
Script para ler arquivos GPX e exportar os dados para um arquivo CSV.

Este script lê um arquivo GPX contendo informações de latitude, longitude, elevação, tempo, velocidade (speed) e cadência (cad).
Os dados são extraídos e exportados para um arquivo CSV, incluindo uma coluna adicional chamada 'time_seconds' que
representa o tempo decorrido em segundos desde o início do percurso.

Uso:
    python readgpx.py <arquivo_gpx> <arquivo_csv>

Argumentos:
    <arquivo_gpx>   Caminho para o arquivo GPX de entrada.
    <arquivo_csv>   Caminho para o arquivo CSV de saída.

Autor: Seu Nome
Data: 2024-10-13
"""

import csv
import logging
import sys
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox
from typing import Any

import gpxpy
import pandas as pd

try:
    import simplekml
except ImportError:
    simplekml = None
import os

import folium
from dateutil.parser import parse as parse_date

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def read_gpx_file(gpx_file_name: str) -> list[dict[str, Any]]:
    """
    Lê um arquivo GPX e retorna uma lista de pontos com dados extraídos.
    """
    points = []

    try:
        with open(gpx_file_name) as gpx_file:
            gpx = gpxpy.parse(gpx_file)

        for track in gpx.tracks:
            for segment in track.segments:
                if not segment.points:
                    continue  # Pula segmentos vazios

                # Obtém o start_time a partir do primeiro ponto com tempo válido
                start_time = next((p.time for p in segment.points if p.time), None)
                if not start_time:
                    logging.warning("Segmento sem tempo inicial válido. Pulando segmento.")
                    continue

                for point in segment.points:
                    time_seconds = None
                    if point.time:
                        time_seconds = (point.time - start_time).total_seconds()

                    # Extraindo dados de extensões (velocidade e cadência)
                    speed = None
                    cad = None
                    if point.extensions:
                        for ext in point.extensions:
                            if ext.tag.endswith("TrackPointExtension"):
                                speed_text = ext.findtext(".//{*}speed")
                                cad_text = ext.findtext(".//{*}cad")
                                # Conversão segura para float
                                speed = float(speed_text) if speed_text else None
                                cad = (
                                    float(cad_text) if cad_text else None
                                )  # Corrigido para aceitar floats

                    # Adiciona os dados do ponto à lista de pontos
                    points.append(
                        {
                            "latitude": point.latitude,
                            "longitude": point.longitude,
                            "elevation": point.elevation,
                            "time": point.time.isoformat() if point.time else None,
                            "speed": speed,
                            "cad": cad,
                            "time_seconds": time_seconds,
                        }
                    )

    except FileNotFoundError:
        logging.error(f"Erro: Arquivo '{gpx_file_name}' não encontrado.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Erro ao ler o arquivo GPX: {e}")
        sys.exit(1)

    return points


def export_to_csv(points: list[dict[str, Any]], output_csv_file: str) -> None:
    """
    Exporta a lista de pontos para um arquivo CSV.
    """
    fieldnames = [
        "latitude",
        "longitude",
        "elevation",
        "time",
        "speed",
        "cad",
        "time_seconds",
    ]

    try:
        with open(output_csv_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for point in points:
                writer.writerow(point)

        logging.info(f"Dados exportados com sucesso para {output_csv_file}")

    except Exception as e:
        logging.error(f"Erro ao escrever o arquivo CSV: {e}")
        sys.exit(1)


def convert_gpx_to_csv():
    """Converts a GPX file to CSV."""
    gpx_file = filedialog.askopenfilename(
        title="Select GPX File", filetypes=[("GPX Files", "*.gpx")]
    )
    if not gpx_file:
        return
    base = os.path.basename(gpx_file)
    name_no_ext, _ = os.path.splitext(base)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(os.path.dirname(gpx_file), f"{name_no_ext}_{timestamp}.csv")

    try:
        points = read_gpx_file(gpx_file)
        export_to_csv(points, output_file)
        messagebox.showinfo("Success", f"GPX data converted and saved to {output_file}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


def convert_gpx_to_kml():
    """Converts a GPX file to KML."""
    gpx_file = filedialog.askopenfilename(
        title="Select GPX File", filetypes=[("GPX Files", "*.gpx")]
    )
    if not gpx_file:
        return
    base = os.path.basename(gpx_file)
    name_no_ext, _ = os.path.splitext(base)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(os.path.dirname(gpx_file), f"{name_no_ext}_{timestamp}.kml")

    try:
        points = read_gpx_file(gpx_file)
        if simplekml is None:
            messagebox.showerror("Error", "simplekml library not installed.")
            return
        kml = simplekml.Kml()
        for point in points:
            kml.newpoint(coords=[(point["longitude"], point["latitude"])])
        kml.save(output_file)
        messagebox.showinfo("Success", f"GPX data converted and saved to {output_file}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


def convert_gpx_to_kmz():
    """Converts a GPX file to KMZ."""
    gpx_file = filedialog.askopenfilename(
        title="Select GPX File", filetypes=[("GPX Files", "*.gpx")]
    )
    if not gpx_file:
        return
    base = os.path.basename(gpx_file)
    name_no_ext, _ = os.path.splitext(base)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(os.path.dirname(gpx_file), f"{name_no_ext}_{timestamp}.kmz")

    try:
        points = read_gpx_file(gpx_file)
        if simplekml is None:
            messagebox.showerror("Error", "simplekml library not installed.")
            return
        kml = simplekml.Kml()
        for point in points:
            kml.newpoint(coords=[(point["longitude"], point["latitude"])])
        kml.savekmz(output_file)
        messagebox.showinfo("Success", f"GPX data converted and saved to {output_file}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


def convert_csv_to_gpx():
    """Converts a CSV to GPX."""
    csv_file = filedialog.askopenfilename(
        title="Select CSV File", filetypes=[("CSV Files", "*.csv")]
    )
    if not csv_file:
        return
    base = os.path.basename(csv_file)
    name_no_ext, _ = os.path.splitext(base)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(os.path.dirname(csv_file), f"{name_no_ext}_{timestamp}.gpx")

    try:
        df = pd.read_csv(csv_file)
        import gpxpy

        gpx = gpxpy.gpx.GPX()
        for _, row in df.iterrows():
            t = parse_date(row["time"]) if "time" in row and pd.notna(row["time"]) else None
            wp = gpxpy.gpx.GPXWaypoint(
                latitude=row["latitude"],
                longitude=row["longitude"],
                elevation=row.get("elevation", None),
                time=t,
            )
            gpx.waypoints.append(wp)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(gpx.to_xml())
        messagebox.showinfo("Success", f"CSV data converted and saved to {output_file}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def convert_csv_to_kml():
    """Converts a CSV to KML."""
    csv_file = filedialog.askopenfilename(
        title="Select CSV File", filetypes=[("CSV Files", "*.csv")]
    )
    if not csv_file:
        return
    base = os.path.basename(csv_file)
    name_no_ext, _ = os.path.splitext(base)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(os.path.dirname(csv_file), f"{name_no_ext}_{timestamp}.kml")

    try:
        df = pd.read_csv(csv_file)
        if simplekml is None:
            messagebox.showerror("Error", "simplekml library not installed.")
            return
        kml = simplekml.Kml()
        for _, row in df.iterrows():
            kml.newpoint(coords=[(row["longitude"], row["latitude"])])
        kml.save(output_file)
        messagebox.showinfo("Success", f"CSV data converted and saved to {output_file}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def convert_csv_to_kmz():
    """Converts a CSV to KMZ."""
    csv_file = filedialog.askopenfilename(
        title="Select CSV File", filetypes=[("CSV Files", "*.csv")]
    )
    if not csv_file:
        return
    base = os.path.basename(csv_file)
    name_no_ext, _ = os.path.splitext(base)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(os.path.dirname(csv_file), f"{name_no_ext}_{timestamp}.kmz")

    try:
        df = pd.read_csv(csv_file)
        if simplekml is None:
            messagebox.showerror("Error", "simplekml library not installed.")
            return
        kml = simplekml.Kml()
        for _, row in df.iterrows():
            kml.newpoint(coords=[(row["longitude"], row["latitude"])])
        kml.savekmz(output_file)
        messagebox.showinfo("Success", f"CSV data converted and saved to {output_file}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def csv_conversion():
    """Opens a dialog to select CSV conversion type."""
    conv_window = tk.Toplevel()
    conv_window.title("CSV Conversion Options")
    conv_window.geometry("400x200")
    label = tk.Label(conv_window, text="Select output format:", font=("Arial", 14))
    label.pack(pady=10)
    mode = tk.StringVar(value="GPX")
    rb1 = tk.Radiobutton(conv_window, text="GPX", variable=mode, value="GPX")
    rb1.pack()
    rb2 = tk.Radiobutton(conv_window, text="KML", variable=mode, value="KML")
    rb2.pack()
    rb3 = tk.Radiobutton(conv_window, text="KMZ", variable=mode, value="KMZ")
    rb3.pack()

    def do_conversion():
        fmt = mode.get()
        conv_window.destroy()
        if fmt == "GPX":
            convert_csv_to_gpx()
        elif fmt == "KML":
            convert_csv_to_kml()
        elif fmt == "KMZ":
            convert_csv_to_kmz()

    btn = tk.Button(conv_window, text="Convert", command=do_conversion)
    btn.pack(pady=10)


def distance_analysis():
    messagebox.showinfo(
        "Distance Analysis", "Distance analysis functionality is not implemented yet."
    )


def unit_conversion():
    messagebox.showinfo("Unit Conversion", "Unit conversion functionality is not implemented yet.")


def speed_analysis():
    messagebox.showinfo("Speed Analysis", "Speed analysis functionality is not implemented yet.")


def trajectory_analysis():
    messagebox.showinfo(
        "Trajectory Analysis",
        "Trajectory analysis functionality is not implemented yet.",
    )


def spatial_analysis():
    messagebox.showinfo(
        "Spatial Analysis", "Spatial analysis functionality is not implemented yet."
    )


def time_series_analysis():
    messagebox.showinfo(
        "Time Series Analysis",
        "Time series analysis functionality is not implemented yet.",
    )


def data_normalization():
    messagebox.showinfo(
        "Data Normalization", "Data normalization functionality is not implemented yet."
    )


def data_visualization():
    messagebox.showinfo(
        "Data Visualization", "Data visualization functionality is not implemented yet."
    )


def batch_processing():
    messagebox.showinfo(
        "Batch Processing", "Batch processing functionality is not implemented yet."
    )


def plot_gnss_data():
    """Plots GNSS GPS data from CSV, GPX, KML, or KMZ."""
    import matplotlib.pyplot as plt

    file_path = filedialog.askopenfilename(
        title="Select GNSS Data File",
        filetypes=[("GNSS Files", "*.csv *.gpx *.kml *.kmz")],
    )
    if not file_path:
        return
    ext = os.path.splitext(file_path)[1].lower()
    lats = []
    lons = []
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
            if "latitude" in df.columns and "longitude" in df.columns:
                lats = df["latitude"].tolist()
                lons = df["longitude"].tolist()
            else:
                messagebox.showerror("Error", "CSV must have 'latitude' and 'longitude' columns.")
                return
        elif ext == ".gpx":
            with open(file_path) as f:
                gpx = gpxpy.parse(f)
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        lats.append(point.latitude)
                        lons.append(point.longitude)
        elif ext in [".kml", ".kmz"]:
            import xml.etree.ElementTree as ET

            if ext == ".kmz":
                import zipfile

                with zipfile.ZipFile(file_path, "r") as z:
                    with z.open("doc.kml") as kml_file:
                        tree = ET.parse(kml_file)
            else:
                tree = ET.parse(file_path)
            root_xml = tree.getroot()
            ns = {"kml": "http://www.opengis.net/kml/2.2"}
            for coord in root_xml.findall(".//kml:coordinates", ns):
                coords_str = coord.text.strip()
                for pair in coords_str.split():
                    parts = pair.split(",")
                    if len(parts) >= 2:
                        lons.append(float(parts[0]))
                        lats.append(float(parts[1]))
        else:
            messagebox.showerror("Error", "Unsupported file type.")
            return
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while reading the file: {e}")
        return

    if not lats or not lons:
        messagebox.showerror("Error", "No valid position data found.")
        return

    # Static Map using Matplotlib (for quick visualization)
    plt.figure(figsize=(10, 8))
    plt.plot(lons, lats, linestyle="-", marker="o", color="blue", markersize=5, linewidth=2)
    plt.scatter(lons[0], lats[0], c="green", s=100, label="Start")
    plt.scatter(lons[-1], lats[-1], c="red", s=100, label="End")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("GNSS Route Map (Static)")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.show()

    # Interactive Map using Folium (see [Folium Getting Started](https://python-visualization.github.io/folium/latest/getting_started.html))
    if lats and lons:
        m = folium.Map(location=[lats[0], lons[0]], zoom_start=12)
        folium.PolyLine(list(zip(lats, lons)), color="blue", weight=2.5, opacity=1).add_to(m)
        folium.Marker([lats[0], lons[0]], popup="Start", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker([lats[-1], lons[-1]], popup="End", icon=folium.Icon(color="red")).add_to(m)
        # Save the map in the same directory as the input file
        base = os.path.basename(file_path)
        name_no_ext, _ = os.path.splitext(base)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        map_file = os.path.join(
            os.path.dirname(file_path), f"{name_no_ext}_{timestamp}_interactive.html"
        )
        m.save(map_file)
        print(f"Interactive map saved as {map_file}")
    else:
        print("No valid GNSS data available.")


def gpx_conversion():
    """Opens a dialog to select GPX conversion type."""
    conv_window = tk.Toplevel()
    conv_window.title("GPX Conversion Options")
    conv_window.geometry("400x200")
    label = tk.Label(conv_window, text="Convert GPX to:", font=("Arial", 14))
    label.pack(pady=10)
    mode = tk.StringVar(value="CSV")
    tk.Radiobutton(conv_window, text="CSV", variable=mode, value="CSV").pack()
    tk.Radiobutton(conv_window, text="KML", variable=mode, value="KML").pack()
    tk.Radiobutton(conv_window, text="KMZ", variable=mode, value="KMZ").pack()

    def do_conversion():
        fmt = mode.get()
        conv_window.destroy()
        if fmt == "CSV":
            convert_gpx_to_csv()
        elif fmt == "KML":
            convert_gpx_to_kml()
        elif fmt == "KMZ":
            convert_gpx_to_kmz()

    tk.Button(conv_window, text="Convert", command=do_conversion).pack(pady=10)


def kml_conversion():
    """Opens a dialog to select KML conversion type."""
    conv_window = tk.Toplevel()
    conv_window.title("KML Conversion Options")
    conv_window.geometry("400x200")
    label = tk.Label(conv_window, text="Convert KML to:", font=("Arial", 14))
    label.pack(pady=10)
    mode = tk.StringVar(value="CSV")
    tk.Radiobutton(conv_window, text="GPX", variable=mode, value="GPX").pack()
    tk.Radiobutton(conv_window, text="CSV", variable=mode, value="CSV").pack()
    tk.Radiobutton(conv_window, text="KMZ", variable=mode, value="KMZ").pack()

    def do_conversion():
        fmt = mode.get()
        conv_window.destroy()
        if fmt == "GPX":
            convert_kml_to_gpx()
        elif fmt == "CSV":
            convert_kml_to_csv()
        elif fmt == "KMZ":
            convert_kml_to_kmz()

    tk.Button(conv_window, text="Convert", command=do_conversion).pack(pady=10)


def kmz_conversion():
    """Opens a dialog to select KMZ conversion type."""
    conv_window = tk.Toplevel()
    conv_window.title("KMZ Conversion Options")
    conv_window.geometry("400x200")
    label = tk.Label(conv_window, text="Convert KMZ to:", font=("Arial", 14))
    label.pack(pady=10)
    mode = tk.StringVar(value="CSV")
    tk.Radiobutton(conv_window, text="GPX", variable=mode, value="GPX").pack()
    tk.Radiobutton(conv_window, text="CSV", variable=mode, value="CSV").pack()
    tk.Radiobutton(conv_window, text="KML", variable=mode, value="KML").pack()

    def do_conversion():
        fmt = mode.get()
        conv_window.destroy()
        if fmt == "GPX":
            convert_kmz_to_gpx()
        elif fmt == "CSV":
            convert_kmz_to_csv()
        elif fmt == "KML":
            convert_kmz_to_kml()

    tk.Button(conv_window, text="Convert", command=do_conversion).pack(pady=10)


def convert_kml_to_gpx():
    """Converts a KML file to GPX."""
    import xml.etree.ElementTree as ET

    import gpxpy.gpx

    kml_file = filedialog.askopenfilename(
        title="Select KML File", filetypes=[("KML Files", "*.kml")]
    )
    if not kml_file:
        return
    try:
        tree = ET.parse(kml_file)
        root = tree.getroot()
        ns = {"kml": "http://www.opengis.net/kml/2.2"}
        coords_elements = root.findall(".//kml:coordinates", ns)
        if not coords_elements:
            messagebox.showerror("Error", "No coordinates found in the KML file.")
            return
        gpx = gpxpy.gpx.GPX()
        for elem in coords_elements:
            coords_text = elem.text.strip()
            # Coordinates may be separated by whitespace
            for pair in coords_text.split():
                parts = pair.split(",")
                if len(parts) >= 2:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    elev = float(parts[2]) if len(parts) >= 3 and parts[2] else None
                    wp = gpxpy.gpx.GPXWaypoint(latitude=lat, longitude=lon, elevation=elev)
                    gpx.waypoints.append(wp)
        base = os.path.basename(kml_file)
        name_no_ext, _ = os.path.splitext(base)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(os.path.dirname(kml_file), f"{name_no_ext}_{timestamp}.gpx")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(gpx.to_xml())
        messagebox.showinfo("Success", f"KML data converted and saved to {output_file}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def convert_kml_to_csv():
    """Converts a KML file to CSV."""
    import csv
    import xml.etree.ElementTree as ET

    kml_file = filedialog.askopenfilename(
        title="Select KML File", filetypes=[("KML Files", "*.kml")]
    )
    if not kml_file:
        return
    try:
        tree = ET.parse(kml_file)
        root = tree.getroot()
        ns = {"kml": "http://www.opengis.net/kml/2.2"}
        coords_elements = root.findall(".//kml:coordinates", ns)
        if not coords_elements:
            messagebox.showerror("Error", "No coordinates found in the KML file.")
            return
        rows = []
        for elem in coords_elements:
            coords_text = elem.text.strip()
            for pair in coords_text.split():
                parts = pair.split(",")
                if len(parts) >= 2:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    elev = float(parts[2]) if len(parts) >= 3 and parts[2] else ""
                    rows.append({"latitude": lat, "longitude": lon, "elevation": elev})
        base = os.path.basename(kml_file)
        name_no_ext, _ = os.path.splitext(base)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(os.path.dirname(kml_file), f"{name_no_ext}_{timestamp}.csv")
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["latitude", "longitude", "elevation"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        messagebox.showinfo("Success", f"KML data converted and saved to {output_file}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def convert_kml_to_kmz():
    """Converts a KML file to KMZ."""
    import zipfile

    kml_file = filedialog.askopenfilename(
        title="Select KML File", filetypes=[("KML Files", "*.kml")]
    )
    if not kml_file:
        return
    try:
        base = os.path.basename(kml_file)
        name_no_ext, _ = os.path.splitext(base)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(os.path.dirname(kml_file), f"{name_no_ext}_{timestamp}.kmz")
        with zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(kml_file, arcname="doc.kml")
        messagebox.showinfo("Success", f"KML data compressed and saved to {output_file}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def convert_kmz_to_gpx():
    """Converts a KMZ file to GPX."""
    import xml.etree.ElementTree as ET
    import zipfile

    import gpxpy.gpx

    kmz_file = filedialog.askopenfilename(
        title="Select KMZ File", filetypes=[("KMZ Files", "*.kmz")]
    )
    if not kmz_file:
        return
    try:
        with zipfile.ZipFile(kmz_file, "r") as z, z.open("doc.kml") as kml_file:
            tree = ET.parse(kml_file)
        root = tree.getroot()
        ns = {"kml": "http://www.opengis.net/kml/2.2"}
        coords_elements = root.findall(".//kml:coordinates", ns)
        if not coords_elements:
            messagebox.showerror("Error", "No coordinates found in the KMZ file.")
            return
        gpx = gpxpy.gpx.GPX()
        for elem in coords_elements:
            coords_text = elem.text.strip()
            for pair in coords_text.split():
                parts = pair.split(",")
                if len(parts) >= 2:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    elev = float(parts[2]) if len(parts) >= 3 and parts[2] else None
                    wp = gpxpy.gpx.GPXWaypoint(latitude=lat, longitude=lon, elevation=elev)
                    gpx.waypoints.append(wp)
        base = os.path.basename(kmz_file)
        name_no_ext, _ = os.path.splitext(base)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(os.path.dirname(kmz_file), f"{name_no_ext}_{timestamp}.gpx")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(gpx.to_xml())
        messagebox.showinfo("Success", f"KMZ data converted and saved to {output_file}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def convert_kmz_to_csv():
    """Converts a KMZ file to CSV."""
    import csv
    import xml.etree.ElementTree as ET
    import zipfile

    kmz_file = filedialog.askopenfilename(
        title="Select KMZ File", filetypes=[("KMZ Files", "*.kmz")]
    )
    if not kmz_file:
        return
    try:
        with zipfile.ZipFile(kmz_file, "r") as z, z.open("doc.kml") as kml_file:
            tree = ET.parse(kml_file)
        root = tree.getroot()
        ns = {"kml": "http://www.opengis.net/kml/2.2"}
        coords_elements = root.findall(".//kml:coordinates", ns)
        if not coords_elements:
            messagebox.showerror("Error", "No coordinates found in the KMZ file.")
            return
        rows = []
        for elem in coords_elements:
            coords_text = elem.text.strip()
            for pair in coords_text.split():
                parts = pair.split(",")
                if len(parts) >= 2:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    elev = float(parts[2]) if len(parts) >= 3 and parts[2] else ""
                    rows.append({"latitude": lat, "longitude": lon, "elevation": elev})
        base = os.path.basename(kmz_file)
        name_no_ext, _ = os.path.splitext(base)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(os.path.dirname(kmz_file), f"{name_no_ext}_{timestamp}.csv")
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["latitude", "longitude", "elevation"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        messagebox.showinfo("Success", f"KMZ data converted and saved to {output_file}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def convert_kmz_to_kml():
    """Converts a KMZ file to KML."""
    import zipfile

    kmz_file = filedialog.askopenfilename(
        title="Select KMZ File", filetypes=[("KMZ Files", "*.kmz")]
    )
    if not kmz_file:
        return
    try:
        with zipfile.ZipFile(kmz_file, "r") as z, z.open("doc.kml") as kml_file:
            content = kml_file.read()
        base = os.path.basename(kmz_file)
        name_no_ext, _ = os.path.splitext(base)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(os.path.dirname(kmz_file), f"{name_no_ext}_{timestamp}.kml")
        with open(output_file, "wb") as f:
            f.write(content)
        messagebox.showinfo("Success", f"KMZ extracted and saved as KML to {output_file}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def run_gnss_analysis_gui():
    root = tk.Tk()
    root.title("Select Analysis Option")
    root.geometry("600x400")

    header = tk.Label(root, text="Select the type of analysis to perform:", font=("Arial", 16))
    header.pack(pady=20)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    buttons = [
        ("GPX Conversion", gpx_conversion),
        ("CSV Conversion", csv_conversion),
        ("KML Conversion", kml_conversion),
        ("KMZ Conversion", kmz_conversion),
        ("Plot GNSS Data", plot_gnss_data),
        ("Distance Analysis", distance_analysis),
        ("Unit Conversion", unit_conversion),
        ("Speed Analysis", speed_analysis),
        ("Trajectory Analysis", trajectory_analysis),
        ("Spatial Analysis", spatial_analysis),
        ("Time Series Analysis", time_series_analysis),
        ("Data Normalization", data_normalization),
        ("Data Visualization", data_visualization),
        ("Batch Processing", batch_processing),
    ]

    rows = 5
    cols = 2
    for idx, (label, cmd) in enumerate(buttons):
        r = idx // cols
        c = idx % cols
        btn = tk.Button(button_frame, text=label, width=25, height=2, command=cmd)
        btn.grid(row=r, column=c, padx=10, pady=10)

    root.mainloop()


if __name__ == "__main__":
    run_gnss_analysis_gui()
