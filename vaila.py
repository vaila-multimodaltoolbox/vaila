"""
vaila.py

Name: Paulo Santiago
Date: 9 Aug 2024

Description:
Main application script for the vailá Multimodal Toolbox. This script provides
a Tkinter-based GUI to access various tools and functionalities for multimodal
analysis, including file management, data analysis, and visualization.

Version: 0.5

Changelog:
v0.5 - Added getpixelvideo load csv data
v0.4 - Changed location rec2d one dlt2d and dlt2d
v0.3 - Changed markerless 2D analysis to use a directory selection dialog.
v0.2 - Added Plot 2D button for 2D plotting of CSV or C3D files using Matplotlib.
       Improved exit functionality to ensure proper shutdown of the application.
       Refactored code for better readability and maintainability.
v0.1 - Initial version with basic GUI layout and functionality.
"""

import os
import signal
import platform
from rich import print
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
from PIL import Image, ImageTk

from vaila import (
    cluster_analysis,
    imu_analysis,
    markerless_2D_analysis,
    markerless_3D_analysis,
    mocap_analysis,
    forceplate_analysis,
    gnss_analysis,
    convert_c3d_to_csv,
    convert_csv_to_c3d,
    rearrange_data_in_directory,
    run_drawboxe,
    count_frames_in_videos,
    import_file,
    export_file,
    copy_file,
    move_file,
    remove_file,
    rename_files,
    tree_file,
    find_file,
    transfer_file,
    show_c3d,
    sync_videos,
    VideoProcessor,
    compress_videos_h264_gui,
    compress_videos_h265_gui,
    cut_videos,
    show_csv,
    getpixelvideo,
    dlt2d,
    rec2d,
    rec2d_one_dlt2d,
    show_vaila_message,
    emg_labiocom,
    plot_2d,
    plot_3d,
    process_videos_gui,
)

text = """
:::::::::'##::::'##::::'###::::'####:'##::::::::::'###::::'####::::::::::
::::::::: ##:::: ##:::'## ##:::. ##:: ##:::::::::'## ##::: ####::::::::::
::::::::: ##:::: ##::'##:. ##::: ##:: ##::::::::'##:. ##::. ##:::::::::::
::::::::: ##:::: ##:'##:::. ##:: ##:: ##:::::::'##:::. ##:'##::::::::::::
:::::::::. ##:: ##:: #########:: ##:: ##::::::: #########:..:::::::::::::
::::::::::. ## ##::: ##.... ##:: ##:: ##::::::: ##.... ##::::::::::::::::
:::::::::::. ###:::: ##:::: ##:'####: ########: ##:::: ##::::::::::::::::
::::::::::::...:::::..:::::..::....::........::..:::::..:::::::::::::::::
Mocap fullbody_c3d        Markerless_3D_videos       Markerless_2D_video
                  \                |                /
                   v               v               v
            +-------------------------------------------+
IMU_csv --> |          vailá - multimodaltoolbox        | <-- Cluster_csv
            +-------------------------------------------+
                                  |
                                  v
                   +-----------------------------+
                   |           Results           |
                   +-----------------------------+
                                  |
                                  v
                       +---------------------+
                       | Visualization/Graph |
                       +---------------------+
============================ File Manager ================================
 Rename | Import | Export | Copy | Move | Remove | Tree | Find | Transfer
========================== Available Multimodal ==========================
1. IMU Analysis
2. Kinematic Cluster Analysis
3. Kinematic Motion Capture Full Body Analysis
4. Markerless 2D with video
5. Markerless 3D with multiple videos
============================== Available Tools ===========================
1. Edit CSV
2. Convert C3D data to CSV
3. Metadata info
4. Cut videos based on list
5. Draw a black box around videos
6. Compress videos to HEVC (H.265)
7. Compress videos to H.264
8. Plot 2D

Type 'h' for help or 'exit' to quit.

Choose an analysis option or file manager command:
"""

print(text)


if platform.system() == "Darwin":
    try:
        import AppKit
    except ImportError:
        AppKit = None


class Vaila(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("vailá - 0.1.0")
        self.geometry("1280x720")

        # Set window icon based on OS
        icon_path_ico = os.path.join(
            os.path.dirname(__file__), "vaila", "images", "vaila.ico"
        )
        icon_path_png = os.path.join(
            os.path.dirname(__file__), "vaila", "images", "vaila_ico_mac.png"
        )

        if platform.system() == "Windows":
            self.iconbitmap(icon_path_ico)
        else:
            img = Image.open(icon_path_png)
            img = ImageTk.PhotoImage(img)
            self.iconphoto(True, img)

        # Set application name for macOS dock
        if platform.system() == "Darwin" and AppKit is not None:
            AppKit.NSBundle.mainBundle().infoDictionary()["CFBundleName"] = "Vaila"

        self.create_widgets()

    def create_widgets(self):
        button_width = 12  # Define a largura dos botões

        # Header with program name and description
        header_frame = tk.Frame(self)
        header_frame.pack(pady=10)

        # Load and place the image
        image_path_preto = os.path.join(
            os.path.dirname(__file__), "vaila", "images", "vaila_logo.png"
        )
        preto_image = Image.open(image_path_preto)
        preto_image = preto_image.resize((87, 87), Image.LANCZOS)
        preto_photo = ImageTk.PhotoImage(preto_image)

        preto_label = tk.Label(header_frame, image=preto_photo)
        preto_label.image = preto_photo
        preto_label.pack(side="left", padx=10)

        header_label = tk.Label(
            header_frame,
            text="vailá - Multimodal Toolbox",
            font=("default", 29, "bold"),
            anchor="center",
        )
        header_label.pack(side="left")

        # Subheader with hyperlink for "vailá"
        subheader_frame = tk.Frame(self)
        subheader_frame.pack(pady=5)

        subheader_label1 = tk.Label(
            subheader_frame,
            text="Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox",
            font=("default", 15),
            anchor="center",
        )
        subheader_label1.pack()

        subheader_label2_frame = tk.Frame(subheader_frame)
        subheader_label2_frame.pack()

        vaila_link = tk.Label(
            subheader_label2_frame,
            text="vailá",
            font=("default", 17, "italic"),
            fg="blue",
            cursor="hand2",
        )
        vaila_link.pack(side="left")
        vaila_link.bind("<Button-1>", lambda e: self.open_link())

        unleash_label = tk.Label(
            subheader_label2_frame,
            text=" and unleash your imagination!",
            font=("default", 17),
            anchor="center",
        )
        unleash_label.pack(side="left")

        # Create a canvas to add scrollbar
        canvas = tk.Canvas(self)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a scrollbar to the canvas
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill="y")

        # Create a frame inside the canvas to hold all content
        scrollable_frame = tk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        # Add the scrollable frame to the canvas
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # File Manager Frame
        file_manager_frame = tk.LabelFrame(
            scrollable_frame,
            text="File Manager",
            padx=5,
            pady=5,
            font=("default", 17),
            labelanchor="n",
        )
        file_manager_frame.pack(pady=10, fill="x")

        file_manager_btn_frame = tk.Frame(file_manager_frame)
        file_manager_btn_frame.pack(pady=5)

        rename_btn = tk.Button(
            file_manager_btn_frame,
            text="Rename",
            command=self.rename_files,
            width=button_width,
        )
        import_btn = tk.Button(
            file_manager_btn_frame,
            text="Import",
            command=self.import_file,
            width=button_width,
        )
        export_btn = tk.Button(
            file_manager_btn_frame,
            text="Export",
            command=self.export_file,
            width=button_width,
        )
        copy_btn = tk.Button(
            file_manager_btn_frame,
            text="Copy",
            command=self.copy_file,
            width=button_width,
        )
        move_btn = tk.Button(
            file_manager_btn_frame,
            text="Move",
            command=self.move_file,
            width=button_width,
        )
        remove_btn = tk.Button(
            file_manager_btn_frame,
            text="Remove",
            command=self.remove_file,
            width=button_width,
        )
        tree_btn = tk.Button(
            file_manager_btn_frame,
            text="Tree",
            command=self.tree_file,
            width=button_width,
        )
        find_btn = tk.Button(
            file_manager_btn_frame,
            text="Find",
            command=self.find_file,
            width=button_width,
        )
        transfer_btn = tk.Button(
            file_manager_btn_frame,
            text="Transfer",
            command=self.transfer_file,
            width=button_width,
        )

        rename_btn.pack(side="left", padx=2, pady=2)
        import_btn.pack(side="left", padx=2, pady=2)
        export_btn.pack(side="left", padx=2, pady=2)
        copy_btn.pack(side="left", padx=2, pady=2)
        move_btn.pack(side="left", padx=2, pady=2)
        remove_btn.pack(side="left", padx=2, pady=2)
        tree_btn.pack(side="left", padx=2, pady=2)
        find_btn.pack(side="left", padx=2, pady=2)
        transfer_btn.pack(side="left", padx=2, pady=2)

        # Analysis Frame
        analysis_frame = tk.LabelFrame(
            scrollable_frame,
            text="Multimodal Analysis",
            padx=5,
            pady=5,
            font=("default", 17),
            labelanchor="n",
        )
        analysis_frame.pack(pady=10, fill="x")

        # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
        ## Insert the buttons for each analysis here
        # Buttons for each Multimodal Toolbox Analysis
        # First row of buttons
        row1_frame = tk.Frame(analysis_frame)
        row1_frame.pack(fill="x")
        imu_analysis_btn = tk.Button(
            row1_frame, text="IMU", width=button_width, command=self.imu_analysis
        )
        cluster_analysis_btn = tk.Button(
            row1_frame,
            text="Motion Capture Cluster",
            width=button_width,
            command=self.cluster_analysis,
        )
        mocap_analysis_btn = tk.Button(
            row1_frame,
            text="Motion Capture Full Body",
            width=button_width,
            command=self.mocap_analysis,
        )
        markerless_2d_analysis_btn = tk.Button(
            row1_frame,
            text="Markerless 2D",
            width=button_width,
            command=self.markerless_2d_analysis,
        )
        markerless_3d_analysis_btn = tk.Button(
            row1_frame,
            text="Markerless 3D",
            width=button_width,
            command=self.markerless_3d_analysis,
        )

        imu_analysis_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        cluster_analysis_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        mocap_analysis_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        markerless_2d_analysis_btn.pack(
            side="left", expand=True, fill="x", padx=2, pady=2
        )
        markerless_3d_analysis_btn.pack(
            side="left", expand=True, fill="x", padx=2, pady=2
        )

        # Second row of buttons
        row2_frame = tk.Frame(analysis_frame)
        row2_frame.pack(fill="x")
        vector_coding_btn = tk.Button(
            row2_frame,
            text="Vector Coding",
            width=button_width,
            command=self.vector_coding,
        )
        emg_analysis_btn = tk.Button(
            row2_frame, text="EMG", width=button_width, command=self.emg_analysis
        )
        forceplate_btn = tk.Button(
            row2_frame,
            text="Force Plate",
            width=button_width,
            command=self.force_analysis,  # Altere para force_analysis
        )
        gnss_btn = tk.Button(
            row2_frame,
            text="GNSS/GPS",
            width=button_width,
            command=self.gnss_analysis,  # Altere para gnss_analysis
        )
        vaila_btn3 = tk.Button(
            row2_frame,
            text="vailá",
            width=button_width,
            command=self.show_vaila_message,
        )

        vector_coding_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        emg_analysis_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        forceplate_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        gnss_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn3.pack(side="left", expand=True, fill="x", padx=2, pady=2)

        # Third row of buttons (if needed)
        row3_frame = tk.Frame(analysis_frame)
        row3_frame.pack(fill="x")
        vaila_btn4 = tk.Button(
            row3_frame,
            text="vailá",
            width=button_width,
            command=self.show_vaila_message,
        )
        vaila_btn5 = tk.Button(
            row3_frame,
            text="vailá",
            width=button_width,
            command=self.show_vaila_message,
        )
        vaila_btn6 = tk.Button(
            row3_frame,
            text="vailá",
            width=button_width,
            command=self.show_vaila_message,
        )
        vaila_btn7 = tk.Button(
            row3_frame,
            text="vailá",
            width=button_width,
            command=self.show_vaila_message,
        )
        vaila_btn8 = tk.Button(
            row3_frame,
            text="vailá",
            width=button_width,
            command=self.show_vaila_message,
        )

        vaila_btn4.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn5.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn6.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn7.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn8.pack(side="left", expand=True, fill="x", padx=2, pady=2)

        # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
        # Tools Frame
        # Create a frame for the tools
        tools_frame = tk.LabelFrame(
            scrollable_frame,
            text="Available Tools",
            padx=5,
            pady=5,
            font=("default", 17),
            labelanchor="n",
        )
        tools_frame.pack(pady=10, fill="x")

        tools_col1 = tk.LabelFrame(
            tools_frame, text="Data Files", padx=5, pady=5, font=("default", 14)
        )
        tools_col2 = tk.LabelFrame(
            tools_frame, text="Video and Image", padx=5, pady=5, font=("default", 14)
        )
        tools_col3 = tk.LabelFrame(
            tools_frame, text="Visualization", padx=5, pady=5, font=("default", 14)
        )

        # Data Files sub-columns
        reorder_csv_btn = tk.Button(
            tools_col1,
            text="Edit CSV",
            command=self.reorder_csv_data,
            width=button_width,
        )
        convert_btn = tk.Button(
            tools_col1,
            text="CSV <--> C3D",
            command=self.convert_c3d_csv,
            width=button_width,
        )
        vaila_btn8to9 = tk.Button(
            tools_col1,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )
        dlt2d_btn = tk.Button(
            tools_col1, text="Make DLT2D", command=self.dlt2d, width=button_width
        )
        rec2d_one_btn = tk.Button(
            tools_col1,
            text="Rec2D 1DLT",
            command=self.rec2d_one_dlt2d,
            width=button_width,
        )
        rec2d_multiple_btn = tk.Button(
            tools_col1, text="Rec2D MultiDLT", command=self.rec2d, width=button_width
        )
        dlt3d_btn = tk.Button(
            tools_col1, text="Make DLT3D", command=self.dlt3d, width=button_width
        )
        rec3d_one_btn = tk.Button(
            tools_col1,
            text="Rec3D 1DLT",
            command=self.rec3d_one_dlt3d,
            width=button_width,
        )
        rec3d_multiple_btn = tk.Button(
            tools_col1, text="Rec3D MultiDLT", command=self.rec3d, width=button_width
        )
        vaila_btn9 = tk.Button(
            tools_col1,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )
        vaila_btn10 = tk.Button(
            tools_col1,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )
        vaila_btn11 = tk.Button(
            tools_col1,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )

        # Packing Data Files buttons
        reorder_csv_btn.grid(row=0, column=0, padx=2, pady=2)
        convert_btn.grid(row=0, column=1, padx=2, pady=2)
        vaila_btn8to9.grid(row=0, column=2, padx=2, pady=2)
        dlt2d_btn.grid(row=1, column=0, padx=2, pady=2)
        rec2d_one_btn.grid(row=1, column=1, padx=2, pady=2)
        rec2d_multiple_btn.grid(row=1, column=2, padx=2, pady=2)
        dlt3d_btn.grid(row=2, column=0, padx=2, pady=2)
        rec3d_one_btn.grid(row=2, column=1, padx=2, pady=2)
        rec3d_multiple_btn.grid(row=2, column=2, padx=2, pady=2)
        vaila_btn9.grid(row=3, column=0, padx=2, pady=2)
        vaila_btn10.grid(row=3, column=1, padx=2, pady=2)
        vaila_btn11.grid(row=3, column=2, padx=2, pady=2)

        tools_col1.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Video sub-columns
        extract_png_btn = tk.Button(
            tools_col2,
            text="Video <--> PNG",
            command=self.extract_png_from_videos,
            width=button_width,
        )
        cut_videos_btn = tk.Button(
            tools_col2, text="Cut Videos", command=self.cut_videos, width=button_width
        )
        draw_box_btn = tk.Button(
            tools_col2, text="Draw Box", command=self.draw_box, width=button_width
        )
        compress_videos_h264_btn = tk.Button(
            tools_col2,
            text="Compress H264",
            command=self.compress_videos_h264_gui,
            width=button_width,
        )
        compress_videos_h265_btn = tk.Button(
            tools_col2,
            text="Compress H265",
            command=self.compress_videos_h265_gui,
            width=button_width,
        )
        sync_videos_btn = tk.Button(
            tools_col2,
            text="Make Sync file",
            command=self.sync_videos,
            width=button_width,
        )
        getpixelvideo_btn = tk.Button(
            tools_col2,
            text="Get Pixel Coords",
            command=self.getpixelvideo,
            width=button_width,
        )
        count_frames_btn = tk.Button(
            tools_col2,
            text="Metadata info",
            command=self.count_frames_in_videos,
            width=button_width,
        )
        video_processing_btn = tk.Button(
            tools_col2,
            text="Merge Videos",
            command=self.process_videos_gui,
            width=button_width,
        )
        vaila_btn13 = tk.Button(
            tools_col2,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )
        vaila_btn14 = tk.Button(
            tools_col2,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )
        vaila_btn15 = tk.Button(
            tools_col2,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )

        # Packing Video buttons
        extract_png_btn.grid(row=0, column=0, padx=2, pady=2)
        cut_videos_btn.grid(row=0, column=1, padx=2, pady=2)
        draw_box_btn.grid(row=0, column=2, padx=2, pady=2)
        compress_videos_h264_btn.grid(row=1, column=0, padx=2, pady=2)
        compress_videos_h265_btn.grid(row=1, column=1, padx=2, pady=2)
        sync_videos_btn.grid(row=1, column=2, padx=2, pady=2)
        getpixelvideo_btn.grid(row=2, column=0, padx=2, pady=2)
        count_frames_btn.grid(row=2, column=1, padx=2, pady=2)
        video_processing_btn.grid(row=2, column=2, padx=2, pady=2)
        vaila_btn13.grid(row=3, column=0, padx=2, pady=2)
        vaila_btn14.grid(row=3, column=1, padx=2, pady=2)
        vaila_btn15.grid(row=3, column=2, padx=2, pady=2)

        tools_col2.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Visualization sub-columns
        show_c3d_btn = tk.Button(
            tools_col3, text="Show C3D", command=self.show_c3d_data, width=button_width
        )
        show_csv_btn = tk.Button(
            tools_col3, text="Show CSV", command=self.show_csv_file, width=button_width
        )
        plot_2d_btn = tk.Button(
            tools_col3, text="Plot 2D", command=self.plot_2d_data, width=button_width
        )
        plot_3d_btn = tk.Button(
            tools_col3,
            text="Plot 3D",
            command=self.plot_3d_data,
            width=button_width,
        )
        vaila_btn16 = tk.Button(
            tools_col3,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )
        vaila_btn17 = tk.Button(
            tools_col3,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )
        vaila_btn18 = tk.Button(
            tools_col3,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )
        vaila_btn19 = tk.Button(
            tools_col3,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )

        # Packing Visualization buttons
        show_c3d_btn.grid(row=0, column=0, padx=2, pady=2)
        show_csv_btn.grid(row=0, column=1, padx=2, pady=2)
        plot_2d_btn.grid(row=1, column=0, padx=2, pady=2)
        plot_3d_btn.grid(row=1, column=1, padx=2, pady=2)
        vaila_btn16.grid(row=2, column=0, padx=2, pady=2)
        vaila_btn17.grid(row=2, column=1, padx=2, pady=2)
        vaila_btn18.grid(row=3, column=0, padx=2, pady=2)
        vaila_btn19.grid(row=3, column=1, padx=2, pady=2)

        tools_col3.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Help and Exit Buttons Frame
        bottom_frame = tk.Frame(scrollable_frame)
        bottom_frame.pack(pady=10)

        help_btn = tk.Button(bottom_frame, text="Help", command=self.display_help)
        exit_btn = tk.Button(bottom_frame, text="Exit", command=self.quit_app)

        help_btn.pack(side="left", padx=5)
        exit_btn.pack(side="left", padx=5)

        license_label = tk.Label(
            scrollable_frame,
            text="© 2024 vailá - Multimodal Toolbox. Licensed under the GNU Lesser General Public License v3.0.",
            font=("default", 11),
            anchor="center",
        )
        license_label.pack(pady=5)

    def rename_files(self):
        rename_files()

    def import_file(self):
        import_file()

    def export_file(self):
        export_file()

    def copy_file(self):
        copy_file()

    def move_file(self):
        move_file()

    def remove_file(self):
        remove_file()

    def tree_file(self):
        tree_file()

    def find_file(self):
        find_file()

    def transfer_file(self):
        transfer_file()

    def imu_analysis(self):
        imu_analysis.analyze_imu_data()

    def cluster_analysis(self):
        cluster_analysis.analyze_cluster_data()

    def mocap_analysis(self):
        mocap_analysis.analyze_mocap_fullbody_data()

    def markerless_2d_analysis(self):
        markerless_2D_analysis.process_videos_in_directory()

    def markerless_3d_analysis(self):
        selected_path = filedialog.askdirectory()
        if selected_path:
            markerless_3D_analysis.analyze_markerless_3D_data(selected_path)

    def vector_coding(self):
        pass

    def emg_analysis(self):
        emg_labiocom.run_emg_gui()

    def force_analysis(self):
        forceplate_analysis.run_force_analysis()

    def gnss_analysis(self):
        gnss_analysis.run_gnss_analysis()

    def reorder_csv_data(self):
        rearrange_data_in_directory()

    def convert_c3d_csv(self):
        action = messagebox.askquestion(
            "Choose Action",
            "Select:\n(No) CSV -> C3D\n(Yes) C3D -> CSV",
            icon="question",
        )
        if action == "yes":
            convert_c3d_to_csv()
        else:
            convert_csv_to_c3d()

    def count_frames_in_videos(self):
        count_frames_in_videos()

    def cut_videos(self):
        cut_videos()

    def draw_box(self):
        run_drawboxe()

    def compress_videos_h264_gui(self):
        compress_videos_h264_gui()

    def compress_videos_h265_gui(self):
        compress_videos_h265_gui()

    def show_c3d_data(self):
        show_c3d()

    def sync_videos(self):
        sync_videos()

    def extract_png_from_videos(self):
        processor = VideoProcessor()
        processor.run()

    def getpixelvideo(self):
        getpixelvideo()

    def dlt2d(self):
        dlt2d()

    def rec2d(self):
        rec2d()

    def rec2d_one_dlt2d(self):
        rec2d_one_dlt2d()

    def dlt3d(self):
        pass  # Aqui você deve adicionar a lógica para o DLT3D

    def rec3d_one_dlt3d(self):
        pass  # Aqui você deve adicionar a lógica para a reconstrução 3D com 1 DLT

    def rec3d(self):
        pass  # Aqui você deve adicionar a lógica para a reconstrução 3D com múltiplos DLTs

    def show_csv_file(self):
        show_csv()

    def plot_2d_data(self):
        plot_2d()

    def plot_3d_data(self):
        plot_3d()

    def process_videos_gui(self):
        process_videos_gui()  # Correctly calls the function from __init__.py

    def display_help(self):
        help_file_path = os.path.join(os.path.dirname(__file__), "docs", "help.html")
        if os.path.exists(help_file_path):
            os.system(
                f"start {help_file_path}"
                if os.name == "nt"
                else f"open {help_file_path}"
            )
        else:
            messagebox.showerror("Error", "Help file not found.")

    def show_vaila_message(self):
        show_vaila_message()

    def open_link(self, event=None):
        import webbrowser

        webbrowser.open("https://github.com/vaila-multimodaltoolbox/vaila")

    def quit_app(self):
        self.destroy()
        os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    app = Vaila()
    app.mainloop()
