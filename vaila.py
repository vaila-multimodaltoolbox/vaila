"""
vaila.py

Author: Paulo Santiago
Date: 7 September 2024

Description:
'vailá' is an open-source multimodal toolbox designed for comprehensive biomechanical analysis. It integrates various types of data and provides advanced tools for researchers and practitioners in the field of biomechanics. This script serves as the main application file, offering a graphical user interface (GUI) built with Tkinter to manage, analyze, and visualize multimodal data effectively.

Key Features:
- **Multimodal Data Analysis**: 
  - Supports analyses across multiple modalities including IMU sensors, motion capture (MoCap), markerless tracking (2D and 3D), force plates, GNSS/GPS, MEG/EEG, and HR/ECG.
- **File Management**: 
  - Comprehensive set of tools for file operations: Rename, Import, Export, Copy, Move, Remove, Tree, Find, and Transfer.
- **Data Conversion**: 
  - Converts between C3D and CSV data formats, and supports DLT (Direct Linear Transformation) methods for 2D and 3D reconstructions.
- **Video Processing**: 
  - Tools for video manipulation including conversion between video and images, cutting, compression (H.264 and HEVC H.265), synchronization, and extraction of pixel coordinates.
- **Data Visualization**: 
  - Provides tools for displaying and plotting 2D and 3D graphs, as well as viewing CSV and C3D data.

Version: 7.9.1822 (First Release)

Changelog:
- **v7.9.1822**:
  - Initial release with fully integrated GUI for multimodal data analysis.
  - Added support for IMU analysis, cluster analysis, full-body motion capture, and markerless 2D/3D video tracking.
  - Included comprehensive file management capabilities and advanced data conversion tools.
  - Developed robust video processing functionalities including compression, synchronization, and pixel data extraction.
  - Implemented advanced visualization tools for 2D and 3D plotting.

Usage:
- Launch the 'vailá' GUI by running this script.
- Utilize the File Manager (Frame A) to handle files.
- Perform various multimodal analyses (Frame B) such as IMU, MoCap, and markerless tracking.
- Use the Available Tools (Frame C) for data conversion, video processing, and visualization.
- Click the [imagination!] button for command-line tools access.

License:
© 2024 'vailá' - Multimodal Toolbox. Licensed under the GNU Lesser General Public License v3.0.
"""

import os
import signal
import platform
import subprocess
from rich import print
import tkinter as tk
from tkinter import messagebox, filedialog, ttk, Toplevel, Label, Button
from PIL import Image, ImageTk
import webbrowser

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
                  \\                |                /
                   v               v               v
            +-------------------------------------------+
IMU_csv --> |          vailá - multimodal toolbox        | <-- Cluster_csv
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

============================ File Manager (Frame A) ========================
A_r1_c1 - Rename          A_r1_c2 - Import           A_r1_c3 - Export
A_r1_c4 - Copy            A_r1_c5 - Move             A_r1_c6 - Remove
A_r1_c7 - Tree            A_r1_c8 - Find             A_r1_c9 - Transfer

========================== Multimodal Analysis (Frame B) ===================
B1_r1_c1 - IMU            B1_r1_c2 - MoCapCluster    B1_r1_c3 - MoCapFullBody
B1_r1_c4 - Markerless2D   B1_r1_c5 - Markerless3D

B2_r2_c1 - Vector Coding  B2_r2_c2 - EMG             B2_r2_c3 - Force Plate
B2_r2_c4 - GNSS/GPS       B2_r2_c5 - MEG/EEG

B3_r3_c1 - HR/ECG         B3_r3_c2 - vailá           B3_r3_c3 - vailá
B3_r3_c4 - vailá          B3_r3_c5 - vailá

============================== Tools Available (Frame C) ===================
C_A: Data Files
C_A_r1_c1 - Edit CSV      C_A_r1_c2 - C3D <--> CSV   C_A_r1_c3 - vailá
C_A_r2_c1 - Make DLT2D    C_A_r2_c2 - Rec2D 1DLT     C_A_r2_c3 - Rec2D MultiDLT
C_A_r3_c1 - Make DLT3D    C_A_r3_c2 - Rec3D 1DLT     C_A_r3_c3 - Rec3D MultiDLT
C_A_r4_c1 - vailá         C_A_r4_c2 - vailá          C_A_r4_c3 - vailá

C_B: Video and Image
C_B_r1_c1 - Video<-->PNG  C_B_r1_c2 - Cut Videos    C_B_r1_c3 - Draw Box
C_B_r2_c1 - CompressH264  C_B_r2_c2 - Compress H265 C_B_r2_c3 - Make Sync file
C_B_r3_c1 - GetPixelCoord C_B_r3_c2 - Metadata info C_B_r3_c3 - Merge Videos
C_B_r4_c1 - vailá         C_B_r4_c2 - vailá         C_B_r4_c3 - vailá

C_C: Visualization
C_C_r1_c1 - Show C3D      C_C_r1_c2 - Show CSV       C_C_r2_c1 - Plot 2D
C_C_r2_c2 - Plot 3D       C_C_r3_c1 - vailá          C_C_r3_c2 - vailá
C_C_r4_c1 - vailá         C_C_r4_c2 - vailá          C_C_r4_c3 - vailá

Type 'h' for help or 'exit' to quit.

Use the button 'imagination!' to access command-line (xonsh) tools for advanced multimodal analysis!
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
        self.title("vailá - 7.9.1822")
        self.geometry("1280x720")

        # Set button dimensions and font size based on OS
        self.set_dimensions_based_on_os()  # Chamada para ajustar dimensões e fonte

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

    def set_dimensions_based_on_os(self):
        if platform.system() == "Darwin":
            # Specific adjustments for macOS
            self.button_width = 10
            self.font_size = 11
        elif platform.system() == "Windows":
            # Specific adjustments for Windows
            self.button_width = 12
            self.font_size = 11
        elif platform.system() == "Linux":
            # Specific adjustments for Linux
            self.button_width = 13
            self.font_size = 11
        else:
            # Default values
            self.button_width = 12
            self.font_size = 11

    def create_widgets(self):
        button_width = self.button_width  # Use the adjusted button width
        font = ("default", self.font_size)  # Use the length-adjusted font

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
            font=font,  # Correct font adjustment
            anchor="center",
        )
        header_label.pack(side="left")

        # Subheader with hyperlink for "vailá"
        subheader_frame = tk.Frame(self)
        subheader_frame.pack(pady=5)

        subheader_label1 = tk.Label(
            subheader_frame,
            text="Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox",
            font=font,  # Correct font adjustment
            anchor="center",
        )
        subheader_label1.pack()

        subheader_label2_frame = tk.Frame(subheader_frame)
        subheader_label2_frame.pack()

        vaila_link = tk.Label(
            subheader_label2_frame,
            text="vailá",
            font=("default", self.font_size, "italic"),
            fg="blue",
            cursor="hand2",
        )
        vaila_link.pack(side="left")
        vaila_link.bind("<Button-1>", lambda e: self.open_link())

        # Keep the button imagination in mind
        unleash_label1 = tk.Label(
            subheader_label2_frame,
            text=" and unleash your ",
            font=font,
            anchor="center",
        )
        unleash_label1.pack(side="left")

        unleash_button = tk.Button(
            subheader_label2_frame,
            text="imagination!",
            font=font,
            command=self.open_terminal_shell,
        )
        unleash_button.pack(side="left")

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

        """
            A - File Manager Avaliable: 
            - Rename
            - Import
            - Export
            - Copy
            - Move
            - Remove
            - Tree
            - Find
            - Transfer
        """
        # A - File Manager Block FRAME
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

        ## VVVVVVVVVV File Manager Buttons VVVVVVVVV
        # A_r1_c1 - File Manager Button: Rename
        rename_btn = tk.Button(
            file_manager_btn_frame,
            text="Rename",
            command=self.rename_files,
            width=button_width,
        )
        # A_r1_c2 - File Manager Button: Import
        import_btn = tk.Button(
            file_manager_btn_frame,
            text="Import",
            command=self.import_file,
            width=button_width,
        )
        # A_r1_c3 - File Manager Button: Export
        export_btn = tk.Button(
            file_manager_btn_frame,
            text="Export",
            command=self.export_file,
            width=button_width,
        )
        # A_r1_c4 - File Manager Button: Copy
        copy_btn = tk.Button(
            file_manager_btn_frame,
            text="Copy",
            command=self.copy_file,
            width=button_width,
        )
        # A_r1_c5 - File Manager Button: Move
        move_btn = tk.Button(
            file_manager_btn_frame,
            text="Move",
            command=self.move_file,
            width=button_width,
        )
        # A_r1_c6 - File Manager Button: Remove
        remove_btn = tk.Button(
            file_manager_btn_frame,
            text="Remove",
            command=self.remove_file,
            width=button_width,
        )
        # A_r1_c7 - File Manager Button: Tree
        tree_btn = tk.Button(
            file_manager_btn_frame,
            text="Tree",
            command=self.tree_file,
            width=button_width,
        )
        # A_r1_c8 - File Manager Button: Find
        find_btn = tk.Button(
            file_manager_btn_frame,
            text="Find",
            command=self.find_file,
            width=button_width,
        )
        # A_r1_c9 - File Manager Button: Transfer
        transfer_btn = tk.Button(
            file_manager_btn_frame,
            text="Transfer",
            command=self.transfer_file,
            width=button_width,
        )
        ## VVVVVVVVVV FILE MANAGER BUTTON VVVVVVVVV
        rename_btn.pack(side="left", padx=2, pady=2)
        import_btn.pack(side="left", padx=2, pady=2)
        export_btn.pack(side="left", padx=2, pady=2)
        copy_btn.pack(side="left", padx=2, pady=2)
        move_btn.pack(side="left", padx=2, pady=2)
        remove_btn.pack(side="left", padx=2, pady=2)
        tree_btn.pack(side="left", padx=2, pady=2)
        find_btn.pack(side="left", padx=2, pady=2)
        transfer_btn.pack(side="left", padx=2, pady=2)

        """
            B - Multimodal Analysis Available:
            B1:
            - IMU
            - Motion Capture Cluster
            - Motion Capture Full Body
            - Markerless 2D
            - Markerless 3D
            B2:
            - Vector Coding
            - EMG
            - Force Plate
            - GNSS/GPS
            - MEG/EEG
            B3:
            - HR/ECG
            - vailá
            - vailá
            - vailá
            - vailá
        """
        # B - Multimodal Analysis FRAME
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
        ## Insert the buttons for each Multimodal Toolbox Analysis
        # Buttons for each Multimodal Toolbox Analysis
        # B1_r1_c1 - IMU
        row1_frame = tk.Frame(analysis_frame)
        row1_frame.pack(fill="x")
        imu_analysis_btn = tk.Button(
            row1_frame, text="IMU", width=button_width, command=self.imu_analysis
        )
        # B1_r1_c2 - Motion Capture
        cluster_analysis_btn = tk.Button(
            row1_frame,
            text="Motion Capture Cluster",
            width=button_width,
            command=self.cluster_analysis,
        )
        # B1_r1_c3 - Motion Capture
        mocap_analysis_btn = tk.Button(
            row1_frame,
            text="Motion Capture Full Body",
            width=button_width,
            command=self.mocap_analysis,
        )
        # B1_r1_c4 - Markerless 2D
        markerless_2d_analysis_btn = tk.Button(
            row1_frame,
            text="Markerless 2D",
            width=button_width,
            command=self.markerless_2d_analysis,
        )
        # B1_r1_c5 - Markerless 3D
        markerless_3d_analysis_btn = tk.Button(
            row1_frame,
            text="Markerless 3D",
            width=button_width,
            command=self.markerless_3d_analysis,
        )

        # Pack the buttons
        imu_analysis_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        cluster_analysis_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        mocap_analysis_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        markerless_2d_analysis_btn.pack(
            side="left", expand=True, fill="x", padx=2, pady=2
        )
        markerless_3d_analysis_btn.pack(
            side="left", expand=True, fill="x", padx=2, pady=2
        )

        # B2 - Multimodal Toolbox: Second row of buttons (EMG, Force Plate, GNSS/GPS, MEG/EEG)
        # B2_r2_c1 - Vector Coding
        row2_frame = tk.Frame(analysis_frame)
        row2_frame.pack(fill="x")
        vector_coding_btn = tk.Button(
            row2_frame,
            text="Vector Coding",
            width=button_width,
            command=self.vector_coding,
        )
        # B2_r2_c2 - EMG
        emg_analysis_btn = tk.Button(
            row2_frame, text="EMG", width=button_width, command=self.emg_analysis
        )
        # B2_r2_c3 - Force Plate
        forceplate_btn = tk.Button(
            row2_frame,
            text="Force Plate",
            width=button_width,
            command=self.force_analysis,
        )
        # B2_r2_c4 - GNSS/GPS
        gnss_btn = tk.Button(
            row2_frame,
            text="GNSS/GPS",
            width=button_width,
            command=self.gnss_analysis,
        )
        # B2_r2_c5 - MEG/EEG
        vaila_btn3 = tk.Button(
            row2_frame,
            text="MEG/EEG",
            width=button_width,
            # Provisory button redirecting to https://mne.tools/dev/auto_tutorials/intro/10_overview.html
            command=lambda: webbrowser.open(
                "https://mne.tools/dev/auto_tutorials/intro/10_overview.html"
            ),
            # command=self.meg_eeg_analysis,
        )

        # Pack the buttons
        vector_coding_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        emg_analysis_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        forceplate_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        gnss_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn3.pack(side="left", expand=True, fill="x", padx=2, pady=2)

        # 3 - Multimodal Toolbox Analysis: Third row of buttons (HR/ECG, vailá, vailá, vailá)
        # B3_r3_c1 - HR/ECG
        row3_frame = tk.Frame(analysis_frame)
        row3_frame.pack(fill="x")
        vaila_btn4 = tk.Button(
            row3_frame,
            text="HR/ECG",
            width=button_width,
            # Provisory button redirecting to https://github.com/paulvangentcom/heartrate_analysis_python
            command=lambda: webbrowser.open(
                "https://github.com/paulvangentcom/heartrate_analysis_python"
            ),
            # command=self.heart_rate_analysis,
        )
        # B3_r3_c2 - vailá
        vaila_btn5 = tk.Button(
            row3_frame,
            text="vailá",
            width=button_width,
            command=self.show_vaila_message,
        )
        # B3_r3_c3 - vailá
        vaila_btn6 = tk.Button(
            row3_frame,
            text="vailá",
            width=button_width,
            command=self.show_vaila_message,
        )
        # B3_r3_c4 - vailá
        vaila_btn7 = tk.Button(
            row3_frame,
            text="vailá",
            width=button_width,
            command=self.show_vaila_message,
        )
        # B3_r3_c5 - vailá
        vaila_btn8 = tk.Button(
            row3_frame,
            text="vailá",
            width=button_width,
            command=self.show_vaila_message,
        )
        # Pack the buttons
        vaila_btn4.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn5.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn6.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn7.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn8.pack(side="left", expand=True, fill="x", padx=2, pady=2)

        ## VVVVVVVVVVVVVVV TOOLS BUTTONS VVVVVVVVVVVVVVVV
        # Tools Frame
        # Create a frame for the tools
        """
            C - Tools Available:
            C_A: Data Files
            C_B: Video and Image
            C_C: Visualization
        """
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

        ## VVVVVVVVVVVVVVV DATA BUTTONS VVVVVVVVVVVVVVVV
        # C_A - Data Files sub-columns
        # C_A_r1_c1 - Data Files: Edit CSV
        reorder_csv_btn = tk.Button(
            tools_col1,
            text="Edit CSV",
            command=self.reorder_csv_data,
            width=button_width,
        )
        # C_A_r1_c2 - Data Files: C3D <--> CSV
        convert_btn = tk.Button(
            tools_col1,
            text="C3D <--> CSV",
            command=self.convert_c3d_csv,
            width=button_width,
        )
        # C_A_r1_c3 - Data Files: vailá
        vaila_btn8to9 = tk.Button(
            tools_col1,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )
        # C_A_r2_c1 - Data Files: Make DLT2D
        dlt2d_btn = tk.Button(
            tools_col1, text="Make DLT2D", command=self.dlt2d, width=button_width
        )
        # C_A_r2_c2 - Data Files: Rec2D 1DLT
        rec2d_one_btn = tk.Button(
            tools_col1,
            text="Rec2D 1DLT",
            command=self.rec2d_one_dlt2d,
            width=button_width,
        )
        # C_A_r2_c3 - Data Files: Rec2D MultiDLT
        rec2d_multiple_btn = tk.Button(
            tools_col1, text="Rec2D MultiDLT", command=self.rec2d, width=button_width
        )
        # C_A_r3_c1 - Data Files: Make DLT3D
        dlt3d_btn = tk.Button(
            tools_col1, text="Make DLT3D", command=self.dlt3d, width=button_width
        )
        # C_A_r3_c2 - Data Files: Rec3D 1DLT
        rec3d_one_btn = tk.Button(
            tools_col1,
            text="Rec3D 1DLT",
            command=self.rec3d_one_dlt3d,
            width=button_width,
        )
        # C_A_r3_c3 - Data Files: Rec3D MultiDLT
        rec3d_multiple_btn = tk.Button(
            tools_col1, text="Rec3D MultiDLT", command=self.rec3d, width=button_width
        )
        # Avaliable blank (vailá) buttons for future tools (10-11)
        # C_A_r4_c1 - Data Files: vailá
        vaila_btn9 = tk.Button(
            tools_col1,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )
        # C_A_r4_c2 - Data Files: vailá
        vaila_btn10 = tk.Button(
            tools_col1,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )
        # C_A_r4_c3 - Data Files: vailá
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

        ## VVVVVVVVVVVVVVV VIDEO BUTTONS VVVVVVVVVVVVVVVV
        # Video sub-columns
        # C_B_r1_c1 - Video: Video <--> PNG
        extract_png_btn = tk.Button(
            tools_col2,
            text="Video <--> PNG",
            command=self.extract_png_from_videos,
            width=button_width,
        )
        # C_B_r1_c2 - Video: Cut Videos
        cut_videos_btn = tk.Button(
            tools_col2, text="Cut Videos", command=self.cut_videos, width=button_width
        )
        # C_B_r1_c3 - Video: Draw Box
        draw_box_btn = tk.Button(
            tools_col2, text="Draw Box", command=self.draw_box, width=button_width
        )
        # C_B_r2_c1 - Video: Compress H.264
        compress_videos_h264_btn = tk.Button(
            tools_col2,
            text="Compress H264",
            command=self.compress_videos_h264_gui,
            width=button_width,
        )
        # C_B_r2_c2 - Video: Compress H.265
        compress_videos_h265_btn = tk.Button(
            tools_col2,
            text="Compress H265",
            command=self.compress_videos_h265_gui,
            width=button_width,
        )
        # C_B_r2_c3 - Video: Make Sync Videos
        sync_videos_btn = tk.Button(
            tools_col2,
            text="Make Sync file",
            command=self.sync_videos,
            width=button_width,
        )
        # C_B_r3_c1 - Video: Get Pixel Coords
        getpixelvideo_btn = tk.Button(
            tools_col2,
            text="Get Pixel Coords",
            command=self.getpixelvideo,
            width=button_width,
        )
        # C_B_r3_c2 - Video: Metadata info
        count_frames_btn = tk.Button(
            tools_col2,
            text="Metadata info",
            command=self.count_frames_in_videos,
            width=button_width,
        )
        # C_B_r3_c3 - Video: Merge Videos
        video_processing_btn = tk.Button(
            tools_col2,
            text="Merge Videos",
            command=self.process_videos_gui,
            width=button_width,
        )
        # Avaliable blank (vailá) buttons for future tools (12-15)
        # C_B_r4_c1 - Video: vailá
        vaila_btn13 = tk.Button(
            tools_col2,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )
        # C_B_r4_c2 - Video: vailá
        vaila_btn14 = tk.Button(
            tools_col2,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )
        # C_B_r4_c3 - Video: vailá
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

        ## VVVVVVVVVVVVVVV VISUALIZATION BUTTONS VVVVVVVVVVVVVVVV
        # Visualization sub-columns (3-6)
        # C_C_r1_c1 - Visualization: Show C3D
        show_c3d_btn = tk.Button(
            tools_col3, text="Show C3D", command=self.show_c3d_data, width=button_width
        )
        # C_C_r1_c2 - Visualization: Show CSV
        show_csv_btn = tk.Button(
            tools_col3, text="Show CSV", command=self.show_csv_file, width=button_width
        )
        # C_C_r2_c1 - Visualization: Plot 2D
        plot_2d_btn = tk.Button(
            tools_col3, text="Plot 2D", command=self.plot_2d_data, width=button_width
        )
        # C_C_r2_c2 - Visualization: Plot 3D
        plot_3d_btn = tk.Button(
            tools_col3,
            text="Plot 3D",
            command=self.plot_3d_data,
            width=button_width,
        )
        # C_C_r3_c1 - Visualization: vailá
        vaila_btn16 = tk.Button(
            tools_col3,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )
        # C_C_r3_c2 - Visualization: vailá
        vaila_btn17 = tk.Button(
            tools_col3,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )
        # C_C_r4_c1 - Visualization: vailá
        vaila_btn18 = tk.Button(
            tools_col3,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )
        # C_C_r4_c2 - Visualization: vailá
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

    # Class definition
    def show_vaila_message(self):
        show_vaila_message()

    # A First FRAME Block
    # A_r1_c1
    def rename_files(self):
        rename_files()

    # A_r1_c2
    def import_file(self):
        import_file()

    # A_r1_c3
    def export_file(self):
        export_file()

    # A_r1_c4
    def copy_file(self):
        copy_file()

    # A_r1_c5
    def move_file(self):
        move_file()

    # A_r1_c6
    def remove_file(self):
        remove_file()

    # A_r1_c7
    def tree_file(self):
        tree_file()

    # A_r1_c8
    def find_file(self):
        find_file()

    # A_r1_c9
    def transfer_file(self):
        transfer_file()

    # B Second FRAME Block
    # B_r1_c1
    def imu_analysis(self):
        imu_analysis.analyze_imu_data()

    # B_r1_c2
    def cluster_analysis(self):
        cluster_analysis.analyze_cluster_data()

    # B_r1_c3
    def mocap_analysis(self):
        mocap_analysis.analyze_mocap_fullbody_data()

    # B_r1_c4
    def markerless_2d_analysis(self):
        markerless_2D_analysis.process_videos_in_directory()

    # B_r1_c5
    def markerless_3d_analysis(self):
        selected_path = filedialog.askdirectory()
        if selected_path:
            markerless_3D_analysis.analyze_markerless_3D_data(selected_path)

    # B_r2_c1
    def vector_coding(self):
        show_vaila_message()
        # vector_coding.run_vector_coding()

    # B_r2_c2
    def emg_analysis(self):
        emg_labiocom.run_emg_gui()

    # B_r2_c3
    def force_analysis(self):
        forceplate_analysis.run_force_analysis()

    # B_r2_c4
    def gnss_analysis(self):
        show_vaila_message()
        # gnss_analysis.run_gnss_analysis()

    # B_r2_c5
    def eeg_analysis(self):
        show_vaila_message()
        # eeg_analysis.run_eeg_analysis()

    # B_r3_c1
    def hr_analysis(self):
        show_vaila_message()
        # hr_analysis.run_hr_analysis()

    # B_r3_c2
    # def vaila

    # B_r3_c3
    # def vaila

    # B_r3_c4
    # def vaila

    # B_r3_c5
    # def vaila

    # C_A_r1_c1
    def reorder_csv_data(self):
        rearrange_data_in_directory()  # Edit CSV

    # C_A_r1_c2
    def convert_c3d_csv(self):
        # Cria uma nova janela para a escolha da ação
        window = Toplevel()
        window.title("Choose Action")
        # Mensagem para o usuário
        label = Label(window, text="Which conversion would you like to perform?")
        label.pack(pady=10)

        # Botão para C3D -> CSV
        button_c3d_to_csv = Button(
            window,
            text="C3D -> CSV",
            command=lambda: [convert_c3d_to_csv(), window.destroy()],
        )
        button_c3d_to_csv.pack(side="left", padx=20, pady=20)

        # Botão para CSV -> C3D
        button_csv_to_c3d = Button(
            window,
            text="CSV -> C3D",
            command=lambda: [convert_csv_to_c3d(), window.destroy()],
        )
        button_csv_to_c3d.pack(side="right", padx=20, pady=20)

    # C_A_r1_c3
    # def vaila(self):
    #     vaila()

    # C_A_r2_c1
    def dlt2d(self):
        dlt2d()

    # C_A_r2_c2
    def rec2d_one_dlt2d(self):
        rec2d_one_dlt2d()

    # C_A_r2_c3 - for multi dlts in rows
    def rec2d(self):
        rec2d()

    # C_A_r3_c1
    def dlt3d(self):
        pass  # Aqui você deve adicionar a lógica para o DLT3D

    # C_A_r3_c2 - for multi dlts in rows
    def rec3d_one_dlt3d(self):
        pass  # Aqui você deve adicionar a lógica para a reconstrução 3D com 1 DLT

    # C_A_r3_c3 - for multi dlts in rows
    def rec3d(self):
        pass  # Aqui você deve adicionar a lógica para a reconstrução 3D com múltiplos DLTs

    # C_A_r4_c1
    # def vaila(self):

    # C_A_r4_c2
    # def vaila(self):
    #     vaila()

    # C_A_r4_c3
    # def vaila(self):
    #     vaila()

    # C_B_r1_c1
    def extract_png_from_videos(self):
        processor = VideoProcessor()
        processor.run()

    # C_B_r1_c2
    def cut_videos(self):
        cut_videos()

    # C_B_r1_c3
    def draw_box(self):
        run_drawboxe()

    # C_B_r2_c1
    def compress_videos_h264_gui(self):
        compress_videos_h264_gui()

    # C_B_r2_c2
    def compress_videos_h265_gui(self):
        compress_videos_h265_gui()

    # C_B_r2_c3
    def sync_videos(self):
        sync_videos()

    # C_B_r3_c1
    def getpixelvideo(self):
        getpixelvideo()

    # C_B_r3_c2
    def count_frames_in_videos(self):
        count_frames_in_videos()

    # C_B_r3_c3
    def process_videos_gui(self):
        process_videos_gui()

    # C_C_r1_c1
    def show_c3d_data(self):
        show_c3d()

    # C_C_r1_c2
    def show_csv_file(self):
        show_csv()

    # C_C_r1_c3
    def plot_2d_data(self):
        plot_2d()

    # C_C_r2_c2
    def plot_3d_data(self):
        plot_3d()

    # C_C_r3_c1
    # def vaila(self):

    # C_C_r3_c2
    # def vaila(self):

    # C_C_r3_c3
    # def vaila(self):

    # C_C_r4_c1
    # def vaila(self):

    # C_C_r4_c2
    # def vaila(self):

    # C_C_r4_c3
    # def vaila(self):

    # Help, Exit and About
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

    def open_link(self, event=None):
        import webbrowser

        webbrowser.open("https://github.com/vaila-multimodaltoolbox/vaila")

    def open_terminal_shell(self):
        # Open a new terminal with the Conda environment activated using xonsh
        if platform.system() == "Darwin":  # For macOS
            # Use osascript to open Terminal and activate the Conda environment, then start xonsh
            subprocess.Popen(
                [
                    "osascript",
                    "-e",
                    'tell application "Terminal" to do script "source ~/anaconda3/etc/profile.d/conda.sh && conda activate vaila && xonsh"',
                ]
            )

        elif platform.system() == "Windows":  # For Windows
            # Open PowerShell and activate the Conda environment
            subprocess.Popen(
                "start powershell -NoExit -Command \"& 'C:\\ProgramData\\anaconda3\\shell\\condabin\\conda-hook.ps1'; conda activate vaila; xonsh\"",
                shell=True,
            )

        elif platform.system() == "Linux":  # For Linux
            # Open a terminal and activate the Conda environment using xonsh
            subprocess.Popen(
                [
                    "x-terminal-emulator",
                    "-e",
                    "bash",
                    "-c",
                    "source ~/anaconda3/etc/profile.d/conda.sh && conda activate vaila && xonsh",
                ],
                start_new_session=True,
            )

    def quit_app(self):
        self.destroy()
        os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    app = Vaila()
    app.mainloop()
