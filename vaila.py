"""
===============================================================================
vaila.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Date:  7 October 2024
Update: 13 June 2025
Version updated: 0.8.1
Python Version: 3.12.11

Description:
------------
vailá (Versatile Anarcho Integrated Liberation Ánalysis) is an open-source,
Python-based multimodal toolbox designed to streamline biomechanical data
analysis. It integrates multiple types of biomechanical data (e.g., IMU, motion
capture, markerless tracking, force plates, GNSS/GPS, EMG) into a unified,
flexible platform for advanced human movement analysis. The software was
developed with a modular architecture to ensure easy expansion, transparency,
and community-driven contributions.

vailá offers batch processing of large datasets, multimodal data analysis,
and cross-platform compatibility (Linux, macOS, Windows). It is developed to
handle complex biomechanical workflows, including kinematic and kinetic data
processing, visualization, and data conversion, as discussed in the associated
paper. The system fosters a collaborative, transparent environment for research,
allowing users to customize and expand the toolbox with new functionalities.

Key Features:
-------------
1. **Multimodal Data Integration**:
   - Supports data from IMUs, markerless tracking (2D and 3D), MoCap systems,
     force plates, GNSS/GPS, EMG, and other biomechanical sensors.

2. **Data Processing and Batch Operations**:
   - Batch processing for large datasets across modalities, including video
     synchronization, pixel extraction, DLT-based 2D/3D reconstructions, and
     force analysis.

3. **Data Conversion and File Management**:
   - Converts between multiple data formats (C3D <--> CSV), automates renaming,
     copying, and managing large sets of biomechanical files.

4. **Visualization**:
   - Includes 2D and 3D plotting of biomechanical data using libraries such as
     Matplotlib and Plotly.

5. **Cross-Platform**:
   - Designed for macOS, Linux, and Windows, with full transparency of execution
     flow through rich terminal outputs and print statements for debugging.

6. **Open Field Test Analysis** (New Feature):
   - Provides tools for analyzing open field test data for rodents, including
     calculations of total distance traveled, speed, and time spent in each zone.

7. **Video Distortion** (New Feature):
   - Provides tools for distorting videos, including adding noise, blurring,
     and other effects.

8. **Audio Video Processing** (New Feature):
   - Provides tools for processing audio and video files, including inserting
     audio into videos and downloading videos from YouTube.

9. **Install .exe Windows**:
   - Provides a script to install the vailá toolbox as an .exe file for Windows.

10. **Soccer Field** (New Feature):
    - Provides a script to create a soccer field video.

11. **Distort Video** (New Feature):
    - Provides a script to distort videos, including adding noise, blurring,
      and other effects.

12. **Get Pixel Coordinates** (New Feature):
    - Provides a script to get the pixel coordinates of a video.

13. **Markerless 2D and 3D and Live with YOLO and MediaPipe** (New Feature):
    - Provides a script to analyze markerless live data, including joint angles
      and movement data.

14. **Relative Angles** (New Feature):
    - Provides a script to calculate relative angles between body segments.

15. **Ultrasound** (New Feature):
    - Provides a script to analyze ultrasound data from images.

Usage:
------
- Run this script to launch the main graphical user interface (GUI) built with
  Tkinter.
- The GUI offers:
  - **File Management (Frame A)**: Tools for renaming, importing, exporting, and
    manipulating large sets of files.
  - **Multimodal Analysis (Frame B)**: Tools for analyzing biomechanical data
    (e.g., MoCap, IMU, and markerless tracking).
  - **Available Tools (Frame C)**: Data conversion, video/image processing,
    DLT-based 2D/3D reconstructions, and visualization tools.
  - **Open Field Test Analysis (Frame D)**: Tools for analyzing open field test
    data, providing insights into animal behavior and movement patterns.

License:
--------
This program is licensed under the GNU Lesser General Public License v3.0.
For more details, visit: https://www.gnu.org/licenses/lgpl-3.0.html
Visit the project repository: https://github.com/vaila-multimodaltoolbox
===============================================================================
"""

# Standard library imports
import os
import signal
import platform
import subprocess
import sys
import webbrowser
from typing import Optional

# Third-party imports
import tkinter as tk
from tkinter import messagebox, filedialog, ttk, Toplevel, Label, Button, simpledialog
from PIL import Image, ImageTk

# Conditionally import platform-specific functionality
# Define a global variable to track if AppKit is available
# This is used to set the application name in the dock for macOS
APPKIT_AVAILABLE = False
if platform.system() == "Darwin":  # macOS
    try:
        import AppKit  # macOS only ignore this line in other OS

        APPKIT_AVAILABLE = True
    except ImportError:
        # Silently continue if AppKit is not available
        print(
            "AppKit not available. Application name in dock might not be set correctly."
        )
        pass

text = r"""
vailá - 13.June.2025 v0.8.1 (Python 3.12.11)
                                             o
                                _,  o |\  _,/
                          |  |_/ |  | |/ / |
                           \/  \/|_/|/|_/\/|_/                   
##########################################################################
Mocap fullbody_c3d           Markerless_3D      Markerless_2D_MP and YOLO
                  \                |                /
                   v               v               v        
   CUBE2D  --> +---------------------------------------+ <-- Vector Coding
   IMU_csv --> |       vailá - multimodal toolbox      | <-- Cluster_csv
Open Field --> +---------------------------------------+ <-- Force Plate
              ^                   |                    ^ <-- YOLOv12 and MediaPipe
        EMG__/                    v                     \__Tracker YOLOv11 or YOLOv12
               +--------------------------------------+
               | Results: Processed Data and Figures  | 
               +--------------------------------------+

============================ File Manager (Frame A) ========================
A_r1_c1 - Rename          A_r1_c2 - Import           A_r1_c3 - Export
A_r1_c4 - Copy            A_r1_c5 - Move             A_r1_c6 - Remove
A_r1_c7 - Tree            A_r1_c8 - Find             A_r1_c9 - Transfer

========================== Multimodal Analysis (Frame B) ===================
B1_r1_c1 - IMU            B1_r1_c2 - MoCapCluster    B1_r1_c3 - MoCapFullBody
B1_r1_c4 - Markerless2D   B1_r1_c5 - Markerless3D

B2_r2_c1 - Vector Coding  B2_r2_c2 - EMG             B2_r2_c3 - Force Plate
B2_r2_c4 - GNSS/GPS       B2_r2_c5 - MEG/EEG

B3_r3_c1 - HR/ECG         B3_r3_c2 - MP_Yolo         B3_r3_c3 - vailá_and_jump
B3_r3_c4 - Cube2D         B3_r3_c5 - Animal Open Field

B3_r4_c1 - Tracker        B3_r4_c2 - ML Walkway      B3_r4_c3 - Markerless Hands
B3_r4_c4 - MP Angles      B3_r4_c5 - Markerless Live

B3_r5_c1 - Ultrasound     B3_r5_c2 - Brainstorm      B3_r5_c3 - vailá
B3_r5_c4 - vailá          B3_r5_c5 - vailá

============================== Tools Available (Frame C) ===================
-> C_A: Data Files
C_A_r1_c1 - Edit CSV      C_A_r1_c2 - C3D <--> CSV   C_A_r1_c3 - Smooth_Fill_Split
C_A_r2_c1 - Make DLT2D    C_A_r2_c2 - Rec2D 1DLT     C_A_r2_c3 - Rec2D MultiDLT
C_A_r3_c1 - Make DLT3D    C_A_r3_c2 - Rec3D 1DLT     C_A_r3_c3 - Rec3D MultiDLT
C_A_r4_c1 - ReID Marker   C_A_r4_c2 - vailá          C_A_r4_c3 - vailá

-> C_B: Video and Image
C_B_r1_c1 - Video<-->PNG  C_B_r1_c2 - Cut Videos    C_B_r1_c3 - Draw Box
C_B_r2_c1 - CompressH264  C_B_r2_c2 - Compress H265 C_B_r2_c3 - Make Sync file
C_B_r3_c1 - GetPixelCoord C_B_r3_c2 - Metadata info C_B_r3_c3 - Merge Videos
C_B_r4_c1 - Distort video C_B_r4_c2 - Cut Video     C_B_r4_c3 - Resize Video
C_B_r5_c1 - YT Downloader C_B_r5_c2 - Insert Audio  C_B_r5_c3 - rm Dup PNG

-> C_C: Visualization
C_C_r1_c1 - Show C3D      C_C_r1_c2 - Show CSV       C_C_r2_c1 - Plot 2D
C_C_r2_c2 - Plot 3D       C_C_r3_c1 - Soccer Field   C_C_r3_c2 - vailá
C_C_r4_c1 - vailá         C_C_r4_c2 - vailá          C_C_r4_c3 - vailá
C_C_r5_c1 - vailá         C_C_r5_c2 - vailá          C_C_r5_c3 - vailá

Type 'h' for help or 'exit' to quit.

Use the button 'imagination!' to access command-line (xonsh) tools for advanced multimodal analysis!
"""

print(text)


class Vaila(tk.Tk):
    def __init__(self):
        """
        Initializes the Vaila application.

        - Sets the window title, geometry, button dimensions, and font size based on the operating system.
        - Configures the window icon based on the operating system.
        - For macOS, sets the application name in the dock if AppKit is available.
        - Creates the widgets for the application.

        """
        super().__init__()
        self.title("vailá - 13.June.2025 v0.8.1 (Python 3.12.11)")

        # Adjust dimensions and layout based on the operating system
        self.set_dimensions_based_on_os()

        # Configure the window icon based on the operating system
        icon_path_ico = os.path.join(
            os.path.dirname(__file__), "vaila", "images", "vaila.ico"
        )
        icon_path_png = os.path.join(
            os.path.dirname(__file__), "vaila", "images", "vaila_ico_mac.png"
        )

        if platform.system() == "Windows":
            self.iconbitmap(icon_path_ico)  # Set .ico file for Windows
        else:
            # Set .png icon for macOS and Linux
            try:
                icon = tk.PhotoImage(file=icon_path_png)
                self.iconphoto(True, icon)
            except Exception as e:
                print(f"Could not set icon: {e}")

        # For macOS, set the application name in the dock if AppKit is available
        if platform.system() == "Darwin" and APPKIT_AVAILABLE:  # macOS with AppKit
            try:
                AppKit.NSBundle.mainBundle().infoDictionary()["CFBundleName"] = "vailá"
            except Exception as e:
                print(f"Could not set application name: {e}")

        # Call method to create the widgets
        self.create_widgets()

    def set_dimensions_based_on_os(self):
        """
        Adjusts the window dimensions, button width, and font size based on the operating system.
        """
        if platform.system() == "Darwin":  # macOS
            self.geometry("1920x1080")  # Wider window for macOS
            self.button_width = 12  # Slightly wider buttons
            self.font_size = 11  # Standard font size
        elif platform.system() == "Windows":  # Windows
            self.geometry("1024x920")  # Compact horizontal size for Windows
            self.button_width = 13  # Narrower buttons for reduced width
            self.font_size = 11  # Standard font size
        elif platform.system() == "Linux":  # Linux
            self.geometry("1300x920")  # Similar to macOS dimensions for Linux
            self.button_width = 15  # Wider buttons
            self.font_size = 11  # Standard font size
        else:  # Default for other systems
            self.geometry("1024x920")  # Default dimensions
            self.button_width = 15  # Default button width
            self.font_size = 11  # Default font size

    def create_widgets(self):
        """
        Creates the widgets of the application.
        """
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

        # Resize the image in the most compatible way
        preto_image = preto_image.resize((87, 87))  # Use default filter

        # Create and keep a reference to the PhotoImage
        preto_photo = ImageTk.PhotoImage(preto_image)

        # Create the label with the image
        preto_label = tk.Label(header_frame, image=preto_photo)  # type: ignore
        self._preto_photo = (
            preto_photo  # Store as instance attribute to prevent garbage collection
        )
        preto_label.pack(side="left", padx=10)

        # Label clickable 'vailá'
        vaila_click = tk.Label(
            header_frame,
            text="vailá",
            font=("default", self.font_size, "italic"),
            fg="blue",
            cursor="hand2",
        )
        vaila_click.pack(side="left")
        # When clicked, open the execution directory
        vaila_click.bind(
            "<Button-1>",
            lambda e: (
                os.startfile(os.getcwd())
                if platform.system() == "Windows"
                else subprocess.Popen(["xdg-open", os.getcwd()])
            ),
        )

        # Static label restante
        toolbox_label = tk.Label(
            header_frame,
            text=" - Multimodal Toolbox",
            font=font,
            anchor="center",
        )
        toolbox_label.pack(side="left")

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
            - Markerless_MP_Yolo
            - vailá_and_jump
            - Cube2D
            - Animal Open Field
            B4:
            - Tracker
            - ML Walkway
            - Markerless Hands
            - MP Angles
            - Markerless Live
            B5:
            - Ultrasound
            - Brainstorm
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

        # Define row4_frame before using it
        row4_frame = tk.Frame(analysis_frame)
        row4_frame.pack(fill="x")

        # Create row5_frame
        row5_frame = tk.Frame(analysis_frame)
        row5_frame.pack(fill="x")

        # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
        ## Insert the buttons for each Multimodal Toolbox Analysis
        # Buttons for each Multimodal Toolbox Analysis
        # B - Multimodal Analysis Buttons
        # B1_r1_c1 - IMU
        row1_frame = tk.Frame(analysis_frame)
        row1_frame.pack(fill="x")
        imu_analysis_btn = tk.Button(
            row1_frame, text="IMU", width=button_width, command=self.imu_analysis
        )

        # B1_r1_c2 - Motion Capture Cluster
        cluster_analysis_btn = tk.Button(
            row1_frame,
            text="Motion Capture Cluster",
            width=button_width,
            command=self.cluster_analysis,
        )

        # B1_r1_c3 - Motion Capture Full Body
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
        row2_frame = tk.Frame(analysis_frame)
        row2_frame.pack(fill="x")
        # B2_r2_c1 - Vector Coding
        vector_coding_btn = tk.Button(
            row2_frame,
            text="Vector Coding",
            width=button_width,
            command=self.vector_coding,
        )

        # B2_r2_c2 - EMG
        emg_analysis_btn = tk.Button(
            row2_frame,
            text="EMG",
            width=button_width,
            command=self.emg_analysis,
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

        # 3 - Multimodal Toolbox Analysis: Third row of buttons (HR/ECG, vailá, vailá_and_jump, vailá)
        # B3_r3_c1 - HR/ECG
        row3_frame = tk.Frame(analysis_frame)
        row3_frame.pack(fill="x")
        ecg_btn = tk.Button(
            row3_frame,
            text="HR/ECG",
            width=button_width,
            # Provisory button redirecting to https://github.com/paulvangentcom/heartrate_analysis_python
            command=lambda: webbrowser.open(
                "https://github.com/paulvangentcom/heartrate_analysis_python"
            ),
            # command=self.heart_rate_analysis,
        )

        # B3_r3_c2 - markerless2d_mpyolo
        markerless2d_mpyolo_btn = tk.Button(
            row3_frame,
            text="Yolo + Markerless_MP",
            width=button_width,
            command=self.markerless2d_mpyolo,
        )

        # B3_r3_c3 - vaila_and_jump
        vailajump_btn = tk.Button(
            row3_frame,
            text="Vertical Jump",
            width=button_width,
            command=self.vailajump,
        )

        # B3_r3_c4 - Cube2D
        cube2d_btn = tk.Button(
            row3_frame,
            text="Cube2D",
            width=button_width,
            command=self.cube2d_kinematics,
        )

        # B3_r3_c5 - Animal Open Field
        vaila_animalof = tk.Button(
            row3_frame,
            text="Animal Open Field",
            width=button_width,
            command=self.animal_open_field,
        )

        # Pack row3 buttons
        ecg_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        markerless2d_mpyolo_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vailajump_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        cube2d_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_animalof.pack(side="left", expand=True, fill="x", padx=2, pady=2)

        # Create row4_frame
        row4_frame = tk.Frame(analysis_frame)
        row4_frame.pack(fill="x")

        # B4_r4_c1 - Tracker
        tracker_btn = tk.Button(
            row4_frame,
            text="Tracker",
            width=button_width,
            command=self.tracker,
        )

        # B4_r4_c2 - ML Walkway
        mlwalkway_btn = tk.Button(
            row4_frame,
            text="ML Walkway",
            width=button_width,
            command=self.ml_walkway,
        )

        # B4_r4_c3 - Markerless Hands
        mphands_btn = tk.Button(
            row4_frame,
            text="Markerless Hands",
            width=button_width,
            command=self.markerless_hands,
        )

        # B4_r4_c4 - MP Angles
        mpangles_btn = tk.Button(
            row4_frame,
            text="MP Angles",
            width=button_width,
            command=self.mp_angles_calculation,
        )

        # B4_r4_c5 - Markerless Live
        markerlesslive_btn = tk.Button(
            row4_frame,
            text="Markerless Live",
            width=button_width,
            command=self.markerless_live,
        )

        # Pack row4 buttons
        tracker_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        mlwalkway_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        mphands_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        mpangles_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        markerlesslive_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)

        # B5 - Fifth row of buttons (Ultrasound, vailá, vailá, vailá, vailá)
        row5_frame = tk.Frame(analysis_frame)
        row5_frame.pack(fill="x")

        # B5_r5_c1 - Ultrasound
        ultrasound_btn = tk.Button(
            row5_frame,
            text="Ultrasound",
            width=button_width,
            command=self.ultrasound,
        )

        # B5_r5_c2 - Brainstorm
        brainstorm_btn = tk.Button(
            row5_frame,
            text="Brainstorm",
            width=button_width,
            command=self.brainstorm,
        )

        # B5_r5_c3 - vailá
        vaila_btn3 = tk.Button(
            row5_frame,
            text="vailá",
            width=button_width,
            command=self.show_vaila_message,
        )

        # B5_r5_c4 - vailá
        vaila_btn4 = tk.Button(
            row5_frame,
            text="vailá",
            width=button_width,
            command=self.show_vaila_message,
        )

        # B5_r5_c5 - vailá
        vaila_btn5 = tk.Button(
            row5_frame,
            text="vailá",
            width=button_width,
            command=self.show_vaila_message,
        )

        # Pack row5 buttons
        ultrasound_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        brainstorm_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn3.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn4.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        vaila_btn5.pack(side="left", expand=True, fill="x", padx=2, pady=2)

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

        # C_A_r1_c3 - Data Files: GapFill - interpolation and split data csv
        gapfill_btn = tk.Button(
            tools_col1,
            text="Smooth_Fill_Split",
            command=self.gapfill_split,  # Assuming this correctly calls the method in the Vaila class
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
            tools_col1, text="Make DLT3D", command=self.run_dlt3d, width=button_width
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

        # Avaliable blank (vailá) buttons for future tools
        # C_A_r4_c1 - Data Files: vailá
        reid_marker_btn = tk.Button(
            tools_col1,
            text="ReID Marker",
            command=self.reid_marker,
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

        # C_A_r5_c1 - Data Files: vailá
        vaila_btn12 = tk.Button(
            tools_col1,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )

        # C_A_r5_c2 - Data Files: vailá
        vaila_btn13 = tk.Button(
            tools_col1,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )

        # C_A_r5_c3 - Data Files: vailá
        vaila_btn14 = tk.Button(
            tools_col1,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )

        # Packing Data Files buttons
        reorder_csv_btn.grid(row=0, column=0, padx=2, pady=2)
        convert_btn.grid(row=0, column=1, padx=2, pady=2)
        gapfill_btn.grid(row=0, column=2, padx=2, pady=2)
        dlt2d_btn.grid(row=1, column=0, padx=2, pady=2)
        rec2d_one_btn.grid(row=1, column=1, padx=2, pady=2)
        rec2d_multiple_btn.grid(row=1, column=2, padx=2, pady=2)
        dlt3d_btn.grid(row=2, column=0, padx=2, pady=2)
        rec3d_one_btn.grid(row=2, column=1, padx=2, pady=2)
        rec3d_multiple_btn.grid(row=2, column=2, padx=2, pady=2)
        reid_marker_btn.grid(row=3, column=0, padx=2, pady=2)
        vaila_btn10.grid(row=3, column=1, padx=2, pady=2)
        vaila_btn11.grid(row=3, column=2, padx=2, pady=2)
        vaila_btn12.grid(row=4, column=0, padx=2, pady=2)
        vaila_btn13.grid(row=4, column=1, padx=2, pady=2)
        vaila_btn14.grid(row=4, column=2, padx=2, pady=2)
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
            tools_col2,
            text="Sync Video",
            command=self.cut_videos,
            width=button_width,
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
            text="Get Pixel Coord",
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
            text="Merge|Split Video",
            command=self.process_videos_gui,
            width=button_width,
        )

        # C_B_r4_c1 - Video: Distort Video/data
        vaila_distortvideo_btn = tk.Button(
            tools_col2,
            text="Distort Video/data",
            command=self.run_distortvideo,
            width=button_width,
        )

        # C_B_r4_c2 - Video: Cut Video
        cut_video_btn = tk.Button(
            tools_col2,
            text="Cut Video",
            command=self.cut_video,
            width=button_width,
        )

        # C_B_r4_c3 - Video: Resize Video
        resize_video_btn = tk.Button(
            tools_col2,
            text="Resize Video",
            command=self.resize_video,
            width=button_width,
        )

        # C_B_r5_c1 - Video: YT Downloader
        ytdownloader_btn = tk.Button(
            tools_col2,
            text="YT Downloader",
            command=self.ytdownloader,
            width=button_width,
        )

        # C_B_r5_c2 - Video: Insert Audio
        iaudiovid_btn = tk.Button(
            tools_col2,
            text="Insert Audio",
            command=self.run_iaudiovid,
            width=button_width,
        )

        # C_B_r5_c3 - Video: Remove Duplicate Frames
        remove_duplicate_frames_btn = tk.Button(
            tools_col2,
            text="rm Dup PNG",
            command=self.remove_duplicate_frames,
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
        vaila_distortvideo_btn.grid(row=3, column=0, padx=2, pady=2)
        cut_video_btn.grid(row=3, column=1, padx=2, pady=2)
        resize_video_btn.grid(row=3, column=2, padx=2, pady=2)
        ytdownloader_btn.grid(row=4, column=0, padx=2, pady=2)
        iaudiovid_btn.grid(row=4, column=1, padx=2, pady=2)
        remove_duplicate_frames_btn.grid(row=4, column=2, padx=2, pady=2)
        tools_col2.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        ## VVVVVVVVVVVVVVV VISUALIZATION BUTTONS VVVVVVVVVVVVVVVV
        # Visualization sub-columns
        # C_C_r1_c1 - Visualization: Show C3D
        show_c3d_btn = tk.Button(
            tools_col3, text="Show C3D", command=self.show_c3d_data, width=button_width
        )

        # C_C_r1_c2 - Visualization: Show CSV
        show_csv_btn = tk.Button(
            tools_col3, text="Show CSV 3D", command=self.show_csv_file, width=button_width
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

        # C_C_r3_c1 - Visualization: Draw Soccer Field
        draw_soccerfield_btn = tk.Button(
            tools_col3,
            text="Draw Soccer Field",
            command=self.draw_soccerfield,
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

        # C_C_r5_c1 - Visualization: vailá
        vaila_btn20 = tk.Button(
            tools_col3,
            text="vailá",
            command=self.show_vaila_message,
            width=button_width,
        )

        # C_C_r5_c2 - Visualization: vailá
        vaila_btn21 = tk.Button(
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
        draw_soccerfield_btn.grid(row=2, column=0, padx=2, pady=2)
        vaila_btn17.grid(row=2, column=1, padx=2, pady=2)
        vaila_btn18.grid(row=3, column=0, padx=2, pady=2)
        vaila_btn19.grid(row=3, column=1, padx=2, pady=2)
        vaila_btn20.grid(row=4, column=0, padx=2, pady=2)
        vaila_btn21.grid(row=4, column=1, padx=2, pady=2)
        tools_col3.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        # Help and Exit Buttons Frame
        bottom_frame = tk.Frame(scrollable_frame)
        bottom_frame.pack(pady=10)

        help_btn = tk.Button(bottom_frame, text="Help", command=self.display_help)
        exit_btn = tk.Button(bottom_frame, text="Exit", command=self.quit_app)

        help_btn.pack(side="left", padx=5)
        exit_btn.pack(side="left", padx=5)

        # Create a frame for the license label
        license_frame = tk.Frame(scrollable_frame)
        license_frame.pack(pady=5)

        # Label for the license text
        license_text = tk.Label(
            license_frame,
            text="© 2025 ",
            font=("default", 11),
            anchor="center",
        )
        license_text.pack(side="left")

        # Label clicável 'vailá'
        vaila_click = tk.Label(
            license_frame,
            text="vailá",
            font=("default", 11, "italic"),  # Set font to italic
            fg="blue",
            cursor="hand2",
        )
        vaila_click.pack(side="left")

        # Ao clicar, abre o link especificado
        vaila_click.bind(
            "<Button-1>",
            lambda e: webbrowser.open(
                "https://doi.org/10.48550/arXiv.2410.07238"
            ),  # Open the link
        )

        # Static label for the rest of the license text
        license_static = tk.Label(
            license_frame,
            text=" - Multimodal Toolbox. Licensed under the GNU Lesser General Public License v3.0.",
            font=("default", 11),
            anchor="center",
        )
        license_static.pack(side="left")

    # Class definition
    def show_vaila_message(self):
        """Runs the Vaila message module.

        This function runs the Vaila message module, which can be used to
        display a message about the Vaila application.

        """
        from vaila.vaila_manifest import show_vaila_message

        show_vaila_message()

    # A First FRAME Block
    # A_r1_c1
    def rename_files(self):
        """Rename files in a directory by replacing a string with another string.

        This function will prompt the user to select a directory containing the files
        to rename and will ask for the text to replace and the replacement text.

        """
        from vaila.filemanager import rename_files

        rename_files()

    # A_r1_c2
    def import_file(self):
        """Import a file from a directory to the current working directory.

        This function will prompt the user to select a source directory and file to
        import. The selected file will be copied to the current working directory.

        """
        from vaila.filemanager import import_file

        import_file()

    # A_r1_c3
    def export_file(self):
        """Export a file from the current working directory to a destination directory.

        This function will prompt the user to select a source file in the current working
        directory and a destination directory. The selected file will be copied from the
        current working directory to the destination directory.

        """
        from vaila.filemanager import export_file

        export_file()

    # A_r1_c4
    def copy_file(self):
        """Copy files from a source directory to a destination directory.

        This function will prompt the user to select a source directory and a destination
        directory. The selected files will be copied from the source directory to the
        destination directory.

        """
        from vaila.filemanager import copy_file

        copy_file()

    # A_r1_c5
    def move_file(self):
        """Move files from a source directory to a destination directory.

        This function will prompt the user to select a source directory and a destination
        directory. The selected files will be moved from the source directory to the
        destination directory.

        """
        from vaila.filemanager import move_file

        move_file()

    # A_r1_c6
    def remove_file(self):
        """Remove files from a directory.

        This function will prompt the user to select a directory and files to remove.
        The selected files will be removed from the directory.

        """
        from vaila.filemanager import remove_file

        remove_file()

    # A_r1_c7
    def tree_file(self):
        """Generate a tree structure of files in a directory.

        This function will prompt the user to select a directory and will generate a
        tree structure of the files in the directory.

        """
        from vaila.filemanager import tree_file

        tree_file()

    # A_r1_c8
    def find_file(self):
        """Find files in a directory.

        This function will prompt the user to select a directory and will find files
        in the directory.

        """
        from vaila.filemanager import find_file

        find_file()

    # A_r1_c9
    def transfer_file(self):
        """Transfer files between a local machine and a remote server using SSH.

        This function will prompt the user to select Upload or Download, and then
        select a source file or directory for upload or specify a destination
        directory for download.

        The function supports both local and remote file transfers, and will
        automatically create subdirectories in the destination directory to
        organize the transferred files.

        """
        from vaila.filemanager import transfer_file

        transfer_file()

    # B Second FRAME Block
    # B_r1_c1
    def imu_analysis(self):
        """Analyze IMU data.

        This function will analyze IMU data from a file.

        """
        from vaila import imu_analysis

        imu_analysis.analyze_imu_data()

    # B_r1_c2
    def cluster_analysis(self):
        """Analyze cluster data.

        This function will analyze cluster data from a file.

        """
        from vaila import cluster_analysis

        cluster_analysis.analyze_cluster_data()

    # B_r1_c3
    def mocap_analysis(self):
        """Analyze motion capture data.

        This function will analyze motion capture data from a file.

        """
        from vaila import mocap_analysis

        mocap_analysis.analyze_mocap_fullbody_data()

    # B_r1_c4
    def markerless_2d_analysis(self):
        """Runs the Markerless 2D Analysis module."""
        import tkinter as tk
        from tkinter import messagebox, simpledialog

        root = tk.Tk()
        root.withdraw()

        # Simple choice dialog
        choices = {
            "1": "Standard (Faster, single-person)",
            "2": "Advanced (Slower, multi-person with YOLO)",
        }

        choice = simpledialog.askstring(
            "Markerless 2D Analysis Version",
            "Select version:\n\n1: Standard (Faster, single-person)\n2: Advanced (Slower, multi-person with YOLO)",
            initialvalue="1",
        )

        if not choice or choice not in choices:
            return

        if choice == "1":
            from vaila.markerless_2D_analysis import process_videos_in_directory
        else:
            from vaila.markerless2d_analysis_v2 import (
                process_videos_in_directory,
            )  # Fixed module name

        process_videos_in_directory()

    # B_r1_c5
    def markerless_3d_analysis(self):
        """Runs the Markerless 3D Analysis module."""
        import tkinter as tk
        from tkinter import messagebox, simpledialog

        root = tk.Tk()
        root.withdraw()

        # Simple choice dialog
        choices = {
            "1": "Standard (Faster, single-person)",
            "2": "Advanced (Slower, multi-person with YOLO)",
        }

        choice = simpledialog.askstring(
            "Markerless 3D Analysis Version",
            "Select version:\n\n1: Standard (Faster, single-person)\n2: Advanced (Slower, multi-person with YOLO)",
            initialvalue="1",
        )

        if not choice or choice not in choices:
            return

        if choice == "1":
            from vaila.markerless_3D_analysis import process_videos_in_directory
        else:
            from vaila.markerless3d_analysis_v2 import process_videos_in_directory

        process_videos_in_directory()

    # B_r2_c1
    def vector_coding(self):
        """Runs the Vector Coding module.

        This function runs the Vector Coding module, which can be used to calculate
        the coupling angle between two joints from a C3D file containing 3D marker
        positions.

        The user will be prompted to select the file, the axis, and the names of the
        two joints. The module will then calculate the coupling angle between the two
        joints and save the result in a CSV file.
        """
        # Import directly from the run_vector_coding.py file
        from vaila.run_vector_coding import run_vector_coding

        run_vector_coding()

    # B_r2_c2
    def emg_analysis(self):
        """Analyze EMG data.

        This function will analyze EMG data from a file.

        """
        from vaila import emg_labiocom

        emg_labiocom.run_emg_gui()

    # B_r2_c3
    def force_analysis(self):
        """Analyze force plate data.

        This function will analyze force plate data from a file.

        """
        from vaila import forceplate_analysis

        forceplate_analysis.run_force_analysis()

    # B_r2_c4
    def gnss_analysis(self):
        """Runs the GNSS Analysis module.

        This function runs the GNSS Analysis module...
        """
        from vaila.gnss_analysis import run_gnss_analysis_gui

        run_gnss_analysis_gui()

    # B_r2_c5
    def eeg_analysis(self):
        """Runs the EEG Analysis module.

        This function runs the EEG Analysis module, which can be used to analyze EEG
        data from CSV files. It processes the EEG data to extract relevant metrics
        such as power spectral density, coherence, and phase-locking values. The
        module will then generate CSV files with the processed results and plots of
        the EEG signals.

        The user will be prompted to select the directory containing the EEG CSV
        files and input the sampling rate and start and end indices for analysis.

        """
        from vaila.vaila_manifest import show_vaila_message

        show_vaila_message()
        # eeg_analysis.run_eeg_analysis()

    # B_r3_c1
    def hr_analysis(self):
        """Runs the Heart Rate Analysis module.

        This function runs the Heart Rate Analysis module, which can be used to analyze
        heart rate data from CSV files. It processes the heart rate data to extract relevant
        metrics such as average heart rate, heart rate variability, and peak heart rate.
        The module will then generate CSV files with the processed results and plots of
        the heart rate signals.

        The user will be prompted to select the directory containing the heart rate CSV
        files and input the sampling rate and start and end indices for analysis.

        """
        from vaila.vaila_manifest import show_vaila_message

        show_vaila_message()
        # hr_analysis.run_hr_analysis()

    # B_r3_c2
    def cube2d_kinematics(self):
        """Run the 2D Cube Kinematics module.

        This function runs the 2D Cube Kinematics module, which can be used to analyze
        2D kinematics of a cube. The analysis will process the data and generate
        graphs and CSV files with the processed results.

        """
        from vaila import cube2d_kinematics

        cube2d_kinematics.run_cube2d_kinematics()

    # B_r3_c3
    def vailajump(self):
        """Runs the VailaJump module.

        This function runs the VailaJump module, which can be used to analyze VailaJump
        data from CSV files. It processes the VailaJump data to extract relevant
        metrics such as acceleration, speed, and distance. The module will then
        generate CSV files with the processed results and plots of the VailaJump
        signals.

        The user will be prompted to select the directory containing the VailaJump CSV
        files and input the sampling rate and start and end indices for analysis.

        """
        from vaila.vaila_and_jump import vaila_and_jump

        vaila_and_jump()

    # B_r3_c4
    def markerless2d_mpyolo(self):
        """Runs the markerless2d_mpyolo analysis."""
        from vaila import markerless2d_mpyolo

        markerless2d_mpyolo.run_markerless2d_mpyolo()

    # B_r3_c5
    def animal_open_field(self):
        """Run the Animal Open Field Analysis module.

        This function runs the Animal Open Field Analysis module, which can be used to analyze
        animal movement data in an open field test. The analysis will process the data and generate
        graphs and CSV files with the processed results.

        """
        from vaila import animal_open_field

        animal_open_field.run_animal_open_field()

    # B_r4_c1
    def tracker(self):
        """Runs the specified YOLO tracking analysis."""
        print(f"Running tracker analysis {os.path.dirname(os.path.abspath(__file__))}")
        print(f"Running tracker analysis {os.path.basename(__file__)}")
        # print number of this line of code
        # Esta linha não vai funcionar como esperado para mostrar a linha exata do código
        # Mas pode ser removida ou apenas exibir o nome do arquivo.
        print("Line number: 1616")  # Melhor assim, ou remova

        # Create a dialog window for tracking version selection
        dialog = tk.Toplevel(self)
        dialog.title("Select YOLO Version")
        dialog.geometry("400x320")
        dialog.transient(self)  # Make dialog modal
        dialog.grab_set()

        tk.Label(dialog, text="Select YOLO tracker version to use:", pady=15).pack()

        def use_yolov12():
            dialog.destroy()
            try:
                # Import inside function to avoid early loading
                import sys

                # Add the site-packages path first to ensure proper numpy loading
                site_packages = os.path.join(
                    os.path.dirname(sys.executable), "Lib", "site-packages"
                )
                if site_packages not in sys.path:
                    sys.path.insert(0, site_packages)

                from vaila import yolov12track

                yolov12track.run_yolov12track()
            except Exception as e:
                messagebox.showerror("Error Running YOLOv12", f"Error: {str(e)}")

        def use_yolov11():
            dialog.destroy()
            try:
                # Import inside function to avoid early loading
                import sys

                # Add the site-packages path first to ensure proper numpy loading
                site_packages = os.path.join(
                    os.path.dirname(sys.executable), "Lib", "site-packages"
                )
                if site_packages not in sys.path:
                    sys.path.insert(0, site_packages)

                from vaila import yolov11track

                yolov11track.run_yolov11track()
            except Exception as e:
                messagebox.showerror(
                    "Error Running YOLOv11",
                    f"Error: {str(e)}",
                )

        def use_train_yolov11():
            dialog.destroy()
            try:
                import sys
                from vaila import yolotrain

                yolotrain.run_yolotrain_gui()
            except Exception as e:
                messagebox.showerror("Error in YOLO Training", f"Error: {str(e)}")

        tk.Button(dialog, text="YOLOv12 Tracker", command=use_yolov12, width=20).pack(
            pady=10
        )
        tk.Button(dialog, text="YOLOv11 Tracker", command=use_yolov11, width=20).pack(
            pady=10
        )
        tk.Button(dialog, text="Train YOLO", command=use_train_yolov11, width=20).pack(
            pady=10
        )
        tk.Button(dialog, text="Cancel", command=dialog.destroy, width=10).pack(pady=10)

        # Wait for the dialog to be closed
        self.wait_window(dialog)

    # B_r4_c2 - ML Walkway
    def ml_walkway(self):
        """Invokes the vaila_mlwalkway module."""
        from vaila import vaila_mlwalkway

        vaila_mlwalkway.run_vaila_mlwalkway_gui()

    # B_r4_c3 - Markerless Hands
    def markerless_hands(self):
        """Invokes the vaila_mphands module."""
        from vaila import mphands

        mphands.run_mphands()

    # B_r4_c4 - MP Angles
    def mp_angles_calculation(self):
        """Runs the MP Angles module."""
        from vaila import mpangles

        mpangles.run_mp_angles()

    # B_r4_c5 - Markerless Live
    def markerless_live(self):
        """Runs the Markerless Live module."""
        from vaila import markerless_live

        markerless_live.run_markerless_live()

    # B_r5_c1 - Ultrasound
    def ultrasound(self):
        """Runs the Ultrasound module."""
        from vaila import usound_biomec1

        usound_biomec1.run_usound()

    # B_r5_c2 - Brainstorm
    def brainstorm(self):
        """Runs the Brainstorm module."""
        from vaila import brainstorm

        brainstorm.run_brainstorm()

    # C_r1_c1
    def reorder_csv_data(self):
        """Runs the Reorder CSV Data module.

        This function runs the Reorder CSV Data module, which can be used to reorder the
        columns of CSV files. It allows the user to select the directory containing the
        CSV files and reorder the columns according to their preference. The module will
        then save the reordered CSV files in a new directory.

        The user will be prompted to select the directory containing the CSV files.

        """
        from vaila import rearrange_data

        rearrange_data.rearrange_data_in_directory()  # Edit CSV

    # C_A_r1_c2
    def convert_c3d_csv(self):
        # Create a new window for the action choice
        """Runs the Convert C3D/CSV module.

        This function runs the Convert C3D/CSV module, which can be used to convert
        C3D files to CSV format and vice versa. It allows the user to select the
        directory containing the C3D files and choose whether to convert them to
        CSV or vice versa.

        The user will be prompted to select the directory containing the C3D files.

        """
        print(f"C_A_r1_c2 - def convert_c3d_csv on code line number 1589")
        from vaila import convert_c3d_to_csv
        from vaila import convert_csv_to_c3d

        window = Toplevel()
        window.title("Choose Action")
        # Message to the user
        label = Label(window, text="Which conversion would you like to perform?")
        label.pack(pady=10)

        # Button for C3D -> CSV
        button_c3d_to_csv = Button(
            window,
            text="C3D -> CSV",
            command=lambda: [convert_c3d_to_csv(), window.destroy()],
        )
        button_c3d_to_csv.pack(side="left", padx=20, pady=20)

        # Button for CSV -> C3D
        button_csv_to_c3d = Button(
            window,
            text="CSV -> C3D",
            command=lambda: [convert_csv_to_c3d(), window.destroy()],
        )
        button_csv_to_c3d.pack(side="right", padx=20, pady=20)

    # C_A_r1_c3
    def gapfill_split(self):
        """Runs the Linear Interpolation or Split Data module.

        This function runs the Linear Interpolation or Split Data module, which can be used to fill
        missing data in CSV files using linear interpolation or to split CSV files into two parts.

        The user will be prompted to select the directory containing the CSV files and choose
        whether to perform linear interpolation or splitting.

        """
        from vaila.interp_smooth_split import run_fill_split_dialog

        run_fill_split_dialog()

    # C_A_r2_c1
    def dlt2d(self):
        """Runs the DLT2D module.

        This function runs the DLT2D module, which can be used to perform 2D direct linear
        transformation (DLT) calibration of a camera. The module will then generate a
        calibration file that can be used for 2D reconstruction.

        The user will be prompted to select the directory containing the CSV files and
        input the sample rate and start and end indices for analysis.

        """
        from vaila import dlt2d

        dlt2d.run_dlt2d()

    # C_A_r2_c2
    def rec2d_one_dlt2d(self):
        """Runs the Reconstruction 2D module with one DLT2D.

        This function runs the Reconstruction 2D module with one DLT2D, which can be used to
        perform 2D reconstruction of 2D coordinates using a single DLT parameters file.

        The user will be prompted to select the directory containing the CSV files and
        input the sample rate and start and end indices for analysis.

        """
        from vaila import rec2d_one_dlt2d

        rec2d_one_dlt2d.run_rec2d_one_dlt2d()

    # C_A_r2_c3 - for multi dlts in rows
    def rec2d(self):
        """Runs the Reconstruction 2D module.

        This function runs the Reconstruction 2D module, which can be used to perform 2D
        reconstruction of 2D coordinates using a set of DLT parameters files.

        The user will be prompted to select the directory containing the CSV files and
        input the sample rate and start and end indices for analysis.

        """
        from vaila import rec2d

        rec2d.run_rec2d()

    # C_A_r3_c1
    def run_dlt3d(self):
        """
        Método para executar o script dlt3d.py.
        Essa função utiliza o interpretador Python (sys.executable) para chamar o script,
        que se encontra na pasta "vaila".
        """
        try:
            # Build the path to the dlt3d.py script
            script_path = os.path.join("vaila", "dlt3d.py")
            # Execute the script in a new process
            subprocess.Popen([sys.executable, script_path])
        except Exception as e:
            messagebox.showerror("Error", f"Error running dlt3d.py: {e}")

    # C_A_r3_c2 - for multi dlts in rows
    def rec3d_one_dlt3d(self):
        """Runs the Reconstruction 3D module with one DLT3D.

        This function runs the Reconstruction 3D module with one DLT3D, which can be used to
        perform 3D reconstruction of 3D coordinates using a single DLT parameters file.

        The user will be prompted to select the directory containing the CSV files and
        input the sample rate and start and end indices for analysis.

        """
        from vaila import rec3d_one_dlt3d

        rec3d_one_dlt3d.run_rec3d_one_dlt3d()

    # C_A_r3_c3 - for multi dlts in rows
    def rec3d(self):
        """Runs the Reconstruction 3D module.

        This function runs the Reconstruction 3D module with multiple DLT3D, which can be used to
        perform 3D reconstruction of 3D coordinates using multiple DLT parameters files.

        The user will be prompted to select the directory containing the CSV files and
        input the sample rate and start and end indices for analysis.

        """
        pass  # Aqui você deve adicionar a lógica para a reconstrução 3D com múltiplos DLTs

    # C_A_r4_c1 - ReID Marker
    def reid_marker(self):
        """Runs the ReID Marker module."""
        from vaila import reid_markers

        reid_markers.create_gui_menu()

    # C_A_r4_c2
    # def vaila(self):

    # C_A_r4_c3
    # def vaila(self):

    # C_B_r1_c1
    def extract_png_from_videos(self):
        """Runs the video to PNG frame extraction module.

        This function runs the video to PNG frame extraction module, which can be used to
        extract PNG frames from video files. The module will prompt the user to select the
        directory containing the video files and input the sample rate and start and end
        indices for analysis.

        """
        from vaila.extractpng import VideoProcessor

        processor = VideoProcessor()
        processor.run()

    # C_B_r1_c2
    def cut_videos(self):
        """Runs the batch video cutting module.

        This function runs the batch video cutting module, which can be used to
        cut videos based on a list of specified time intervals. The module will
        prompt the user to select the directory containing the video files and
        input the list of time intervals for analysis.

        """
        from vaila import cutvideo

        cutvideo.run_cutvideo()

    # C_B_r1_c3
    def draw_box(self):
        """Runs the video box drawing module.

        This function runs the video box drawing module, which can be used to
        draw a box around a region of interest in video files. The module will
        prompt the user to select the directory containing the video files and
        input the coordinates for the box.

        """
        from vaila import drawboxe

        drawboxe.run_drawboxe()

    # C_B_r2_c1
    def compress_videos_h264_gui(self):
        """Runs the video compression module for H.264 format.

        This function runs the video compression module for H.264 format, which can be used to
        compress video files. The module will prompt the user to select the directory containing
        the video files and input the sample rate and start and end indices for analysis.

        """
        from vaila import compress_videos_h264

        compress_videos_h264.compress_videos_h264_gui()

    # C_B_r2_c2
    def compress_videos_h265_gui(self):
        """Runs the video compression module for H.265 format.

        This function runs the video compression module for H.265 format, which can be used to
        compress video files. The module will prompt the user to select the directory containing
        the video files and input the sample rate and start and end indices for analysis.

        """
        from vaila import compress_videos_h265

        compress_videos_h265.compress_videos_h265_gui()

    # C_B_r2_c3
    def sync_videos(self):
        """Runs the video synchronization module.

        This function runs the video synchronization module, which can be used to
        synchronize multiple video files based on a flash or brightness change.
        The module will prompt the user to select the directory containing the
        video files and input the sample rate and start and end indices for
        analysis.

        """
        from vaila import syncvid

        syncvid.sync_videos()

    # C_B_r3_c1
    def getpixelvideo(self):
        """Runs the video pixel marking module.

        This function runs the video pixel marking module, which can be used to
        mark specific pixels in a video file. The module will prompt the user to
        select the directory containing the video files and input the coordinates
        and sample rate for analysis.

        """
        from vaila import getpixelvideo

        getpixelvideo.run_getpixelvideo()

    # C_B_r3_c2
    def count_frames_in_videos(self):
        """Runs the video frame counting module.

        This function runs the video frame counting module, which can be used to
        count the number of frames in a video file. The module will prompt the
        user to select the directory containing the video files and input the
        sample rate and start and end indices for analysis.

        """
        from vaila import numberframes

        numberframes.count_frames_in_videos()

    # C_B_r3_c3
    def process_videos_gui(self):
        """Runs the video processing module.

        This function runs the video processing module, which can be used to
        process video files. The module will prompt the user to select the
        directory containing the video files and input the processing parameters.

        """
        from vaila import videoprocessor

        videoprocessor.process_videos_gui()

    # C_B_r4_c1
    def run_distortvideo(self):
        """Runs the Lens Distortion Correction Module.

        This method provides options to correct lens distortion in either videos or CSV/DAT coordinate files,
        or via an interactive GUI. All use the same camera calibration parameters.
        """
        print(f"C_B_r4_c1 - def run_distortvideo on code line number 1846")
        # bring in everything we need
        from tkinter import Toplevel, Label, Button
        from vaila import (
            vaila_lensdistortvideo,
            vaila_distortvideo_gui,
            vaila_datdistort,
        )

        # Create dialog window
        dialog = Toplevel(self)
        dialog.title("Choose Distortion Correction Type")
        dialog.geometry("300x300")  # enlarged for the third option

        # Descriptive label
        Label(dialog, text="Select the type of distortion correction:", pady=10).pack()

        # Video Correction
        Button(
            dialog,
            text="Video Correction",
            command=lambda: (
                vaila_lensdistortvideo.run_distortvideo(),
                dialog.destroy(),
            ),
        ).pack(pady=5)

        # Interactive GUI Correction
        Button(
            dialog,
            text="Interactive Distortion Correction",
            command=lambda: (
                vaila_distortvideo_gui.run_distortvideo_gui(),
                dialog.destroy(),
            ),
        ).pack(pady=5)

        # CSV/DAT Coordinates Correction
        Button(
            dialog,
            text="CSV/DAT Coordinates Correction",
            command=lambda: (
                vaila_datdistort.run_datdistort(),
                dialog.destroy(),
            ),
        ).pack(pady=5)

        # Cancel button
        Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=10)

        # Make dialog modal
        dialog.transient(self)
        dialog.grab_set()
        self.wait_window(dialog)

    # C_B_r4_c2
    def cut_video(self):
        """Runs the video cutting module.

        This function runs the video cutting module, which can be used to
        cut a video file. The module will prompt the user to select the video
        file and input the start and end times for cutting.

        """
        from vaila import cutvideo

        cutvideo.run_cutvideo()

    def resize_video(self):
        """Runs the video resizing module.

        This function runs the video resizing module, which can be used to
        resize video files. The module will prompt the user to select the
        directory containing the video files and input the sample rate and start
        and end indices for analysis.
        """
        from vaila import resize_video

        resize_video.run_resize_video()

    # C_B_r5_c1
    def ytdownloader(self):
        """Runs the YouTube downloader module.

        This function runs the YouTube downloader module, which can be used to
        download videos from YouTube. The module will prompt the user to select
        the directory containing the video files and input the sample rate and start
        and end indices for analysis.
        """
        from vaila import vaila_ytdown

        vaila_ytdown.run_ytdown()

    # C_B_r5_c2
    def run_iaudiovid(self):
        """Runs the Insert Audio module.

        This function runs the Insert Audio module, which can be used to
        insert audio into video files. The module will prompt the user to select
        the directory containing the video files and input the sample rate and start
        and end indices for analysis.
        """
        from vaila import vaila_iaudiovid

        vaila_iaudiovid.run_iaudiovid()

    # C_B_r5_c3
    def remove_duplicate_frames(self):
        """Runs the Remove Duplicate Frames module.

        This function runs the Remove Duplicate Frames module, which can be used to
        remove duplicate frames from video files.
        """
        from vaila import rm_duplicateframes

        rm_duplicateframes.run_rm_duplicateframes()

    # C_C_r1_c1
    def show_c3d_data(self):
        """Runs the C3D file viewer module.

        This function runs the C3D file viewer module, which can be used to
        view C3D files. The module will prompt the user to select the C3D file
        to view.

        """
        from vaila.viewc3d import run_viewc3d

        run_viewc3d()

    # C_C_r1_c2
    def show_csv_file(self):
        """Runs the CSV file viewer module.

        This function runs the CSV file viewer module, which can be used to
        view CSV files. The module will prompt the user to select the CSV file
        to view.

        """
        from vaila.readcsv import show_csv

        show_csv()

    # C_C_r1_c3
    def plot_2d_data(self):
        """Runs the plot_2d_data module.

        This function runs the plot_2d_data module, which can be used to
        visualize data from CSV files using Matplotlib, with column selection
        interface and line animation.

        """
        from vaila import vailaplot2d

        vailaplot2d.run_plot_2d()

    # C_C_r2_c2
    def plot_3d_data(self):
        """Runs the 3D data plotting module.

        This function runs the 3D data plotting module, which can be used to
        plot 3D data from CSV files. The module will prompt the user to select
        the CSV file and input the plotting parameters.

        """
        from vaila import vailaplot3d

        vailaplot3d.run_plot_3d()

    # C_C_r3_c1
    def draw_soccerfield(self):
        """Runs the soccer field drawing module.

        This function runs the soccer field drawing module, which can be used to
        visualize a soccer field using Matplotlib.

        """
        from vaila import soccerfield

        soccerfield.run_soccerfield()

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
        """Displays the help file for the Multimodal Toolbox.

        The help file is a static HTML file located in the "docs" directory
        of the Multimodal Toolbox source code. If the file is not found,
        an error message is shown.
        """
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
        """Opens a new terminal with the Conda environment activated using xonsh.

        The Multimodal Toolbox provides a convenient way to open a new terminal with
        the Conda environment activated. On macOS, the Terminal app is used. On Windows,
        PowerShell 7 is used, and on Linux, the default terminal emulator is used.

        The Conda environment is activated using the `conda activate` command, and then
        the xonsh shell is started. This allows you to access the Conda environment and
        any packages installed in it, as well as the xonsh shell features.

        Note that this function does not work if the Multimodal Toolbox is not installed
        in the default location. If you have installed the Multimodal Toolbox in a custom
        location, you should use the `conda activate` command manually to activate the
        environment and then start xonsh.
        """
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
            # Open PowerShell 7 and activate the Conda environment
            subprocess.Popen(
                "start pwsh -NoExit -Command \"& 'C:\\ProgramData\\anaconda3\\shell\\condabin\\conda-hook.ps1'; conda activate vaila; xonsh\"",
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
        """Quits the Multimodal Toolbox application.

        This method is called when the user clicks the "Exit" menu option.
        It first calls the `destroy` method on the root window to close the
        application window, and then kills the process using the `os.kill`
        method to ensure that any spawned processes are also terminated.

        """
        self.destroy()
        os.kill(os.getpid(), signal.SIGTERM)

    def plot_2d(self):
        """Runs the 2D plotting module.

        This function runs the 2D plotting module, which can be used to
        create 2D plots from data files.

        """
        from vaila.vailaplot2d import run_plot_2d

        run_plot_2d()

    def plot_3d(self):
        """Runs the 3D plotting module.

        This function runs the 3D plotting module, which can be used to
        create 3D plots from data files.

        """
        from vaila.vailaplot3d import run_plot_3d

        run_plot_3d()


if __name__ == "__main__":
    app = Vaila()
    app.mainloop()
