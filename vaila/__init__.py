"""
Module: vaila
File: __init__.py

Description:
This is the initialization script for the vaila multimodal toolbox, which serves as the entry point for all modules and functions within the package. This file aggregates all the essential functionalities and provides a streamlined interface for importing and using the various tools available in the package.

Version: 2024-08-16 12:00:00

Changelog:
- 2024-08-16: Added `process_videos_gui` functionality for video processing. This update integrates the video processing GUI into the package, allowing users to define and concatenate video segments through a user-friendly interface.
- 2024-10-09: Added `linear_interpolation_split` functionality for applying linear interpolation and data splitting. This update provides enhancing data_processingta cleaning and manipulation features.

Author: Prof. Paulo Santiago
"""

from .ellipse import plot_ellipse_pca, plot_cop_pathway_with_ellipse
from .data_processing import read_cluster_csv, read_mocap_csv
from .filter_utils import butter_filter
from .plotting import plot_orthonormal_bases
from .rotation import rotdata, createortbase, calcmatrot, rotmat2euler
from .readcsv import (
    headersidx,
    reshapedata,
    select_file,
    select_headers_gui,
    get_csv_headers,
    show_csv,
)
from .rearrange_data import rearrange_data_in_directory
from .readc3d_export import convert_c3d_to_csv
from .modifylabref import modify_lab_coords, get_labcoord_angles
from .numberframes import count_frames_in_videos
from .batchcut import batch_cut_videos
from .drawboxe import run_drawboxe
from . import cluster_analysis
from . import imu_analysis
from . import markerless_3D_analysis
from . import mocap_analysis
from . import forceplate_analysis
from .filemanager import (
    import_file,
    export_file,
    copy_file,
    move_file,
    remove_file,
    rename_files,
    tree_file,
    find_file,
    transfer_file,
)
from .showc3d import show_c3d
from .syncvid import sync_videos
from .compress_videos_h264 import compress_videos_h264_gui
from .compress_videos_h265 import compress_videos_h265_gui
from .extractpng import VideoProcessor
from .readcsv_export import create_c3d_from_csv, convert_csv_to_c3d
from .vaila_manifest import show_vaila_message
from .emg_labiocom import run_emg_gui
from .vailaplot2d import run_plot_2d as plot_2d
from .vailaplot3d import run_plot_3d as plot_3d
from .mergestack import merge_csv_files, stack_csv_files
from .videoprocessor import process_videos_gui
from .sync_flash import get_median_brightness
from .spectral_features import (
    total_power,
    power_frequency_50,
    power_frequency_95,
    power_mode,
    centroid_frequency,
    frequency_dispersion,
    energy_content_below_0_5,
    energy_content_0_5_2,
    energy_content_above_2,
    frequency_quotient,
)

__all__ = [
    "plot_ellipse_pca",
    "plot_cop_pathway_with_ellipse",
    "read_cluster_csv",
    "read_mocap_csv",
    "butter_filter",
    "plot_orthonormal_bases",
    "rotdata",
    "createortbase",
    "calcmatrot",
    "rotmat2euler",
    "headersidx",
    "reshapedata",
    "rearrange_data_in_directory",
    "count_frames_in_videos",
    "batch_cut_videos",
    "run_drawboxe",
    "compress_videos_h264_gui",
    "compress_videos_h265_gui",
    "convert_c3d_to_csv",
    "create_c3d_from_csv",
    "convert_csv_to_c3d",
    "modify_lab_coords",
    "get_labcoord_angles",
    "cluster_analysis",
    "imu_analysis",
    "markerless_2D_analysis",
    "markerless_3D_analysis",
    "mocap_analysis",
    "forceplate_analysis",
    "gnss_analysis",
    "import_file",
    "export_file",
    "copy_file",
    "move_file",
    "remove_file",
    "rename_files",
    "tree_file",
    "find_file",
    "transfer_file",
    "show_c3d",
    "vector_coding",
    "sync_videos",
    "VideoProcessor",
    "select_file",
    "show_csv",
    "get_csv_headers",
    "select_headers_gui",
    "getpixelvideo",
    "show_vaila_message",
    "run_emg_gui",
    "plot_2d",
    "plot_3d",
    "merge_csv_files",
    "stack_csv_files",
    "process_videos_gui",
    "get_median_brightness",
    "total_power",
    "power_frequency_50",
    "power_frequency_95",
    "power_mode",
    "centroid_frequency",
    "frequency_dispersion",
    "energy_content_below_0_5",
    "energy_content_0_5_2",
    "energy_content_above_2",
    "frequency_quotient",
]
