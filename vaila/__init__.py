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

from . import cluster_analysis, forceplate_analysis, imu_analysis
from .batchcut import batch_cut_videos
from .compress_videos_h264 import compress_videos_h264_gui
from .compress_videos_h265 import compress_videos_h265_gui
from .data_processing import read_cluster_csv, read_mocap_csv
from .drawboxe import run_drawboxe
from .extractpng import VideoProcessor
from .filemanager import (
    copy_file,
    export_file,
    find_file,
    import_file,
    move_file,
    remove_file,
    rename_files,
    transfer_file,
    tree_file,
)
from .filter_utils import butter_filter
from .mergestack import merge_csv_files, stack_csv_files
from .modifylabref import get_labcoord_angles, modify_lab_coords
from .plotting import plot_orthonormal_bases
from .readc3d_export import convert_c3d_to_csv
from .readcsv import (
    get_csv_headers,
    headersidx,
    reshapedata,
    select_file,
    select_headers_gui,
    show_csv,
)
from .readcsv_export import convert_csv_to_c3d, create_c3d_from_csv
from .rearrange_data import rearrange_data_in_directory
from .rotation import calcmatrot, createortbase, rotdata, rotmat2euler
from .showc3d import show_c3d
from .spectral_features import (
    centroid_frequency,
    energy_content_0_5_2,
    energy_content_above_2,
    energy_content_below_0_5,
    frequency_dispersion,
    frequency_quotient,
    power_frequency_50,
    power_frequency_95,
    power_mode,
    total_power,
)
from .vailaplot2d import run_plot_2d as plot_2d
from .vailaplot3d import run_plot_3d as plot_3d
from .videoprocessor import process_videos_gui

__all__ = [
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
    "forceplate_analysis",
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
    "VideoProcessor",
    "select_file",
    "show_csv",
    "get_csv_headers",
    "select_headers_gui",
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
