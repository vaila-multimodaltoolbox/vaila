"""
Module: vaila
File: __init__.py

Description:
This is the initialization script for the vaila multimodal toolbox, which serves as the entry point for all modules and functions within the package. This file aggregates all the essential functionalities and provides a streamlined interface for importing and using the various tools available in the package.

Version: 2024-08-16 12:00:00

Changelog:
- 2024-08-16: Added `process_videos_gui` functionality for video processing. This update integrates the video processing GUI into the package, allowing users to define and concatenate video segments through a user-friendly interface.

Author: Prof. Paulo Santiago
"""

from .data_processing import read_cluster_csv, read_mocap_csv
from .filtering import apply_filter
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
from .batchcut import batch_cut_videos, cut_videos
from .drawboxe import run_drawboxe
from . import cluster_analysis
from . import imu_analysis
from . import markerless_2D_analysis
from . import markerless_3D_analysis
from . import mocap_analysis
from .filemanager import import_file, export_file, copy_file, move_file, remove_file
from .showc3d import show_c3d
from .vector_coding import vector_coding
from .syncvid import sync_videos
from .compress_videos_h264 import compress_videos_h264_gui
from .compress_videos_h265 import compress_videos_h265_gui
from .extractpng import VideoProcessor
from .readcsv_export import create_c3d_from_csv, convert_csv_to_c3d
from .getpixelvideo import main as getpixelvideo
from .dlt2d import main as dlt2d
from .rec2d import main as rec2d
from .rec2d_one_dlt2d import main as rec2d_one_dlt2d
from .vaila_manifest import show_vaila_message
from .emg_labiocom import run_emg_gui
from .vailaplot2d import plot_2d
from .mergestack import merge_csv_files, stack_csv_files
from .videoprocessor import process_videos_gui


__all__ = [
    "read_cluster_csv",
    "read_mocap_csv",
    "apply_filter",
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
    "cut_videos",
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
    "import_file",
    "export_file",
    "copy_file",
    "move_file",
    "remove_file",
    "show_c3d",
    "vector_coding",
    "sync_videos",
    "VideoProcessor",  # Agora exportando a classe VideoProcessor
    "select_file",
    "show_csv",
    "get_csv_headers",
    "select_headers_gui",
    "getpixelvideo",
    "dlt2d",
    "rec2d",
    "rec2d_one_dlt2d",
    "show_vaila_message",
    "run_emg_gui",
    "plot_2d",
    "merge_csv_files",
    "stack_csv_files",
    "process_videos_gui",  # Export the GUI function for video processing
]
