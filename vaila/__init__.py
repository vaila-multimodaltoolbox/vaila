"""Public vailá entry points, imported only when requested.

Version: 0.3.137
Update Date: 11 September 2026
"""

from importlib import import_module

_EXPORTS = {
    "cluster_analysis": ("cluster_analysis", None),
    "forceplate_analysis": ("forceplate_analysis", None),
    "imu_analysis": ("imu_analysis", None),
    "batch_cut_videos": ("batchcut", "batch_cut_videos"),
    "compress_videos_h264_gui": ("compress_videos_h264", "compress_videos_h264_gui"),
    "compress_videos_h265_gui": ("compress_videos_h265", "compress_videos_h265_gui"),
    "read_cluster_csv": ("data_processing", "read_cluster_csv"),
    "read_mocap_csv": ("data_processing", "read_mocap_csv"),
    "run_drawboxe": ("drawboxe", "run_drawboxe"),
    "run_edit_csv_c3d": ("edit_csv_c3d", "run_edit_csv_c3d"),
    "VideoProcessor": ("extractpng", "VideoProcessor"),
    "copy_file": ("filemanager", "copy_file"),
    "export_file": ("filemanager", "export_file"),
    "find_file": ("filemanager", "find_file"),
    "import_file": ("filemanager", "import_file"),
    "move_file": ("filemanager", "move_file"),
    "remove_file": ("filemanager", "remove_file"),
    "rename_files": ("filemanager", "rename_files"),
    "transfer_file": ("filemanager", "transfer_file"),
    "tree_file": ("filemanager", "tree_file"),
    "butter_filter": ("filter_utils", "butter_filter"),
    "merge_csv_files": ("mergestack", "merge_csv_files"),
    "stack_csv_files": ("mergestack", "stack_csv_files"),
    "get_labcoord_angles": ("modifylabref", "get_labcoord_angles"),
    "modify_lab_coords": ("modifylabref", "modify_lab_coords"),
    "plot_orthonormal_bases": ("plotting", "plot_orthonormal_bases"),
    "convert_c3d_to_csv": ("readc3d_export", "convert_c3d_to_csv"),
    "get_csv_headers": ("readcsv", "get_csv_headers"),
    "headersidx": ("readcsv", "headersidx"),
    "reshapedata": ("readcsv", "reshapedata"),
    "select_file": ("readcsv", "select_file"),
    "select_headers_gui": ("readcsv", "select_headers_gui"),
    "show_csv": ("readcsv", "show_csv"),
    "convert_csv_to_c3d": ("readcsv_export", "convert_csv_to_c3d"),
    "create_c3d_from_csv": ("readcsv_export", "create_c3d_from_csv"),
    "rearrange_data_in_directory": ("rearrange_data", "rearrange_data_in_directory"),
    "calcmatrot": ("rotation", "calcmatrot"),
    "createortbase": ("rotation", "createortbase"),
    "rotdata": ("rotation", "rotdata"),
    "rotmat2euler": ("rotation", "rotmat2euler"),
    "show_c3d": ("showc3d", "show_c3d"),
    "show_points_3d": ("showc3d", "show_points_3d"),
    "plot_2d": ("vailaplot2d", "run_plot_2d"),
    "plot_3d": ("vailaplot3d", "run_plot_3d"),
    "process_videos_gui": ("videoprocessor", "process_videos_gui"),
}

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
    "run_edit_csv_c3d",
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
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, attribute) if attribute else module
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
