"""CLI run hints for vailá menu entries (headless ``--cli`` mode)."""

from __future__ import annotations

from dataclasses import dataclass

__version__ = "0.3.113"
__updated__ = "24 August 2026"


@dataclass(frozen=True, slots=True)
class CliRunHint:
    commands: tuple[str, ...]
    note: str = ""
    invoke_handler: bool = False  # rare: HELP browser, etc.


_GUI_FALLBACK_NOTE = (
    "Tkinter GUI — launch the desktop app and use this button, "
    "or run the module CLI below when available."
)

# Handler → copy-paste launcher(s). Chooser buttons expand to sub-tools.
CLI_HINTS_BY_HANDLER: dict[str, CliRunHint] = {
    # Frame A — file manager (GUI dialogs; no standalone CLI)
    "rename_files": CliRunHint(("uv run vaila.py",), _GUI_FALLBACK_NOTE),
    "import_file": CliRunHint(("uv run vaila.py",), _GUI_FALLBACK_NOTE),
    "export_file": CliRunHint(("uv run vaila.py",), _GUI_FALLBACK_NOTE),
    "copy_file": CliRunHint(("uv run vaila.py",), _GUI_FALLBACK_NOTE),
    "move_file": CliRunHint(("uv run vaila.py",), _GUI_FALLBACK_NOTE),
    "remove_file": CliRunHint(("uv run vaila.py",), _GUI_FALLBACK_NOTE),
    "tree_file": CliRunHint(("uv run vaila.py",), _GUI_FALLBACK_NOTE),
    "find_file": CliRunHint(("uv run vaila.py",), _GUI_FALLBACK_NOTE),
    "transfer_file": CliRunHint(
        ("uv run vaila/transfer.sh",),
        "SSH/rsync transfer — interactive shell script.",
    ),
    # Frame B
    "imu_analysis": CliRunHint(("uv run vaila/imu_analysis.py",), "Opens IMU analysis GUI."),
    "cluster_analysis": CliRunHint(("uv run vaila/cluster_analysis.py",), "MoCap cluster GUI."),
    "mocap_analysis": CliRunHint(("uv run vaila/mocap_analysis.py",), "Full-body MoCap GUI."),
    "markerless_2d_analysis": CliRunHint(
        (
            "uv run vaila/markerless_2d_analysis.py",
            "uv run vaila/markerless2d_analysis_v2.py",
            "uv run vaila/markerless2d_yolo26.py",
            "uv run vaila/markerless2d_mpyolo.py",
            "uv run python -u -m vaila.yolov26track",
            "uv run python -u -m vaila.yolotrain",
            "uv run python -u vaila/vaila_sam.py",
            "uv run python -u vaila/vaila_sapiens.py",
            "uv run python -u vaila/sam3sapiens2.py",
            "uv run python -u vaila/sam3sapiens2_visualize.py",
            "uv run vaila/mphands.py",
            "uv run vaila/mpangles.py",
            "uv run vaila/mp_facemesh.py",
            "uv run vaila/markerless_live.py",
        ),
        "Markerless 2D chooser (coringa) — Standard/Advanced/YOLOv26 MediaPipe, "
        "Yolo + Markerless_MP, YOLOv26 Tracker/Pose/Seg/Train, SAM 3, Sapiens2, "
        "SAM3+Sapiens2 and its Visualize ID, Markerless Hands, MP Angles, Face Mesh, "
        "Markerless Live. Each tool prints its own Run CLI after dialogs.",
    ),
    "markerless_3d_analysis": CliRunHint(
        (
            "uv run vaila/markerless_3d_analysis.py",
            "uv run vaila/markerless3d_analysis_v2.py",
            "uv run python -u vaila/sam3dinov3.py",
            "uv run python -u vaila/sam3dinov3_visualize.py",
            "uv run python -u vaila/sapiens2_3d.py",
            "uv run python -m vaila.monocular_dlt_align",
        ),
        "Markerless 3D chooser (coringa) — Standard/Advanced MediaPipe+YOLO lift, "
        "SAM3+DINOv3 3D and its Visualize ID, Sapiens2 3D Pose (Sapiens2-keypoint-guided "
        "bbox tightening feeding the same SAM 3D Body mesh), plus Monocular -> DLT world "
        "(places a 1-camera reconstruction in the calibrated lab frame). Each tool prints "
        "its own Run CLI after dialogs.",
    ),
    "vector_coding": CliRunHint(("uv run vaila/vector_coding.py",), "Vector coding GUI."),
    "emg_analysis": CliRunHint(("uv run vaila/emg_analysis.py",), "EMG analysis GUI."),
    "force_analysis": CliRunHint(("uv run vaila/force_analysis.py",), "Force plate GUI."),
    "gnss_analysis": CliRunHint(("uv run vaila/gnss_analysis.py",), "GNSS/GPS GUI."),
    "vailajump": CliRunHint(
        ("uv run vaila/vaila_and_jump.py --help",),
        "Vertical jump — CLI flags or GUI from main app.",
    ),
    "cube2d_kinematics": CliRunHint(("uv run vaila/cube2d_kinematics.py",), "Cube2D GUI."),
    "animal_open_field": CliRunHint(
        ("uv run vaila/animal_open_field.py",), "Animal open field GUI."
    ),
    "ml_walkway": CliRunHint(("uv run vaila/walkway_ml_prediction.py",), "ML walkway GUI."),
    "ultrasound": CliRunHint(("uv run vaila/ultrasound.py",), "Ultrasound GUI."),
    "brainstorm": CliRunHint(("uv run vaila/brainstorm.py",), "Brainstorm integration GUI."),
    "scout": CliRunHint(("uv run vaila/scout.py",), "Scout GUI."),
    "startblock": CliRunHint(("uv run vaila/startblock.py",), "Start block GUI."),
    "pynalty": CliRunHint(("uv run vaila/pynalty.py",), "Pynalty GUI."),
    "sprint": CliRunHint(("uv run vaila/vailasprint.py",), "Sprint analysis GUI."),
    "tugturn": CliRunHint(
        ("uv run vaila/tugturn.py --help",),
        "TUG/Turn clinical analysis — CSV + TOML via CLI or GUI.",
    ),
    "soccer_tools": CliRunHint(
        (
            "uv run vaila/soccerfield_keypoints_ai.py",
            "uv run vaila/soccerfield_calib.py",
            "uv run vaila/fifa_to_dlt.py",
            "uv run vaila/fifa_dataset_builder.py",
            "uv run vaila/fifa_manual_merge.py",
        ),
        "Soccer Tools chooser — field KPs, calibration, FIFA DLT/dataset utilities.",
    ),
    "deadlift_analysis": CliRunHint(
        ("uv run vaila/vaila_deadlift.py --help",),
        "Deadlift analysis — pose CSV via CLI or GUI.",
    ),
    "treadmill_lc": CliRunHint(
        ("uv run vaila/treadmill_lc.py --help",), "Treadmill load-cell CLI/GUI."
    ),
    "show_vaila_message": CliRunHint((), "Placeholder — future vailá module."),
    # Frame C_A
    "reorder_csv_data": CliRunHint(
        ("uv run vaila/edit_csv_c3d.py --help",), "Edit/reorder CSV and C3D GUI."
    ),
    "convert_c3d_csv": CliRunHint(
        ("uv run vaila.py",), "C3D ↔ CSV conversion — GUI chooser in main app."
    ),
    "gapfill_split": CliRunHint(
        ("uv run vaila/interp_smooth_split.py --help",),
        "Smooth, filter, gap-fill — TOML config + directory batch CLI.",
    ),
    "dlt_rec_toolkit": CliRunHint(
        (
            "uv run vaila/dlt2d.py",
            "uv run vaila/rec2d_one_dlt2d.py",
            "uv run vaila/rec2d.py",
            "uv run vaila/dlt3d.py",
            "uv run vaila/rec3d_one_dlt3d.py",
            "uv run vaila/rec3d.py",
        ),
        "DLT/REC chooser (coringa) — Make DLT2D, Rec2D 1DLT/MultiDLT, "
        "Make DLT3D, Rec3D 1DLT/MultiDLT. Each keeps its own GUI/CLI unchanged.",
    ),
    "reid_marker": CliRunHint(("uv run vaila/reid_markers.py",), "Marker Re-ID GUI."),
    "run_sapiens3d_kinematics": CliRunHint(
        (
            "uv run python -m vaila.sapiens3d_kinematics --help",
            "uv run python -m vaila.sapiens3d_kinematics -i /path/to/rec3d_sapiens.c3d",
        ),
        "Sapiens2 3D Kinematics — joint rotation matrices, quaternions, and Cardan angles from REC3D C3D.",
    ),
    # Frame C_B
    "extract_png_from_videos": CliRunHint(
        (
            "uv run vaila/extractpng.py",
            "uv run vaila/extractpng.py extract -i /path/to/videos",
            "uv run vaila/extractpng.py create -i /path/to/png_dirs --fps 30 --codec 264",
            "uv run vaila/extractpng.py frames -i VIDEO.mp4 --frames 0,3,5",
        ),
        "Video ↔ PNG (GUI or CLI).",
    ),
    "crop_faces_atletas": CliRunHint(
        (
            "uv run vaila/crop_faces_atletas.py --download-model",
            "uv run vaila/crop_faces_atletas.py",
        ),
        "Crop Face — download model then GUI/CLI.",
    ),
    "draw_box": CliRunHint(("uv run vaila/drawbox.py",), "Draw bounding boxes GUI."),
    "compress_videos_gui": CliRunHint(
        (
            "uv run vaila/compress_videos_h264.py",
            "uv run vaila/compress_videos_h265.py",
            "uv run vaila/compress_videos_h266.py",
        ),
        "Video compression — pick codec script or use GUI chooser.",
    ),
    "sync_videos": CliRunHint(("uv run vaila/syncvid.py",), "Multi-camera sync file GUI."),
    "getpixelvideo": CliRunHint(
        ("uv run vaila/getpixelvideo.py",), "Pixel coordinate labeling (pygame GUI)."
    ),
    "count_frames_in_videos": CliRunHint(("uv run vaila/numberframes.py",), "Video metadata GUI."),
    "process_videos_gui": CliRunHint(
        ("uv run vaila/videoprocessor.py",), "Merge/split videos GUI."
    ),
    "run_distortvideo": CliRunHint(
        ("uv run vaila/vaila_distortvideo_gui.py",),
        "Lens distortion GUI.",
    ),
    "cut_video": CliRunHint(("uv run vaila/cutvideo.py",), "Cut video GUI."),
    "resize_video": CliRunHint(("uv run vaila/resize_video.py",), "Resize video GUI."),
    "ytdownloader": CliRunHint(("uv run vaila/vaila_ytdown.py",), "YouTube downloader GUI."),
    "run_iaudiovid": CliRunHint(
        ("uv run vaila/vaila_iaudiovid.py",), "Insert audio into video GUI."
    ),
    "remove_duplicate_frames": CliRunHint(
        ("uv run vaila/remove_frames2sync.py",),
        "Remove duplicate PNG frames GUI.",
    ),
    # Frame C_C
    "show_c3d_data": CliRunHint(
        ("uv run vaila/viewc3d.py", "uv run vaila/viewc3d_pyvista.py"),
        "C3D viewer — Open3D or PyVista.",
    ),
    "show_csv_file": CliRunHint(("uv run vaila/vpython_c3d.py",), "CSV 3D viewer GUI."),
    "animation_blender": CliRunHint(
        ("uv run python -m vaila.blender_viz -i RUN_DIR",),
        "Open a rec3d run in Blender (BVH + mesh + bones + scene rate, all set up).",
    ),
    "plot_2d_data": CliRunHint(("uv run vaila/vailaplot2d.py",), "2D plot GUI."),
    "plot_3d_data": CliRunHint(("uv run vaila/vailaplot3d.py",), "3D plot GUI."),
    "draw_sports_fields_courts": CliRunHint(
        ("uv run vaila/drawsportsfields.py --help",),
        "Sports field/court plots — CLI --field or GUI.",
    ),
    "run_stroboscopic": CliRunHint(
        ("uv run vaila/vaila_stroboscopic.py",), "Stroboscopic video GUI."
    ),
    # Global
    "run_gpu_test": CliRunHint(
        ("uv run python vaila/gputest.py",),
        "Runs GPU, PyTorch, CUDA, and AI stack diagnostics (SAM 3, Sapiens2, YOLO).",
        invoke_handler=True,
    ),
    "display_help": CliRunHint(
        ("uv run vaila.py",),
        "Opens help index in browser from GUI, or see vaila/help/index.html",
        invoke_handler=True,
    ),
    "quit_app": CliRunHint((), invoke_handler=True),
    "open_terminal_shell": CliRunHint(
        (),
        "Opens the system terminal with the vailá .venv activated (imagination! button).",
        invoke_handler=True,
    ),
}


def get_cli_hint(*, handler: str, code: str = "", label: str = "") -> CliRunHint:
    if handler.startswith("external:"):
        return CliRunHint(
            (), "Opens an external documentation link in your browser.", invoke_handler=True
        )
    return CLI_HINTS_BY_HANDLER.get(
        handler,
        CliRunHint(
            ("uv run vaila.py",),
            f"{_GUI_FALLBACK_NOTE} ({code} — {label})" if code else _GUI_FALLBACK_NOTE,
        ),
    )
