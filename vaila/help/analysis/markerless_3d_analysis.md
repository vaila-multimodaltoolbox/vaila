# markerless_3d_analysis

## 📋 Informações do Módulo

- **Categoria:** Analysis
- **Arquivo:** `vaila\markerless_3d_analysis.py`
- **Linhas:** 1075
- **Tamanho:** 36486 caracteres


- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


markerless_3d_gui.py — GUI configurator + TOML + batch runner for monocular 3D (meters by default)

What it does
- Launches a Tkinter GUI to collect all parameters and save a .toml config file (with sensible defaults).
- Batch processes a directory of inputs:
    • If the input is a VIDEO: extracts 2D MediaPipe → lifts 2D→3D (VideoPose3D) → anchors to ground (DLT2D/JSON/Click) → calibrates vertical by leg length → optional DLT3D refine → exports CSV/C3D.
    • If the input is a CSV (MediaPipe pixels, shape [T, 33*2]): skips extraction and continues the same pipeline. Width/Height/FPS for CSVs can be provided in the GUI (CSV defaults).

Defaults set for your case
- Units = METERS ("m") and conversions handled internally.
- DLT2D default: /mnt/data/cam2_calib2D.dlt2d
- DLT3D default: /mnt/data/cam2_calib3D.dlt3d
- Input dir default: /mnt/data
- Pattern default: "*.mp4;*.mov;*.avi;*_mp_pixel.csv" (videos + existing 2D CSVs)
- Leg length default = 0.42 m (can be changed per participant in...

## 🔧 Funções Principais

**Total de funções encontradas:** 20

- `dict_to_toml`
- `normalize_screen_coordinates`
- `mediapipe_to_coco17`
- `build_vp3d_model`
- `load_vp3d_checkpoint`
- `sliding_windows`
- `infer_vp3d`
- `dlt8_to_H`
- `pick_points_on_image`
- `image_to_ground_xy`
- `detect_foot_contacts`
- `fit_plane_normal`
- `rotation_between`
- `anchor_with_vertical`
- `load_dlt3d_from_file`
- `project_dlt3d`
- `euler_to_Rxyz`
- `refine_with_dlt3d`
- `extract_mediapipe_csv`
- `load_csv33`




---

📅 **Gerado automaticamente em:** 08/10/2025 10:07:00  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
