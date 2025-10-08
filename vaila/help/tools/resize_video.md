# resize_video

## 📋 Informações do Módulo

- **Categoria:** Tools
- **Arquivo:** `vaila\resize_video.py`
- **Linhas:** 1293
- **Tamanho:** 46650 caracteres
- **Versão:** --------
- **Autor:** -------
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


resize_video.py

Description:
-----------
This script provides tools for improving pose detection in videos:
1. Batch resize videos to higher resolutions (2x-8x)
2. Crop specific regions of interest and resize them
3. Convert MediaPipe, YOLO, and vailá pixel coordinates back to original video coordinates

Version:
--------
0.4.1
Create: 27 April 2025
update: 26 May 2025

Author:
-------
Prof. PhD. Paulo R. P. Santiago

License:
--------
This code is licensed under the GNU General Public License v3.0.

Dependencies:
-------------
- Python 3.12.9
- opencv-python
- tkinter
- pandas (for coordinates conversion)


## 🔧 Funções Principais

**Total de funções encontradas:** 20

- `get_video_info`
- `resize_with_opencv`
- `convert_coordinates`
- `convert_coordinates_by_format`
- `convert_mediapipe_coordinates`
- `validate_scale_factor`
- `batch_resize_videos`
- `run_resize_video`
- `select_roi`
- `select_input_dir`
- `select_output_dir`
- `revert_coordinates_gui`
- `crop_and_resize_single`
- `start_batch_processing`
- `mouse_callback`
- `set_format_and_highlight`
- `update_progress`
- `select_metadata_file`
- `select_pixel_csv_file`
- `select_output_csv_file`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:18:44  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
