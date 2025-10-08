# merge_multivideos

## 📋 Informações do Módulo

- **Categoria:** Tools
- **Arquivo:** `vaila/merge_multivideos.py`
- **Linhas:** 2172
- **Tamanho:** 89274 caracteres
- **Versão:** updated: 0.2.0
- **Autor:** Paulo R. P. Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


merge_multivideos.py

Author: Paulo R. P. Santiago
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Licensed under GNU Lesser General Public License v3.0

Created: 25 February 2025
Update: 13 March 2025
Version updated: 0.2.0

Description:
This script allows users to merge multiple video files into a single video in a specified order.
It provides two methods of selecting videos:
1. By choosing multiple video files through a file selection dialog and arranging their order.
2. By providing a text file with a list of video files to merge.

The script processes videos using FFmpeg without requiring GPU acceleration, making it compatible
with all systems.

Key Features:
- Graphical User Interface (GUI) for selecting and arranging multiple videos
- Option to load video list from a text file
- Preview of selected videos and their order
- Ability to reorder videos before processing
- Detailed console output for tracking progress and handling errors
- Crea...

## 🔧 Funções Principais

**Total de funções encontradas:** 20

- `run_merge_multivideos`
- `select_multiple_videos`
- `load_video_metadata`
- `load_from_text_file`
- `set_output_directory`
- `clear_all`
- `update_video_list`
- `select_video`
- `move_up`
- `move_down`
- `remove_selected`
- `start_merge`
- `do_merge`
- `do_precise_merge`
- `do_fast_merge`
- `do_frame_accurate_merge`
- `show_mode_help`
- `merge_complete`
- `create_mode_buttons`
- `select_mode`




---

📅 **Gerado automaticamente em:** 08/10/2025 14:00:12  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
