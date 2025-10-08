# numberframes

## 📋 Informações do Módulo

- **Categoria:** Tools
- **Arquivo:** `vaila\numberframes.py`
- **Linhas:** 409
- **Tamanho:** 15222 caracteres
- **Versão:** 0.1.2
- **Autor:** Paulo R. P. Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


numberframes.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 10 October 2024
Update Date: 13 August 2025
Version: 0.1.2
Python Version: 3.12.11

Description:
------------
This script allows users to analyze video files within a selected directory and extract metadata such as frame count, frame rate (FPS), resolution, codec, and duration. The script generates a summary of this information, displays it in a user-friendly graphical interface, and saves the metadata to text files. The "basic" file contains essential metadata, while the "full" file includes all possible metadata extracted using `ffprobe`.

Key Features:
-------------
1. Fast metadata extraction using a single ffprobe JSON call.
2. Detection of capture FPS via Android tag com.android.capture.fps when present.
3. Parallel processing of multiple videos for faster a...

## 🔧 Funções Principais

**Total de funções encontradas:** 7

- `get_video_info`
- `display_video_info`
- `save_basic_metadata_to_file`
- `save_full_metadata_to_file`
- `show_save_success_message`
- `count_frames_in_videos`
- `on_closing`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:18:44  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
