# batchcut

## 📋 Informações do Módulo

- **Categoria:** Tools
- **Arquivo:** `vaila\batchcut.py`
- **Linhas:** 191
- **Tamanho:** 6825 caracteres
- **Versão:** 1.1
- **Autor:** Prof. PhD. Paulo Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


Batch Video Cutting Script with GPU Acceleration
Author: Prof. PhD. Paulo Santiago
Date: September 29, 2024
Version: 1.1

Description:
This script performs batch video cutting by processing a list of videos, extracting specified segments
based on frame ranges, and saving them in a structured output directory. The script supports GPU
acceleration via NVIDIA NVENC when available, defaulting to CPU-based processing if a GPU is not detected.

The script reads a list file where each line specifies the original video name, the desired name
for the cut video, the start frame, and the end frame. The videos are processed and saved in a "cut_videos"
subdirectory inside the specified output directory.

List file format:
<original_name> <new_name> <start_frame> <end_frame>

Example:
PC001_STS_02_FLIRsagital.avi PC001_STS_02_FLIRsagital_cut.mp4 100 300

The script automatically removes duplicate ".mp4" extensions from the new file name if necessary.

### Key Features:
1. **Batch Video Processing**...

## 🔧 Funções Principais

**Total de funções encontradas:** 3

- `is_nvidia_gpu_available`
- `batch_cut_videos`
- `cut_videos`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:53:50  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
