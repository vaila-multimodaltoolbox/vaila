# vaila_datdistort

## 📋 Informações do Módulo

- **Categoria:** Tools
- **Arquivo:** `vaila\vaila_datdistort.py`
- **Linhas:** 269
- **Tamanho:** 9032 caracteres
- **Versão:** 0.0.2
- **Autor:** Prof. Dr. Paulo R. P. Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


================================================================================
vaila_datdistort.py
================================================================================
vailá - Multimodal Toolbox
Author: Prof. Dr. Paulo R. P. Santiago
https://github.com/paulopreto/vaila-multimodaltoolbox
Date: 03 April 2025
Update: 24 July 2025
Version: 0.0.2
Python Version: 3.12.11

Description:
------------
This tool applies lens distortion correction to 2D coordinates from a DAT file
using the same camera calibration parameters as vaila_lensdistortvideo.py.

New Features in This Version:
------------------------------
1. Fixed issue with column order in output file.
2. Improved error handling.
3. Added more detailed error logging.

How to use:
------------
1. Select the distortion parameters CSV file.
2. Select the directory containing CSV/DAT files to process.
3. The script will process all CSV and DAT files in the directory and save the
   results in the output directory.

python vai...

## 🔧 Funções Principais

**Total de funções encontradas:** 6

- `load_distortion_parameters`
- `undistort_points`
- `process_dat_file`
- `select_file`
- `select_directory`
- `run_datdistort`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:53:50  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
