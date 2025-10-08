# fixnoise

## 📋 Informações do Módulo

- **Categoria:** Tools
- **Arquivo:** `vaila\fixnoise.py`
- **Linhas:** 273
- **Tamanho:** 8988 caracteres
- **Versão:** 0.02
- **Autor:** Prof. Dr. Paulo R. P. Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


fixnoise_batch.py
version 0.02

This script is designed to batch process CSV files, applying a noise filter (fixnoise) interactively.
The user selects a directory containing CSV files and, from the first file, chooses the headers of interest.
The script applies the same header selection for all subsequent files and allows the user to mark start and end points
for specific segment replacements. If no points are selected, the file is saved without changes.

Author: Prof. Dr. Paulo R. P. Santiago
Date: September 13, 2024

Modifications:
- Added batch processing for multiple CSV files.
- Unified header selection based on the first file.
- Saved processed files in a new directory with a timestamp.

This script was created to facilitate the processing of large volumes of experimental data, allowing fine adjustments
through a user-friendly graphical interface.

Instructions:
1. Run the script.
2. Select the directory containing the CSV files to be processed.
3. Follow the instructions in the...

## 🔧 Funções Principais

**Total de funções encontradas:** 13

- `read_csv_full`
- `select_headers_and_load_data`
- `makefig1`
- `replace_segments`
- `process_files_in_directory`
- `main`
- `get_csv_headers`
- `on_select`
- `select_all`
- `unselect_all`
- `on_key_press`
- `on_key_release`
- `onclick`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:53:50  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
