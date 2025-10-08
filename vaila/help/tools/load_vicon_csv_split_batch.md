# load_vicon_csv_split_batch

## 📋 Informações do Módulo

- **Categoria:** Tools
- **Arquivo:** `vaila\load_vicon_csv_split_batch.py`
- **Linhas:** 219
- **Tamanho:** 8327 caracteres
- **Versão:** 1.1
- **Autor:** Prof. Dr. Paulo R. P. Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


================================================================================
VICON CSV Split Batch Processor
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 2024-09-20
Version: 1.1

Description:
------------
This Python script processes CSV files generated from the VICON Nexus system and splits the data into separate files by device. It processes only the CSV files in the specified directory (first level, without entering subdirectories) and exports them into a user-specified output folder. The header information is cleaned and sanitized, and the files are saved with additional timestamp information for traceability.

Main Features:
--------------
1. **Batch Processing**: The script automatically finds and processes all CSV files in the specified directory (without subdirectories).
2. **Header Merging and Cleaning**: It merges multiple header rows, replaces problematic characters, and sanitizes unit symb...

## 🔧 Funções Principais

**Total de funções encontradas:** 6

- `clean_header`
- `merge_headers`
- `get_file_creation_datetime`
- `read_csv_devs`
- `select_directory`
- `process_csv_files_first_level`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:18:44  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
