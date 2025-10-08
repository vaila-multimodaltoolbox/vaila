# readcsv_export

## 📋 Informações do Módulo

- **Categoria:** Processing
- **Arquivo:** `vaila\readcsv_export.py`
- **Linhas:** 971
- **Tamanho:** 36912 caracteres
- **Versão:** 25 September 2024
- **Autor:** Prof. Paulo R. P. Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


===============================================================================
readcsv_export.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Version: 25 September 2024
Update: 04 September 2025
Version updated: 0.1.1
Python Version: 3.12.11

Description:
This script provides functionality to convert CSV files containing point and analog data
into the C3D format, commonly used for motion capture data analysis. The script uses the
ezc3d library to create C3D files from CSV inputs while sanitizing and formatting the data
to ensure compatibility with the C3D standard.

Main Features:
- Reads point and analog data from user-selected CSV files.
- Sanitizes headers to remove unwanted characters and ensure proper naming conventions.
- Handles user input for data rates, unit conversions, and sorting preferences.
- Converts the CSV data into a C3D file with appropriately formatted point and analog data.
- Provides a user in...

## 🔧 Funções Principais

**Total de funções encontradas:** 8

- `sanitize_header`
- `validate_and_filter_columns`
- `get_conversion_factor`
- `convert_csv_to_c3d`
- `create_c3d_from_csv`
- `batch_convert_csv_to_c3d`
- `auto_create_c3d_from_csv`
- `on_submit`




---

📅 **Gerado automaticamente em:** 08/10/2025 10:07:00  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
