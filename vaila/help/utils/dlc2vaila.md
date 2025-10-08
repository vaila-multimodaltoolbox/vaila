# dlc2vaila

## 📋 Informações do Módulo

- **Categoria:** Utils
- **Arquivo:** `vaila\dlc2vaila.py`
- **Linhas:** 192
- **Tamanho:** 7719 caracteres
- **Versão:** 1.0.0
- **Autor:** Prof. Dr. Paulo Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


Script: dlc2vaila.py
Author: Prof. Dr. Paulo Santiago
Version: 1.0.0
Last Updated: December 9, 2024

Description:
    This script converts DLC (DeepLabCut) CSV files into a format compatible with
    the vailá multimodal toolbox. It processes all CSV files from a specified input
    directory, adjusts their structure, and saves the converted files in a newly
    created directory with a timestamped name.

    The conversion process includes:
    - Retaining only the third header line from the original DLC file.
    - Removing the first column temporarily, processing the remaining data, and re-adding the first column.
    - Excluding every third column from the data.
    - Generating a new header with the format: 'frame, p1_x, p1_y, p2_x, p2_y, ...'.
    - Saving the processed files in a dedicated output directory with a timestamp.

Usage:
    - Run the script to select an input directory containing DLC CSV files.
    - The script will process each file and save the converted outputs i...

## 🔧 Funções Principais

**Total de funções encontradas:** 2

- `process_csv_files_with_numpy`
- `batch_convert_dlc`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:53:50  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
