# process_gait_features

## 📋 Informações do Módulo

- **Categoria:** Tools
- **Arquivo:** `vaila\process_gait_features.py`
- **Linhas:** 241
- **Tamanho:** 8731 caracteres
- **Versão:** 3.12.9
- **Autor:** Abel Gonçalves Chinaglia
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


===============================================================================
process_gait_features.py
===============================================================================
Author: Abel Gonçalves Chinaglia
Ph.D. Candidate in PPGRDF - FMRP - USP
Date: 05 Feb. 2025
Update: 24.Feb.2025
Python Version: 3.12.9

Description:
------------
This script extracts spatial and temporal features from gait analysis data
stored in .csv files. It computes metrics such as mean, variance, range, speed,
and step length for each individual and trial, organizing the results in a final
.csv file for further analysis.

Key Features:
-------------
1. Asks for the participant's name to save it in the results.
2. Requests the number of steps per trial to divide data into blocks for analysis.
3. Calculates spatial, temporal, and kinematic features for each trial and individual.
4. Automatically processes all .csv files in the selected directory.
5. Saves the results in an output .csv file for further...

## 🔧 Funções Principais

**Total de funções encontradas:** 4

- `calculate_features`
- `divide_into_blocks`
- `process_files_and_save`
- `run_process_gait_features`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:18:44  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
