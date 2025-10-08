# force_cube_fig

## 📋 Informações do Módulo

- **Categoria:** Tools
- **Arquivo:** `vaila/force_cube_fig.py`
- **Linhas:** 1930
- **Tamanho:** 65744 caracteres
- **Versão:** 0.5
- **Autor:** Prof. Dr. Paulo R. P. Santiago Ligia
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


================================================================================
Force Platform Data Analysis Toolkit - force_cube_fig.py
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago Ligia
Date: 9 September 2024
Version: 0.5
Python Version: 3.11

Description:
------------
This script processes biomechanical data from force platforms, analyzing
the vertical ground reaction force (VGRF) to compute key metrics, including:
- Peak forces, time intervals, impulse, rate of force development (RFD),
  and stiffness parameters.
The results are visualized through interactive plots and saved to CSV files for
further analysis. The script allows batch processing of multiple files and provides
descriptive statistics for all analyzed data.

Key Functionalities:
---------------------
1. Data Selection:
   - Allows the user to select input CSV files containing biomechanical data.
   - Prompts the user to specify output directori...

## 🔧 Funções Principais

**Total de funções encontradas:** 20

- `select_source_directory`
- `select_output_directory`
- `select_body_weight`
- `process_file`
- `batch_process_directory`
- `select_headers_and_load_data`
- `create_main_output_directory`
- `prompt_user_input`
- `butterworthfilt`
- `calculate_median`
- `find_active_indices`
- `build_headers`
- `calculate_loading_rates`
- `logistic_ogive`
- `fit_stiffness_models`
- `calculate_cube_values`
- `makefig1`
- `makefig2`
- `makefig3`
- `makefig4`




---

📅 **Gerado automaticamente em:** 08/10/2025 14:00:12  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
