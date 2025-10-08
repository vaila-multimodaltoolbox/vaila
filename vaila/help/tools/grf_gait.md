# grf_gait

## 📋 Informações do Módulo

- **Categoria:** Tools
- **Arquivo:** `vaila\grf_gait.py`
- **Linhas:** 2007
- **Tamanho:** 68150 caracteres
- **Versão:** 1.0
- **Autor:** [Your Name]
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


================================================================================
Gait Data Analysis Toolkit - grf_gait.py
================================================================================
Author: [Your Name]
Date: [Current Date]
Version: 1.0
Python Version: 3.x

Description:
------------
This script processes gait data from force platforms, analyzing a single
contact strike to compute key metrics, including:
- Peak forces, impulse, rate of force development (RFD), and contact time.

The results are visualized through interactive plots and saved to CSV files for
further analysis.

Key Functionalities:
---------------------
1. Data Selection:
   - Allows the user to select an input CSV file containing gait data.
   - Prompts the user to specify output directories.
   - Prompts the user for input parameters (sampling frequency, thresholds, etc.).
2. Data Processing:
   - Normalizes data, applies Butterworth filters, and computes key biomechanical metrics.
3. Visualization:...

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

📅 **Gerado automaticamente em:** 08/10/2025 10:07:00  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
