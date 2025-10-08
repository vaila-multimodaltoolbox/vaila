# stabilogram_analysis

## 📋 Informações do Módulo

- **Categoria:** Tools
- **Arquivo:** `vaila\stabilogram_analysis.py`
- **Linhas:** 389
- **Tamanho:** 13025 caracteres
- **Versão:** 1.3
- **Autor:** Prof. Dr. Paulo R. P. Santiago
- **Interface Gráfica:** ❌ Não

## 📖 Descrição


Module: stabilogram_analysis.py
Description:
This module provides a comprehensive set of functions for analyzing Center of Pressure (CoP) data obtained from force plate measurements in postural control studies. The functions included in this module allow for the calculation of various stabilometric parameters and the generation of stabilogram plots to evaluate balance and stability.

The main features of this module include:
- **Root Mean Square (RMS) Calculation**: Computes the RMS of the CoP displacement in the mediolateral (ML) and anteroposterior (AP) directions, providing a measure of postural sway.
- **Speed Computation**: Calculates the instantaneous speed of the CoP using a Savitzky-Golay filter to smooth the data, allowing for analysis of the dynamics of postural control.
- **Power Spectral Density (PSD) Analysis**: Computes the PSD of the CoP signals using Welch's method, providing insight into the frequency components of postural sway.
- **Mean Square Displacement (MSD) Cal...

## 🔧 Funções Principais

**Total de funções encontradas:** 11

- `compute_rms`
- `compute_speed`
- `compute_power_spectrum`
- `compute_msd`
- `count_zero_crossings`
- `count_peaks`
- `compute_sway_density`
- `compute_total_path_length`
- `plot_stabilogram`
- `plot_power_spectrum`
- `save_metrics_to_csv`




---

📅 **Gerado automaticamente em:** 08/10/2025 10:07:00  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
