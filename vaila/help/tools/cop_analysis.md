# cop_analysis

## 📋 Informações do Módulo

- **Categoria:** Tools
- **Arquivo:** `vaila\cop_analysis.py`
- **Linhas:** 703
- **Tamanho:** 24820 caracteres
- **Versão:** 1.4 Date: 2024-09-12
- **Autor:** Prof. Dr. Paulo R. P. Santiago Version: 1.4 Date: 2024-09-12
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


# Plot the CoP pathway with points only (no connecting lines)
ax2.plot(X_n, Y_n, label='CoP Pathway', color='blue', marker='.', markersize=3, linestyle='None')
Module: cop_analysis.py
Description: This module provides a comprehensive set of tools for analyzing Center of Pressure (CoP) data from force plate measurements.
             CoP data is critical in understanding balance and postural control in various fields such as biomechanics, rehabilitation, and sports science.

             The module includes:
             - Functions for data filtering, specifically using a Butterworth filter, to remove noise and smooth the CoP data.
             - GUI components to allow users to select relevant data headers interactively for analysis.
             - Methods to calculate various postural stability metrics such as Mean Square Displacement (MSD), Power Spectral Density (PSD), and sway density.
             - Spectral feature analysis to quantify different frequency components of the CoP ...

## 🔧 Funções Principais

**Total de funções encontradas:** 10

- `convert_to_cm`
- `read_csv_full`
- `select2headers`
- `plot_final_figure`
- `analyze_data_2d`
- `main`
- `get_csv_headers`
- `on_select`
- `select_all`
- `unselect_all`




---

📅 **Gerado automaticamente em:** 08/10/2025 10:07:00  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
