# cluster_analysis

## 📋 Informações do Módulo

- **Categoria:** Analysis
- **Arquivo:** `vaila\cluster_analysis.py`
- **Linhas:** 515
- **Tamanho:** 18621 caracteres
- **Versão:** 1.0
- **Autor:** Prof. Dr. Paulo R. P. Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


================================================================================
cluster_analysis.py

Cluster Data Analysis Toolkit for Motion Capture
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 2024-07-19
Version: 1.0

Overview:

This Python script processes motion capture data for trunk and pelvis rotations using clusters of anatomical markers. It reads CSV files with marker positions, computes orthonormal bases for the clusters, calculates Euler angles, and optionally compares these angles with anatomical reference data. The script generates 3D visualizations of clusters and saves results to CSV for further analysis.

Main Features:

    1. Data Input and Filtering:
        - Reads CSV files containing 3D motion capture data of anatomical markers.
        - Applies a Butterworth filter for noise reduction.

    2. Orthonormal Bases and Euler Angles:
        - Calculates orthonormal bases for the anato...

## 🔧 Funções Principais

**Total de funções encontradas:** 3

- `save_results_to_csv`
- `read_anatomical_csv`
- `analyze_cluster_data`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:53:50  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
