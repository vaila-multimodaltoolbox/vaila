# gnss_analysis

## 📋 Module Information

- **Category:** Analysis
- **File:** `vaila\gnss_analysis.py`
- **Lines:** 850
- **Size:** 29908 characters


- **GUI Interface:** ✅ Yes

## 📖 Description


Script para ler files GPX e exportar os dados para um file CSV.

Este script lê um file GPX contendo informações de latitude, longitude, elevação, tempo, velocidade (speed) e cadência (cad).
Os dados são extraídos e exportados para um file CSV, incluindo uma coluna adicional chamada 'time_seconds' que
representa o tempo decorrido em segundos desde o início do percurso.

Uso:
    python readgpx.py <file_gpx> <file_csv>

Argumentos:
    <file_gpx>   Caminho para o file GPX de entrada.
    <file_csv>   Caminho para o file CSV de saída.

Author: Seu Nome
Data: 2024-10-13


## 🔧 Main Functions

**Total functions found:** 20

- `read_gpx_file`
- `export_to_csv`
- `convert_gpx_to_csv`
- `convert_gpx_to_kml`
- `convert_gpx_to_kmz`
- `convert_csv_to_gpx`
- `convert_csv_to_kml`
- `convert_csv_to_kmz`
- `csv_conversion`
- `distance_analysis`
- `unit_conversion`
- `speed_analysis`
- `trajectory_analysis`
- `spatial_analysis`
- `time_series_analysis`
- `data_normalization`
- `data_visualization`
- `batch_processing`
- `plot_gnss_data`
- `gpx_conversion`




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
