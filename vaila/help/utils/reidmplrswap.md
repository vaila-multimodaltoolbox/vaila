# reidmplrswap

## 📋 Informações do Módulo

- **Categoria:** Utils
- **Arquivo:** `vaila\reidmplrswap.py`
- **Linhas:** 830
- **Tamanho:** 27842 caracteres


- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


reidmplrswap.py — Detect and fix abrupt left/right swaps in marker tracks

This tool analyzes 2D marker CSVs (e.g., MediaPipe pixel outputs in vailá
format) to detect short, abrupt swaps between left/right marker pairs and
apply corrections. It supports:

1) Automatic detection and correction
   - Side consistency via the sign of (x_right - x_left)
   - Optional continuity check to prefer the swap that minimizes total motion
   - Produces a report with suggested swap frame ranges per marker pair

2) Manual correction
   - User specifies a marker pair and a frame interval to swap left<->right

Inputs
------
- CSV with columns like: marker_x, marker_y [, marker_z]. Left/right markers
  should be identifiable via tokens like L/R or left/right (prefix/suffix).
- Optional video path: if provided, the script exports short preview clips
  around suspected swap intervals to assist visual validation.

Outputs
-------
- Corrected CSV saved alongside original with suffix "_reidswap.csv"
- Text r...

## 🔧 Funções Principais

**Total de funções encontradas:** 17

- `find_lr_pairs`
- `propose_swaps_for_pair`
- `apply_swap_for_pair`
- `auto_fix_swaps`
- `load_csv`
- `save_csv_with_suffix`
- `write_report`
- `export_preview_clips`
- `interactive_review`
- `run_auto`
- `run_manual`
- `main`
- `on_trackbar`
- `get_current_base`
- `in_any_interval`
- `draw_overlay`
- `print_help`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:53:50  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
