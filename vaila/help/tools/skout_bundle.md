# skout_bundle

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/skout_bundle.py`
- **Lines:** 504
- **Size:** 14819 characters


- **GUI Interface:** ❌ No

## 📖 Description


Skout bundle: convert Skout.exe ASCII export to vailá scout CSV and
emit a ready-to-use vailá Scout config TOML in one pass.

Created by Paulo Roberto Pereira Santiago
Date: 2025-08-19
Updated by: Paulo Roberto Pereira Santiago

Usage examples
--------------
# Basic: infer time from period/min/second, write sibling files
python skout_bundle.py braxing.txt

# Custom outputs + team name
python skout_bundle.py braxing.txt \
  --csv-out jogo1_preto_serjao.csv \
  --toml-out vaila_scout_config_preto.toml \
  --team HOME

# If your Skout export lacks minute/second, use a sequential clock (1 s steps)
python skout_bundle.py braxing.txt --time-mode sequence --dt 1.0

# If you know the Skout grid extents (screen coordinates), set them explicitly
python skout_bundle.py braxing.txt --grid-width 320 --grid-height 210

Outputs
-------
1) CSV (compatible with vailá scout):
   timestamp_s, team, player_name, player, action, action_code, result, pos_x_m, pos_y_m
2) TOML config aligned to vailá_scout, ...

## 🔧 Main Functions

**Total functions found:** 7

- `parse_players`
- `parse_actions`
- `parse_events`
- `load_skout_ascii`
- `convert_to_vaila_csv`
- `build_config_toml`
- `main`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
