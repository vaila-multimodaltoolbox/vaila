### vailá Scout (Annotation + Analysis)

The vailá Scout tool lets you annotate soccer events on a virtual field and generate quick analyses (heatmaps) in a separate window.

Features
- Field drawn to scale (default 105 x 68 m) in the main window.
- Click to annotate events with team, player, action, and result.
- Configurable teams, players, actions, results, and field size via TOML.
- Save/Load events as CSV.
- Heatmap shown in a new window when requested.
- Hotkeys for faster workflows.
- Player numbering modes: auto-number and unique-per-click.
- Automatic players summary CSV when saving events.
 - Rename teams quickly (Tools → Rename Teams or Ctrl+T). Quick numeric entry to set player number.

How to run
```bash
python -m vaila.scout_vaila
```

CSV format
```text
timestamp_s,team,player,action,result,pos_x_m,pos_y_m,player_name
```

Players summary CSV (auto-created on save)
```text
team,player,num_events,first_timestamp_s,last_timestamp_s,timestamps
```

Config (TOML)
- Auto-created at `~/.vaila/vaila_scout_config.toml` on first run.
- You can Load/Save or Edit the config in the app.

Default schema
```toml
[program]
name = "vaila_scout"
version = "0.1.0"
author = "Paulo Roberto Pereira Santiago"
email = "paulosantiago@usp.br"
description = "Integrated scouting (annotation + analysis) for soccer."
repository = "https://github.com/vaila-multimodaltoolbox/vaila"
license = "GPL-3.0-or-later"
homepage = "https://github.com/vaila-multimodaltoolbox/vaila"
created = "2025-08-12"
updated = "2025-08-12"

[project]
name = "vaila_scout"
field_width_m = 105.0
field_height_m = 68.0
sport = "soccer"
field_units = "meters"
field_standard = "FIFA"
description = "Default soccer field and teams for scouting."

[teams.home]
name = "HOME"
players = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]

[teams.away]
name = "AWAY"
players = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]

actions = [
  "Passing",
  "First touch and receiving",
  "Ball control",
  "Dribbling",
  "Shielding",
  "Shooting and finishing",
  "Heading",
  "Crossing",
  "Tackling",
  "Interceptions",
  "Goalkeeping skills: catching, shot stopping, positioning, distribution",
]
results = ["success", "fail", "neutral"]
```

UI
- Left: field (click to annotate)
- Right top: selectors for team, player, action, result; CSV buttons; Show Heatmap; Reset Time
- Right mid: config buttons (Load/Save/Edit)
- Heatmap opens in a new window when clicking Show Heatmap (or press H)
 - Tools menu includes Rename Teams (Ctrl+T)

Hotkeys
- Ctrl+S: Save CSV
- Ctrl+O: Load CSV
- Ctrl+K: Clear events
- H: Show heatmap (new window)
- R: Reset timer
- Ctrl+E: Edit config
- Ctrl+L: Load config
- Ctrl+Shift+S: Save config
- N: Toggle auto-number players
- U: Toggle unique player per click
- T: Toggle current team (home/away)
- ,/. or ←/→: Next/previous player
 - Ctrl+T: Rename teams
 - Digits (0–9): buffer a player number; Enter to apply; Backspace to edit; Esc to clear; + / - to increment/decrement current player

Notes
- Field coordinates use meters in the range [0, field_width] and [0, field_height].
- Heatmap uses KDE from seaborn.


