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
timestamp_s,team,player,action,result,pos_x_m,pos_y_m
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
[project]
name = "vaila_scout"
field_width_m = 105.0
field_height_m = 68.0

[teams.home]
name = "HOME"
players = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]

[teams.away]
name = "AWAY"
players = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]

actions = ["pass", "shot", "dribble", "tackle", "interception", "foul", "assist", "clearance", "cross"]
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
- T: Reset timer
- Ctrl+E: Edit config
- Ctrl+L: Load config
- Ctrl+Shift+S: Save config
- N: Toggle auto-number players
- U: Toggle unique player per click
- F1: Open this help (HTML)
 - F5: Toggle current team (home/away)
 - Ctrl+T: Rename teams
 - Digits (0–9): buffer a player number; Enter to apply; Backspace to edit; Esc to clear; + / - to increment/decrement current player

Notes
- Field coordinates use meters in the range [0, field_width] and [0, field_height].
- Heatmap uses KDE from seaborn.


