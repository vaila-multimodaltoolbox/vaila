### vailá Scout (Annotation + Analysis)

The vailá Scout tool lets you annotate soccer events on a virtual field and generate quick analyses (heatmaps) in a separate window.

Features
- Field drawn to scale (default 105 x 68 m) in the main window.
- Click to annotate events with team, player, action, and result.
- Configurable teams, players, actions, results, and field size via TOML.
- Save/Load events as CSV with action codes and player names.
- Heatmap shown in a new window when requested.
- Hotkeys for faster workflows (numeric input for actions).
- Player rosters displayed side-by-side (Home/Away).
- Automatic players summary CSV when saving events.
- Rename teams quickly (Tools → Rename Teams or Ctrl+T).
- Action symbols drawn on field by default for better distinction.
- Remove events with Ctrl + right click functionality.

How to run
```bash
python vaila.py           # Click the "Scout" button in the vailá GUI
python -m vaila.scout_vaila   # or run the module directly
python vaila/scout_vaila.py   # or run the script file
```

CSV format
```text
timestamp_s,team,player_name,player,action,action_code,result,pos_x_m,pos_y_m
```

Players summary CSV (auto-created on save)
```text
team,player,num_events,first_timestamp_s,last_timestamp_s,timestamps
```

Config (TOML)
- The app searches for `vaila_scout_config.toml` next to the script or under `vaila/models`.
- If none is found, a commented template is created under `vaila/models`.
- You can Load/Save or Create Template the config in the app.

Template summary
```toml
[program]
name = "vaila_scout"
version = "0.1.0"
...

[project]
field_width_m = 105.0
field_height_m = 68.0
...

[teams.home]
name = "HOME"
color = "#1f77b4"
players = ["1","2",...,"23"]

[teams.home.players_names]
"10" = "Pelé"  # optional mapping number → name

[teams.away]
name = "AWAY"
color = "#d62728"
players = ["1","2",...,"23"]

[teams.away.players_names]
"10" = "Maradona"

[drawing]
player_circle_radius_m = 0.6
player_edge_color = "black"
player_number_color = "white"
player_number_size = 8
action_symbol_size = 90.0
show_player_name = true
player_name_size = 8
show_action_symbol = true  # draw action symbols on field

results = ["success", "fail", "neutral"]

[[actions]]        # structured actions with codes
name = "Pass"
code = 1
symbol = "o"
key = "p"
color = "#FFD700"

[[actions]]
name = "Shot"
code = 6
symbol = "*"
key = "g"
color = "#FF4500"
...
```

UI
- Left: field (click to annotate)
- Right controls: Action and Result selectors, CSV buttons, Clock (Start/Pause/Reset), Config (Load/Save/Create Template), Quick action code input.
- Right bottom: Home and Away rosters (No/Name) side by side with team names displayed.
- Heatmap opens in a new window when clicking Show Heatmap (or press H)
- Tools menu includes Rename Teams (Ctrl+T)

Hotkeys
- Ctrl+S: Save CSV
- Ctrl+O: Load CSV
- Ctrl+K: Clear events
- H: Show heatmap (new window)
- R: Reset timer
- Space: Start/Pause clock
- Ctrl+L: Load config
- Ctrl+Shift+S: Save config
- T: Toggle current team (home/away)
- Ctrl+T: Rename teams
- Digits (0–9): Enter action code; Enter to apply; Backspace to edit; Esc to clear
- Mouse: Left click = success, Right click = fail, Middle click = neutral
- Ctrl + Right click: Remove the most recent event near clicked position (hold Ctrl, right click)

Notes
- Field coordinates use meters in the range [0, field_width] and [0, field_height].
- Heatmap uses KDE from seaborn.
- Action codes are numeric identifiers for quick input (1-11 by default).
- Player markers show team color with result-based outline (blue=success, red=fail, white=neutral).
- Action symbols are drawn by default next to player circles with enhanced visibility.
- Event removal works within a 1-meter radius and removes only the most recent event.


