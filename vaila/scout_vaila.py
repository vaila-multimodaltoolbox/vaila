"""
Project: vailá Multimodal Toolbox
Script: scout_vaila.py - Integrated Sports Scouting (Annotation + Analysis)

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 12 August 2025
Update Date: 14 August 2025
Version: 0.1.2

Description:
    Integrated GUI to annotate sports events on a virtual soccer field and generate
    quick analyses (e.g., heatmaps). Inspired by manual scouting tools and designed
    to fit the vailá project style. No external field image is required; the field is
    drawn to scale using standard FIFA dimensions (105m x 68m).

Usage:
    Click in button Scout in the vaila GUI
    python vaila.py
    or
    Run from the command line:
        python -m vaila.scout_vaila
    or enter the vaila directory and run:
        python scout_vaila.py

    
Requirements:
    - Python 3.x
    - tkinter (GUI)
    - matplotlib
    - seaborn
    - pandas
    - rich (for console prints)
    - toml (for writing config) and tomllib/tomli (for reading config)

License:
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from rich import print
import webbrowser


# --- Config I/O helpers (TOML) -------------------------------------------------
try:  # Python 3.11+
    import tomllib as _toml_reader  # type: ignore
except Exception:  # noqa: BLE001
    try:
        import tomli as _toml_reader  # type: ignore
    except Exception:  # noqa: BLE001
        _toml_reader = None

try:
    import toml as _toml_writer  # writing
except Exception:  # noqa: BLE001
    _toml_writer = None


DEFAULT_CFG_FILENAME = "vaila_scout_config.toml"


def _default_config() -> Dict:
    return {
        "program": {
            "name": "vaila_scout",
            "version": "0.1.0",
            "author": "Paulo Roberto Pereira Santiago",
            "email": "paulosantiago@usp.br",
            "description": "Integrated scouting (annotation + analysis) for soccer.",
            "repository": "https://github.com/vaila-multimodaltoolbox/vaila",
            "license": "GPL-3.0-or-later",
            "homepage": "https://github.com/vaila-multimodaltoolbox/vaila",
            "created": "2025-08-12",
            "updated": "2025-08-12",
        },
        "project": {
            "name": "vaila_scout",
            "field_width_m": 105.0,
            "field_height_m": 68.0,
            "sport": "soccer",
            "field_units": "meters",
            "field_standard": "FIFA",
            "description": "Default soccer field and teams for scouting.",
        },
        "teams": {
            "home": {
                "name": "HOME",
                "color": "#1f77b4",  # blue
                # 11 starters minimum; allow more numbers to support up to 5 substitutions
                "players": [str(n) for n in range(1, 24)],
                "players_names": {
                    "1": "Taffarel",
                    "2": "Cafu",
                    "3": "Aldair",
                    "4": "Djalma Santos",
                    "5": "Falcão",
                    "6": "Roberto Carlos",
                    "7": "Garrincha",
                    "8": "Sócrates",
                    "9": "Ronaldo",
                    "10": "Pelé",
                    "11": "Neymar",
                    "12": "Rivaldo",
                    "13": "Nilton Santos",
                    "14": "Zico",
                    "15": "Romário",
                    "16": "Rivelino",
                    "17": "Kaká",
                    "18": "Jairzinho",
                    "19": "Tostão",
                    "20": "Didi"
                },
            },
            "away": {
                "name": "AWAY",
                "color": "#d62728",  # red
                # 11 starters minimum; allow more numbers to support up to 5 substitutions
                "players": [str(n) for n in range(1, 24)],
                "players_names": {
                    "1": "Gianluigi Buffon",
                    "2": "Philipp Lahm",
                    "3": "Paolo Maldini",
                    "4": "Franz Beckenbauer",
                    "5": "Bobby Moore",
                    "6": "Franco Baresi",
                    "7": "Cristiano Ronaldo",
                    "8": "Andrés Iniesta",
                    "9": "Marco van Basten",
                    "10": "Diego Maradona",
                    "11": "Lionel Messi",
                    "12": "Zinedine Zidane",
                    "13": "Eusébio",
                    "14": "Johan Cruyff",
                    "15": "Thierry Henry",
                    "16": "Michel Platini",
                    "17": "George Best",
                    "18": "Ferenc Puskás",
                    "19": "Alfredo Di Stéfano",
                    "20": "Xavi"
                },
            },
        },
        # Prefer structured actions with symbol and color. If user provides a simple list of strings,
        # the app will convert them to this structure with defaults.
        "actions": [
            {"name": "Pass", "code": 1, "symbol": "o", "color": "#FFD700"},
            {"name": "First touch", "code": 2, "symbol": "P", "color": "#8c564b"},
            {"name": "Control", "code": 3, "symbol": "s", "color": "#9467bd"},
            {"name": "Dribble", "code": 4, "symbol": "D", "color": "#17becf"},
            {"name": "Shield", "code": 5, "symbol": "h", "color": "#7f7f7f"},
            {"name": "Shot", "code": 6, "symbol": "*", "color": "#FF4500"},
            {"name": "Header", "code": 7, "symbol": "^", "color": "#2ca02c"},
            {"name": "Cross", "code": 8, "symbol": "+", "color": "#1f77b4"},
            {"name": "Tackle", "code": 9, "symbol": "x", "color": "#d62728"},
            {"name": "Interception", "code": 10, "symbol": "X", "color": "#bcbd22"},
            {"name": "Goalkeeping", "code": 11, "symbol": "s", "color": "#8c564b"},
        ],
        "results": ["success", "fail", "neutral"],
        "drawing": {
            "player_circle_radius_m": 0.6,
            "player_edge_color": "black",
            "player_number_color": "white",
            "player_number_size": 8,
            "action_symbol_size": 90,
            "show_player_name": True,
            "player_name_size": 8,
            "show_action_symbol": True,
        },
    }


def read_toml_config(path: Path) -> Optional[Dict]:
    if not path or not path.exists():
        return None
    if _toml_reader is None:
        return None
    try:
        with path.open("rb") as f:
            return _toml_reader.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"[bold red]Error reading config[/]: {exc}")
        return None


def write_toml_config(path: Path, cfg: Dict) -> bool:
    if _toml_writer is None:
        print("[yellow]toml package not available; skipping config write[/]")
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            toml_str = _toml_writer.dumps(cfg)
            if not toml_str.endswith("\n"):
                toml_str += "\n"
            f.write(toml_str)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[bold red]Error writing config[/]: {exc}")
        return False


# --- Data structures ------------------------------------------------------------
@dataclass
class ScoutEvent:
    # CSV order: timestamp_s, team, player_name, player, action, action_code, result, pos_x_m, pos_y_m
    timestamp_s: float
    team: str
    player_name: str
    player: str
    action: str
    action_code: int
    result: str
    pos_x_m: float
    pos_y_m: float

    def to_row(self) -> Dict[str, object]:
        return asdict(self)


# --- Friendly TOML template writer ---------------------------------------------
def _toml_template_with_comments(cfg: Dict) -> str:
    """Generate a human-friendly TOML template with comments in English."""
    program = cfg.get("program", {})
    project = cfg.get("project", {})
    teams = cfg.get("teams", {})
    drawing = cfg.get("drawing", {})
    results = cfg.get("results", [])
    actions = cfg.get("actions", [])

    def _quote_list_str(items):
        return ", ".join(f'"{str(i)}"' for i in items)

    home = teams.get("home", {})
    away = teams.get("away", {})
    home_players = home.get("players", [])
    away_players = away.get("players", [])
    home_names = home.get("players_names", {})
    away_names = away.get("players_names", {})

    # players_names objects
    def _players_names_block(mapping: Dict[str, str]) -> str:
        if not mapping:
            return ""
        lines = []
        for k, v in mapping.items():
            lines.append(f'"{k}" = "{v}"')
        return "\n".join(lines)

    # Actions as array of tables
    actions_blocks = []
    if actions and isinstance(actions[0], dict):
        for a in actions:
            name = str(a.get("name", "Action"))
            code = a.get("code", "")
            symbol = str(a.get("symbol", "o"))
            color = str(a.get("color", "#FFD700"))
            actions_blocks.append(
                "\n".join([
                    "[[actions]]",
                    f'name = "{name}"',
                    (f"code = {int(code)}" if str(code).isdigit() else f'code = "{code}"') if code != "" else "code = -1",
                    f'symbol = "{symbol}"',
                    f'color = "{color}"',
                    "",
                ])
            )
    else:
        # fallback list of names
        for n in actions:
            actions_blocks.append(
                "\n".join([
                    "[[actions]]",
                    f'name = "{str(n)}"',
                    "code = -1",
                    'symbol = "o"',
                    'color = "#FFD700"',
                    "",
                ])
            )

    template = f"""
# vailá Scout configuration (TOML)
# This file controls the integrated scouting GUI behavior and visuals.
# All values are in English and documented below for clarity.

[program]
# Metadata about this tool
name = "{program.get('name', 'vaila_scout')}"
version = "{program.get('version', '0.1.0')}"
author = "{program.get('author', '')}"
email = "{program.get('email', '')}"
description = "{program.get('description', 'Integrated scouting (annotation + analysis) for soccer.')}"
repository = "{program.get('repository', '')}"
license = "{program.get('license', '')}"
homepage = "{program.get('homepage', '')}"
created = "{program.get('created', '')}"
updated = "{program.get('updated', '')}"

[project]
# Field size in meters (FIFA standard is 105 x 68)
name = "{project.get('name', 'vaila_scout')}"
field_width_m = {float(project.get('field_width_m', 105.0))}
field_height_m = {float(project.get('field_height_m', 68.0))}
sport = "{project.get('sport', 'soccer')}"
field_units = "{project.get('field_units', 'meters')}"
field_standard = "{project.get('field_standard', 'FIFA')}"
description = "{project.get('description', 'Default soccer field and teams for scouting.')}"

[teams.home]
# Home team settings: display name, color, roster, and player name mapping
name = "{home.get('name', 'HOME')}"
color = "{home.get('color', '#1f77b4')}"  # jersey color used for player circles
players = [{_quote_list_str(home_players)}]

[teams.home.players_names]
# Map shirt numbers to player names (optional). Example: "10" = "Playmaker"
{_players_names_block(home_names) if _players_names_block(home_names) else '# Add entries like: "10" = "Pelé"'}

[teams.away]
name = "{away.get('name', 'AWAY')}"
color = "{away.get('color', '#d62728')}"
players = [{_quote_list_str(away_players)}]

[teams.away.players_names]
{_players_names_block(away_names) if _players_names_block(away_names) else '# Add entries like: "10" = "Maradona"'}

[drawing]
# Visual defaults for drawing players and action symbols on the field
player_circle_radius_m = {float(drawing.get('player_circle_radius_m', 0.6))}
player_edge_color = "{drawing.get('player_edge_color', 'black')}"
player_number_color = "{drawing.get('player_number_color', 'white')}"
player_number_size = {int(drawing.get('player_number_size', 8))}
action_symbol_size = {float(drawing.get('action_symbol_size', 90))}
show_player_name = {str(bool(drawing.get('show_player_name', True))).lower()}  # draw small label near circle
player_name_size = {int(drawing.get('player_name_size', 8))}

# Results are generic and independent from actions. Use any labels you need.
results = [{_quote_list_str(results)}]

# Actions are listed at the end for readability. Each action has a display name,
# a plot symbol/marker, and a color used for the symbol.
# Common symbols: o, *, +, x, ^, s (square), P (plus filled), D (diamond), h (hexagon).
{''.join(actions_blocks)}
""".strip() + "\n"
    return template


def write_toml_template(path: Path, cfg: Dict) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(_toml_template_with_comments(cfg))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[bold red]Error writing template config[/]: {exc}")
        return False

# --- Soccer field drawing -------------------------------------------------------
def draw_soccer_field(ax: plt.Axes, field_w: float, field_h: float) -> None:
    """Draws a standard soccer field (FIFA 105m x 68m) to scale on the axis.

    Field origin is bottom-left at (0, 0) and top-right at (field_w, field_h).
    """
    margin = 2.0
    ax.clear()
    ax.set_xlim(-margin, field_w + margin)
    ax.set_ylim(-margin, field_h + margin)
    ax.set_aspect("equal")
    ax.axis("off")

    # Grass
    ax.add_patch(
        patches.Rectangle(
            (0, 0), field_w, field_h, facecolor="forestgreen", edgecolor="none", zorder=0
        )
    )

    line_color = "white"
    lw = 2

    # Perimeter
    ax.add_patch(
        patches.Rectangle((0, 0), field_w, field_h, fill=False, edgecolor=line_color, linewidth=lw)
    )

    # Halfway line
    ax.plot([field_w / 2, field_w / 2], [0, field_h], color=line_color, linewidth=lw)

    # Center circle
    center = (field_w / 2.0, field_h / 2.0)
    center_circle_r = 9.15  # 10 yd ≈ 9.15 m
    ax.add_patch(patches.Circle(center, center_circle_r, fill=False, edgecolor=line_color, linewidth=lw))

    # Center spot
    ax.add_patch(patches.Circle(center, 0.2, color=line_color))

    # Penalty areas and goal areas (dimensions per FIFA; standard layout)
    penalty_w = 40.32
    penalty_d = 16.5
    goal_w = 18.32
    goal_d = 5.5

    # Left penalty area
    ax.add_patch(
        patches.Rectangle((0, (field_h - penalty_w) / 2), penalty_d, penalty_w, fill=False, edgecolor=line_color, linewidth=lw)
    )
    # Right penalty area
    ax.add_patch(
        patches.Rectangle((field_w - penalty_d, (field_h - penalty_w) / 2), penalty_d, penalty_w, fill=False, edgecolor=line_color, linewidth=lw)
    )

    # Left goal area
    ax.add_patch(
        patches.Rectangle((0, (field_h - goal_w) / 2), goal_d, goal_w, fill=False, edgecolor=line_color, linewidth=lw)
    )
    # Right goal area
    ax.add_patch(
        patches.Rectangle((field_w - goal_d, (field_h - goal_w) / 2), goal_d, goal_w, fill=False, edgecolor=line_color, linewidth=lw)
    )

    # Penalty spots
    ax.add_patch(patches.Circle((11, field_h / 2.0), 0.2, color=line_color))
    ax.add_patch(patches.Circle((field_w - 11, field_h / 2.0), 0.2, color=line_color))

    # Penalty arcs (outside the penalty area, facing field center)
    # Left side (x=11): arc around 0° (points to +x)
    arc_left = patches.Arc((11, field_h / 2.0), 2 * 9.15, 2 * 9.15, theta1=310, theta2=50, edgecolor=line_color, linewidth=lw)
    ax.add_patch(arc_left)
    # Right side (x=field_w-11): arc around 180° (points to -x)
    arc_right = patches.Arc((field_w - 11, field_h / 2.0), 2 * 9.15, 2 * 9.15, theta1=130, theta2=230, edgecolor=line_color, linewidth=lw)
    ax.add_patch(arc_right)


# --- Main Application -----------------------------------------------------------
class ScoutApp(tk.Tk):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.title("vailá - Scout - v0.1.0")
        self.geometry("1280x800")

        self.cfg = cfg
        self.cfg_path: Optional[Path] = None
        self.field_w = float(cfg["project"]["field_width_m"])  # meters
        self.field_h = float(cfg["project"]["field_height_m"])  # meters
        self.start_time = time.time()
        self.events: List[ScoutEvent] = []

        # --- Menu bar
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save CSV	Ctrl+S", command=self.save_csv)
        file_menu.add_command(label="Load CSV	Ctrl+O", command=self.load_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Show Heatmap (H)", command=self.draw_heatmap)
        menubar.add_cascade(label="View", menu=view_menu)

        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Load Config	Ctrl+L", command=self.load_config)
        tools_menu.add_command(label="Save Config	Ctrl+Shift+S", command=self.save_config)
        tools_menu.add_command(label="Create Template", command=self._create_template_config)
        tools_menu.add_command(label="Rename Teams	Ctrl+T", command=self.rename_teams)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Help (?)", command=self.open_help)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.open_shortcuts)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

        # --- Layout frames
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(container)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        right_frame = ttk.Frame(container)
        right_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=8, pady=8)

        # --- Controls (top-right)
        controls = ttk.Frame(right_frame)
        controls.pack(side=tk.TOP, fill=tk.X)

        # Internal state for team/player (not shown in the UI)
        teams = [cfg["teams"]["home"]["name"], cfg["teams"]["away"]["name"]]
        self.team_var = tk.StringVar(value=teams[0])
        self.player_var = tk.StringVar(value="")
        # Hidden combobox to reuse existing logic without showing controls
        self.player_cb = ttk.Combobox(controls, textvariable=self.player_var, values=[], width=1, state="readonly")
        self.player_cb.grid_remove()

        # Action selector (normalized structured actions)
        ttk.Label(controls, text="Action:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        self.actions_cfg = self._normalize_actions_config(cfg.get("actions", []))
        self.action_names = [a["name"] for a in self.actions_cfg]
        self.name_to_action = {a["name"]: a for a in self.actions_cfg}
        first_action = self.action_names[0] if self.action_names else ""
        self.action_var = tk.StringVar(value=first_action)
        self.action_cb = ttk.Combobox(controls, textvariable=self.action_var, values=self.action_names, width=16, state="readonly")
        self.action_cb.grid(row=1, column=1, padx=2, pady=2)

        # Result selector
        ttk.Label(controls, text="Result:").grid(row=1, column=2, sticky=tk.W, padx=2, pady=2)
        self.results_list = list(cfg.get("results", _default_config()["results"]))
        if not self.results_list:
            self.results_list = _default_config()["results"]
        self.result_var = tk.StringVar(value=self.results_list[0])
        self.result_cb = ttk.Combobox(controls, textvariable=self.result_var, values=self.results_list, width=10, state="readonly")
        self.result_cb.grid(row=1, column=3, padx=2, pady=2)

        # Timer display and controls
        timer_frame = ttk.Frame(controls)
        timer_frame.grid(row=2, column=0, columnspan=4, sticky=tk.W, padx=2, pady=(2, 4))
        ttk.Label(timer_frame, text="Clock:").pack(side=tk.LEFT)
        self.clock_var = tk.StringVar(value="00:00.0")
        self.clock_label = ttk.Label(timer_frame, textvariable=self.clock_var, font=("Segoe UI", 10, "bold"))
        self.clock_label.pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(timer_frame, text="Start", command=self.start_timer).pack(side=tk.LEFT, padx=2)
        ttk.Button(timer_frame, text="Pause", command=self.pause_timer).pack(side=tk.LEFT, padx=2)
        ttk.Button(timer_frame, text="Reset", command=self.reset_timer).pack(side=tk.LEFT, padx=2)
        self.manual_time_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(timer_frame, text="Manual time", variable=self.manual_time_var).pack(side=tk.LEFT, padx=(8, 2))
        self.manual_time_entry = ttk.Entry(timer_frame, width=8)
        self.manual_time_entry.insert(0, "0.0")
        self.manual_time_entry.pack(side=tk.LEFT)
        ttk.Button(timer_frame, text="Set=Now", command=self._set_manual_time_now).pack(side=tk.LEFT, padx=(4, 0))

        # Buttons
        btns = ttk.Frame(controls)
        btns.grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=2, pady=(6, 4))
        ttk.Button(btns, text="Save CSV", command=self.save_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Load CSV", command=self.load_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Clear Events", command=self.clear_events).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Show Heatmap", command=self.draw_heatmap).pack(side=tk.LEFT, padx=12)
        ttk.Button(btns, text="Reset Time", command=self.reset_timer).pack(side=tk.LEFT, padx=12)

        # Config buttons (without Edit Config)
        cfg_btns = ttk.Frame(controls)
        cfg_btns.grid(row=4, column=0, columnspan=4, sticky=tk.W, padx=2, pady=(2, 6))
        ttk.Button(cfg_btns, text="Load Config", command=self.load_config).pack(side=tk.LEFT, padx=4)
        ttk.Button(cfg_btns, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=4)
        ttk.Button(cfg_btns, text="Create Template", command=self._create_template_config).pack(side=tk.LEFT, padx=4)

        # --- Field figure (left)
        self.field_fig, self.field_ax = plt.subplots(figsize=(6.8, 4.3))
        self.field_canvas = FigureCanvasTkAgg(self.field_fig, master=left_frame)
        self.field_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._redraw_field()
        self.field_canvas.mpl_connect("button_press_event", self._on_field_click)

        # Remove auto-number/unique-per-click (not needed)
        self.auto_number_players = False
        self.unique_player_per_click = False
        self.next_player_number_by_team: Dict[str, int] = {}
        
        # Ctrl state tracking for remove functionality
        self._ctrl_pressed = False

        # Sync players list on team change
        # No visible team combobox; keep hotkey 'T' to toggle team

        # Hotkeys (simple)
        self.bind_all("<Control-s>", lambda e: self.save_csv())
        self.bind_all("<Control-o>", lambda e: self.load_csv())
        self.bind_all("<Control-k>", lambda e: self.clear_events())
        self.bind_all("h", lambda e: self.draw_heatmap())
        self.bind_all("H", lambda e: self.draw_heatmap())
        self.bind_all("r", lambda e: self.reset_timer())
        self.bind_all("R", lambda e: self.reset_timer())
        self.bind_all("?", lambda e: self.open_help())
        # self.bind_all("<Control-e>", lambda e: self.edit_config())  # disabled - removed Edit Config
        self.bind_all("<Control-l>", lambda e: self.load_config())
        self.bind_all("<Control-S>", lambda e: self.save_config())
        # Additional hotkeys
        self.bind_all("t", lambda e: self._toggle_team())
        self.bind_all("T", lambda e: self._toggle_team())
        # self.bind_all("<Control-t>", lambda e: self.rename_teams())  # optional
        
        # Ctrl state tracking for remove functionality
        self.bind_all("<KeyPress-Control_L>", lambda e: self._set_ctrl_pressed(True))
        self.bind_all("<KeyPress-Control_R>", lambda e: self._set_ctrl_pressed(True))
        self.bind_all("<KeyRelease-Control_L>", lambda e: self._set_ctrl_pressed(False))
        self.bind_all("<KeyRelease-Control_R>", lambda e: self._set_ctrl_pressed(False))
        
        # Space toggles timer - will be bound after _toggle_timer is defined
        # Action hotkeys from config (single-letter) - DISABLED, using numeric codes instead
        # for k, act in self.key_to_action.items():
        #     if not k:
        #         continue
        #     def _mk_set_action(name: str):
        #         return lambda _e: self._set_action_by_name(name)
        #     self.bind_all(k, _mk_set_action(act["name"]))

        # Quick action code buffer (digits map to action codes)
        self.quick_action_buffer = ""
        self.quick_label = ttk.Label(controls, text="Quick action code: (none)")
        self.quick_label.grid(row=6, column=0, columnspan=4, sticky=tk.W, padx=2, pady=(2, 2))
        for _d in list("0123456789"):
            self.bind_all(_d, self._on_digit_action)
        self.bind_all("<BackSpace>", self._on_backspace_action)
        self.bind_all("<Escape>", self._on_escape_action)
        self.bind_all("<Return>", self._on_enter_apply_action)

        # --- Rosters (bottom-right): Home and Away side by side
        rosters = ttk.Frame(right_frame)
        rosters.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=2, pady=(4, 2))

        home_box = ttk.Labelframe(rosters, text=f"Home ({self.cfg['teams']['home']['name']})")
        home_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        away_box = ttk.Labelframe(rosters, text=f"Away ({self.cfg['teams']['away']['name']})")
        away_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        self.home_tv = ttk.Treeview(home_box, columns=("no", "name"), show="headings", height=12)
        self.home_tv.heading("no", text="No")
        self.home_tv.heading("name", text="Name")
        self.home_tv.column("no", width=40, anchor=tk.CENTER)
        self.home_tv.column("name", width=120, anchor=tk.W)
        self.home_tv.pack(fill=tk.BOTH, expand=True)
        self.home_tv.bind("<<TreeviewSelect>>", lambda e: self._on_roster_select("home"))

        self.away_tv = ttk.Treeview(away_box, columns=("no", "name"), show="headings", height=12)
        self.away_tv.heading("no", text="No")
        self.away_tv.heading("name", text="Name")
        self.away_tv.column("no", width=40, anchor=tk.CENTER)
        self.away_tv.column("name", width=120, anchor=tk.W)
        self.away_tv.pack(fill=tk.BOTH, expand=True)
        self.away_tv.bind("<<TreeviewSelect>>", lambda e: self._on_roster_select("away"))

        # Populate initial players list based on current team
        self._on_team_changed()
        # Populate rosters
        self._populate_rosters()
        # Timer state
        self._timer_running = False
        self._timer_start_epoch = time.time()
        self._timer_paused_elapsed = 0.0
        self._tick_timer()
        
        # Now bind Space key after _toggle_timer is defined
        self.bind_all("<space>", lambda e: self._toggle_timer())

    def destroy(self):
        try:
            if hasattr(self, "_tick_after_id") and self._tick_after_id:
                self.after_cancel(self._tick_after_id)
        except Exception:
            pass
        super().destroy()

    # --- UI helpers
    def _on_team_changed(self, _evt=None):
        team_name = self.team_var.get()
        team_key = self._team_key_from_name(team_name)
        self._refresh_player_combobox()
        self._update_player_name_label()
        self._populate_rosters()
    
    def _on_player_changed(self, _evt=None):
        self._update_player_name_label()

    def _redraw_field(self):
        draw_soccer_field(self.field_ax, self.field_w, self.field_h)
        self.field_fig.tight_layout()
        self.field_canvas.draw()

    # --- Timer helpers
    def _format_time(self, seconds: float) -> str:
        seconds = max(0.0, seconds)
        m = int(seconds // 60)
        s = seconds - 60 * m
        return f"{m:02d}:{s:04.1f}"

    def _tick_timer(self):
        # Update clock label periodically
        if hasattr(self, "clock_var"):
            self.clock_var.set(self._format_time(self._get_current_time_s()))
        try:
            self._tick_after_id = self.after(200, self._tick_timer)
        except Exception:
            pass

    def _get_current_time_s(self) -> float:
        if getattr(self, "manual_time_var", None) and self.manual_time_var.get():
            try:
                return float(self.manual_time_entry.get())
            except Exception:
                return 0.0
        if not getattr(self, "_timer_running", False):
            return float(getattr(self, "_timer_paused_elapsed", 0.0))
        return float(self._timer_paused_elapsed + (time.time() - self._timer_start_epoch))

    def _get_live_time_s(self) -> float:
        # Live timer ignoring manual mode
        if not getattr(self, "_timer_running", False):
            return float(getattr(self, "_timer_paused_elapsed", 0.0))
        return float(self._timer_paused_elapsed + (time.time() - self._timer_start_epoch))

    def start_timer(self):
        if not getattr(self, "_timer_running", False):
            self._timer_start_epoch = time.time()
            self._timer_running = True

    def pause_timer(self):
        if getattr(self, "_timer_running", False):
            self._timer_paused_elapsed = self._get_current_time_s()
            self._timer_running = False

    def _toggle_timer(self):
        """Toggle timer between running and paused states."""
        # If manual mode is on, disable it so the live clock is visible
        if getattr(self, "manual_time_var", None) and self.manual_time_var.get():
            self.manual_time_var.set(False)
        if getattr(self, "_timer_running", False):
            self.pause_timer()
        else:
            self.start_timer()

    def _set_manual_time_now(self) -> None:
        try:
            current = self._get_live_time_s()
        except Exception:
            current = 0.0
        self.manual_time_entry.delete(0, tk.END)
        self.manual_time_entry.insert(0, f"{float(current):.1f}")

    def _init_next_player_numbers(self) -> Dict[str, int]:
        def next_from_list(players: List[str]) -> int:
            nums: List[int] = []
            for p in players:
                try:
                    nums.append(int(p))
                except Exception:  # noqa: BLE001
                    continue
            return (max(nums) + 1) if nums else 1

        home = self.cfg["teams"]["home"]["name"]
        away = self.cfg["teams"]["away"]["name"]
        return {
            home: next_from_list(self._get_players_numbers("home")),
            away: next_from_list(self._get_players_numbers("away")),
        }

    # --- Event handling
    def _on_field_click(self, event):
        if event.inaxes != self.field_ax:
            return
        try:
            # Matplotlib gives data coordinates directly
            x_m = float(event.xdata)
            y_m = float(event.ydata)
        except Exception:  # noqa: BLE001
            return

        if not (0 <= x_m <= self.field_w and 0 <= y_m <= self.field_h):
            return

        # Check for Ctrl + right click to remove events
        btn = int(getattr(event, "button", 1) or 1)
        
        # Check for Ctrl + right click to remove events
        if btn == 3 and self._ctrl_pressed:
            # Ctrl + right click: remove events near this position
            self._remove_events_near_position(x_m, y_m)
            return

        timestamp = self._get_current_time_s()
        # Result via mouse button: 1=left→success, 2=middle→neutral, 3=right→fail
        if btn == 1:
            clicked_result = "success"
        elif btn == 3:
            clicked_result = "fail"
        else:
            clicked_result = "neutral"
        self.result_var.set(clicked_result)
        # Player assignment rules
        team_name = self.team_var.get()
        player_str = self.player_var.get()
        if self.unique_player_per_click or self.auto_number_players:
            next_num = self.next_player_number_by_team.get(team_name, 1)
            player_str = str(next_num)
            self.next_player_number_by_team[team_name] = next_num + 1
            self._ensure_player_in_team(team_name, player_str)
            # Reflect in combobox
            current_vals = list(self.player_cb.cget("values"))
            if player_str not in current_vals:
                current_vals.append(player_str)
                self.player_cb.config(values=current_vals)
            self.player_var.set(player_str)
        elif hasattr(self, 'quick_player_buffer') and self.quick_player_buffer:
            # Apply numeric buffer
            player_str = self.quick_player_buffer
            self._ensure_player_in_team(team_name, player_str)
            current_vals = list(self.player_cb.cget("values"))
            if player_str not in current_vals:
                current_vals.append(player_str)
                self.player_cb.config(values=current_vals)
            self.player_var.set(player_str)
            self.quick_player_buffer = ""
            self._update_quick_label()

        # Action code lookup
        _act = next((a for a in self.actions_cfg if a["name"] == self.action_var.get()), None)
        _act_code = int(_act.get("code", -1)) if _act else -1

        ev = ScoutEvent(
            timestamp_s=round(timestamp, 2),
            team=team_name,
            player_name=self._get_player_name(self._team_key_from_name(team_name),
                                      self._parse_display_to_number(self.player_var.get())),
            player=self._parse_display_to_number(self.player_var.get()),
            action=self.action_var.get(),
            action_code=_act_code,
            result=self.result_var.get(),
            pos_x_m=round(x_m, 2),
            pos_y_m=round(y_m, 2),
        )
        self.events.append(ev)

        # Visual marker with team color, number, name (optional) and action symbol
        self._draw_player_marker_with_action(x_m, y_m, ev)
        self.field_canvas.draw()

    # --- Actions
    def reset_timer(self):
        self._timer_running = False
        self._timer_paused_elapsed = 0.0
        self._timer_start_epoch = time.time()
        self.clock_var.set(self._format_time(0.0))

    def clear_events(self):
        if not self.events:
            return
        if not messagebox.askyesno("Confirm", "Clear all events?"):
            return
        self.events.clear()
        self._redraw_field()

    def save_csv(self):
        if not self.events:
            messagebox.showwarning("No data", "No events to save.")
            return
        df = pd.DataFrame([e.to_row() for e in self.events])
        # Ensure CSV columns order as requested
        desired_cols = [
            "timestamp_s",
            "team",
            "player_name",
            "player",
            "action",
            "action_code",
            "result",
            "pos_x_m",
            "pos_y_m",
        ]
        ordered = [c for c in desired_cols if c in df.columns] + [c for c in df.columns if c not in desired_cols]
        df = df[ordered]
        path = filedialog.asksaveasfilename(
            title="Save Events CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            df.to_csv(path, index=False)
            self._save_players_summary(Path(path))
            messagebox.showinfo("Saved", f"Saved: {path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Failed to save CSV: {exc}")

    def load_csv(self):
        path = filedialog.askopenfilename(
            title="Load Events CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
            required = {"timestamp_s", "team", "player", "action", "result", "pos_x_m", "pos_y_m"}
            if not required.issubset(df.columns):
                messagebox.showerror("Error", "Invalid CSV structure.")
                return
            has_name = "player_name" in df.columns
            has_action_code = "action_code" in df.columns
            self.events = [
                ScoutEvent(
                    timestamp_s=float(row["timestamp_s"]),
                    team=str(row["team"]),
                    player_name=str(row["player_name"]) if has_name and not pd.isna(row["player_name"]) else "",
                    player=str(row["player"]),
                    action=str(row["action"]),
                    action_code=int(row["action_code"]) if has_action_code and not pd.isna(row["action_code"]) else -1,
                    result=str(row["result"]),
                    pos_x_m=float(row["pos_x_m"]),
                    pos_y_m=float(row["pos_y_m"]),
                )
                for _, row in df.iterrows()
            ]
            # Update names mapping from CSV if present
            if has_name:
                for e in self.events:
                    if e.player_name:
                        k = self._team_key_from_name(e.team)
                        self.cfg["teams"][k].setdefault("players_names", {})[str(e.player)] = e.player_name
            # Re-plot on field
            self._redraw_field()
            if self.events:
                for e in self.events:
                    self._draw_player_marker_with_action(e.pos_x_m, e.pos_y_m, e)
                self.field_canvas.draw()
            messagebox.showinfo("Loaded", f"Loaded: {path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Failed to load CSV: {exc}")

    def draw_heatmap(self):
        if not self.events:
            messagebox.showwarning("No data", "No events available for heatmap.")
            return
        # Window with team/player filters
        win = tk.Toplevel(self)
        win.title("Scout Heatmap")
        win.resizable(True, True)
        win.minsize(480, 360)
        ctrl = ttk.Frame(win)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        # Team filter
        ttk.Label(ctrl, text="Team:").pack(side=tk.LEFT)
        ALL = "All"
        team_values = [ALL, self.cfg["teams"]["home"]["name"], self.cfg["teams"]["away"]["name"]]
        team_var = tk.StringVar(value=ALL)
        team_cb = ttk.Combobox(ctrl, textvariable=team_var, values=team_values, state="readonly", width=14)
        team_cb.pack(side=tk.LEFT, padx=6)

        # Player filter
        ttk.Label(ctrl, text="Player:").pack(side=tk.LEFT)
        player_var = tk.StringVar(value=ALL)
        player_cb = ttk.Combobox(ctrl, textvariable=player_var, values=[ALL], state="readonly", width=12)
        player_cb.pack(side=tk.LEFT, padx=6)

        # Plot area
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def refresh_players(*_):
            sel_team = team_var.get()
            if sel_team == ALL:
                player_cb.config(values=[ALL])
                player_var.set(ALL)
            else:
                key = "home" if sel_team == self.cfg["teams"]["home"]["name"] else "away"
                values = [ALL] + list(self.cfg["teams"][key].get("players", []))
                player_cb.config(values=values)
                player_var.set(ALL)

        def plot_heatmap():
            try:
                df = pd.DataFrame([e.to_row() for e in self.events])
                filt = df
                sel_team = team_var.get()
                sel_player = player_var.get()
                if sel_team != ALL:
                    filt = filt[filt["team"] == sel_team]
                if sel_player != ALL:
                    filt = filt[filt["player"].astype(str) == str(sel_player)]
                ax.clear()
                draw_soccer_field(ax, self.field_w, self.field_h)
                if len(filt) > 0:
                    sns.kdeplot(
                        data=filt,
                        x="pos_x_m",
                        y="pos_y_m",
                        cmap="Reds",
                        fill=True,
                        alpha=0.6,
                        bw_method="scott",
                        thresh=0.05,
                        ax=ax,
                    )
                ax.set_xlim(0, self.field_w)
                ax.set_ylim(0, self.field_h)
                title = "Heatmap"
                if sel_team != ALL:
                    title += f" - {sel_team}"
                if sel_player != ALL:
                    title += f" (player {sel_player})"
                ax.set_title(title)
                canvas.draw()
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Error", f"Failed to generate heatmap: {exc}")

        ttk.Button(ctrl, text="Show", command=plot_heatmap).pack(side=tk.LEFT, padx=8)
        team_cb.bind("<<ComboboxSelected>>", refresh_players)
        plot_heatmap()

    def _toggle_auto(self):
        self.auto_number_players = bool(self.chk_auto_var.get()) if hasattr(self, "chk_auto_var") else not self.auto_number_players
        if self.auto_number_players and not self.next_player_number_by_team:
            self.next_player_number_by_team = self._init_next_player_numbers()

    def _toggle_unique(self):
        self.unique_player_per_click = bool(self.chk_unique_var.get()) if hasattr(self, "chk_unique_var") else not self.unique_player_per_click
        if self.unique_player_per_click and not self.next_player_number_by_team:
            self.next_player_number_by_team = self._init_next_player_numbers()

    def _ensure_player_in_team(self, team_name: str, player: str) -> None:
        key = self._team_key_from_name(team_name)
        players = self.cfg["teams"][key].setdefault("players", [])
        if player not in players:
            players.append(player)
        self.cfg["teams"][key].setdefault("players_names", {}).setdefault(str(player), "")
    
    def _remove_events_near_position(self, x_m: float, y_m: float, radius_m: float = 1.0) -> None:
        """Remove the most recent event within radius_m meters of the given position."""
        if not self.events:
            return
            
        # Find events to remove (sorted by timestamp, most recent first)
        events_to_remove = []
        for ev in self.events:
            distance = ((ev.pos_x_m - x_m) ** 2 + (ev.pos_y_m - y_m) ** 2) ** 0.5
            if distance <= radius_m:
                events_to_remove.append(ev)
        
        if not events_to_remove:
            return
            
        # Sort by timestamp (most recent first) and remove only the most recent one
        events_to_remove.sort(key=lambda ev: ev.timestamp_s, reverse=True)
        event_to_remove = events_to_remove[0]
        
        # Remove the most recent event
        self.events.remove(event_to_remove)
        
        # Redraw field
        self._redraw_field()
        for ev in self.events:
            self._draw_player_marker_with_action(ev.pos_x_m, ev.pos_y_m, ev)
        self.field_canvas.draw()
        
        print(f"Removed event at ({event_to_remove.pos_x_m:.1f}, {event_to_remove.pos_y_m:.1f}) - {event_to_remove.team} {event_to_remove.player} {event_to_remove.action}")
    
    def _set_ctrl_pressed(self, pressed: bool) -> None:
        """Set the Ctrl key pressed state."""
        self._ctrl_pressed = pressed
    
    # --- Drawing helpers
    def _get_team_color(self, team_name: str) -> str:
        key = self._team_key_from_name(team_name)
        return str(self.cfg["teams"][key].get("color", "#1f77b4"))

    def _draw_player_marker_with_action(self, x_m: float, y_m: float, ev: "ScoutEvent") -> None:
        draw_cfg = self.cfg.get("drawing", {})
        circle_r = float(draw_cfg.get("player_circle_radius_m", 0.6))
        edge_default = str(draw_cfg.get("player_edge_color", "black"))
        number_color = str(draw_cfg.get("player_number_color", "white"))
        number_size = int(draw_cfg.get("player_number_size", 8))
        action_symbol_size = float(draw_cfg.get("action_symbol_size", 90))
        show_player_name = bool(draw_cfg.get("show_player_name", True))
        player_name_size = int(draw_cfg.get("player_name_size", 8))

        # Edge color by result
        result_edge = {
            "success": "#1E90FF",  # blue
            "fail": "#FF0000",     # red
            "neutral": "#FFFFFF",  # white
        }.get(str(ev.result).lower(), edge_default)

        team_color = self._get_team_color(ev.team)
        circle = patches.Circle((x_m, y_m), radius=circle_r, facecolor=team_color, edgecolor=result_edge, linewidth=1.6, zorder=6)
        self.field_ax.add_patch(circle)

        # Number in center
        self.field_ax.text(x_m, y_m, str(ev.player), color=number_color, ha="center", va="center", fontsize=number_size, zorder=7)

        # Optional player name label
        if show_player_name and getattr(ev, "player_name", ""):
            self.field_ax.text(
                x_m,
                y_m - circle_r * 1.4,
                ev.player_name,
                color="white",
                ha="center",
                va="top",
                fontsize=player_name_size,
                zorder=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6),
            )

        # Action symbol - always show for better distinction
        act = self.name_to_action.get(ev.action, None)
        if act is not None:
            symbol = str(act.get("symbol", "o"))
            a_color = str(act.get("color", "yellow"))
            off = circle_r * 1.2
            mx, my = x_m + off, y_m + off
            
            # Enhanced symbol drawing with better visibility
            marker_map = {"o": "o", "*": "*", "+": "+", "x": "x", "X": "x", "^": "^", "s": "s", "P": "P", "D": "D", "h": "h"}
            m = marker_map.get(symbol, None)
            
            if m is not None and len(m) == 1:
                unfilled = m in {"+", "x"}
                if unfilled:
                    # For unfilled markers, use larger size and add background
                    self.field_ax.scatter([mx], [my], s=action_symbol_size * 1.5, c=a_color, marker=m, 
                                        linewidth=2, edgecolors="black", zorder=8)
                else:
                    # For filled markers, add edge for better visibility
                    self.field_ax.scatter([mx], [my], s=action_symbol_size, c=a_color, 
                                        edgecolors="black", linewidth=1, marker=m, zorder=8)
            else:
                # For text symbols, add background for better visibility
                self.field_ax.text(mx, my, symbol, color=a_color, fontsize=number_size + 3, 
                                 weight="bold", zorder=8, ha="center", va="center",
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    # --- Actions helpers
    def _normalize_actions_config(self, actions_cfg: List) -> List[Dict[str, str]]:
        """Accepts either list[str] or list[dict] and returns list of dicts with keys name,code,symbol,color."""
        out: List[Dict[str, str]] = []
        if not actions_cfg:
            return _default_config()["actions"]
        if isinstance(actions_cfg, list) and actions_cfg and isinstance(actions_cfg[0], dict):
            # ensure defaults
            for a in actions_cfg:
                out.append({
                    "name": str(a.get("name", "Action")),
                    "code": int(a.get("code", -1)) if str(a.get("code", "")).isdigit() else a.get("code", -1),
                    "symbol": str(a.get("symbol", "o")),
                    "color": str(a.get("color", "#FFD700")),
                })
            return out
        # assume list of names
        for idx, n in enumerate(actions_cfg, start=1):
            name = str(n)
            out.append({"name": name, "code": idx, "symbol": "o", "color": "#FFD700"})
        return out

    def _set_action_by_name(self, name: str) -> None:
        if name in self.action_names:
            self.action_var.set(name)

    def _cycle_result(self) -> None:
        if not hasattr(self, "results_list") or not self.results_list:
            self.results_list = _default_config()["results"]
        current = self.result_var.get()
        try:
            idx = self.results_list.index(current)
        except ValueError:
            idx = -1
        self.result_var.set(self.results_list[(idx + 1) % len(self.results_list)])

    # --- Config management
    def load_config(self):
        path_str = filedialog.askopenfilename(
            title="Select TOML config",
            filetypes=(("TOML files", "*.toml"), ("All files", "*.*")),
        )
        if not path_str:
            return
        path = Path(path_str)
        cfg = read_toml_config(path)
        if cfg is None:
            messagebox.showerror("Error", "Failed to read selected config.")
            return
        # Backwards compatibility: accept old schema without drawing or structured actions
        if "drawing" not in cfg:
            cfg["drawing"] = _default_config()["drawing"]
        if "results" not in cfg:
            cfg["results"] = _default_config()["results"]
        # Normalize actions to structured form
        cfg["actions"] = self._normalize_actions_config(cfg.get("actions", _default_config()["actions"]))
        self.cfg = cfg
        self.cfg_path = path
        # Update UI based on new config
        self.field_w = float(cfg["project"]["field_width_m"])
        self.field_h = float(cfg["project"]["field_height_m"])
        # Refresh actions
        self.actions_cfg = self._normalize_actions_config(cfg.get("actions", []))
        self.action_names = [a["name"] for a in self.actions_cfg]
        self.name_to_action = {a["name"]: a for a in self.actions_cfg}
        self.action_cb.config(values=self.action_names)
        if self.action_names:
            self.action_var.set(self.action_names[0])
        teams = [cfg["teams"]["home"]["name"], cfg["teams"]["away"]["name"]]
        self.team_var.set(teams[0])
        self._on_team_changed()
        self._redraw_field()
        self.next_player_number_by_team = {}

    def save_config(self):
        if self.cfg_path is None:
            path_str = filedialog.asksaveasfilename(
                title="Save TOML config",
                defaultextension=".toml",
                filetypes=(("TOML files", "*.toml"), ("All files", "*.*")),
            )
            if not path_str:
                return
            self.cfg_path = Path(path_str)
        ok = write_toml_config(self.cfg_path, self.cfg)
        if ok:
            messagebox.showinfo("Config", f"Config saved to\n{self.cfg_path}")
        else:
            messagebox.showwarning("Config", "Config was not saved (writer unavailable).")

    def _create_template_config(self):
        here = Path(__file__).resolve()
        models_dir = here.parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / DEFAULT_CFG_FILENAME
        # Use current defaults but keep existing teams/actions/results if already loaded
        cfg = {
            "program": dict(self.cfg.get("program", _default_config()["program"])),
            "project": dict(self.cfg.get("project", _default_config()["project"])),
            "teams": {
                "home": dict(self.cfg.get("teams", {}).get("home", _default_config()["teams"]["home"])),
                "away": dict(self.cfg.get("teams", {}).get("away", _default_config()["teams"]["away"])),
            },
            "drawing": dict(self.cfg.get("drawing", _default_config()["drawing"])),
            "results": list(self.cfg.get("results", _default_config()["results"])),
            "actions": list(self.cfg.get("actions", _default_config()["actions"])),
        }
        ok = write_toml_template(out_path, cfg)
        if ok:
            messagebox.showinfo("Config", f"Template written to:\n{out_path}")
        else:
            messagebox.showerror("Config", "Failed to write template")

    def edit_config(self):
        dialog = ConfigDialog(self, self.cfg)
        self.wait_window(dialog)
        if getattr(dialog, "result", None):
            # Update internal config and UI
            self.cfg = dialog.result
            self.field_w = float(self.cfg["project"]["field_width_m"])
            self.field_h = float(self.cfg["project"]["field_height_m"])
            teams = [self.cfg["teams"]["home"]["name"], self.cfg["teams"]["away"]["name"]]
            self.team_cb.config(values=teams)
            # keep current team if possible
            if self.team_var.get() not in teams:
                self.team_var.set(teams[0])
            self._on_team_changed()
            self._redraw_field()
            self.next_player_number_by_team = self._init_next_player_numbers()

    def open_help(self):
        here = Path(__file__).resolve()
        help_dir = here.parent / "help"
        html_path = help_dir / "scout_vaila.html"
        md_path = help_dir / "scout_vaila.md"
        if html_path.exists():
            webbrowser.open_new_tab(html_path.as_uri())
            return
        if md_path.exists():
            try:
                webbrowser.open_new_tab(md_path.as_uri())
            except Exception as exc:  # noqa: BLE001
                messagebox.showinfo("Help", f"Help markdown is at vaila/help/scout_vaila.md. Could not open automatically: {exc}")
            return
        messagebox.showinfo("Help", "Help page not found at vaila/help/scout_vaila.html or .md")

    def open_shortcuts(self):
        win = tk.Toplevel(self)
        win.title("Keyboard Shortcuts")
        win.resizable(True, True)
        win.minsize(360, 200)

        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Permitir que as colunas do grid cresçam no resize
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=3)

        ttk.Label(frame, text="Keyboard Shortcuts", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))

        shortcuts = [
            ("Ctrl+S", "Save CSV"),
            ("Ctrl+O", "Load CSV"),
            ("Ctrl+K", "Clear events"),
            ("H", "Show heatmap"),
            ("R", "Reset timer"),
            ("Space", "Start/Pause clock"),
            ("?", "Open help"),
            ("T", "Toggle current team (home/away)"),
            ("Ctrl+L", "Load config"),
            ("Ctrl+Shift+S", "Save config"),
            ("Ctrl+T", "Rename teams"),
            ("Digits 0–9", "Enter action code; Enter apply; Backspace edit; Esc clear"),
            ("Mouse", "Left=success, Right=fail, Middle=neutral"),
            ("Ctrl+Right Click", "Remove events near clicked position (hold Ctrl, right click)"),
        ]

        for i, (keys, desc) in enumerate(shortcuts, start=1):
            ttk.Label(frame, text=keys, width=16).grid(row=i, column=0, sticky=tk.W, padx=(0, 12), pady=2)
            ttk.Label(frame, text=desc).grid(row=i, column=1, sticky=tk.W, pady=2)

        ttk.Button(frame, text="Close", command=win.destroy).grid(row=len(shortcuts) + 1, column=1, sticky=tk.E, pady=(10, 0))

    def _save_players_summary(self, events_csv_path: Path) -> None:
        try:
            if not self.events:
                return
            df = pd.DataFrame([e.to_row() for e in self.events])
            rows: List[Dict[str, str]] = []
            for (team, player), g in df.groupby(["team", "player"], sort=True):
                times = sorted(g["timestamp_s"].astype(float).tolist())
                rows.append({
                    "team": str(team),
                    "player": str(player),
                    "num_events": str(len(times)),
                    "first_timestamp_s": f"{times[0]:.2f}" if times else "",
                    "last_timestamp_s": f"{times[-1]:.2f}" if times else "",
                    "timestamps": ";".join(f"{t:.2f}" for t in times),
                })
            out_df = pd.DataFrame(rows).sort_values(["team", "player"]).reset_index(drop=True)
            out_path = events_csv_path.with_name(events_csv_path.stem + "_players_summary.csv")
            out_df.to_csv(out_path, index=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[yellow]Warning: could not write players summary[/]: {exc}")

    def _toggle_team(self):
        names = [self.cfg["teams"]["home"]["name"], self.cfg["teams"]["away"]["name"]]
        self.team_var.set(names[1] if self.team_var.get() == names[0] else names[0])
        self._on_team_changed()

    def _on_digit(self, event):
        pass

    def _on_backspace(self, _event):
        pass

    def _on_escape(self, _event):
        pass

    def _on_enter_apply_player(self, _event):
        pass

    def _on_plus_inc(self, _event):
            pass

    def _on_minus_dec(self, _event):
            pass

    def _on_next_player(self, _event=None):
        pass

    def _on_prev_player(self, _event=None):
        pass

    def _update_quick_label(self):
        buf = self.quick_action_buffer if hasattr(self, 'quick_action_buffer') and self.quick_action_buffer else "(none)"
        self.quick_label.config(text=f"Quick action code: {buf}")

    def _update_player_name_label(self):
        team_key = self._team_key_from_name(self.team_var.get())
        num = self._parse_display_to_number(self.player_var.get())
        name = self._get_player_name(team_key, num)
        if hasattr(self, 'player_name_label'):
            self.player_name_label.config(text=f"Name: {name if name else '-'}")

    # --- Quick action code mapping
    def _on_digit_action(self, event):
        if event.char and event.char.isdigit() and len(getattr(self, 'quick_action_buffer', '')) < 4:
            self.quick_action_buffer += event.char
            self._update_quick_label()

    def _on_backspace_action(self, _event):
        if hasattr(self, 'quick_action_buffer') and self.quick_action_buffer:
            self.quick_action_buffer = self.quick_action_buffer[:-1]
            self._update_quick_label()

    def _on_escape_action(self, _event):
        if hasattr(self, 'quick_action_buffer') and self.quick_action_buffer:
            self.quick_action_buffer = ""
            self._update_quick_label()

    def _on_enter_apply_action(self, _event):
        if not hasattr(self, 'quick_action_buffer') or not self.quick_action_buffer:
            return
        try:
            code = int(self.quick_action_buffer)
        except Exception:
            self.quick_action_buffer = ""
            self._update_quick_label()
            return
        # Find action by code field
        chosen = None
        for a in self.actions_cfg:
            if int(a.get("code", -1)) == code:
                chosen = a
                break
        if chosen is not None:
            self.action_var.set(chosen["name"])
        self.quick_action_buffer = ""
        self._update_quick_label()

    def _populate_rosters(self):
        # Fill home roster
        for item in self.home_tv.get_children():
            self.home_tv.delete(item)
        for no in self._get_players_numbers("home"):
            self.home_tv.insert("", tk.END, values=(no, self._get_player_name("home", no)))
        # Fill away roster
        for item in self.away_tv.get_children():
            self.away_tv.delete(item)
        for no in self._get_players_numbers("away"):
            self.away_tv.insert("", tk.END, values=(no, self._get_player_name("away", no)))

    def _on_roster_select(self, which: str):
        tv = self.home_tv if which == "home" else self.away_tv
        sel = tv.selection()
        if not sel:
            return
        no, name = tv.item(sel[0], "values")
        # Switch team if needed
        target_team_name = self.cfg["teams"][which]["name"]
        if self.team_var.get() != target_team_name:
            self.team_var.set(target_team_name)
            self._on_team_changed()
        # Ensure player exists and select it
        self._ensure_player_in_team(target_team_name, str(no))
        self._set_player_selection_by_number(which, str(no))
        self._update_player_name_label()

    def _team_key_from_name(self, team_name: str) -> str:
        return "home" if team_name == self.cfg["teams"]["home"]["name"] else "away"

    def _get_players_numbers(self, team_key: str) -> List[str]:
        return list(self.cfg["teams"][team_key].get("players", []))

    def _get_player_name(self, team_key: str, number: str) -> str:
        return str(self.cfg["teams"][team_key].get("players_names", {}).get(str(number), "")).strip()

    def _display_for(self, team_key: str, number: str) -> str:
        name = self._get_player_name(team_key, number)
        # Keep compact display: number and optional short name (first word)
        if name:
            short = name.split()[0]
            return f"{number} {short}"
        return str(number)

    def _players_display_list(self, team_key: str) -> List[str]:
        return [self._display_for(team_key, num) for num in self._get_players_numbers(team_key)]

    def _parse_display_to_number(self, display: str) -> str:
        return display.strip().split()[0] if display else ""

    def _refresh_player_combobox(self):
        team_key = self._team_key_from_name(self.team_var.get())
        values = self._players_display_list(team_key)
        self.player_cb.config(values=values)
        if values:
            if self.player_var.get() not in values:
                self.player_var.set(values[0])

    def _set_player_selection_by_number(self, team_key: str, number: str):
        disp = self._display_for(team_key, number)
        values = list(self.player_cb.cget("values"))
        if disp not in values:
            values.append(disp)
            self.player_cb.config(values=values)
        self.player_var.set(disp)

    def rename_teams(self):
        dlg = RenameTeamsDialog(self, self.cfg)
        self.wait_window(dlg)
        if getattr(dlg, "result", None):
            self.cfg = dlg.result
            teams = [self.cfg["teams"]["home"]["name"], self.cfg["teams"]["away"]["name"]]
            if self.team_var.get() not in teams:
                self.team_var.set(teams[0])
            # Update roster labels with new team names
            self.home_tv.master.configure(text=f"Home ({self.cfg['teams']['home']['name']})")
            self.away_tv.master.configure(text=f"Away ({self.cfg['teams']['away']['name']})")
            self._on_team_changed()


class RenameTeamsDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, cfg: Dict):
        super().__init__(parent)
        self.title("Rename Teams")
        self.resizable(True, True)
        self.result = None
        self.cfg = cfg

        frame = ttk.Frame(self, padding=8)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Home team name:").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        self.home_var = tk.StringVar(value=cfg["teams"]["home"].get("name", "HOME"))
        ttk.Entry(frame, textvariable=self.home_var, width=24).grid(row=0, column=1, padx=4, pady=4)

        ttk.Label(frame, text="Away team name:").grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        self.away_var = tk.StringVar(value=cfg["teams"]["away"].get("name", "AWAY"))
        ttk.Entry(frame, textvariable=self.away_var, width=24).grid(row=1, column=1, padx=4, pady=4)

        btns = ttk.Frame(frame)
        btns.grid(row=2, column=0, columnspan=2, pady=(8, 0), sticky=tk.E)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Apply", command=self._apply).pack(side=tk.RIGHT, padx=4)

        self.grab_set()
        self.transient(parent)

    def _apply(self):
        home = self.home_var.get().strip() or "HOME"
        away = self.away_var.get().strip() or "AWAY"
        self.cfg["teams"]["home"]["name"] = home
        self.cfg["teams"]["away"]["name"] = away
        self.result = self.cfg
        self.destroy()


class ConfigDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, cfg: Dict):
        super().__init__(parent)
        self.title("Edit Scout Config")
        self.resizable(True, True)
        self.result: Optional[Dict] = None

        # Clone config
        self.cfg = {
            "program": dict(cfg.get("program", {})),
            "project": dict(cfg.get("project", {})),
            "teams": {
                "home": dict(cfg.get("teams", {}).get("home", {})),
                "away": dict(cfg.get("teams", {}).get("away", {})),
            },
            "actions": list(cfg.get("actions", [])),
            "results": list(cfg.get("results", [])),
        }

        frame = ttk.Frame(self, padding=8)
        frame.pack(fill=tk.BOTH, expand=True)

        # Field size
        ttk.Label(frame, text="Field width (m):").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        self.field_w_var = tk.StringVar(value=str(self.cfg["project"].get("field_width_m", 105)))
        ttk.Entry(frame, textvariable=self.field_w_var, width=12).grid(row=0, column=1, padx=4, pady=4)

        ttk.Label(frame, text="Field height (m):").grid(row=0, column=2, sticky=tk.W, padx=4, pady=4)
        self.field_h_var = tk.StringVar(value=str(self.cfg["project"].get("field_height_m", 68)))
        ttk.Entry(frame, textvariable=self.field_h_var, width=12).grid(row=0, column=3, padx=4, pady=4)

        # Home team
        ttk.Label(frame, text="Home team name:").grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        self.home_name_var = tk.StringVar(value=self.cfg["teams"]["home"].get("name", "HOME"))
        ttk.Entry(frame, textvariable=self.home_name_var, width=20).grid(row=1, column=1, padx=4, pady=4)
        ttk.Label(frame, text="Home color (#hex or name):").grid(row=1, column=2, sticky=tk.W, padx=4, pady=4)
        self.home_color_var = tk.StringVar(value=self.cfg["teams"]["home"].get("color", "#1f77b4"))
        ttk.Entry(frame, textvariable=self.home_color_var, width=20).grid(row=1, column=3, padx=4, pady=4)

        ttk.Label(frame, text="Home players (comma-separated):").grid(row=2, column=0, sticky=tk.W, padx=4, pady=4)
        self.home_players_var = tk.StringVar(value=",".join(self.cfg["teams"]["home"].get("players", [])))
        ttk.Entry(frame, textvariable=self.home_players_var, width=60).grid(row=2, column=1, columnspan=3, padx=4, pady=4, sticky=tk.W)

        # After existing players fields
        ttk.Label(frame, text="Home player names (num:name; ...):").grid(row=2, column=4, sticky=tk.W, padx=4, pady=4)
        home_names_map = self.cfg["teams"]["home"].get("players_names", {})
        self.home_names_var = tk.StringVar(value=";".join(f"{k}:{v}" for k, v in home_names_map.items()))
        ttk.Entry(frame, textvariable=self.home_names_var, width=40).grid(row=2, column=5, padx=4, pady=4, sticky=tk.W)

        # Away team
        ttk.Label(frame, text="Away team name:").grid(row=3, column=0, sticky=tk.W, padx=4, pady=4)
        self.away_name_var = tk.StringVar(value=self.cfg["teams"]["away"].get("name", "AWAY"))
        ttk.Entry(frame, textvariable=self.away_name_var, width=20).grid(row=3, column=1, padx=4, pady=4)
        ttk.Label(frame, text="Away color (#hex or name):").grid(row=3, column=2, sticky=tk.W, padx=4, pady=4)
        self.away_color_var = tk.StringVar(value=self.cfg["teams"]["away"].get("color", "#d62728"))
        ttk.Entry(frame, textvariable=self.away_color_var, width=20).grid(row=3, column=3, padx=4, pady=4)

        ttk.Label(frame, text="Away players (comma-separated):").grid(row=4, column=0, sticky=tk.W, padx=4, pady=4)
        self.away_players_var = tk.StringVar(value=",".join(self.cfg["teams"]["away"].get("players", [])))
        ttk.Entry(frame, textvariable=self.away_players_var, width=60).grid(row=4, column=1, columnspan=3, padx=4, pady=4, sticky=tk.W)

        # After existing players fields
        ttk.Label(frame, text="Away player names (num:name; ...):").grid(row=4, column=4, sticky=tk.W, padx=4, pady=4)
        away_names_map = self.cfg["teams"]["away"].get("players_names", {})
        self.away_names_var = tk.StringVar(value=";".join(f"{k}:{v}" for k, v in away_names_map.items()))
        ttk.Entry(frame, textvariable=self.away_names_var, width=40).grid(row=4, column=5, padx=4, pady=4, sticky=tk.W)

        # Actions and results
        _acts = self.cfg.get("actions", [])
        if _acts and isinstance(_acts[0], dict):
            act_names_value = ",".join(a.get("name", "") for a in _acts)
            self._original_actions_struct = _acts
        else:
            act_names_value = ",".join(str(a) for a in _acts)
            self._original_actions_struct = None
        ttk.Label(frame, text="Actions (names, comma-separated):").grid(row=5, column=0, sticky=tk.W, padx=4, pady=4)
        self.actions_var = tk.StringVar(value=act_names_value)
        ttk.Entry(frame, textvariable=self.actions_var, width=60).grid(row=5, column=1, columnspan=3, padx=4, pady=4, sticky=tk.W)

        ttk.Label(frame, text="Results (comma-separated):").grid(row=6, column=0, sticky=tk.W, padx=4, pady=4)
        self.results_var = tk.StringVar(value=",".join(self.cfg.get("results", [])))
        ttk.Entry(frame, textvariable=self.results_var, width=60).grid(row=6, column=1, columnspan=3, padx=4, pady=4, sticky=tk.W)

        # Buttons
        btns = ttk.Frame(frame)
        btns.grid(row=7, column=0, columnspan=4, pady=(8, 0), sticky=tk.E)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Apply", command=self._apply).pack(side=tk.RIGHT, padx=4)

        self.grab_set()
        self.transient(parent)

    def _apply(self):
        try:
            fw = float(self.field_w_var.get())
            fh = float(self.field_h_var.get())
        except ValueError:
            messagebox.showerror("Error", "Field size must be numeric.")
            return
        home_players = [p.strip() for p in self.home_players_var.get().split(",") if p.strip()]
        away_players = [p.strip() for p in self.away_players_var.get().split(",") if p.strip()]
        actions = [a.strip() for a in self.actions_var.get().split(",") if a.strip()]
        results = [r.strip() for r in self.results_var.get().split(",") if r.strip()]

        # Parse names mappings
        def parse_names(s: str) -> Dict[str, str]:
            out: Dict[str, str] = {}
            for pair in [p.strip() for p in s.split(";") if p.strip()]:
                if ":" in pair:
                    k, v = pair.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    if k:
                        out[k] = v
            return out

        home_names = parse_names(self.home_names_var.get())
        away_names = parse_names(self.away_names_var.get())

        # Ensure players include keys from names
        for k in home_names:
            if k not in home_players:
                home_players.append(k)
        for k in away_names:
            if k not in away_players:
                away_players.append(k)

        self.cfg["project"]["field_width_m"] = fw
        self.cfg["project"]["field_height_m"] = fh
        self.cfg["teams"]["home"]["name"] = self.home_name_var.get().strip() or "HOME"
        self.cfg["teams"]["home"]["players"] = home_players or [str(n) for n in range(1, 24)]
        self.cfg["teams"]["home"]["players_names"] = home_names
        self.cfg["teams"]["away"]["name"] = self.away_name_var.get().strip() or "AWAY"
        self.cfg["teams"]["away"]["players"] = away_players or [str(n) for n in range(1, 24)]
        self.cfg["teams"]["away"]["players_names"] = away_names
        self.cfg["actions"] = actions or _default_config()["actions"]
        self.cfg["results"] = results or _default_config()["results"]

        self.result = self.cfg
        self.destroy()


def _locate_or_init_config() -> Dict:
    """Find the TOML config alongside the script, or under vaila/models, else create template there."""
    here = Path(__file__).resolve()
    script_dir = here.parent
    models_dir = script_dir / "models"
    script_cfg = script_dir / DEFAULT_CFG_FILENAME
    models_cfg = models_dir / DEFAULT_CFG_FILENAME

    candidate_paths = [script_cfg, models_cfg]
    for cand in candidate_paths:
        cfg = read_toml_config(cand)
        if cfg is not None:
            print(f"Using config: {cand}")
            return cfg

    # Not found → create a default template in vaila/models
    cfg = _default_config()
    models_dir.mkdir(parents=True, exist_ok=True)
    if write_toml_template(models_cfg, cfg):
        print(f"Default config template created at: {models_cfg}")
    else:
        print("[yellow]Proceeding with in-memory default config (not saved).[/]")
    print(f"Using config: {models_cfg}")
    return cfg


def run_scout():
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Running Scout vailá")
    print("================================================")
    
    cfg = _locate_or_init_config()
    app = ScoutApp(cfg)
    app.mainloop()


if __name__ == "__main__":
    run_scout()


