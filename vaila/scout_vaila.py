"""
Project: vailá Multimodal Toolbox
Script: scout_vaila.py - Integrated Sports Scouting (Annotation + Analysis)

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 12 August 2025
Update Date: 12 August 2025
Version: 0.1.0

Description:
    Integrated GUI to annotate sports events on a virtual soccer field and generate
    quick analyses (e.g., heatmaps). Inspired by manual scouting tools and designed
    to fit the vailá project style. No external field image is required; the field is
    drawn to scale using standard FIFA dimensions (105m x 68m).

Usage:
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
from typing import Dict, List, Optional

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
                "players": [str(n) for n in range(1, 24)],
                "players_names": {},  # "10": "Player Name"
            },
            "away": {
                "name": "AWAY",
                "players": [str(n) for n in range(1, 24)],
                "players_names": {},
            },
        },
        "actions": [
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
        ],
        "results": ["success", "fail", "neutral"],
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
    timestamp_s: float
    team: str
    player: str
    action: str
    result: str
    pos_x_m: float
    pos_y_m: float
    player_name: str = ""

    def to_row(self) -> Dict[str, object]:
        return asdict(self)


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

    # Penalty arcs
    for cx in (11, field_w - 11):
        arc = patches.Arc((cx, field_h / 2.0), 2 * 9.15, 2 * 9.15, theta1=310, theta2=50, edgecolor=line_color, linewidth=lw)
        ax.add_patch(arc)


# --- Main Application -----------------------------------------------------------
class ScoutApp(tk.Tk):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.title("vailá - Scout Integrated")
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
        view_menu.add_command(label="Show Heatmap	H", command=self.draw_heatmap)
        menubar.add_cascade(label="View", menu=view_menu)

        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Load Config	Ctrl+L", command=self.load_config)
        tools_menu.add_command(label="Save Config	Ctrl+Shift+S", command=self.save_config)
        tools_menu.add_command(label="Edit Config	Ctrl+E", command=self.edit_config)
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

        # --- Controls (top of right frame)
        controls = ttk.Frame(right_frame)
        controls.pack(side=tk.TOP, fill=tk.X)

        # Team selector
        ttk.Label(controls, text="Team:").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        teams = [cfg["teams"]["home"]["name"], cfg["teams"]["away"]["name"]]
        self.team_var = tk.StringVar(value=teams[0])
        self.team_cb = ttk.Combobox(controls, textvariable=self.team_var, values=teams, width=16, state="readonly")
        self.team_cb.grid(row=0, column=1, padx=4, pady=4)

        # Player selector (uses display with optional names)
        ttk.Label(controls, text="Player:").grid(row=0, column=2, sticky=tk.W, padx=4, pady=4)
        self.player_var = tk.StringVar(value="")
        self.player_cb = ttk.Combobox(controls, textvariable=self.player_var, values=[], width=24, state="readonly")
        self.player_cb.grid(row=0, column=3, padx=4, pady=4)

        # Action selector
        ttk.Label(controls, text="Action:").grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        self.action_var = tk.StringVar(value=cfg["actions"][0])
        self.action_cb = ttk.Combobox(controls, textvariable=self.action_var, values=cfg["actions"], width=16)
        self.action_cb.grid(row=1, column=1, padx=4, pady=4)

        # Result selector
        ttk.Label(controls, text="Result:").grid(row=1, column=2, sticky=tk.W, padx=4, pady=4)
        self.result_var = tk.StringVar(value=cfg["results"][0])
        self.result_cb = ttk.Combobox(controls, textvariable=self.result_var, values=cfg["results"], width=12)
        self.result_cb.grid(row=1, column=3, padx=4, pady=4)

        # Buttons
        btns = ttk.Frame(controls)
        btns.grid(row=2, column=0, columnspan=4, sticky=tk.W, padx=4, pady=(8, 4))
        ttk.Button(btns, text="Save CSV", command=self.save_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Load CSV", command=self.load_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Clear Events", command=self.clear_events).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Show Heatmap", command=self.draw_heatmap).pack(side=tk.LEFT, padx=12)
        ttk.Button(btns, text="Reset Time", command=self.reset_timer).pack(side=tk.LEFT, padx=12)

        # Config buttons
        cfg_btns = ttk.Frame(controls)
        cfg_btns.grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=4, pady=(4, 8))
        ttk.Button(cfg_btns, text="Load Config", command=self.load_config).pack(side=tk.LEFT, padx=4)
        ttk.Button(cfg_btns, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=4)
        ttk.Button(cfg_btns, text="Edit Config", command=self.edit_config).pack(side=tk.LEFT, padx=4)

        # --- Field figure (left)
        self.field_fig, self.field_ax = plt.subplots(figsize=(6.5, 4.2))
        self.field_canvas = FigureCanvasTkAgg(self.field_fig, master=left_frame)
        self.field_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._redraw_field()
        self.field_canvas.mpl_connect("button_press_event", self._on_field_click)

        # --- Modes for player numbering
        self.auto_number_players = False
        self.unique_player_per_click = False
        self.next_player_number_by_team: Dict[str, int] = self._init_next_player_numbers()
        modes = ttk.Frame(controls)
        modes.grid(row=4, column=0, columnspan=4, sticky=tk.W, padx=4, pady=(0, 4))
        self.chk_auto_var = tk.BooleanVar(value=self.auto_number_players)
        self.chk_unique_var = tk.BooleanVar(value=self.unique_player_per_click)
        ttk.Checkbutton(modes, text="Auto-number players (N)", variable=self.chk_auto_var, command=self._toggle_auto).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(modes, text="Unique player per click (U)", variable=self.chk_unique_var, command=self._toggle_unique).pack(side=tk.LEFT, padx=6)

        # Sync players list on team change
        self.team_cb.bind("<<ComboboxSelected>>", self._on_team_changed)

        # Hotkeys (simple)
        self.bind_all("<Control-s>", lambda e: self.save_csv())
        self.bind_all("<Control-o>", lambda e: self.load_csv())
        self.bind_all("<Control-k>", lambda e: self.clear_events())
        self.bind_all("h", lambda e: self.draw_heatmap())
        self.bind_all("H", lambda e: self.draw_heatmap())
        self.bind_all("r", lambda e: self.reset_timer())
        self.bind_all("R", lambda e: self.reset_timer())
        self.bind_all("?", lambda e: self.open_help())
        self.bind_all("<Control-e>", lambda e: self.edit_config())
        self.bind_all("<Control-l>", lambda e: self.load_config())
        self.bind_all("<Control-S>", lambda e: self.save_config())
        # Additional hotkeys
        self.bind_all("n", lambda e: self._toggle_auto())
        self.bind_all("N", lambda e: self._toggle_auto())
        self.bind_all("u", lambda e: self._toggle_unique())
        self.bind_all("U", lambda e: self._toggle_unique())
        self.bind_all("t", lambda e: self._toggle_team())
        self.bind_all("T", lambda e: self._toggle_team())
        # Use comma/period to avoid matplotlib key conflicts with 'k'
        self.bind_all(",", lambda e: self._on_prev_player())
        self.bind_all(".", lambda e: self._on_next_player())
        # Arrow keys for player navigation
        self.bind_all("<Right>", lambda e: self._on_next_player())
        self.bind_all("<Left>", lambda e: self._on_prev_player())
        self.bind_all("<Control-t>", lambda e: self.rename_teams())

        # Quick player numeric buffer UI and keybinds
        self.quick_player_buffer: str = ""
        self.quick_label = ttk.Label(controls, text="Quick player input: (none)")
        self.quick_label.grid(row=5, column=0, columnspan=4, sticky=tk.W, padx=4, pady=(2, 2))
        for _d in list("0123456789"):
            self.bind_all(_d, self._on_digit)
        self.bind_all("<BackSpace>", self._on_backspace)
        self.bind_all("<Escape>", self._on_escape)
        self.bind_all("<Return>", self._on_enter_apply_player)
        self.bind_all("+", self._on_plus_inc)
        self.bind_all("-", self._on_minus_dec)

        # Populate initial players list based on current team
        self._on_team_changed()

    # --- UI helpers
    def _on_team_changed(self, _evt=None):
        team_name = self.team_var.get()
        team_key = self._team_key_from_name(team_name)
        self._refresh_player_combobox()

    def _redraw_field(self):
        draw_soccer_field(self.field_ax, self.field_w, self.field_h)
        self.field_fig.tight_layout()
        self.field_canvas.draw()

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

        timestamp = float(time.time() - self.start_time)
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
        elif self.quick_player_buffer:
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

        ev = ScoutEvent(
            timestamp_s=round(timestamp, 2),
            team=team_name,
            player=self._parse_display_to_number(self.player_var.get()),
            action=self.action_var.get(),
            result=self.result_var.get(),
            pos_x_m=round(x_m, 2),
            pos_y_m=round(y_m, 2),
            player_name=self._get_player_name(self._team_key_from_name(team_name),
                                      self._parse_display_to_number(self.player_var.get())),
        )
        self.events.append(ev)

        # Visual marker
        self.field_ax.scatter([x_m], [y_m], s=40, c="yellow", edgecolors="black", zorder=5)
        self.field_canvas.draw()

    # --- Actions
    def reset_timer(self):
        self.start_time = time.time()
        messagebox.showinfo("Timer", "Timer reset to 0s.")

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
            self.events = [
                ScoutEvent(
                    timestamp_s=float(row["timestamp_s"]),
                    team=str(row["team"]),
                    player=str(row["player"]),
                    action=str(row["action"]),
                    result=str(row["result"]),
                    pos_x_m=float(row["pos_x_m"]),
                    pos_y_m=float(row["pos_y_m"]),
                    player_name=str(row["player_name"]) if has_name and not pd.isna(row["player_name"]) else "",
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
                xs = [e.pos_x_m for e in self.events]
                ys = [e.pos_y_m for e in self.events]
                self.field_ax.scatter(xs, ys, s=40, c="yellow", edgecolors="black", zorder=5)
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
        self.cfg = cfg
        self.cfg_path = path
        # Update UI based on new config
        self.field_w = float(cfg["project"]["field_width_m"])
        self.field_h = float(cfg["project"]["field_height_m"])
        teams = [cfg["teams"]["home"]["name"], cfg["teams"]["away"]["name"]]
        self.team_cb.config(values=teams)
        self.team_var.set(teams[0])
        self._on_team_changed()
        self._redraw_field()
        self.next_player_number_by_team = self._init_next_player_numbers()

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
        win.resizable(False, False)
        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Keyboard Shortcuts", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))

        shortcuts = [
            ("Ctrl+S", "Save CSV"),
            ("Ctrl+O", "Load CSV"),
            ("Ctrl+K", "Clear events"),
            ("H", "Show heatmap"),
            ("R", "Reset timer"),
            ("?", "Open help"),
            ("T", "Toggle current team (home/away)"),
            (", / .", "Previous / Next player"),
            ("← / →", "Previous / Next player"),
            ("+ / -", "Increment / Decrement current player number"),
            ("N", "Toggle auto-number players"),
            ("U", "Toggle unique player per click"),
            ("Ctrl+E", "Edit config"),
            ("Ctrl+L", "Load config"),
            ("Ctrl+Shift+S", "Save config"),
            ("Ctrl+T", "Rename teams"),
            ("Digits 0–9", "Buffer player number; Enter apply; Backspace edit; Esc clear"),
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
        if event.char and event.char.isdigit() and len(self.quick_player_buffer) < 3:
            self.quick_player_buffer += event.char
            self._update_quick_label()

    def _on_backspace(self, _event):
        if self.quick_player_buffer:
            self.quick_player_buffer = self.quick_player_buffer[:-1]
            self._update_quick_label()

    def _on_escape(self, _event):
        if self.quick_player_buffer:
            self.quick_player_buffer = ""
            self._update_quick_label()

    def _on_enter_apply_player(self, _event):
        if not self.quick_player_buffer:
            return
        team_name = self.team_var.get()
        player_str = self.quick_player_buffer
        self._ensure_player_in_team(team_name, player_str)
        vals = list(self.player_cb.cget("values"))
        if player_str not in vals:
            vals.append(player_str)
            self.player_cb.config(values=vals)
        self.player_var.set(player_str)
        self.quick_player_buffer = ""
        self._update_quick_label()

    def _on_plus_inc(self, _event):
        try:
            current_num = self._parse_display_to_number(self.player_var.get())
            current = int(current_num) if current_num.isdigit() else 0
            new_val = str(current + 1)
            self._ensure_player_in_team(self.team_var.get(), new_val)
            self._set_player_selection_by_number(self._team_key_from_name(self.team_var.get()), new_val)
            self._update_quick_label()
        except Exception:
            pass

    def _on_minus_dec(self, _event):
        try:
            current_num = self._parse_display_to_number(self.player_var.get())
            current = int(current_num) if current_num.isdigit() else 1
            new_val = str(max(1, current - 1))
            self._ensure_player_in_team(self.team_var.get(), new_val)
            self._set_player_selection_by_number(self._team_key_from_name(self.team_var.get()), new_val)
            self._update_quick_label()
        except Exception:
            pass

    def _on_next_player(self, _event=None):
        values = list(self.player_cb.cget("values"))
        if not values:
            return
        try:
            idx = values.index(self.player_var.get())
        except ValueError:
            idx = -1
        self.player_var.set(values[(idx + 1) % len(values)])

    def _on_prev_player(self, _event=None):
        values = list(self.player_cb.cget("values"))
        if not values:
            return
        try:
            idx = values.index(self.player_var.get())
        except ValueError:
            idx = 0
        self.player_var.set(values[(idx - 1) % len(values)])

    def _update_quick_label(self):
        buf = self.quick_player_buffer if self.quick_player_buffer else "(none)"
        self.quick_label.config(text=f"Quick player input: {buf}")

    def _team_key_from_name(self, team_name: str) -> str:
        return "home" if team_name == self.cfg["teams"]["home"]["name"] else "away"

    def _get_players_numbers(self, team_key: str) -> List[str]:
        return list(self.cfg["teams"][team_key].get("players", []))

    def _get_player_name(self, team_key: str, number: str) -> str:
        return str(self.cfg["teams"][team_key].get("players_names", {}).get(str(number), "")).strip()

    def _display_for(self, team_key: str, number: str) -> str:
        name = self._get_player_name(team_key, number)
        return f"{number} {name}" if name else str(number)

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
            self.team_cb.config(values=teams)
            if self.team_var.get() not in teams:
                self.team_var.set(teams[0])
            self._on_team_changed()


class RenameTeamsDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, cfg: Dict):
        super().__init__(parent)
        self.title("Rename Teams")
        self.resizable(False, False)
        self.result = None
        self.cfg = cfg

        frame = ttk.Frame(self, padding=8)
        frame.pack(fill=tk.BOTH, expand=True)

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
        self.resizable(False, False)
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

        ttk.Label(frame, text="Away players (comma-separated):").grid(row=4, column=0, sticky=tk.W, padx=4, pady=4)
        self.away_players_var = tk.StringVar(value=",".join(self.cfg["teams"]["away"].get("players", [])))
        ttk.Entry(frame, textvariable=self.away_players_var, width=60).grid(row=4, column=1, columnspan=3, padx=4, pady=4, sticky=tk.W)

        # After existing players fields
        ttk.Label(frame, text="Away player names (num:name; ...):").grid(row=4, column=4, sticky=tk.W, padx=4, pady=4)
        away_names_map = self.cfg["teams"]["away"].get("players_names", {})
        self.away_names_var = tk.StringVar(value=";".join(f"{k}:{v}" for k, v in away_names_map.items()))
        ttk.Entry(frame, textvariable=self.away_names_var, width=40).grid(row=4, column=5, padx=4, pady=4, sticky=tk.W)

        # Actions and results
        ttk.Label(frame, text="Actions (comma-separated):").grid(row=5, column=0, sticky=tk.W, padx=4, pady=4)
        self.actions_var = tk.StringVar(value=",".join(self.cfg.get("actions", [])))
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
    """Find a TOML config near repository root or create one in user's home."""
    # Try repo root next to this file
    here = Path(__file__).resolve()
    repo_root = here.parents[1] if len(here.parents) >= 2 else here.parent
    candidate_paths = [
        repo_root / DEFAULT_CFG_FILENAME,
        Path.cwd() / DEFAULT_CFG_FILENAME,
        Path.home() / ".vaila" / DEFAULT_CFG_FILENAME,
    ]
    for cand in candidate_paths:
        cfg = read_toml_config(cand)
        if cfg is not None:
            print(f"[green]Using config[/]: {cand}")
            return cfg

    # Not found → create a default one under ~/.vaila
    cfg = _default_config()
    default_path = Path.home() / ".vaila" / DEFAULT_CFG_FILENAME
    if write_toml_config(default_path, cfg):
        print(f"[yellow]Default config created at[/]: {default_path}")
    else:
        print("[yellow]Proceeding with in-memory default config (not saved).[/]")
    return cfg


def run_scout():
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    cfg = _locate_or_init_config()
    app = ScoutApp(cfg)
    app.mainloop()


if __name__ == "__main__":
    run_scout()


