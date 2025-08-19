"""
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
2) TOML config aligned to vailá_scout, prefilled with players and actions read from Skout.

This script has no external dependencies beyond the Python stdlib + pandas.

Explanation
-----------
Skout.exe ASCII export is a text file with three sections:
- Player names
- Action codes
- Events
Each section is a block of lines, delimited by curly braces.
The first line of each section is the section name.
The second line of each section is the number of lines in the section.
The remaining lines are the section content.
The section content is a list of lines, each line is a player name, action code, and event result.
The player name is a string, the action code is an integer, and the event result is a string.
The event result is a string, either 'T' or 'F'.
The event result is 'T' if the event was successful, and 'F' if the event was not successful.
The event result is 'T' if the event was successful, and 'F' if the event was not successful.


License: GPL-3.0-or-later
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

# --------------------------------- Parsing ------------------------------------

def _strip_comments(line: str) -> str:
    return line.strip()


def _parse_section(lines: List[str], start_index: int) -> Tuple[str, int]:
    """Return (section_name, index_after_open_brace)."""
    for i in range(start_index, len(lines)):
        line = _strip_comments(lines[i])
        if not line or line.startswith(";"):
            continue
        if line.endswith("{"):
            name = line.split("{")[0].strip()
            return name, i + 1
    raise ValueError("No section header with '{' found")


def _read_block(lines: List[str], start_idx: int) -> Tuple[List[str], int]:
    block: List[str] = []
    i = start_idx
    while i < len(lines):
        ln = _strip_comments(lines[i])
        if ln.startswith("}"):
            return block, i + 1
        if ln and not ln.startswith(";"):
            block.append(ln)
        i += 1
    raise ValueError("Missing closing '}' for section")


def parse_players(block: List[str]) -> Dict[int, str]:
    players: Dict[int, str] = {}
    pat = re.compile(r'^\s*(\d+)\s*,\s*"(.*)"\s*$')
    for ln in block:
        m = pat.match(ln)
        if m:
            pid = int(m.group(1))
            name = m.group(2)
            players[pid] = name
    return players


def parse_actions(block: List[str]) -> Dict[int, str]:
    actions: Dict[int, str] = {}
    pat = re.compile(r'^\s*(\d+)\s*,\s*"(.*)"\s*$')
    for ln in block:
        m = pat.match(ln)
        if m:
            code = int(m.group(1))
            name = m.group(2)
            actions[code] = name
    return actions


@dataclass
class SkoutRecord:
    period: int
    minute: int
    second: int
    x: float
    y: float
    player_id: int
    action_code: int
    result_flag: str  # 'T' or 'F'


def parse_events(block: List[str]) -> List[SkoutRecord]:
    events: List[SkoutRecord] = []
    for ln in block:
        parts = [p.strip() for p in ln.split(',')]
        if len(parts) < 8:
            continue
        try:
            events.append(SkoutRecord(
                int(parts[0]), int(parts[1]), int(parts[2]),
                float(parts[3]), float(parts[4]), int(parts[5]), int(parts[6]),
                parts[7].upper()[:1] if parts[7] else 'T'
            ))
        except Exception:
            continue
    return events


def load_skout_ascii(path: Path) -> Tuple[Dict[int, str], Dict[int, str], List[SkoutRecord]]:
    lines = path.read_text(encoding="latin-1", errors="replace").splitlines()
    idx = 0
    players: Dict[int, str] = {}
    actions: Dict[int, str] = {}
    events: List[SkoutRecord] = []

    seen = set()
    while idx < len(lines) and len(seen) < 3:
        try:
            name, idx_after = _parse_section(lines, idx)
        except ValueError:
            break
        block, next_idx = _read_block(lines, idx_after)
        if name.lower().startswith('player'):
            players = parse_players(block); seen.add('players')
        elif name.lower().startswith('action'):
            actions = parse_actions(block); seen.add('actions')
        elif name.lower().startswith('event'):
            events = parse_events(block); seen.add('events')
        idx = next_idx
    return players, actions, events


# ------------------------------ CSV conversion --------------------------------

def convert_to_vaila_csv(
    players: Dict[int, str],
    actions: Dict[int, str],
    events: List[SkoutRecord],
    *,
    team_name: str = "HOME",
    field_width_m: float = 105.0,
    field_height_m: float = 68.0,
    grid_width: Optional[float] = None,
    grid_height: Optional[float] = None,
    time_mode: str = "infer",  # infer | sequence | zero
    dt: float = 1.0
) -> pd.DataFrame:
    max_x = max((e.x for e in events), default=1.0)
    max_y = max((e.y for e in events), default=1.0)
    gx = float(grid_width) if grid_width else max(1.0, max_x)
    gy = float(grid_height) if grid_height else max(1.0, max_y)

    rows = []
    seq_time = 0.0
    for ev in events:
        # time
        if time_mode == 'zero':
            t_s = 0.0
        elif time_mode == 'sequence':
            t_s = seq_time; seq_time += dt
        else:  # infer
            if (ev.minute, ev.second) != (0, 0):
                t_s = (max(ev.period, 1) - 1) * 45 * 60 + ev.minute * 60 + ev.second
            else:
                t_s = seq_time; seq_time += dt
        # position scaling
        # Skout uses top-left origin (Y down), vailá uses bottom-left origin (Y up)
        pos_x_m = (ev.x / gx) * field_width_m
        pos_y_m = ((gy - ev.y) / gy) * field_height_m  # Invert Y coordinate
        # lookups
        player_name = players.get(ev.player_id, f"#{ev.player_id}")
        action_name = actions.get(ev.action_code, f"code_{ev.action_code}")
        result = 'success' if ev.result_flag == 'T' else 'fail'
        rows.append({
            'timestamp_s': float(t_s),
            'team': team_name,
            'player_name': player_name,
            'player': str(ev.player_id),
            'action': action_name,
            'action_code': int(ev.action_code),
            'result': result,
            'pos_x_m': float(pos_x_m),
            'pos_y_m': float(pos_y_m),
        })

    return pd.DataFrame(rows, columns=[
        'timestamp_s','team','player_name','player','action','action_code','result','pos_x_m','pos_y_m'
    ])


# ------------------------------ TOML emission ---------------------------------

def _toml_escape(s: str) -> str:
    return s.replace('"', '\\"')


def build_config_toml(
    *,
    players: Dict[int, str],
    actions: Dict[int, str],
    team_home: str = "HOME",
    field_width_m: float = 105.0,
    field_height_m: float = 68.0,
    input_glob: str = "*.csv",
    project_name: str = "vaila_scout"
) -> str:
    symbol_map = {
        "Passe":"o", "Cruzamento":"+","Drible":"D","Desarme":"x","Finalização":"*",
        "Falta":"s","Recepção":"P","Gol":"^","Condução":"s"
    }
    color_map = {
        "Passe":"#FFD700","Cruzamento":"#1f77b4","Drible":"#17becf","Desarme":"#d62728","Finalização":"#FF4500",
        "Falta":"#7f7f7f","Recepção":"#8c564b","Gol":"#2ca02c","Condução":"#9467bd"
    }

    player_ids = [str(x) for x in sorted(players.keys())]

    lines: List[str] = []
    lines += [
        '# vailá Scout configuration (TOML) — generated by skout_bundle.py',
        '',
        '[program]',
        'name = "vaila_scout"',
        'version = "0.2.0"',
        'author = "Paulo Roberto Pereira Santiago"',
        'email = "paulosantiago@usp.br"',
        'description = "Integrated scouting (annotation + analysis) for soccer."',
        'repository = "https://github.com/vaila-multimodaltoolbox/vaila"',
        'license = "GPL-3.0-or-later"',
        '',
        '[project]',
        f'name = "{project_name}"',
        'sport = "soccer"',
        'field_standard = "FIFA"',
        'field_units = "meters"',
        f'field_width_m = {field_width_m}',
        f'field_height_m = {field_height_m}',
        '',
        '[io]',
        f'input_glob = ["{input_glob}"]',
        'output_dir = "reports"',
        'encoding = "utf-8"',
        'delimiter = ","',
        'decimal = "."',
        '',
        '[columns]',
        'timestamp = "timestamp_s"',
        'team = "team"',
        'player_id = "player"',
        'player_name = "player_name"',
        'action = "action"',
        'action_code = "action_code"',
        'result = "result"',
        'x = "pos_x_m"',
        'y = "pos_y_m"',
        '',
        '[conversion.skout]',
        'team = "HOME"',
        'time_mode = "infer"',
        'dt = 1.0',
        f'field_width_m = {field_width_m}',
        f'field_height_m = {field_height_m}',
        '# grid_width = 320  # uncomment if known',
        '# grid_height = 210 # uncomment if known',
        '',
        '[teams.home]',
        f'name = "{team_home}"',
        'color = "#1f77b4"',
        'players = [' + ', '.join('"%s"' % p for p in player_ids) + ']',
        '',
        '[teams.home.players_names]'
    ]

    for pid in player_ids:
        lines.append(f'"{pid}" = "{_toml_escape(players[int(pid)])}"')

    lines += [
        '',
        '[teams.away]',
        'name = "AWAY"',
        'color = "#d62728"',
        'players = ["1","2","3","4","5","6","7","8","9","10","11"]',
        '',
        '[drawing]',
        'player_circle_radius_m = 0.6',
        'player_edge_color = "black"',
        'player_number_color = "white"',
        'player_number_size = 8',
        'action_symbol_size = 90.0',
        'show_player_name = true',
        'player_name_size = 8',
        'results = ["success", "fail", "neutral"]',
        ''
    ]

    for code, name in sorted(actions.items()):
        sym = symbol_map.get(name, 'o')
        col = color_map.get(name, '#1f77b4')
        lines += [
            '[[actions]]',
            f'name = "{_toml_escape(name)}"',
            f'code = {code}',
            f'symbol = "{sym}"',
            f'color = "{col}"',
            ''
        ]

    return "\n".join(lines)


# ---------------------------------- CLI ---------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Skout bundle: create CSV + TOML from Skout ASCII export")
    ap.add_argument('input', type=str, help='Path to Skout ASCII export (e.g., braxing.txt)')
    ap.add_argument('--csv-out', type=str, default='', help='Output CSV path (default: input basename + .csv)')
    ap.add_argument('--toml-out', type=str, default='', help='Output TOML path (default: input basename + _config.toml)')

    ap.add_argument('--team', type=str, default='HOME', help='Team name to write in CSV and TOML [HOME]')
    ap.add_argument('--time-mode', type=str, default='infer', choices=['infer','sequence','zero'],
                    help='Timestamp mode: infer from period/min/sec, or sequential, or all-zero [infer]')
    ap.add_argument('--dt', type=float, default=1.0, help='Seconds per event when using sequence/implicit times [1.0]')

    ap.add_argument('--field-width', type=float, default=105.0, help='Field width in meters [105]')
    ap.add_argument('--field-height', type=float, default=68.0, help='Field height in meters [68]')
    ap.add_argument('--grid-width', type=float, default=None, help='Skout grid width (pixels/units); default: observed max X')
    ap.add_argument('--grid-height', type=float, default=None, help='Skout grid height (pixels/units); default: observed max Y')

    args = ap.parse_args()

    in_path = Path(args.input)
    csv_out = Path(args.csv_out) if args.csv_out else in_path.with_suffix('.csv')
    toml_out = Path(args.toml_out) if args.toml_out else in_path.with_name(in_path.stem + '_config.toml')

    players, actions, events = load_skout_ascii(in_path)

    # CSV
    df = convert_to_vaila_csv(
        players, actions, events,
        team_name=args.team,
        field_width_m=args.field_width,
        field_height_m=args.field_height,
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        time_mode=args.time_mode,
        dt=args.dt,
    )
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False, quoting=csv.QUOTE_MINIMAL, encoding='utf-8')

    # TOML
    toml_text = build_config_toml(
        players=players,
        actions=actions,
        team_home=args.team,
        field_width_m=args.field_width,
        field_height_m=args.field_height,
        input_glob='*.csv',
        project_name='vaila_scout'
    )
    toml_out.parent.mkdir(parents=True, exist_ok=True)
    toml_out.write_text(toml_text, encoding='utf-8')

    print(f"Saved CSV : {csv_out}")
    print(f"Saved TOML: {toml_out}")


if __name__ == '__main__':
    main()
