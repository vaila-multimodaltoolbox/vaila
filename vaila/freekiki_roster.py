"""
================================================================================
Script: freekiki_roster.py - the user labels the people (teams, keeper, kicker, referees)
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.19
Created: 23 November 2026
Update Date: 23 November 2026

Description:
    The automatic team split of :mod:`freekiki_teams` (torso colour in four HSV bins) fails when
    the shirts are not green, blue, orange or yellow (a red-black team against a white one, a
    light-blue goalkeeper read as a referee). The roster moves that decision to the person who
    watched the video: after the 2D stage (field detection + SAM 3 + SAM 3D Body) and before the
    3D stage, a contact sheet shows every SAM id and the user says who is who.

    * ``write_roster_sheet`` writes ``roster_ids.png`` (per id three crops: first, middle and last
      third of its track, box drawn, frame number on each crop, so a SAM track that swaps people is
      visible), ``roster_context_<fff>.png`` (full frames with every box and id), one tile per id
      in ``roster_crops/`` (used by the dialog) and ``roster.csv``, the template to fill. A second
      call refreshes the sheet and the frame columns but KEEPS every label already typed.
    * ``read_roster`` validates the csv (:class:`RosterError` names the sam_id and the rule) and
      returns a :class:`Roster`; ``write_roster`` saves it.
    * ``roster_dialog`` is the Tkinter editor (Toplevel when a parent exists, own root otherwise).

    Data format, ``<kick_dir>/roster/<stem>/roster.csv``, columns
    ``sam_id,team,role,name,first_frame,last_frame,frames,note``:

    * ``team``: blank (not labelled), ``A``, ``B``, ``REF`` or ``OUT`` (not a person on the pitch or
      not used: ball, flag, crowd, bench, duplicate). Read case-insensitively, written upper-case.
    * ``role``: blank, ``player``, ``keeper``, ``kicker``, ``referee_main``, ``referee_aux`` (read
      case-insensitively, written lower-case). Blank with A/B means player; blank with REF means a
      referee ranked by the distance to the ball. ``keeper``/``kicker`` need team A or B,
      ``referee_*`` need team REF.
    * ``name``: free text shown in the labels; ``note`` free text; the frame columns are written
      by the sheet and ignored on read.

    Once the roster is *filled* (at least one non-empty team) an id left blank stays unlabelled and
    is never guessed from the colour.

Usage:
    python -m vaila.freekiki_roster sheet --predictions P.json.gz --video V.mp4 --out DIR
    python -m vaila.freekiki_roster check ROSTER_CSV
    python -m vaila.freekiki_roster gui ROSTER_CSV [--sheet-dir DIR]
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .freekiki_teams import read_predictions
except ImportError:
    from freekiki_teams import read_predictions  # ty: ignore[unresolved-import]

TEAM_CODES = ("A", "B", "REF", "OUT")
ROLE_CODES = ("player", "keeper", "kicker", "referee_main", "referee_aux")
ROSTER_FIELDS = ["sam_id", "team", "role", "name", "first_frame", "last_frame", "frames", "note"]
ROSTER_DIR = "roster"  # <kick_dir>/roster/<stem>/
ROSTER_CSV = "roster.csv"
SHEET_PNG = "roster_ids.png"
CROPS_DIR = "roster_crops"  # one tile per id, read by the dialog

# Roles each team accepts ("" = blank).
ROLES_BY_TEAM: dict[str, tuple[str, ...]] = {
    "": ("",),
    "A": ("", "player", "keeper", "kicker"),
    "B": ("", "player", "keeper", "kicker"),
    "REF": ("", "referee_main", "referee_aux"),
    "OUT": ("",),
}
MAX_PER_TEAM = 11  # keeper included
MAX_REFEREES = 3
OVERLAP_MAX = 0.15  # a "clean" crop overlaps another box by less than 15 % of its own area
DEFAULT_CONTEXT_FRAMES = 5

# Contact-sheet geometry (pixels).
CELL_H, CELL_W, LABEL_H = 230, 120, 46
_BG = 30
_FRAME_BORDER = 80


class RosterError(ValueError):
    """The roster csv breaks a rule; the message names the sam_id and the rule."""


@dataclass(frozen=True)
class RosterEntry:
    """What the user decided about one SAM object id."""

    sam_id: int
    team: str
    role: str
    name: str = ""


@dataclass
class Roster:
    """The labels of one camera, keyed by SAM object id."""

    path: Path | None
    entries: dict[int, RosterEntry] = field(default_factory=dict)

    @property
    def filled(self) -> bool:
        """True when at least one id has a team (only then the roster replaces the colour split)."""
        return any(e.team for e in self.entries.values())

    def team_of(self, sam_id: int) -> str:
        """Team code of an id, ``""`` when it is absent or not labelled."""
        entry = self.entries.get(int(sam_id))
        return entry.team if entry else ""

    def out_ids(self) -> set[int]:
        """Ids labelled OUT."""
        return {s for s, e in self.entries.items() if e.team == "OUT"}

    def labelled_ids(self) -> set[int]:
        """Ids labelled A, B or REF."""
        return {s for s, e in self.entries.items() if e.team in ("A", "B", "REF")}

    def kicker(self) -> int | None:
        """SAM id with role ``kicker`` (None when nobody has it)."""
        for sam_id in sorted(self.entries):
            if self.entries[sam_id].role == "kicker":
                return sam_id
        return None

    def keeper(self, team: str | None = None) -> int | None:
        """SAM id with role ``keeper`` in ``team`` (any team when None)."""
        want = normalize_team(team) if team is not None else None
        for sam_id in sorted(self.entries):
            e = self.entries[sam_id]
            if e.role == "keeper" and (want is None or e.team == want):
                return sam_id
        return None

    def key(self) -> dict[str, Any]:
        """Stable, json-able digest for stage inputs: only labelled ids, sorted by sam_id."""
        labels = [
            [e.sam_id, e.team, e.role, e.name] for _, e in sorted(self.entries.items()) if e.team
        ]
        return {"path": str(self.path) if self.path else "", "labels": labels}


# ---------------------------------------------------------------------------------------------
# normalisation, validation and csv I/O
# ---------------------------------------------------------------------------------------------


def normalize_team(text: object) -> str:
    """Team text as stored: stripped and upper-case (``" ref "`` -> ``"REF"``)."""
    return str(text if text is not None else "").strip().upper()


def normalize_role(text: object) -> str:
    """Role text as stored: stripped and lower-case."""
    return str(text if text is not None else "").strip().lower()


def roles_for_team(team: str) -> tuple[str, ...]:
    """Roles a team accepts, blank first (the dialog fills its role box with this)."""
    return ROLES_BY_TEAM.get(normalize_team(team), ("",))


def roster_dir(kick_dir: str | Path, stem: str) -> Path:
    """``<kick_dir>/roster/<stem>``."""
    return Path(kick_dir) / ROSTER_DIR / stem


def find_roster(kick_dir: str | Path, stem: str) -> Path | None:
    """``roster_dir(...)/roster.csv`` when it exists, else None."""
    path = roster_dir(kick_dir, stem) / ROSTER_CSV
    return path if path.is_file() else None


def validate_entries(entries: Iterable[RosterEntry]) -> None:
    """Raise :class:`RosterError` for the first broken rule (message names the sam_id)."""
    people = sorted(entries, key=lambda e: e.sam_id)
    for e in people:
        if e.team not in ROLES_BY_TEAM:
            raise RosterError(
                f"SAM {e.sam_id}: unknown team {e.team!r} (use A, B, REF, OUT or leave it blank)"
            )
        if e.role and e.role not in ROLE_CODES:
            raise RosterError(
                f"SAM {e.sam_id}: unknown role {e.role!r} (use {', '.join(ROLE_CODES)} or blank)"
            )
        if e.role not in ROLES_BY_TEAM[e.team]:
            where = f"team {e.team}" if e.team else "an unlabelled person (no team)"
            raise RosterError(
                f"SAM {e.sam_id}: role {e.role!r} is not allowed for {where} "
                f"(allowed: {', '.join(repr(r) for r in ROLES_BY_TEAM[e.team])})"
            )

    def ids(pred) -> list[int]:
        return [e.sam_id for e in people if pred(e)]

    kickers = ids(lambda e: e.role == "kicker")
    if len(kickers) > 1:
        raise RosterError(f"more than one kicker: SAM {', SAM '.join(map(str, kickers))}")
    for team in ("A", "B"):
        keepers = ids(lambda e, t=team: e.team == t and e.role == "keeper")
        if len(keepers) > 1:
            raise RosterError(
                f"more than one keeper in team {team}: SAM {', SAM '.join(map(str, keepers))}"
            )
        members = ids(lambda e, t=team: e.team == t)
        if len(members) > MAX_PER_TEAM:
            raise RosterError(
                f"team {team} has {len(members)} people (maximum {MAX_PER_TEAM}, keeper included): "
                f"SAM {', SAM '.join(map(str, members))}"
            )
    mains = ids(lambda e: e.role == "referee_main")
    if len(mains) > 1:
        raise RosterError(f"more than one referee_main: SAM {', SAM '.join(map(str, mains))}")
    refs = ids(lambda e: e.team == "REF")
    if len(refs) > MAX_REFEREES:
        raise RosterError(
            f"{len(refs)} people labelled REF (maximum {MAX_REFEREES}): "
            f"SAM {', SAM '.join(map(str, refs))}"
        )


def _read_rows(path: Path) -> dict[int, dict[str, str]]:
    """Raw rows of a roster csv by sam_id (no rule check besides int ids and duplicates)."""
    rows: dict[int, dict[str, str]] = {}
    seen_at: dict[int, int] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return rows
        reader.fieldnames = [str(n).strip().lower() for n in reader.fieldnames]
        if "sam_id" not in reader.fieldnames:
            raise RosterError(f"{path}: no 'sam_id' column (header: {reader.fieldnames})")
        for number, raw in enumerate(reader, start=2):  # line 1 is the header
            if not any((v or "").strip() for v in raw.values() if isinstance(v, str)):
                continue  # empty line
            text = (raw.get("sam_id") or "").strip()
            try:
                sam_id = int(float(text)) if text.replace(".", "", 1).isdigit() else int(text)
            except ValueError:
                raise RosterError(
                    f"{path} line {number}: sam_id {text!r} is not an integer"
                ) from None
            if sam_id in rows:
                raise RosterError(
                    f"duplicate sam_id {sam_id} (lines {seen_at[sam_id]} and {number} of {path})"
                )
            seen_at[sam_id] = number
            rows[sam_id] = {k: (v or "").strip() for k, v in raw.items() if isinstance(v, str)}
    return rows


def _row_to_entry(sam_id: int, row: dict[str, str]) -> RosterEntry:
    return RosterEntry(
        sam_id=sam_id,
        team=normalize_team(row.get("team", "")),
        role=normalize_role(row.get("role", "")),
        name=row.get("name", "").strip(),
    )


def _write_rows(path: Path, rows: dict[int, dict[str, str]]) -> Path:
    """Write the rows (sorted by sam_id) atomically; team upper-case, role lower-case."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ROSTER_FIELDS, lineterminator="\n")
        writer.writeheader()
        for sam_id in sorted(rows):
            row = {k: rows[sam_id].get(k, "") for k in ROSTER_FIELDS}
            row["sam_id"] = str(sam_id)
            row["team"] = normalize_team(row["team"])
            row["role"] = normalize_role(row["role"])
            writer.writerow(row)
    os.replace(tmp, path)
    return path


def _frame_columns(info: dict[str, Any] | None) -> dict[str, str]:
    if not info:
        return {}
    return {
        "first_frame": str(info["first"]),
        "last_frame": str(info["last"]),
        "frames": str(info["frames"]),
    }


def read_roster(path: str | Path) -> Roster:
    """Read and validate a roster csv; raise :class:`RosterError` on any broken rule."""
    path = Path(path)
    rows = _read_rows(path)
    entries = {sam_id: _row_to_entry(sam_id, row) for sam_id, row in rows.items()}
    validate_entries(entries.values())
    return Roster(path=path, entries=entries)


def write_roster(
    path: str | Path, roster: Roster, summary: dict[int, dict[str, Any]] | None = None
) -> Path:
    """Save ``roster`` to ``path``.

    The frame columns come from ``summary`` (output of :func:`track_summary`) when given, else from
    the file already there; ``note`` is kept from the file already there; rows of the file whose id
    is not in ``roster`` are kept too, so no label is ever lost.
    """
    path = Path(path)
    rows: dict[int, dict[str, str]] = _read_rows(path) if path.is_file() else {}
    for sam_id, e in roster.entries.items():
        row = rows.setdefault(sam_id, {})
        row.update(team=e.team, role=e.role, name=e.name)
        if summary and sam_id in summary:
            row.update(_frame_columns(summary[sam_id]))
    return _write_rows(path, rows)


# ---------------------------------------------------------------------------------------------
# the contact sheet
# ---------------------------------------------------------------------------------------------


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _overlap(box: Sequence[float], others: Iterable[Sequence[float]]) -> float:
    """Largest share of ``box`` (by its own area) covered by another box."""
    worst = 0.0
    own = max(_area(box), 1.0)
    for o in others:
        ix = max(0.0, min(box[2], o[2]) - max(box[0], o[0]))
        iy = max(0.0, min(box[3], o[3]) - max(box[1], o[1]))
        worst = max(worst, ix * iy / own)
    return worst


def track_summary(frames: dict[int, list[dict]]) -> dict[int, dict[str, Any]]:
    """Per SAM id: ``{"first", "last", "frames", "picks": [(frame, bbox_xyxy) x up to 3]}``.

    The picks are the largest box in each third of the track (first, middle, last), among the
    frames where the box overlaps another box by less than 15 % of its own area; a third with no
    such frame falls back to all its frames.
    """
    tracks: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    boxes_of: dict[int, list[list[float]]] = {}
    for f in sorted(frames):
        boxes_of[f] = [list(map(float, i["bbox_xyxy"])) for i in frames[f]]
        for inst, box in zip(frames[f], boxes_of[f], strict=True):
            tracks[int(inst["sam_obj_id"])].append((f, box))
    summary: dict[int, dict[str, Any]] = {}
    for sam_id, items in tracks.items():
        n = len(items)
        picks: list[tuple[int, list[float]]] = []
        for part in range(3):
            lo, hi = part * n // 3, (part + 1) * n // 3
            if hi <= lo:
                continue
            third = items[lo:hi]
            clean = [
                (f, b)
                for f, b in third
                if _overlap(b, (o for o in boxes_of[f] if o is not b)) < OVERLAP_MAX
            ]
            best = max(clean or third, key=lambda fb: _area(fb[1]))
            if all(best[0] != p[0] for p in picks):
                picks.append(best)
        summary[sam_id] = {
            "first": items[0][0],
            "last": items[-1][0],
            "frames": n,
            "picks": picks,
        }
    return dict(sorted(summary.items()))


def default_context_frames(
    frames: dict[int, list[dict]], count: int = DEFAULT_CONTEXT_FRAMES
) -> list[int]:
    """Up to ``count`` frames spread over the video, each the one with most boxes in its part."""
    if not frames:
        return []
    ordered = sorted(frames)
    chosen: list[int] = []
    for k in range(count):
        part = ordered[k * len(ordered) // count : (k + 1) * len(ordered) // count]
        if part:
            chosen.append(max(part, key=lambda f: (len(frames[f]), -f)))
    return sorted(set(chosen))


def _blank_cell() -> np.ndarray:
    return np.full((CELL_H, CELL_W, 3), _BG, np.uint8)


def _put_text(
    img: np.ndarray, text: str, org: tuple[int, int], scale: float, colour, thick: int
) -> None:
    import cv2

    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def _crop_cell(image: np.ndarray, box: Sequence[float], frame: int) -> np.ndarray:
    """One 120 x 230 cell: the person with a margin, the box drawn, the frame number."""
    import cv2

    h, w = image.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x0, y0 = int(max(0, box[0] - 0.25 * bw)), int(max(0, box[1] - 0.15 * bh))
    x1, y1 = int(min(w, box[2] + 0.25 * bw)), int(min(h, box[3] + 0.15 * bh))
    cell = _blank_cell()
    if x1 - x0 < 2 or y1 - y0 < 2:
        return cell
    crop = image[y0:y1, x0:x1].copy()
    cv2.rectangle(
        crop,
        (int(box[0] - x0), int(box[1] - y0)),
        (int(box[2] - x0), int(box[3] - y0)),
        (0, 255, 255),
        1,
    )
    scale = min(CELL_W / crop.shape[1], CELL_H / crop.shape[0])
    size = (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale)))
    crop = cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    oy, ox = (CELL_H - crop.shape[0]) // 2, (CELL_W - crop.shape[1]) // 2
    cell[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop
    _put_text(cell, f"f{frame}", (4, CELL_H - 6), 0.45, (255, 255, 255), 1)
    return cell


def _tile(cells: list[np.ndarray], sam_id: int, info: dict[str, Any]) -> np.ndarray:
    """Three cells side by side plus the label strip ``ID n`` / ``frames a-b (n)``."""
    cells = cells + [_blank_cell()] * (3 - len(cells))
    body = np.hstack(cells[:3])
    tile = np.full((CELL_H + LABEL_H, body.shape[1], 3), _BG, np.uint8)
    tile[:CELL_H] = body
    _put_text(tile, f"ID {sam_id}", (6, CELL_H + 30), 0.9, (255, 255, 255), 2)
    _put_text(
        tile,
        f"frames {info['first']}-{info['last']} ({info['frames']})",
        (110, CELL_H + 30),
        0.5,
        (200, 200, 200),
        1,
    )
    return tile


def _grid(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    rows = []
    for i in range(0, len(tiles), columns):
        row = tiles[i : i + columns]
        row = row + [np.full_like(tiles[0], _BG)] * (columns - len(row))
        rows.append(
            np.hstack(
                [np.pad(t, ((2, 2), (2, 2), (0, 0)), constant_values=_FRAME_BORDER) for t in row]
            )
        )
    return np.vstack(rows)


def _context_image(image: np.ndarray, instances: list[dict], frame: int) -> np.ndarray:
    import cv2

    img = image.copy()
    for inst in instances:
        b = [int(v) for v in inst["bbox_xyxy"]]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
        _put_text(
            img, str(int(inst["sam_obj_id"])), (b[0], max(14, b[1] - 6)), 0.8, (0, 255, 255), 2
        )
    _put_text(img, f"frame {frame}", (20, 40), 1.1, (255, 255, 255), 3)
    return img


def write_roster_sheet(
    predictions: str | Path,
    video: str | Path,
    out_dir: str | Path,
    *,
    columns: int = 4,
    context_frames: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Write the contact sheet and the roster template of a camera; return the paths.

    Writes in ``out_dir``: ``roster_ids.png``, ``roster_context_<fff>.png`` (default: up to 5 frames
    spread over the video), ``roster_crops/id_<nnn>.png`` (one tile per id, for the dialog) and
    ``roster.csv``. If ``roster.csv`` already exists its labels are kept: new ids are added blank
    and the frame columns refreshed (the user's work is never lost). The video is read once, in
    order, and only the needed frames are kept.

    Returns ``{"sheet": png, "csv": csv, "context": [pngs...]}``.
    """
    import cv2

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames, _ = read_predictions(predictions)
    summary = track_summary(frames)
    if not summary:
        raise RosterError(f"{predictions}: no SAM instance in the predictions")
    wanted_context = (
        sorted({int(f) for f in context_frames})
        if context_frames is not None
        else default_context_frames(frames)
    )
    need = {f for info in summary.values() for f, _ in info["picks"]} | set(wanted_context)

    cells: dict[tuple[int, int], np.ndarray] = {}
    context_paths: list[Path] = []
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open the video {video}")
    try:
        last = max(need)
        for index in range(last + 1):
            if not cap.grab():
                break
            if index not in need:
                continue
            ok, image = cap.retrieve()
            if not ok:
                continue
            for sam_id, info in summary.items():
                for f, box in info["picks"]:
                    if f == index:
                        cells[(sam_id, f)] = _crop_cell(image, box, f)
            if index in wanted_context:
                path = out / f"roster_context_{index:03d}.png"
                cv2.imwrite(str(path), _context_image(image, frames.get(index, []), index))
                context_paths.append(path)
    finally:
        cap.release()

    crops = out / CROPS_DIR
    crops.mkdir(exist_ok=True)
    tiles = []
    for sam_id, info in summary.items():
        tile = _tile(
            [cells[(sam_id, f)] for f, _ in info["picks"] if (sam_id, f) in cells], sam_id, info
        )
        cv2.imwrite(str(crops / f"id_{sam_id:03d}.png"), tile)
        tiles.append(tile)
    sheet = out / SHEET_PNG
    cv2.imwrite(str(sheet), _grid(tiles, max(1, int(columns))))

    csv_path = out / ROSTER_CSV
    rows: dict[int, dict[str, str]] = _read_rows(csv_path) if csv_path.is_file() else {}
    for sam_id, info in summary.items():
        rows.setdefault(sam_id, {}).update(_frame_columns(info))
    _write_rows(csv_path, rows)
    return {"sheet": sheet, "csv": csv_path, "context": sorted(context_paths)}


# ---------------------------------------------------------------------------------------------
# the Tkinter dialog
# ---------------------------------------------------------------------------------------------


def _open_with_system(path: Path) -> bool:
    """Open a file with the desktop default viewer; False when nothing could be launched."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError:
        return False
    return True


# Colours of the team buttons and of the card borders (a glance at the grid shows what is left).
TEAM_COLOURS = {
    "A": "#6f9bff",  # blue-ish
    "B": "#ff9f4a",  # orange-ish
    "REF": "#ffe14d",  # yellow
    "OUT": "#8a8a8a",  # grey
    "": "#c4c4c4",  # not labelled
}
_BAD_TEAM_COLOUR = "#d9534f"  # a team text the dialog does not know (wrong csv)
CARD_CROP_H = 160  # height in pixels of the three crops of a card


def team_colour(team: str) -> str:
    """Colour of a team code in the dialog (red for an unknown text)."""
    return TEAM_COLOURS.get(normalize_team(team), _BAD_TEAM_COLOUR)


def roster_dialog(
    roster_csv: str | Path, sheet_dir: str | Path, *, parent: Any = None
) -> Roster | None:
    """Edit the roster in a window; return the saved :class:`Roster`, None when cancelled.

    Tkinter only. With a ``parent`` the window is a ``Toplevel`` of it (never a second ``Tk``);
    without one it owns a ``Tk`` root. The window opens at about 90 % of the screen and shows one
    card per id in a grid that reflows with the window width: the three crops written by
    :func:`write_roster_sheet`, ``ID n - frames a-b``, team buttons ``A B REF OUT -`` (the selected
    one coloured), a role box limited to what the team accepts and a name entry. The card border
    carries the team colour. Save checks the rules and writes the csv (a broken rule is shown and
    the window stays open); the file is read leniently so a wrong one can be fixed here.

    Widget names (for scripted use): ``card<id>`` with children ``team_A``, ``team_B``,
    ``team_REF``, ``team_OUT``, ``team_none``, ``role``, ``name_entry``; buttons ``save`` and
    ``cancel``.
    """
    import tkinter as tk
    from tkinter import messagebox, ttk

    csv_path = Path(roster_csv)
    sheet_path = Path(sheet_dir)
    rows = _read_rows(csv_path) if csv_path.is_file() else {}
    if not rows:
        raise RosterError(f"{csv_path}: no ids; run the 'sheet' step first")

    own_root = parent is None
    win = tk.Tk() if own_root else tk.Toplevel(parent)
    win.title(f"vailá - label the people - {csv_path}")
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    width, height = int(sw * 0.9), int(sh * 0.9)
    win.geometry(f"{width}x{height}+{(sw - width) // 2}+{(sh - height) // 2}")
    result: dict[str, Roster | None] = {"roster": None}
    photos: list[Any] = []  # keep a reference or Tk drops the images

    def load_photo(path: Path) -> Any:
        """The three crops of an id (label strip cut off), scaled to ``CARD_CROP_H`` high."""
        if not path.is_file():
            return None
        try:
            try:
                from PIL import Image, ImageTk
            except ImportError:
                return tk.PhotoImage(master=win, file=str(path)).subsample(2)
            img = Image.open(path).convert("RGB")
            img = img.crop((0, 0, img.width, min(CELL_H, img.height)))
            scale = CARD_CROP_H / img.height
            img = img.resize(
                (max(1, int(img.width * scale)), CARD_CROP_H), Image.Resampling.LANCZOS
            )
            return ImageTk.PhotoImage(img, master=win)
        except (OSError, ValueError, tk.TclError):
            return None

    # header: counter + one-line legend
    top = tk.Frame(win, padx=8, pady=4)
    top.pack(side="top", fill="x")
    counter = tk.StringVar(master=win)
    tk.Label(top, textvariable=counter, font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
    tk.Label(
        top,
        text=(
            "A / B = the two teams   REF = referee   OUT = not on the pitch   - = unknown   "
            "keeper / kicker need A or B   check the 3 crops are one person"
        ),
    ).pack(anchor="w")

    bottom = tk.Frame(win, padx=8, pady=6)
    bottom.pack(side="bottom", fill="x")
    body = tk.Frame(win)
    body.pack(side="top", fill="both", expand=True)
    canvas = tk.Canvas(body, highlightthickness=0, yscrollincrement=24)
    vbar = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
    hbar = ttk.Scrollbar(body, orient="horizontal", command=canvas.xview)
    canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
    vbar.pack(side="right", fill="y")
    hbar.pack(side="bottom", fill="x")
    canvas.pack(side="left", fill="both", expand=True)
    inner = tk.Frame(canvas)
    canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))

    ids = sorted(rows)
    cards: list[Any] = []
    team_vars: dict[int, Any] = {}
    role_vars: dict[int, Any] = {}
    name_vars: dict[int, Any] = {}

    def refresh_counter(*_a: Any) -> None:
        count = dict.fromkeys(("", *TEAM_CODES), 0)
        for s in ids:
            t = normalize_team(team_vars[s].get())
            count[t if t in count else ""] += 1
        counter.set(
            f"A: {count['A']}   B: {count['B']}   REF: {count['REF']}   OUT: {count['OUT']}   "
            f"not labelled: {count['']}   (of {len(ids)} ids)"
        )

    for sam_id in ids:
        row = rows[sam_id]
        e = _row_to_entry(sam_id, row)
        card = tk.Frame(
            inner,
            name=f"card{sam_id}",
            highlightthickness=4,
            highlightbackground=team_colour(e.team),
            highlightcolor=team_colour(e.team),
            padx=3,
            pady=3,
        )
        cards.append(card)
        photo = load_photo(sheet_path / CROPS_DIR / f"id_{sam_id:03d}.png")
        if photo is not None:
            photos.append(photo)
            tk.Label(card, image=photo).pack()
        else:
            tk.Label(card, text="(no crop: run the sheet step)", height=8).pack()
        tk.Label(
            card,
            text=f"ID {sam_id} · frames {row.get('first_frame', '?')}-{row.get('last_frame', '?')}",
            font=("TkDefaultFont", 10, "bold"),
        ).pack()
        tv = tk.StringVar(master=win, value=e.team)
        rv = tk.StringVar(master=win, value=e.role)
        nv = tk.StringVar(master=win, value=e.name)
        team_vars[sam_id], role_vars[sam_id], name_vars[sam_id] = tv, rv, nv

        bar = tk.Frame(card)
        bar.pack(pady=2)
        for code in (*TEAM_CODES, ""):
            tk.Radiobutton(
                bar,
                name=f"team_{code or 'none'}",
                text=code or "-",
                value=code,
                variable=tv,
                indicatoron=False,
                selectcolor=TEAM_COLOURS[code],
                width=4 if code == "REF" else 3,
                padx=2,
                pady=1,
            ).pack(side="left", padx=1)
        role_row = tk.Frame(card)
        role_row.pack(fill="x", pady=1)
        tk.Label(role_row, text="role").pack(side="left")
        role_box = ttk.Combobox(
            role_row,
            name="role",
            textvariable=rv,
            values=roles_for_team(e.team),
            width=14,
            state="readonly",
        )
        role_box.pack(side="left", padx=4)
        name_row = tk.Frame(card)
        name_row.pack(fill="x", pady=1)
        tk.Label(name_row, text="name").pack(side="left")
        tk.Entry(name_row, name="name_entry", textvariable=nv, width=14).pack(side="left", padx=4)

        def on_team(*_a: Any, card=card, tv=tv, rv=rv, role_box=role_box) -> None:
            colour = team_colour(tv.get())
            card.configure(highlightbackground=colour, highlightcolor=colour)
            allowed = roles_for_team(tv.get())
            role_box.configure(values=allowed)
            if normalize_role(rv.get()) not in allowed:
                rv.set("")
            refresh_counter()

        tv.trace_add("write", on_team)
    refresh_counter()

    # reflow: as many cards per row as fit in the canvas width
    layout = {"cols": 0}

    def reflow(_e: Any = None) -> None:
        cell = max(c.winfo_reqwidth() for c in cards) + 12
        cols = max(1, canvas.winfo_width() // cell)
        if cols == layout["cols"]:
            return
        layout["cols"] = cols
        for i, c in enumerate(cards):
            c.grid(row=i // cols, column=i % cols, padx=4, pady=4, sticky="n")

    canvas.bind("<Configure>", reflow)

    # mouse wheel on the canvas and every widget inside it (Linux Button-4/5, Windows/macOS Wheel)
    def on_wheel(event: Any) -> str:
        up = getattr(event, "num", 0) == 4 or getattr(event, "delta", 0) > 0
        canvas.yview_scroll(-3 if up else 3, "units")
        return "break"  # also stops a Combobox from changing value under the wheel

    def bind_wheel(widget: Any) -> None:
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(seq, on_wheel)
        for child in widget.winfo_children():
            bind_wheel(child)

    bind_wheel(canvas)

    def save() -> None:
        entries = {
            s: RosterEntry(
                s,
                normalize_team(team_vars[s].get()),
                normalize_role(role_vars[s].get()),
                name_vars[s].get().strip(),
            )
            for s in ids
        }
        try:
            validate_entries(entries.values())
        except RosterError as exc:
            messagebox.showerror("Roster", str(exc), parent=win)
            return
        roster = Roster(path=csv_path, entries=entries)
        write_roster(csv_path, roster)
        result["roster"] = roster
        win.destroy()

    def open_sheet() -> None:
        sheet = sheet_path / SHEET_PNG
        if not sheet.is_file() or not _open_with_system(sheet):
            messagebox.showinfo("Roster", f"Open this file by hand:\n{sheet}", parent=win)

    ttk.Button(bottom, name="save", text="Save", command=save).pack(side="left", padx=4)
    ttk.Button(bottom, name="cancel", text="Cancel", command=win.destroy).pack(side="left", padx=4)
    ttk.Button(bottom, name="open_sheet", text="Open full sheet", command=open_sheet).pack(
        side="left", padx=4
    )
    tk.Label(bottom, text=str(csv_path)).pack(side="right")
    win.protocol("WM_DELETE_WINDOW", win.destroy)

    if own_root:
        win.mainloop()
    else:
        win.transient(parent)
        with contextlib.suppress(tk.TclError):  # not viewable yet on some window managers
            win.grab_set()
        parent.wait_window(win)
    return result["roster"]


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _check(path: Path) -> int:
    try:
        roster = read_roster(path)
    except (RosterError, OSError) as exc:
        print(f"roster INVALID: {exc}", file=sys.stderr)
        return 2
    count = dict.fromkeys(("", *TEAM_CODES), 0)
    for e in roster.entries.values():
        count[e.team] += 1
    print(f"roster: {path}")
    print(f"filled: {'yes' if roster.filled else 'no'}   ids: {len(roster.entries)}")
    for team in ("A", "B"):
        keeper = roster.keeper(team)
        print(
            f"team {team}: {count[team]}   keeper: {'SAM ' + str(keeper) if keeper is not None else 'none'}"
        )
    mains = [s for s, e in sorted(roster.entries.items()) if e.role == "referee_main"]
    print(f"REF: {count['REF']}   referee_main: {'SAM ' + str(mains[0]) if mains else 'none'}")
    print(f"OUT: {count['OUT']}   not labelled: {count['']}")
    kicker = roster.kicker()
    print(f"kicker: {'SAM ' + str(kicker) if kicker is not None else 'none'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vaila.freekiki_roster",
        description="Label the SAM ids of a FreeKiki run (teams, keeper, kicker, referees).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_sheet = sub.add_parser("sheet", help="write the contact sheet and the roster.csv template")
    p_sheet.add_argument("--predictions", required=True, help="*_sam3dinov3_predictions.json.gz")
    p_sheet.add_argument("--video", required=True, help="video of that camera")
    p_sheet.add_argument("--out", required=True, help="output folder (roster/<stem>)")
    p_sheet.add_argument("--columns", type=int, default=4, help="ids per sheet row (default 4)")
    p_check = sub.add_parser("check", help="validate a roster.csv and print the counts")
    p_check.add_argument("roster_csv")
    p_gui = sub.add_parser("gui", help="open the Tkinter editor")
    p_gui.add_argument("roster_csv")
    p_gui.add_argument("--sheet-dir", default=None, help="folder of the sheet (default: the csv's)")
    args = parser.parse_args(argv)

    if args.cmd == "sheet":
        paths = write_roster_sheet(args.predictions, args.video, args.out, columns=args.columns)
        print(f"sheet: {paths['sheet']}")
        print(f"csv: {paths['csv']}")
        for p in paths["context"]:
            print(f"context: {p}")
        print(f"next: python -m vaila.freekiki_roster gui {paths['csv']}")
        return 0
    if args.cmd == "check":
        return _check(Path(args.roster_csv))
    csv_path = Path(args.roster_csv)
    saved = roster_dialog(csv_path, args.sheet_dir or csv_path.parent)
    print("saved" if saved is not None else "cancelled (nothing written)")
    return 0 if saved is not None else 1


if __name__ == "__main__":
    sys.exit(main())
