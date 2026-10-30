"""
================================================================================
Script: freekiki_player_ids.py - one player id across the cameras of the same kick
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.12
Created: 27 October 2026
Update Date: 29 October 2026

Description:
    Each camera numbers its people with the SAM 3 object id, so the same player can carry
    different ids in two cameras of the same kick. This module gives every player one id,
    ``player_id``, that is the same in every camera:

    * 0: the goalkeeper, the person on the ground nearest the goal the ball is attacking, at the
      first frame.
    * 1: the kicker, the person whose foot is nearest the ball's rest point when the ball starts
      to move (the contact). When no foot is within reach there, the first later frame in which
      the ball moves toward that goal is used.
    * 2, 3, ...: everyone else, by the distance from their nearest foot to the same point at the
      same frame, nearest first. The order is fixed for the whole clip.

    The tracks of two cameras are linked as tracks, not frame by frame: the median ground distance
    of two tracks over their common frames is the cost, a link is kept when it is below
    ``PAIR_GATE_M``, and the links are added cheapest first. Two tracks of the same camera that
    exist at the same time are two people, so they never share a group; a person seen in pieces
    (tracks that do not overlap) joins one group. A camera without a partner keeps its own tracks.

    The frames of the second camera are read through the sync name of each cut
    (``..._sync_<n>_frames_<a>_to_<b>``); without it the frame index is used as it is.

    The pure functions take plain data and are tested without files. :func:`run_player_ids`
    reads the 3D folders and writes ``player_map.csv``, ``poses_3d_on_field_players.csv`` and
    ``ground_tracks_players.csv`` in each folder; the original files are not changed.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from .freekiki_auto_flight import GOAL_X_M, kick_onset, rest_point
    from .freekiki_field3d import FEET_NAMES
    from .freekiki_report_players import ROLE_RADIUS_M
except ImportError:
    from freekiki_auto_flight import (  # ty: ignore[unresolved-import]
        GOAL_X_M,
        kick_onset,
        rest_point,
    )
    from freekiki_field3d import FEET_NAMES  # ty: ignore[unresolved-import]
    from freekiki_report_players import ROLE_RADIUS_M  # ty: ignore[unresolved-import]

KEEPER_ID = 0
KICKER_ID = 1
FIRST_OTHER_ID = 2
KEEPER_RADIUS_M = 6.0  # the goalkeeper stands on the goal line at the first frame
KICKER_RADIUS_M = ROLE_RADIUS_M["kicker"]  # 2 m: the foot that touches the ball
PAIR_GATE_M = 1.5  # the largest median ground distance of two tracks of the same person
MIN_OVERLAP_FRAMES = 10  # common frames needed to compare two tracks
NEAREST_WINDOW = 3  # a missing frame is read from the closest observed frame within this many
BALL_TOWARD_GOAL_MPS = 1.0  # the ball counts as moving toward the goal above this speed

FeetMap = dict[tuple[str, int], dict[int, np.ndarray]]  # (camera, sam id) -> frame -> (k, 2)


def _nearest[T](values: dict[int, T], frame: int) -> T | None:
    """The value at ``frame``, or at the closest frame within ``NEAREST_WINDOW``; None if none."""
    for step in range(NEAREST_WINDOW + 1):
        for f in (frame - step, frame + step):
            if f in values:
                return values[f]
    return None


@dataclass
class Track:
    """One track of one camera: the ground position (metres) of a SAM id per frame of the camera."""

    camera: str
    sam_id: int
    xy: dict[int, tuple[float, float]]


@dataclass
class Group:
    """Tracks that are one person: at most one track per camera at a time."""

    tracks: list[Track]
    costs: list[float] = field(default_factory=list)  # the link costs that joined the group


def pair_cost(a: Track, b: Track, offset: int = 0) -> tuple[float, int]:
    """Median ground distance (m) of two tracks over their common frames, and how many there are.

    ``offset`` is the frame of ``b`` minus the frame of ``a`` for the same instant. Fewer than
    ``MIN_OVERLAP_FRAMES`` common frames give an infinite cost: the pair cannot be judged.
    """
    dists = []
    for f, p in a.xy.items():
        q = b.xy.get(f + offset)
        if q is not None:
            dists.append(math.hypot(p[0] - q[0], p[1] - q[1]))
    if len(dists) < MIN_OVERLAP_FRAMES:
        return math.inf, len(dists)
    return float(np.median(dists)), len(dists)


def _coexist(a: Track, b: Track) -> bool:
    """True when two tracks of the same camera exist at the same frame: two different people."""
    return bool(a.xy.keys() & b.xy.keys())


def _compatible(g1: Group, g2: Group) -> bool:
    """Two groups can merge when no two of their tracks of one camera exist at the same time."""
    return not any(a.camera == b.camera and _coexist(a, b) for a in g1.tracks for b in g2.tracks)


def link_tracks(
    tracks: list[Track], offsets: dict[str, int], *, gate_m: float = PAIR_GATE_M
) -> list[Group]:
    """Groups of tracks that are one person, greedy by pair cost under the one-place-per-time rule.

    ``offsets[camera]`` is the frame index of that camera for the same instant as frame 0 of the
    reference camera (the reference has zero): frame ``f`` of camera ``a`` and frame
    ``f + offsets[b] - offsets[a]`` of camera ``b`` are one instant.
    """
    edges: list[tuple[float, int, int]] = []
    for i, a in enumerate(tracks):
        for j in range(i + 1, len(tracks)):
            b = tracks[j]
            if a.camera == b.camera:
                continue
            cost, _ = pair_cost(a, b, offsets[b.camera] - offsets[a.camera])
            if cost <= gate_m:
                edges.append((cost, i, j))
    edges.sort()
    owner = {i: i for i in range(len(tracks))}  # track index -> group key
    groups: dict[int, Group] = {i: Group([t]) for i, t in enumerate(tracks)}
    for cost, i, j in edges:
        gi, gj = owner[i], owner[j]
        if gi == gj or not _compatible(groups[gi], groups[gj]):
            continue
        groups[gi].tracks.extend(groups[gj].tracks)
        groups[gi].costs.extend(groups[gj].costs)
        groups[gi].costs.append(cost)
        for k, g in owner.items():
            if g == gj:
                owner[k] = gi
        del groups[gj]
    return list(groups.values())


def _at(track: Track, index: int) -> tuple[float, float] | None:
    return _nearest(track.xy, index)


def _group_ground_at(group: Group, frame: int, offsets: dict[str, int]) -> np.ndarray | None:
    """Mean ground point of the group at the reference frame ``frame`` (None when none is seen)."""
    pts = [p for t in group.tracks if (p := _at(t, frame + offsets[t.camera])) is not None]
    return None if not pts else np.mean(np.asarray(pts, dtype=float), axis=0)


def _foot_distance(
    group: Group, frame: int, offsets: dict[str, int], feet: FeetMap, point: np.ndarray
) -> float | None:
    """Distance from the nearest foot of any track of the group to ``point`` at ``frame``."""
    best: float | None = None
    for t in group.tracks:
        pts = _nearest(feet.get((t.camera, t.sam_id), {}), frame + offsets[t.camera])
        if pts is None or len(pts) == 0:
            continue
        d = float(np.min(np.hypot(pts[:, 0] - point[0], pts[:, 1] - point[1])))
        best = d if best is None else min(best, d)
    return best


def _keeper(
    groups: list[Group], offsets: dict[str, int], goal_point: np.ndarray, frame: int
) -> int | None:
    """The group whose ground point at ``frame`` is nearest the goal, within ``KEEPER_RADIUS_M``."""
    best: tuple[float, int] | None = None
    for gi, g in enumerate(groups):
        centre = _group_ground_at(g, frame, offsets)
        if centre is None:
            continue
        d = float(np.hypot(*(centre - goal_point)))
        if d <= KEEPER_RADIUS_M and (best is None or d < best[0]):
            best = (d, gi)
    return None if best is None else best[1]


def _moving_toward_goal(
    ball_xy: dict[int, tuple[float, float]], frame: int, side: float, fps: float
) -> bool:
    """True when the ball moves at the frame faster than ``BALL_TOWARD_GOAL_MPS`` toward the goal."""
    if frame not in ball_xy or frame - 1 not in ball_xy:
        return False
    vx = (ball_xy[frame][0] - ball_xy[frame - 1][0]) * fps
    vy = (ball_xy[frame][1] - ball_xy[frame - 1][1]) * fps
    return side * vx > 0 and math.hypot(vx, vy) > BALL_TOWARD_GOAL_MPS


def _kick_reference(
    groups: list[Group],
    *,
    skip: set[int],
    offsets: dict[str, int],
    feet: FeetMap,
    rest: tuple[float, float],
    contact: int,
    side: float,
    ball_xy: dict[int, tuple[float, float]],
    fps: float,
    last_frame: int,
) -> tuple[int, np.ndarray, int | None, float | None]:
    """``(frame, point, kicker group, foot distance)`` of the reference.

    The contact is the frame of the rest point. When no foot is within reach there, the first
    later frame in which the ball moves toward the goal is used, with the ball at that frame.
    """
    rest_pt = np.asarray(rest, dtype=float)

    def kicker_at(frame: int, point: np.ndarray) -> tuple[int, float] | None:
        best: tuple[float, int] | None = None
        for gi, g in enumerate(groups):
            if gi in skip:
                continue
            d = _foot_distance(g, frame, offsets, feet, point)
            if d is not None and d <= KICKER_RADIUS_M and (best is None or d < best[0]):
                best = (d, gi)
        return None if best is None else (best[1], best[0])

    found = kicker_at(contact, rest_pt)
    if found is not None:
        return contact, rest_pt, found[0], found[1]
    for frame in range(contact + 1, last_frame + 1):
        if not _moving_toward_goal(ball_xy, frame, side, fps):
            continue
        point = np.asarray(ball_xy[frame], dtype=float)
        found = kicker_at(frame, point)
        if found is not None:
            return frame, point, found[0], found[1]
    return contact, rest_pt, None, None


def assign_player_ids(
    groups: list[Group],
    *,
    offsets: dict[str, int],
    feet: FeetMap,
    rest: tuple[float, float],
    contact: int,
    side: float,
    ball_xy: dict[int, tuple[float, float]],
    fps: float,
    last_frame: int,
    first_frame: int = 0,
) -> tuple[dict[int, int], dict[int, str], dict]:
    """The player id and role of every group: ``(ids, roles, info)``.

    ``ids`` and ``roles`` map the index of a group to its id (0, 1, 2, ...) and role
    (``keeper``, ``kicker``, ``other``). The ids 0 and 1 are reserved: when no keeper or no kicker
    is found, that id is simply absent and the others keep their numbers from 2. ``side`` is +1
    or -1: the goal the ball is attacking is at ``x = side * GOAL_X_M``.
    """
    goal_point = np.array([side * GOAL_X_M, 0.0])
    keeper = _keeper(groups, offsets, goal_point, first_frame)
    skip = {keeper} if keeper is not None else set()
    frame, point, kicker, kick_d = _kick_reference(
        groups,
        skip=skip,
        offsets=offsets,
        feet=feet,
        rest=rest,
        contact=contact,
        side=side,
        ball_xy=ball_xy,
        fps=fps,
        last_frame=last_frame,
    )
    if kicker is not None:
        skip.add(kicker)

    ids: dict[int, int] = {}
    roles: dict[int, str] = {}
    if keeper is not None:
        ids[keeper], roles[keeper] = KEEPER_ID, "keeper"
    if kicker is not None:
        ids[kicker], roles[kicker] = KICKER_ID, "kicker"

    ranked: list[tuple[float, int, int]] = []
    for gi, g in enumerate(groups):
        if gi in skip:
            continue
        d = _foot_distance(g, frame, offsets, feet, point)
        if d is None:
            centre = _group_ground_at(g, frame, offsets)
            d = math.inf if centre is None else float(np.hypot(*(centre - point)))
        first = min(min(t.xy) for t in g.tracks)
        ranked.append((d, first, gi))
    ranked.sort()
    for n, (_, _, gi) in enumerate(ranked, start=FIRST_OTHER_ID):
        ids[gi], roles[gi] = n, "other"

    info = {
        "contact_frame": int(contact),
        "reference_frame": int(frame),
        "keeper_found": keeper is not None,
        "kicker_found": kicker is not None,
        "kicker_distance_m": None if kick_d is None else round(float(kick_d), 3),
    }
    return ids, roles, info


def _conflicts(tracks: list[Track], player_of: dict[tuple[str, int], int]) -> int:
    """Pairs of tracks of one camera that share a player id while both exist: 0 when consistent."""
    count = 0
    for i, a in enumerate(tracks):
        for b in tracks[i + 1 :]:
            if (
                a.camera == b.camera
                and player_of[(a.camera, a.sam_id)] == player_of[(b.camera, b.sam_id)]
                and _coexist(a, b)
            ):
                count += 1
    return count


def read_ground_tracks(path: Path) -> dict[int, dict[int, tuple[float, float]]]:
    """SAM id -> frame -> ``(x, y)`` metres, from a 3D folder's ``ground_tracks.csv``."""
    out: dict[int, dict[int, tuple[float, float]]] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                pid, frame = int(row["person_id"]), int(row["frame"])
                x, y = float(row["x_m"]), float(row["y_m"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                out.setdefault(pid, {})[frame] = (x, y)
    return out


def read_feet(path: Path) -> dict[int, dict[int, np.ndarray]]:
    """SAM id -> frame -> ``(k, 2)`` metres of the foot keypoints, from ``poses_3d_on_field.csv``."""
    raw: dict[int, dict[int, list[tuple[float, float]]]] = {}
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["kpt_name"] not in FEET_NAMES:
                continue
            try:
                pid, frame = int(row["person_id"]), int(row["frame"])
                x, y = float(row["x_m"]), float(row["y_m"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                raw.setdefault(pid, {}).setdefault(frame, []).append((x, y))
    return {
        pid: {f: np.asarray(pts, dtype=float) for f, pts in frames.items()}
        for pid, frames in raw.items()
    }


def read_ball_pixels(path: Path) -> dict[int, tuple[float, float]]:
    """Frame -> ``(cx, cy)`` of the ball in pixels, from a 3D folder's ``ball.csv``."""
    out: dict[int, tuple[float, float]] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("cx"):
                out[int(row["frame"])] = (float(row["cx"]), float(row["cy"]))
    return out


def read_ball_metres(path: Path) -> dict[int, dict[str, float]]:
    """Frame -> ``{"x_m", "y_m"}`` of the ball on the pitch, from a 3D folder's ``ball_3d.csv``."""
    out: dict[int, dict[str, float]] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                x, y = float(row["x_m"]), float(row["y_m"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                out[int(row["frame"])] = {"x_m": x, "y_m": y}
    return out


def _write_rows(path: Path, header: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _cut_sync(folder: Path) -> tuple[int, int] | None:
    """``(sync frame, cut start)`` from the folder name, or None when the name has no sync.

    The video of the folder (its README) is the second source: two videos of the event equalizer
    carry their sync in ``event_equalizer.json``, not in their names.
    """
    try:
        from .freekiki_detect3d import _video_from_readme
        from .freekiki_stereo import parse_sync_name, video_sync
    except ImportError:
        from freekiki_detect3d import _video_from_readme  # ty: ignore[unresolved-import]
        from freekiki_stereo import (  # ty: ignore[unresolved-import]
            parse_sync_name,
            video_sync,
        )
    named = parse_sync_name(folder.name)
    if named is not None:
        return named
    video = _video_from_readme(folder)
    return video_sync(video) if video is not None else None


def run_player_ids(folders: list[Path]) -> dict:
    """Give one ``player_id`` to every person of the 3D folders of one kick.

    ``folders[0]`` is the reference camera. Each folder gets ``player_map.csv`` (one row per
    track of that camera), ``poses_3d_on_field_players.csv`` and ``ground_tracks_players.csv``
    (the original rows plus ``player_id``). Returns the summary the CLI prints.
    """
    if not folders:
        raise ValueError("at least one 3D folder is needed")
    names = [f.name for f in folders]
    sync = [_cut_sync(f) for f in folders]
    offsets: dict[str, int] = {names[0]: 0}
    no_sync: list[str] = []
    for name, s in zip(names[1:], sync[1:], strict=True):
        if s is None or sync[0] is None:
            offsets[name] = 0
            no_sync.append(name)
        else:
            offsets[name] = (s[0] - s[1]) - (sync[0][0] - sync[0][1])

    ball_px = read_ball_pixels(folders[0] / "ball.csv")
    ball_m = read_ball_metres(folders[0] / "ball_3d.csv")
    if len(ball_px) < 2 or len(ball_m) < 2:
        raise ValueError(f"no ball track in {folders[0]}: the kick cannot be located")
    contact = kick_onset(ball_px)
    if contact is None:
        raise ValueError(f"the ball does not move in {folders[0]}")
    if contact not in ball_m:
        raise ValueError(f"no ball position at the contact frame {contact} in {folders[0]}")
    rest = rest_point(ball_m, contact)  # the still frames before the contact, or the contact itself
    last = max(ball_m)
    side = 1.0 if ball_m[last]["x_m"] - float(rest[0]) >= 0 else -1.0
    ball_xy = {f: (r["x_m"], r["y_m"]) for f, r in ball_m.items()}

    onsets: dict[str, int | None] = {names[0]: contact}
    tracks: list[Track] = []
    feet: FeetMap = {}
    for folder, name in zip(folders, names, strict=True):
        ground = read_ground_tracks(folder / "ground_tracks.csv")
        for pid, xy in ground.items():
            tracks.append(Track(camera=name, sam_id=pid, xy=xy))
        for pid, frames in read_feet(folder / "poses_3d_on_field.csv").items():
            feet[(name, pid)] = frames
        if name != names[0]:
            px = read_ball_pixels(folder / "ball.csv")
            onsets[name] = kick_onset(px) if len(px) >= 2 else None

    return _finish(
        folders=folders,
        names=names,
        tracks=tracks,
        feet=feet,
        offsets=offsets,
        no_sync=no_sync,
        onsets=onsets,
        contact=int(contact),
        rest=(float(rest[0]), float(rest[1])),
        side=side,
        ball_xy=ball_xy,
        last=int(last),
    )


def _finish(
    *,
    folders: list[Path],
    names: list[str],
    tracks: list[Track],
    feet: FeetMap,
    offsets: dict[str, int],
    no_sync: list[str],
    onsets: dict[str, int | None],
    contact: int,
    rest: tuple[float, float],
    side: float,
    ball_xy: dict[int, tuple[float, float]],
    last: int,
) -> dict:
    fps, fps_source = _fps_of(folders[0])
    groups = link_tracks(tracks, offsets)
    ids, roles, info = assign_player_ids(
        groups,
        offsets=offsets,
        feet=feet,
        rest=rest,
        contact=contact,
        side=side,
        ball_xy=ball_xy,
        fps=fps,
        last_frame=last,
    )
    player_of: dict[tuple[str, int], int] = {}
    for gi, g in enumerate(groups):
        for t in g.tracks:
            player_of[(t.camera, t.sam_id)] = ids[gi]

    for folder, name in zip(folders, names, strict=True):
        own = [(gi, t) for gi, g in enumerate(groups) for t in g.tracks if t.camera == name]
        rows = []
        for gi, t in own:
            g = groups[gi]
            partners = ";".join(f"{u.camera}:{u.sam_id}" for u in g.tracks if u.camera != name)
            rows.append(
                {
                    "player_id": ids[gi],
                    "role": roles[gi],
                    "camera": name,
                    "sam_id": t.sam_id,
                    "first_frame": min(t.xy),
                    "last_frame": max(t.xy),
                    "frames": len(t.xy),
                    "partner": partners,
                    "median_match_m": (round(float(np.median(g.costs)), 3) if g.costs else ""),
                    "group": gi,
                }
            )
        rows.sort(key=lambda r: (r["player_id"], r["sam_id"]))
        _write_rows(
            folder / "player_map.csv",
            [
                "player_id",
                "role",
                "camera",
                "sam_id",
                "first_frame",
                "last_frame",
                "frames",
                "partner",
                "median_match_m",
                "group",
            ],
            rows,
        )
        _write_players_table(
            folder / "poses_3d_on_field.csv",
            folder / "poses_3d_on_field_players.csv",
            name,
            player_of,
        )
        _write_players_table(
            folder / "ground_tracks.csv",
            folder / "ground_tracks_players.csv",
            name,
            player_of,
        )

    cross = [g for g in groups if len({t.camera for t in g.tracks}) > 1]
    medians = [c for g in cross for c in g.costs]
    return {
        "cameras": names,
        "offsets": offsets,
        "no_sync": no_sync,
        "onsets": onsets,
        "players": len(groups),
        "cross_camera_players": len(cross),
        "median_match_m": round(float(np.median(medians)), 3) if medians else None,
        "conflicts": _conflicts(tracks, player_of),
        "onset_expected": {n: contact + offsets[n] for n in names[1:]},
        "fps": fps,
        "fps_source": fps_source,
        "reference": info,
        "keeper": _who(groups, ids, KEEPER_ID),
        "kicker": _who(groups, ids, KICKER_ID),
        "ids": {f"{c}:{s}": pid for (c, s), pid in sorted(player_of.items())},
    }


def _who(groups: list[Group], ids: dict[int, int], wanted: int) -> list[str]:
    """``camera:sam id`` of the tracks that carry ``wanted``, for the summary."""
    for gi, pid in ids.items():
        if pid == wanted:
            return [f"{t.camera}:{t.sam_id}" for t in groups[gi].tracks]
    return []


def _fps_of(folder: Path) -> tuple[float, str]:
    """``(fps, source)``: the frame rate of the run's video, the ``video:`` line of its README.

    Falls back to 30 fps when the video cannot be read; the source says which one was used.
    """
    readme = folder / "README.txt"
    if readme.is_file():
        for line in readme.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("video:"):
                video = Path(line.split(":", 1)[1].strip())
                if video.is_file():
                    import cv2

                    cap = cv2.VideoCapture(str(video))
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    cap.release()
                    if fps > 0:
                        return fps, "video"
                break
    return 30.0, "default"


def _write_players_table(src: Path, dst: Path, camera: str, player_of: dict) -> None:
    """Copy a per-person table (``person_id`` column) and add ``player_id`` next to it."""
    if not src.is_file():
        return
    with src.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        out_header = [*header, "player_id"]
        with dst.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=out_header)
            writer.writeheader()
            for row in reader:
                try:
                    pid = player_of.get((camera, int(row["person_id"])), "")
                except (KeyError, TypeError, ValueError):
                    pid = ""
                writer.writerow({**row, "player_id": pid})
