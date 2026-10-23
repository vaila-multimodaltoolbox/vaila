"""Shared clock sync between *getpixelvideo* (player) and *scout_vaila* (annotation).

Update Date: 22 October 2026
Version: 0.4.10

GetPixelVideo is the video viewer; Scout marks fundamentos on the field diagram.
While the video plays / seeks, GetPixelVideo writes the current media time here;
Scout polls the same file and mirrors that time on its clock.

Default path: ``~/.vaila/scout_getpixel_sync.json`` (override with env
``VAILA_SCOUT_GETPIXEL_SYNC``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

SYNC_ENV = "VAILA_SCOUT_GETPIXEL_SYNC"
DEFAULT_NAME = "scout_getpixel_sync.json"


def sync_path() -> Path:
    raw = os.environ.get(SYNC_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".vaila" / DEFAULT_NAME


def write_sync_state(
    *,
    time_s: float,
    frame: int,
    fps: float,
    paused: bool,
    video_path: str | None = None,
    source: str = "getpixelvideo",
) -> Path:
    """Atomically write the current player clock for Scout to follow."""
    path = sync_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": 1,
        "source": source,
        "time_s": float(max(0.0, time_s)),
        "frame": int(max(0, frame)),
        "fps": float(fps) if fps and fps > 0 else 0.0,
        "paused": bool(paused),
        "playing": not bool(paused),
        "video_path": str(video_path) if video_path else None,
        "updated_at": time.time(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def read_sync_state() -> dict[str, Any] | None:
    """Return latest sync payload, or None if missing/invalid."""
    path = sync_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or "time_s" not in data:
        return None
    return data
