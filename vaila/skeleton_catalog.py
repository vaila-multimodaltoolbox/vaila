"""Skeleton pose template catalog for getpixelvideo ``Tpl:`` and related tools.

Loads canonical JSON presets from ``vaila/skeletons/`` (excluding soccer-field
layouts, which use the dedicated Soccer-Kiki pitch-guide path).

Update Date: 16 September 2026
Version: 0.4.3
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_P_INDEX_RE = re.compile(r"^p(\d+)$", re.IGNORECASE)

# Fixed dialog order (body / hand presets). soccerfield_* intentionally omitted.
_POSE_TEMPLATE_ORDER: tuple[tuple[str, str, str, str], ...] = (
    # filename, id, short_label (button), dialog_label
    ("mediapipe_pose33.json", "mediapipe", "MediaPipe33", "MediaPipe Pose (33)"),
    ("yolo_coco17.json", "yolo", "YOLO17", "YOLO COCO-17 (17)"),
    ("openpose_body25.json", "openpose25", "OpenPose25", "OpenPose Body-25 (25)"),
    ("halpe26.json", "halpe26", "Halpe26", "Halpe 26 (26)"),
    ("fifa_body15.json", "fifa15", "FIFA15", "FIFA Body-15 (15)"),
    ("sam3dinov3_mhr70.json", "sam3d70", "SAM3D70", "SAM3+DINOv3 MHR-70 (70)"),
    ("sapiens2_goliath308.json", "sapiens308", "Sapiens308", "Sapiens2 Goliath (308)"),
    ("mediapipe_hand21.json", "hand21", "Hand21", "MediaPipe Hand (21)"),
    ("mediapipe_hands42.json", "hands42", "Hands42", "MediaPipe Hands (42)"),
    ("mediapipe_holistic75.json", "holistic75", "Holistic75", "MediaPipe Holistic (75)"),
    ("coco_wholebody133.json", "wholebody133", "WholeBody133", "COCO WholeBody (133)"),
)

# Special (non-JSON) modes always available in the Tpl dialog.
# Internal id stays ``fifa`` for CLI/TOML/dataset compatibility; UI label is Soccer-Kiki.
SPECIAL_TEMPLATE_IDS: frozenset[str] = frozenset({"free", "fifa"})

SPECIAL_TEMPLATE_LABELS: dict[str, str] = {
    "free": "Free",
    "fifa": "Soccer-Kiki",
}


@dataclass(frozen=True)
class TemplateSpec:
    """One pose / hand skeleton preset loaded from ``vaila/skeletons/``."""

    id: str
    label: str
    dialog_label: str
    path: Path
    num_keypoints: int
    keypoints: tuple[str, ...]
    connections: tuple[tuple[int, int], ...]  # 0-based marker indices
    schema: str = ""


def skeletons_dir() -> Path:
    """Return the on-disk ``vaila/skeletons/`` directory next to this package."""
    return Path(__file__).resolve().parent / "skeletons"


def _parse_p_index(token: str) -> int | None:
    """Convert ``pN`` (1-based) to 0-based index, or None if invalid."""
    match = _P_INDEX_RE.match(str(token).strip())
    if not match:
        return None
    n = int(match.group(1))
    if n < 1:
        return None
    return n - 1


def _parse_connections(
    raw: list[list[str]] | list[tuple[str, str]],
    *,
    num_keypoints: int,
) -> tuple[tuple[int, int], ...]:
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        a = _parse_p_index(pair[0])
        b = _parse_p_index(pair[1])
        if a is None or b is None:
            continue
        if a >= num_keypoints or b >= num_keypoints or a == b:
            continue
        edge = (a, b) if a < b else (b, a)
        if edge in seen:
            continue
        seen.add(edge)
        edges.append((a, b))
    return tuple(edges)


def _load_one(
    path: Path,
    *,
    template_id: str,
    label: str,
    dialog_label: str,
) -> TemplateSpec:
    data = json.loads(path.read_text(encoding="utf-8"))
    keypoints_raw = data.get("keypoints")
    if not isinstance(keypoints_raw, list) or not keypoints_raw:
        raise ValueError(f"{path.name}: missing non-empty 'keypoints' list")
    keypoints = tuple(str(name) for name in keypoints_raw)
    num_kp = int(data.get("num_keypoints") or len(keypoints))
    if num_kp != len(keypoints):
        # Prefer the explicit names list length for slot allocation.
        num_kp = len(keypoints)
    connections_raw = data.get("connections") or []
    if not isinstance(connections_raw, list):
        connections_raw = []
    connections = _parse_connections(connections_raw, num_keypoints=num_kp)
    schema = str(data.get("schema") or "")
    return TemplateSpec(
        id=template_id,
        label=label,
        dialog_label=dialog_label,
        path=path,
        num_keypoints=num_kp,
        keypoints=keypoints,
        connections=connections,
        schema=schema,
    )


@lru_cache(maxsize=1)
def list_pose_templates() -> tuple[TemplateSpec, ...]:
    """Return pose/hand templates in fixed dialog order (soccerfield excluded)."""
    root = skeletons_dir()
    specs: list[TemplateSpec] = []
    for filename, template_id, label, dialog_label in _POSE_TEMPLATE_ORDER:
        path = root / filename
        if not path.is_file():
            continue
        specs.append(
            _load_one(path, template_id=template_id, label=label, dialog_label=dialog_label)
        )
    return tuple(specs)


def get_template(template_id: str) -> TemplateSpec | None:
    """Lookup a pose template by stable id (e.g. ``sam3d70``)."""
    tid = str(template_id or "").strip().lower()
    for spec in list_pose_templates():
        if spec.id == tid:
            return spec
    return None


def template_labels() -> dict[str, str]:
    """Short button captions for Free / Soccer-Kiki / all pose presets."""
    labels = dict(SPECIAL_TEMPLATE_LABELS)
    for spec in list_pose_templates():
        labels[spec.id] = spec.label
    return labels


def dialog_choices() -> list[tuple[str, str]]:
    """``(id, dialog_line)`` rows for the Tpl picker (excludes numbering)."""
    rows: list[tuple[str, str]] = [
        ("free", "Free (variable markers)"),
        ("fifa", "Soccer-Kiki (pitch guide)"),
    ]
    for spec in list_pose_templates():
        rows.append((spec.id, spec.dialog_label))
    return rows


def format_template_dialog_prompt() -> str:
    """Multi-line prompt for ``show_input_dialog`` (one option per line)."""
    lines = ["Tpl — choose skeleton:"]
    for idx, (_tid, label) in enumerate(dialog_choices()):
        lines.append(f"{idx} = {label}")
    return "\n".join(lines)


def resolve_dialog_choice(answer: str | None) -> str | None:
    """Map user dialog input to a template id, or None if cancelled/invalid."""
    if answer is None:
        return None
    raw = str(answer).strip().lower()
    if not raw:
        return None
    choices = dialog_choices()
    # Numeric index.
    if raw.isdigit():
        idx = int(raw)
        if 0 <= idx < len(choices):
            return choices[idx][0]
        return None
    # Direct id or label substring.
    for tid, label in choices:
        if raw == tid or raw == label.lower():
            return tid
        if raw in tid or raw in label.lower():
            return tid
    return None


def clear_catalog_cache() -> None:
    """Drop the cached template list (tests / after regenerating JSONs)."""
    list_pose_templates.cache_clear()


if __name__ == "__main__":
    for i, (tid, label) in enumerate(dialog_choices()):
        print(f"{i:2d}  {tid:14s}  {label}")
    print(f"skeletons_dir={skeletons_dir()}")
    print(f"pose templates loaded={len(list_pose_templates())}")
