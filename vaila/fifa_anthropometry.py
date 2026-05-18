"""Anthropometric helpers for the FIFA skeletal baseline.

The baseline only consumes ``boxes[frame, person, 4]`` and skeleton tensors, so
this module keeps the integration lightweight:

1. Resolve the match-specific anthropometric CSV for a sequence.
2. Assign player rows to person slots using long-term box height ranking.
3. Derive a conservative uniform scale factor from segment-length priors.

This does not replace player identity tracking. It provides a reproducible hook
for anthropometric regularization when per-match player tables are available.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

TEAM_CODE_TO_CANONICAL: dict[str, str] = {
    "ARG": "argentina",
    "BRA": "brazil",
    "CRO": "croatia",
    "ENG": "england",
    "FRA": "france",
    "KOR": "south korea",
    "MOR": "morocco",
    "NET": "netherlands",
    "POR": "portugal",
}

TEAM_ALIASES: dict[str, set[str]] = {
    "argentina": {"argentina"},
    "brazil": {"brazil", "brasil"},
    "croatia": {"croacia", "croatia"},
    "england": {"england", "inglaterra"},
    "france": {"franca", "france"},
    "morocco": {"marrocos", "morocco"},
    "netherlands": {"holanda", "netherlands", "the netherlands"},
    "portugal": {"portugal"},
    "south korea": {
        "coreia do sul",
        "korea republic",
        "republic of korea",
        "south korea",
        "southkorea",
    },
}

CSV_COLUMN_ALIASES: dict[str, str] = {
    "nome": "player_name",
    "player": "player_name",
    "pais": "team",
    "team": "team",
    "numero": "number",
    "number": "number",
    "altura_m": "height_m",
    "height_m": "height_m",
    "height_cm": "height_cm",
    "envergadura_m": "wingspan_m",
    "coxa_m": "thigh_m",
    "perna_m": "shank_m",
    "tronco_m": "trunk_m",
    "braco_m": "upper_arm_m",
    "antebraco_m": "forearm_m",
}

RIGHT_SHOULDER = 1
LEFT_SHOULDER = 2
RIGHT_ELBOW = 3
LEFT_ELBOW = 4
RIGHT_WRIST = 5
LEFT_WRIST = 6
RIGHT_HIP = 7
LEFT_HIP = 8
RIGHT_KNEE = 9
LEFT_KNEE = 10
RIGHT_ANKLE = 11
LEFT_ANKLE = 12


@dataclass(frozen=True)
class AnthropometricProfile:
    """Per-slot anthropometric target assigned to one tracked person slot."""

    slot_index: int
    player_name: str
    team: str
    number: str
    source_csv: Path
    height_m: float | None = None
    wingspan_m: float | None = None
    thigh_m: float | None = None
    shank_m: float | None = None
    trunk_m: float | None = None
    upper_arm_m: float | None = None
    forearm_m: float | None = None
    slot_box_height_px: float | None = None


def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def _normalize_token(text: str) -> str:
    text = _strip_accents(str(text)).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_column_key(text: str) -> str:
    text = _strip_accents(str(text)).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def canonical_team_name(name: str) -> str:
    token = _normalize_token(name)
    for canonical, aliases in TEAM_ALIASES.items():
        if token in aliases:
            return canonical
    return token


def sequence_team_names(sequence: str) -> tuple[str, str] | None:
    parts = sequence.upper().split("_")
    if len(parts) < 2:
        return None
    a = TEAM_CODE_TO_CANONICAL.get(parts[0])
    b = TEAM_CODE_TO_CANONICAL.get(parts[1])
    if a is None or b is None:
        return None
    return a, b


def _candidate_csv_paths(csv_dir: Path) -> list[Path]:
    full = sorted(csv_dir.glob("*_full.csv"))
    coarse = sorted(csv_dir.glob("*_2022.csv"))
    return full + coarse


@lru_cache(maxsize=256)
def _load_csv_table(csv_path_str: str) -> pd.DataFrame:
    csv_path = Path(csv_path_str)
    df = pd.read_csv(csv_path)
    renamed = {
        col: CSV_COLUMN_ALIASES.get(_normalize_column_key(col), _normalize_column_key(col))
        for col in df.columns
    }
    df = df.rename(columns=renamed).copy()

    for col in (
        "height_m",
        "height_cm",
        "wingspan_m",
        "thigh_m",
        "shank_m",
        "trunk_m",
        "upper_arm_m",
        "forearm_m",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "height_m" not in df.columns and "height_cm" in df.columns:
        df["height_m"] = df["height_cm"] / 100.0
    elif "height_m" in df.columns and "height_cm" in df.columns:
        df["height_m"] = df["height_m"].fillna(df["height_cm"] / 100.0)

    if "player_name" not in df.columns:
        df["player_name"] = ""
    if "team" not in df.columns:
        df["team"] = ""
    if "number" not in df.columns:
        df["number"] = ""

    df["team_canonical"] = df["team"].map(canonical_team_name)
    df["player_name"] = df["player_name"].astype(str).fillna("")
    df["number"] = df["number"].astype(str).fillna("")
    return df


def resolve_sequence_csv(sequence: str, csv_dir: Path) -> Path | None:
    teams = sequence_team_names(sequence)
    if teams is None:
        return None
    target = frozenset(teams)
    csv_dir = csv_dir.expanduser().resolve()
    if not csv_dir.is_dir():
        return None

    for csv_path in _candidate_csv_paths(csv_dir):
        df = _load_csv_table(str(csv_path))
        present = frozenset(
            team for team in df["team_canonical"].dropna().astype(str).tolist() if team
        )
        if present == target:
            return csv_path
    return None


def _nanmedian_or_none(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def _box_slot_heights(boxes: np.ndarray) -> np.ndarray:
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"Expected boxes[F, P, 4], got {boxes.shape}")
    heights = boxes[..., 3] - boxes[..., 1]
    heights[~np.isfinite(heights)] = np.nan
    return np.nanmedian(heights, axis=0)


def assign_profiles_to_slots(
    boxes: np.ndarray,
    table: pd.DataFrame,
    source_csv: Path,
) -> list[AnthropometricProfile | None]:
    """Assign player rows to slots by descending long-term box height."""
    num_slots = int(boxes.shape[1])
    slot_heights = _box_slot_heights(boxes)
    profiles: list[AnthropometricProfile | None] = [None] * num_slots

    usable = table.copy()
    usable = usable[
        usable[
            [
                col
                for col in (
                    "height_m",
                    "thigh_m",
                    "shank_m",
                    "trunk_m",
                    "upper_arm_m",
                    "forearm_m",
                )
                if col in usable.columns
            ]
        ]
        .notna()
        .any(axis=1)
    ].copy()
    if usable.empty:
        return profiles

    usable["sort_height_m"] = pd.to_numeric(usable.get("height_m"), errors="coerce").fillna(0.0)
    usable = usable.sort_values(
        by=["sort_height_m", "player_name", "number"],
        ascending=[False, True, True],
        kind="mergesort",
    )

    valid_slots = [idx for idx, h in enumerate(slot_heights.tolist()) if np.isfinite(h) and h > 0]
    valid_slots.sort(key=lambda idx: float(slot_heights[idx]), reverse=True)

    for slot_idx, row in zip(valid_slots, usable.itertuples(index=False), strict=False):
        profiles[slot_idx] = AnthropometricProfile(
            slot_index=slot_idx,
            player_name=str(getattr(row, "player_name", "")),
            team=str(getattr(row, "team", "")),
            number=str(getattr(row, "number", "")),
            source_csv=source_csv,
            height_m=_nan_to_none(getattr(row, "height_m", np.nan)),
            wingspan_m=_nan_to_none(getattr(row, "wingspan_m", np.nan)),
            thigh_m=_nan_to_none(getattr(row, "thigh_m", np.nan)),
            shank_m=_nan_to_none(getattr(row, "shank_m", np.nan)),
            trunk_m=_nan_to_none(getattr(row, "trunk_m", np.nan)),
            upper_arm_m=_nan_to_none(getattr(row, "upper_arm_m", np.nan)),
            forearm_m=_nan_to_none(getattr(row, "forearm_m", np.nan)),
            slot_box_height_px=float(slot_heights[slot_idx]),
        )
    return profiles


def load_sequence_profiles(
    sequence: str,
    boxes: np.ndarray,
    csv_dir: Path,
) -> tuple[list[AnthropometricProfile | None] | None, Path | None]:
    csv_path = resolve_sequence_csv(sequence, csv_dir)
    if csv_path is None:
        return None, None
    table = _load_csv_table(str(csv_path))
    return assign_profiles_to_slots(boxes, table, csv_path), csv_path


def _nan_to_none(value: object) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    return val


def _joint_distance(skel: np.ndarray, a: int, b: int) -> float | None:
    pa = skel[a]
    pb = skel[b]
    if not np.isfinite(pa).all() or not np.isfinite(pb).all():
        return None
    dist = float(np.linalg.norm(pa - pb))
    if dist <= 0:
        return None
    return dist


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None and np.isfinite(v)]
    if not valid:
        return None
    return float(np.mean(valid))


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    return (a + b) / 2.0


def _anchor_point(skel: np.ndarray) -> np.ndarray | None:
    hip_center = _midpoint(skel[RIGHT_HIP], skel[LEFT_HIP])
    if hip_center is not None:
        return hip_center
    valid = skel[np.isfinite(skel).all(axis=1)]
    if len(valid) == 0:
        return None
    return valid.mean(axis=0)


def estimate_segment_lengths(skel: np.ndarray) -> dict[str, float | None]:
    shoulder_center = _midpoint(skel[RIGHT_SHOULDER], skel[LEFT_SHOULDER])
    hip_center = _midpoint(skel[RIGHT_HIP], skel[LEFT_HIP])
    trunk_m = None
    if shoulder_center is not None and hip_center is not None:
        trunk_m = float(np.linalg.norm(shoulder_center - hip_center))

    return {
        "thigh_m": _mean_or_none(
            [
                _joint_distance(skel, RIGHT_HIP, RIGHT_KNEE),
                _joint_distance(skel, LEFT_HIP, LEFT_KNEE),
            ]
        ),
        "shank_m": _mean_or_none(
            [
                _joint_distance(skel, RIGHT_KNEE, RIGHT_ANKLE),
                _joint_distance(skel, LEFT_KNEE, LEFT_ANKLE),
            ]
        ),
        "upper_arm_m": _mean_or_none(
            [
                _joint_distance(skel, RIGHT_SHOULDER, RIGHT_ELBOW),
                _joint_distance(skel, LEFT_SHOULDER, LEFT_ELBOW),
            ]
        ),
        "forearm_m": _mean_or_none(
            [
                _joint_distance(skel, RIGHT_ELBOW, RIGHT_WRIST),
                _joint_distance(skel, LEFT_ELBOW, LEFT_WRIST),
            ]
        ),
        "trunk_m": trunk_m,
    }


def estimate_height_like(skel: np.ndarray) -> float | None:
    seg = estimate_segment_lengths(skel)
    parts = [seg["trunk_m"], seg["thigh_m"], seg["shank_m"]]
    valid = [p for p in parts if p is not None and np.isfinite(p)]
    if len(valid) < 2:
        return None
    return float(sum(valid))


def compute_profile_scale(
    skel: np.ndarray,
    profile: AnthropometricProfile,
    *,
    min_scale: float = 0.75,
    max_scale: float = 1.35,
) -> float:
    ratios: list[float] = []
    current = estimate_segment_lengths(skel)
    for key in ("thigh_m", "shank_m", "trunk_m", "upper_arm_m", "forearm_m"):
        target = getattr(profile, key)
        observed = current.get(key)
        if target is None or observed is None or observed <= 1e-8:
            continue
        ratios.append(float(target / observed))

    if not ratios and profile.height_m is not None:
        current_height = estimate_height_like(skel)
        if current_height is not None and current_height > 1e-8:
            ratios.append(float(profile.height_m / current_height))

    if not ratios:
        return 1.0

    scale = float(np.median(np.asarray(ratios, dtype=np.float64)))
    if not np.isfinite(scale):
        return 1.0
    return float(np.clip(scale, min_scale, max_scale))


def scale_skeleton_to_profile(
    skel: np.ndarray,
    profile: AnthropometricProfile | None,
    *,
    min_scale: float = 0.75,
    max_scale: float = 1.35,
) -> tuple[np.ndarray, float]:
    if profile is None:
        return skel, 1.0
    anchor = _anchor_point(skel)
    if anchor is None:
        return skel, 1.0
    scale = compute_profile_scale(skel, profile, min_scale=min_scale, max_scale=max_scale)
    if abs(scale - 1.0) < 1e-6:
        return skel, 1.0
    scaled = (skel - anchor) * scale + anchor
    return scaled.astype(np.float32), scale
