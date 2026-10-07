"""
================================================================================
Script: external.py - external soccer-field datasets to kiki49 samples
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.7
Created: 06 October 2026
Update Date: 06 October 2026

Description:
    Loaders that turn external field datasets into kiki49 ``Sample`` objects
    (``freekiki.py extend`` stages and commits them):

    * Roboflow Universe keypoint projects (COCO export). Their keypoint layout
      is unknown and their names unreliable, so every project keypoint is
      aligned to a kiki point by **voting**: the FreeKiki network predicts the
      49 points on a subset of images and each project keypoint votes for the
      accepted prediction nearest to it. Only consistent, one-to-one votes map.
      Square exports are usually 16:9 broadcast frames stretched by Roboflow;
      the aspect whose fitted camera is the most pinhole-like is used.
    * SoccerNet Game State Reconstruction (``gamestate-2024``): per-frame pitch
      line annotations, converted like SoccerNet calibration.
    * SoccerNet calibration images rejected by the kiki49 build, rescued with
      the network + fitted camera and accepted only when the human line
      annotations agree with the projected field lines.

    Pure functions over arrays and files; the network runs in ``freekiki``.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import Options
from .camera import apply_h, fit_homography
from .clicks import clicks_to_sample
from .kiki49 import AUX3D_POINTS, N_KPT, load_kiki49
from .sample import Reject, Sample

RARE_POINTS = (5, 16, 29, 39, 47)
NON_SOCCER = (
    "rugby",
    "cricket",
    "handball",
    "futsal",
    "goalball",
    "basket",
    "hockey",
    "tennis",
    "volley",
    "padel",
)
VOTE_RADIUS_PX = 12.0  # px at 1920: project click vs network prediction
MIN_VOTES = 10
MIN_AGREEMENT = 0.7
MIN_MAPPED = 6
GEOM_RADIUS_M = 1.5  # metres on the pitch: back-projected click vs kiki point
GEOM_MIN_VOTES = 3
GEOM_MIN_AGREEMENT = 0.8
MAPPING_CSV = "mapping.csv"


# --------------------------------------------------------------------------- #
# Roboflow COCO exports
# --------------------------------------------------------------------------- #
@dataclass
class Clicked:
    """One annotated image of an external dataset (clicks in the project layout)."""

    image: Path
    width: int
    height: int
    clicks: np.ndarray  # (K, 3) pixels + visibility (0 = not clicked)
    name: str  # original file name (before Roboflow's ``.rf.<hash>`` suffix)


def is_soccer_project(result: dict) -> bool:
    text = " ".join(str(result.get(k, "")) for k in ("name", "url", "annotation")).lower()
    return result.get("type") == "keypoint-detection" and not any(w in text for w in NON_SOCCER)


def read_coco_clicks(export_dir: Path) -> tuple[list[Clicked], list[str]]:
    """Every split of a Roboflow COCO keypoint export -> (images, keypoint names).

    An image with several instances keeps the one with most clicked points.
    """
    out: list[Clicked] = []
    names: list[str] = []
    for ann_path in sorted(Path(export_dir).glob("*/_annotations.coco.json")):
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        cats = [c for c in data.get("categories", []) if c.get("keypoints")]
        if not cats:
            continue
        names = names or [str(n) for n in cats[0]["keypoints"]]
        k = len(names)
        best: dict[int, np.ndarray] = {}
        for a in data.get("annotations", []):
            kp = np.asarray(a.get("keypoints") or [], dtype=float)
            if kp.size != 3 * k:
                continue
            kp = kp.reshape(k, 3)
            kp[kp[:, 2] <= 0] = 0.0
            prev = best.get(a["image_id"])
            if prev is None or (kp[:, 2] > 0).sum() > (prev[:, 2] > 0).sum():
                best[a["image_id"]] = kp
        for im in data.get("images", []):
            kp = best.get(im["id"])
            if kp is None or not (kp[:, 2] > 0).any():
                continue
            original = (im.get("extra") or {}).get("name") or im["file_name"].split(".rf.")[0]
            out.append(
                Clicked(
                    image=ann_path.parent / im["file_name"],
                    width=int(im["width"]),
                    height=int(im["height"]),
                    clicks=kp,
                    name=str(original),
                )
            )
    return out, names


def roboflow_group(project: str, name: str) -> str:
    """Match/clip group of an image: the 6-hex clip id of the Roboflow sports
    footage (shared with the ``martinjolif`` source, so split reservations hold),
    else the project."""
    m = re.match(r"([0-9a-f]{6})_", Path(name).name)
    if m:
        return f"martinjolif:{m.group(1)}"
    return f"rf:{project}"


# --------------------------------------------------------------------------- #
# Keypoint alignment by voting
# --------------------------------------------------------------------------- #
def vote_mapping(
    clicks: list[np.ndarray],
    pred_xy: list[np.ndarray],
    pred_kc: list[np.ndarray],
    widths: list[int],
    *,
    kp_conf: float = 0.5,
    radius_px: float = VOTE_RADIUS_PX,
    min_votes: int = MIN_VOTES,
    min_agreement: float = MIN_AGREEMENT,
) -> tuple[dict[int, int], list[dict]]:
    """Map project keypoints to kiki indices from network predictions.

    ``clicks[i]`` is (K, 3) for image i; ``pred_xy[i]`` (49, 2) and
    ``pred_kc[i]`` (49,) are the network's raw output on the same image
    (same pixel frame). Each clicked project point votes for the accepted
    prediction nearest to it within ``radius_px`` (px at 1920). A point maps
    when it has ``min_votes`` votes and its best kiki index holds
    ``min_agreement`` of them; two points claiming one kiki index keep the
    stronger. Returns ({j: k}, one table row per project keypoint).
    """
    k_proj = clicks[0].shape[0] if clicks else 0
    votes: dict[int, Counter] = defaultdict(Counter)
    seen = Counter()
    for c, xy, kc, w in zip(clicks, pred_xy, pred_kc, widths, strict=True):
        if xy is None or kc is None:
            continue
        ok = np.nan_to_num(np.asarray(kc, dtype=float), nan=-1.0) >= kp_conf
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        scale = 1920.0 / float(w)
        for j in np.flatnonzero(c[:, 2] > 0):
            seen[int(j)] += 1
            d = np.hypot(*(xy[idx] - c[j, :2]).T) * scale
            m = int(np.argmin(d))
            if d[m] <= radius_px:
                votes[int(j)][int(idx[m])] += 1
    rows, claims = [], {}
    for j in range(k_proj):
        total = sum(votes[j].values())
        k, n = votes[j].most_common(1)[0] if total else (-1, 0)
        agree = n / total if total else 0.0
        row = {
            "j": j,
            "clicked": seen[j],
            "votes": total,
            "k": k,
            "agreement": round(agree, 3),
            "mapped": total >= min_votes and agree >= min_agreement,
            "method": "network",
        }
        rows.append(row)
        if row["mapped"] and (k not in claims or n > claims[k][1]):
            claims[k] = (j, n)
    winners = {j for j, _ in claims.values()}
    mapping = {}
    for row in rows:
        if row["mapped"] and row["j"] not in winners:
            row["mapped"] = False  # lost a one-to-one conflict
        if row["mapped"]:
            mapping[row["j"]] = row["k"]
    return mapping, rows


def geometric_votes(
    clicks: list[np.ndarray],
    mapping: dict[int, int],
    rows: list[dict],
    *,
    radius_m: float = GEOM_RADIUS_M,
    min_votes: int = GEOM_MIN_VOTES,
    min_agreement: float = GEOM_MIN_AGREEMENT,
) -> dict[int, int]:
    """Extend ``mapping`` to points the network rarely accepts (the rare ones).

    Per image, the already mapped ground clicks fit a homography; every
    unmapped clicked point is sent back to the pitch and votes for the nearest
    free ground kiki point within ``radius_m``. Updates ``rows`` in place
    (``method`` = ``geometry``) and returns the extended mapping.
    """
    xyz = load_kiki49().xyz
    ground = np.flatnonzero(load_kiki49().planar)
    ground = np.array([k for k in ground if k not in AUX3D_POINTS])
    free = np.array([k for k in ground if k not in set(mapping.values())])
    if not len(free):
        return dict(mapping)
    votes: dict[int, Counter] = defaultdict(Counter)
    for c in clicks:
        fit = [(j, k) for j, k in mapping.items() if c[j, 2] > 0 and k in set(ground)]
        if len(fit) < 5:
            continue
        js, ks = zip(*fit, strict=True)
        H, _ = fit_homography(xyz[list(ks), :2], c[list(js), :2], 3.0)
        if H is None:
            continue
        Hinv = np.linalg.inv(H)
        for j in np.flatnonzero(c[:, 2] > 0):
            if int(j) in mapping:
                continue
            world = apply_h(Hinv, c[j : j + 1, :2])[0]
            d = np.hypot(*(xyz[free, :2] - world).T)
            m = int(np.argmin(d))
            if d[m] <= radius_m:
                votes[int(j)][int(free[m])] += 1
    out = dict(mapping)
    claimed = set(out.values())
    for row in rows:
        j = row["j"]
        if j in out or not votes[j]:
            continue
        k, n = votes[j].most_common(1)[0]
        total = sum(votes[j].values())
        if n >= min_votes and n / total >= min_agreement and k not in claimed:
            out[j] = k
            claimed.add(k)
            row.update(k=k, votes=total, agreement=round(n / total, 3), mapped=True)
            row["method"] = "geometry"
    return out


def write_mapping(path: Path, rows: list[dict], names: list[str]) -> None:
    kiki = load_kiki49().names
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["j", "name", "clicked", "votes", "k", "kiki_name", "agreement", "method", "mapped"]
        )
        for r in rows:
            k = int(r["k"])
            w.writerow(
                [
                    r["j"],
                    names[r["j"]] if r["j"] < len(names) else "",
                    r["clicked"],
                    r["votes"],
                    k,
                    kiki[k] if 0 <= k < N_KPT else "",
                    r["agreement"],
                    r.get("method", "") if r["mapped"] else "",
                    int(bool(r["mapped"])),
                ]
            )


def read_mapping(path: Path) -> dict[int, int]:
    """``{j: k}`` of the rows marked mapped (the user may edit the file)."""
    with Path(path).open(encoding="utf-8") as f:
        return {
            int(r["j"]): int(r["k"])
            for r in csv.DictReader(f)
            if str(r.get("mapped", "")).strip() in ("1", "true", "True")
        }


def mapped_clicks(clicks: np.ndarray, mapping: dict[int, int], sx: float = 1.0) -> np.ndarray:
    """Project-layout clicks -> (49, 3) kiki clicks; x scaled by ``sx`` (un-stretch)."""
    ann = np.zeros((N_KPT, 3))
    for j, k in mapping.items():
        if clicks[j, 2] > 0:
            ann[k] = (clicks[j, 0] * sx, clicks[j, 1], 2.0)
    return ann


# --------------------------------------------------------------------------- #
# Stretched exports
# --------------------------------------------------------------------------- #
def load_image(item: Clicked, aspect: float) -> tuple[np.ndarray | None, float]:
    """Image resized to ``aspect`` (width changes) and the x scale applied."""
    img = cv2.imread(str(item.image))
    if img is None:
        return None, 1.0
    h, w = img.shape[:2]
    new_w = int(round(h * aspect))
    if new_w == w:
        return img, 1.0
    return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_CUBIC), new_w / w


def pick_aspect(export_dir: Path, items: list[Clicked]) -> tuple[float, str]:
    """Original aspect of an export (images are resized back to it).

    Roboflow writes its preprocessing in ``README.roboflow.txt``; a square
    "Resize to NxN (Stretch)" export of broadcast footage was 16:9. Exports
    that were not stretched keep their own aspect.
    """
    if not items:
        return 1.0, "empty"
    native = items[0].width / items[0].height
    readme = Path(export_dir) / "README.roboflow.txt"
    text = readme.read_text(encoding="utf-8", errors="replace") if readme.is_file() else ""
    m = re.search(r"Resize to (\d+)x(\d+) \((\w+)", text)
    if m and m.group(3).lower() == "stretch" and abs(native - 1.0) < 0.02:
        return 16 / 9, f"README: {m.group(0)}) -> 16:9"
    return native, f"README: {m.group(0) + ')' if m else 'no resize'} -> native"


def roboflow_samples(
    project: str,
    items: list[Clicked],
    mapping: dict[int, int],
    aspect: float,
    opts: Options,
    source: str = "roboflow",
):
    """Yield ``(Sample | Reject, image)`` for every clicked image; the image array
    is given only when it was resized (otherwise the file is copied as is)."""
    tag = project.replace("/", "__")
    for it in items:
        uid = f"{tag}__{Path(it.name).stem}"
        img, sx = load_image(it, aspect)
        if img is None:
            yield Reject(source, uid, "unreadable_image"), None
            continue
        ann = mapped_clicks(it.clicks, mapping, sx)
        if int((ann[:, 2] > 0).sum()) < 4:
            yield Reject(source, uid, "few_mapped_clicks"), None
            continue
        s = clicks_to_sample(
            source, uid, "train", roboflow_group(project, it.name), ann, img, it.image, opts
        )
        yield s, (img if isinstance(s, Sample) and sx != 1.0 else None)


# --------------------------------------------------------------------------- #
# SoccerNet Game State Reconstruction (gamestate-2024)
# --------------------------------------------------------------------------- #
GSR_SPLITS = ("train", "valid", "test")  # "challenge" has no labels
GSR_EVERY = 25  # 1 frame per second at 25 fps
GSR_RARE_EVERY = 5  # denser where a rare point is in view


def gsr_clips(root: Path) -> list[Path]:
    """Clip folders (``<split>/SNGS-xxx``) that carry a ``Labels-GameState.json``."""
    return sorted(
        p.parent for s in GSR_SPLITS for p in (Path(root) / s).glob("*/Labels-GameState.json")
    )


def gsr_samples(clip: Path, opts: Options, source: str = "soccernet_gsr"):
    """Yield ``(Sample | Reject, None)`` for the sampled frames of one GSR clip.

    Every ``GSR_EVERY``-th labelled frame is kept, and every
    ``GSR_RARE_EVERY``-th one whose labels show a rare point (bottom corners,
    bottom flags, bottom midfield). Group = split + game id (the GSR files do
    not name the SoccerNet match).
    """
    from .soccernet import sample_from_lines

    data = json.loads((clip / "Labels-GameState.json").read_text(encoding="utf-8"))
    info = data.get("info", {})
    split = clip.parent.name
    group = f"gsr:{split}:game{info.get('game_id', clip.name)}"
    pitch = {
        a["image_id"]: a["lines"]
        for a in data.get("annotations", [])
        if a.get("supercategory") == "pitch" and a.get("lines")
    }
    im_dir = clip / str(info.get("im_dir", "img1"))
    for n, im in enumerate(data.get("images", [])):
        if n % GSR_RARE_EVERY or not im.get("has_labeled_pitch"):
            continue
        lines = pitch.get(im["image_id"])
        if lines is None:
            continue
        if isinstance(lines, str):  # some releases store the dict as its repr
            import ast

            lines = ast.literal_eval(lines)
        uid = f"{split}_{clip.name}_{Path(im['file_name']).stem}"
        s = sample_from_lines(
            lines,
            int(im["width"]),
            int(im["height"]),
            source=source,
            uid=uid,
            split="train",
            group=group,
            image=im_dir / im["file_name"],
            opts=opts,
        )
        if n % GSR_EVERY and not (
            isinstance(s, Sample) and any(s.kps[k, 2] > 0 for k in RARE_POINTS)
        ):
            continue
        yield s, None


# --------------------------------------------------------------------------- #
# SoccerNet calibration images the kiki49 build rejected
# --------------------------------------------------------------------------- #
RESCUE_REASONS = ("no_homography", "underdetermined", "residual")
RESCUE_MIN_SEED = 4


def rescue_tasks(root: Path, limit: int | None = None) -> list[tuple[Path, str]]:
    """Official **train** calibration-2023 images with their match group."""
    d = Path(root) / "train"
    info = json.loads((d / "match_info.json").read_text(encoding="utf-8"))
    out = []
    for j in sorted(d.glob("*.json")):
        if not j.stem.isdigit() or not j.with_suffix(".jpg").exists():
            continue
        m = info.get(f"{j.stem}.jpg", {})
        group = "|".join(str(m.get(k, "")).strip() for k in ("league", "season", "match", "date"))
        out.append((j, group or f"unknown:train:{j.stem}"))
        if limit is not None and len(out) >= limit:
            break
    return out


def rescue_sample(
    task: tuple[Path, str],
    predict,
    opts: Options,
    *,
    kp_conf: float = 0.5,
    source: str = "soccernet_rescue",
):
    """``(Sample | Reject, None)`` for one rejected calibration image, else None.

    ``predict(image BGR) -> (xy (49, 2) | None, conf (49,) | None)`` is the
    FreeKiki network. Its accepted ground points anchor the plane; the human
    lines must agree with it (the usual SoccerNet residual gates) and the
    projected field must sit on painted lines (line support >= tau).
    """
    from .line_support import line_distance_map, support_score
    from .sample import project_plane
    from .soccernet import sample_from_lines

    j, group = task
    uid = f"train_{j.stem}"
    ann = json.loads(j.read_text(encoding="utf-8"))
    img = cv2.imread(str(j.with_suffix(".jpg")))
    if img is None:
        return Reject(source, uid, "unreadable_image"), None
    height, width = img.shape[:2]
    image = j.with_suffix(".jpg")

    def convert(seed: dict | None) -> Sample | Reject:
        return sample_from_lines(
            ann,
            width,
            height,
            source=source,
            uid=uid,
            split="train",
            group=group,
            image=image,
            opts=opts,
            seed_points=seed,
        )

    first = convert(None)
    if isinstance(first, Sample) or first.reason not in RESCUE_REASONS:
        return None  # accepted by the build already, or rejected for another reason
    xy, kc = predict(img)
    if xy is None or kc is None:
        return Reject(source, uid, "no_network_points"), None
    planar = load_kiki49().planar
    ok = (np.nan_to_num(np.asarray(kc, dtype=float), nan=-1.0) >= kp_conf) & planar
    ok[list(AUX3D_POINTS)] = False
    seed = {int(k): np.asarray(xy[k], dtype=float) for k in np.flatnonzero(ok)}
    if len(seed) < RESCUE_MIN_SEED:
        return Reject(source, uid, "few_network_points"), None
    s = convert(seed)
    if isinstance(s, Reject):
        return s, None
    H = s.H
    assert H is not None
    qa = support_score(line_distance_map(img), lambda X: project_plane(H, X[:, :2], width, height))
    s.stats["support"] = qa
    s.stats["rescued_from"] = float(RESCUE_REASONS.index(first.reason))
    if not qa >= opts.tau:
        return Reject(source, uid, "line_support", s.stats), None
    return s, None
