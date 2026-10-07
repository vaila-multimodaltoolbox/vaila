"""FreeKiki external datasets: keypoint alignment by voting, GSR loader, commit gates.

Version: 0.4.7
Update Date: 06 October 2026
"""

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from vaila import freekiki

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vaila"))

from kiki49_build import Options  # noqa: E402
from kiki49_build import external as ext  # noqa: E402
from kiki49_build.sample import Sample  # noqa: E402
from test_getpixelvideo_freekiki import _workspace  # noqa: E402
from test_kiki49_build import _soccernet_json, left_camera, projected_kiki  # noqa: E402


def _layout_dataset(cam, perm: np.ndarray, n: int, rng) -> tuple[list, list, list, list]:
    """Project clicks in a permuted layout (project j = kiki perm[j]) + network output."""
    uv, ok = projected_kiki(cam)
    clicks, xs, ks, ws = [], [], [], []
    for _ in range(n):
        jitter = rng.normal(0.0, 0.5, uv.shape)
        c = np.zeros((len(perm), 3))
        for j, k in enumerate(perm):
            if ok[k]:
                c[j] = (*(uv[k] + jitter[k]), 2.0)
        clicks.append(c)
        kc = np.where(ok, 0.9, 0.01)
        kc[[5, 29]] = 0.2  # the network rarely accepts the corners
        xs.append(uv + jitter)
        ks.append(kc)
        ws.append(1280)
    return clicks, xs, ks, ws


def test_vote_mapping_recovers_a_permuted_layout():
    rng = np.random.default_rng(0)
    perm = rng.permutation(32)
    clicks, xs, ks, ws = _layout_dataset(left_camera(), perm, 20, rng)
    mapping, rows = ext.vote_mapping(clicks, xs, ks, ws)
    assert mapping, rows
    for j, k in mapping.items():
        assert perm[j] == k
    assert all(r["agreement"] == 1.0 for r in rows if r["mapped"])


def test_vote_mapping_drops_an_ambiguous_point():
    rng = np.random.default_rng(1)
    perm = np.arange(32)
    clicks, xs, ks, ws = _layout_dataset(left_camera(), perm, 20, rng)
    uv, ok = projected_kiki(left_camera())
    j = next(k for k in range(32) if ok[k] and k not in (5, 29))
    other = next(k for k in range(32) if ok[k] and k not in (5, 29, j))
    for i, c in enumerate(clicks):  # point j clicked on two different kiki points
        c[j, :2] = uv[j] if i % 2 else uv[other]
    mapping, rows = ext.vote_mapping(clicks, xs, ks, ws)
    assert j not in mapping
    assert rows[j]["agreement"] <= 0.5 + 1e-9


def test_geometric_votes_map_points_the_network_misses():
    rng = np.random.default_rng(2)
    perm = np.arange(32)
    cam = left_camera()
    _, ok = projected_kiki(cam)
    clicks, xs, ks, ws = _layout_dataset(cam, perm, 20, rng)
    for kc in ks:
        kc[[1, 2]] = 0.1  # accepted nowhere: only geometry can map them
    mapping, rows = ext.vote_mapping(clicks, xs, ks, ws)
    assert 1 not in mapping and 2 not in mapping
    full = ext.geometric_votes(clicks, mapping, rows)
    assert full[1] == 1 and full[2] == 2
    assert rows[1]["method"] == "geometry"


def test_gsr_loader_keeps_rare_frames_densely(tmp_path):
    cam = left_camera()
    lines = _soccernet_json(cam)
    clip = tmp_path / "train" / "SNGS-001"
    (clip / "img1").mkdir(parents=True)
    images, annotations = [], []
    for n in range(60):
        fid = f"{n + 1:06d}"
        images.append(
            {
                "image_id": fid,
                "file_name": f"{fid}.jpg",
                "width": 1280,
                "height": 720,
                "has_labeled_pitch": True,
            }
        )
        annotations.append({"image_id": fid, "supercategory": "pitch", "lines": lines})
    (clip / "Labels-GameState.json").write_text(
        json.dumps(
            {
                "info": {"game_id": "7", "im_dir": "img1"},
                "images": images,
                "annotations": annotations,
            }
        )
    )
    assert ext.gsr_clips(tmp_path) == [clip]
    got = [s for s, _ in ext.gsr_samples(clip, Options())]
    assert got and all(isinstance(s, Sample) for s in got)
    assert {s.group for s in got} == {"gsr:train:game7"}
    rare_in_view = any(projected_kiki(cam)[1][k] for k in ext.RARE_POINTS)
    # one per second (frames 0, 25, 50), or every 5th frame when a rare point shows
    assert len(got) == (12 if rare_in_view else 3)


def _stage(ws: Path, folder: Path, frames: list[tuple[str, np.ndarray, str]]) -> Path:
    """Minimal staging folder: (name, image, group) frames with a valid label each."""
    for d in ("images", "labels"):
        (folder / d).mkdir(parents=True, exist_ok=True)
    rows = []
    for name, image, group in frames:
        cv2.imwrite(str(folder / "images" / f"{name}.png"), image)
        pts = [[10, 10], [30, 20]] + [None] * 47
        (folder / "labels" / f"{name}.txt").write_text(freekiki.pose_label_line(pts, 48, 32))
        rows.append(
            {
                "split": "train",
                "image": f"images/{name}.png",
                "label": f"labels/{name}.txt",
                "source": "fixture_ext",
                "group": group,
                "origin": "0" * 49,
                "n_visible": "2",
                "aux3d": "0",
                "qa_score": "1.0",
            }
        )
    with (folder / freekiki.EXTERNAL_ROWS).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=freekiki.EXTERNAL_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return folder


def test_commit_external_drops_leaks_and_duplicates(tmp_path):
    ws = _workspace(tmp_path / "workspace")
    ds = ws / "datasets/kiki49"
    val_img = cv2.imread(str(ds / "images/val/val.png"))
    rng = np.random.default_rng(9)
    fresh = rng.integers(0, 255, (32, 48, 3), dtype=np.uint8)
    other = rng.integers(0, 255, (32, 48, 3), dtype=np.uint8)
    src = _stage(
        ws,
        tmp_path / "staging",
        [
            ("dup_of_val", val_img, "NEW1"),  # near-duplicate of a val image
            ("held_group", other, "test"),  # group reserved by the test split
            ("good", fresh, "NEW2"),
            ("good_again", fresh.copy(), "NEW3"),  # duplicate of an earlier staged frame
        ],
    )
    before = (ds / "manifest.csv").read_text()
    report = freekiki.commit_external(ws, src)
    assert report["frames"] == 1 and not report["committed"]
    assert report["dropped"] == {
        "group_reserved_by_val_test_hard": 1,
        "near_dup_val_test_hard": 1,
        "near_dup_staged": 1,
    }
    assert (ds / "manifest.csv").read_text() == before  # preview writes nothing

    report = freekiki.commit_external(ws, src, commit=True)
    assert report["committed"]
    with (ds / "manifest.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["source"] == "fixture_ext"]
    assert [r["split"] for r in rows] == ["train"]
    assert (ds / rows[0]["image"]).is_file() and (ds / rows[0]["label"]).is_file()
    again = freekiki.commit_external(ws, src)  # safe re-run
    assert again["frames"] == 0 and again["dropped"]["already_registered"] == 1


def test_extend_cli_parses_stage_and_commit():
    p = freekiki.build_parser()
    a = p.parse_args(
        ["extend", "-w", "W", "--source", "roboflow", "--project", "a/b@3", "--project", "c/d"]
    )
    assert a.source == "roboflow" and a.project == ["a/b@3", "c/d"] and not a.commit
    a = p.parse_args(["extend", "-w", "W", "--src", "S", "--commit"])
    assert a.src == "S" and a.commit
    with pytest.raises(SystemExit):
        p.parse_args(["extend", "-w", "W", "--source", "kaggle"])
