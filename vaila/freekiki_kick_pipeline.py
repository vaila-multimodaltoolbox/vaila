"""
================================================================================
Script: freekiki_kick_pipeline.py - one click: the whole kick analysis from two cameras
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.15
Created: 28 October 2026
Update Date: 09 November 2026

Description:
    ``freekiki.py kick-analysis`` (and the One-click button of section 5) turns two cut videos of
    one kick, with their corrected ball tracks, into one new folder with the whole analysis:

    * E1 per camera: the field keypoints screened (the camera must be one the network was trained
      for), the DLT3D of every frame, SAM 3 + DINOv3 (SAM 3D Body) with the corrected ball;
    * E2 the people paired between the cameras (``players-link``): one ``player_id`` per person;
    * E3 the ball in metres from the two cameras (``rec3d-binocular``);
    * E4 the keypoints of the paired people triangulated with the DLT3D of each frame (the Vitruvian
      refit on the weak frames);
    * E5 the kicker's metrics (:mod:`freekiki_kicker`);
    * E6 the press report of the binocular fit (``report_tv``, as before);
    * E7 ``README.txt`` with the parameters, the sync, the contact and the limits.

    Each stage writes its state in ``pipeline_status.json``. A stage that finished with the same
    inputs is reused when the folder is run again (``reuse``), so an interrupted run continues from
    the last finished stage. A camera that the screen rejects stops the run with a message; the
    pipeline never drops a camera silently.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from .freekiki_stereo import equalizer_videos, nominal_clock, parse_sync_name, video_sync
except ImportError:
    from freekiki_stereo import (  # ty: ignore[unresolved-import]
        equalizer_videos,
        nominal_clock,
        parse_sync_name,
        video_sync,
    )

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
STATUS_NAME = "pipeline_status.json"
KICK_FOOT_CHOICES = ("auto", "left", "right")
SHOT_OUTCOME_CHOICES = ("auto", "goal", "save", "miss", "woodwork")
# the pynalty metrics shown in the README of a penalty run (the rest is in data/penalty_metrics.csv)
PENALTY_KEEP = ("shot_outcome", "vel_kmh", "dist", "coord_x", "coord_z", "gk_response_time",
                "gk_dive_time_s", "gk_vel_ms")  # fmt: skip
# the ball path of a penalty (freekiki_penalty.penalty_path); a change here reruns the penalty stage
PENALTY_MODEL = "save-v12"  # v12: speed summary (sub-frame time base, one mean speed); v11: GK goal-plane 2D-ray pose, geodesic crouch stature, save anchor
# the kicker stage: a change here reruns it (the contact of the fitted flight, the goal side)
PLANAR_KINDS = ("open_play", "dribble", "pass")  # no shot at goal: no goal (flight) report
REPORT_SKIPPED = "skipped: no shot at goal (dribble / pass: freekiki.py openplay-report)"
REPORT_AFTER_PENALTY = "rendered after the penalty metrics it quotes"
KICKER_MODEL = "contact-fit-v2"  # v2: a panning camera reads the marker contact on the pitch
# the only people a penalty reads: ID0 (the goalkeeper) and ID1 (the kicker), as in player_map.csv
PENALTY_PLAYERS = ("keeper", "kicker")
FPS_TOLERANCE = 0.01  # the two cameras must share a frame rate (the clocks are frame / fps)


class KickInputError(ValueError):
    """An input the pipeline cannot use; the message says what to change."""


class KickStageError(RuntimeError):
    """A stage failed; the message names the stage and the cause."""


def _module(name: str):
    """A sibling module, imported when it is needed (package import first, then the plain name)."""
    try:
        return importlib.import_module(f".{name}", __package__)
    except (ImportError, TypeError):
        return importlib.import_module(name)


def _log(message: str) -> None:
    print(f"[freekiki kick-analysis] {message}", flush=True)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def cut_videos_in(folder: Path) -> list[Path]:
    """The cut videos of a folder: the video files whose names carry a sync (``_sync_N_frames_A_to_B``).

    A folder written by the event equalizer (``event_equalizer.json``) has no sync in the names: its
    two videos share one grid and the contact frame of the manifest is the sync (reference camera
    first).
    """
    named = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES and video_sync(p) is not None
    )
    return named or equalizer_videos(folder)


def video_files_in(folder: Path) -> list[Path]:
    """Every video file of a folder, sync in the name or not."""
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)


def pick_videos(source) -> list[Path]:
    """One or two videos from a folder (one or two cut videos) or the files given.

    A folder whose only video has no sync in its name is the monocular analysis of that video.
    """
    path = Path(source).expanduser()
    if path.is_dir():
        videos = cut_videos_in(path)
        plain = video_files_in(path)
        if not videos and len(plain) == 1:
            return plain  # one video alone, no sync needed: the monocular analysis of it
        if len(videos) not in (1, 2):
            names = ", ".join(v.name for v in videos) or "none"
            raise KickInputError(
                f"{path} needs one or two cut videos (found {len(videos)}: {names}). Two videos "
                "are named '..._sync_N_frames_A_to_B...' or come from the event equalizer "
                "(event_equalizer.json). Give the files with --videos instead."
            )
        return videos
    if path.is_file():
        return [path]  # one video file: the monocular analysis of that camera
    raise KickInputError(f"{path} is neither a folder nor a video file")


class CameraInput:
    """One cut video and what the pipeline reads from it."""

    def __init__(self, video: Path, markers: Path, sync_frame: int, cut_start: int, fps: float,
                 n_frames: int, keypoints: Path | None = None) -> None:  # fmt: skip
        self.video = video
        self.stem = video.stem
        self.markers = markers
        self.sync_frame = sync_frame
        self.cut_start = cut_start
        self.fps = fps
        self.n_frames = n_frames
        # hand-marked 49 keypoints (<stem>_kiki49.csv): the network does not judge this camera
        self.keypoints = keypoints


def validate_inputs(
    videos: list[Path], reference: int, *, sync: tuple[int, int] | None = None
) -> list[CameraInput]:
    """E0: two cut videos, each with its sync and its corrected ball markers.

    The sync is in the name (``_sync_N_frames_A_to_B``) or, for the two videos of the event equalizer,
    in the ``event_equalizer.json`` beside them. ``reference`` (1 or 2) is the camera whose frames are
    the output frames. ``sync`` is ``(sync_frame, cut_start)`` for one video whose name carries no sync
    numbers; the name wins when it has them.
    """
    video_fps = _module("freekiki_detect3d")._video_fps

    if reference not in (1, 2):
        raise KickInputError("--reference must be 1 or 2")
    if len(videos) not in (1, 2):
        raise KickInputError(
            "the kick analysis needs one video (monocular) or two (one per camera)"
        )
    if sync is not None and len(videos) != 1:
        raise KickInputError("--sync and --cut-start are for one video (monocular) only")
    if len(videos) == 2 and videos[0].resolve() == videos[1].resolve():
        raise KickInputError("the two videos are the same file: give one video per camera")
    cams: list[CameraInput] = []
    missing: list[Path] = []
    for video in videos:
        video = Path(video).expanduser().resolve()
        if not video.is_file():
            raise KickInputError(f"video not found: {video}")
        parsed = video_sync(video)
        if len(videos) == 1 and sync is not None and parse_sync_name(video.stem) is None:
            parsed = sync  # the sync given for one video beats the equalizer's, not the name's
        if parsed is None and len(videos) == 1:
            parsed = (0, 0)  # one camera alone: no sync is needed unless it is given
        if parsed is None:
            raise KickInputError(
                f"{video.name} has no sync in its name ('_sync_N_frames_A_to_B') and no "
                "event_equalizer.json beside it; rename the cut video as the cut tool wrote it"
            )
        markers = video.with_name(f"{video.stem}_markers.csv")
        if not markers.is_file():
            missing.append(video)  # every camera is listed at once, not one run per camera
            continue
        with markers.open(encoding="utf-8") as fh:
            is_keypoints = "p1_x" in fh.readline()
        if is_keypoints:
            raise KickInputError(
                f"{markers.name} holds 49 keypoints, not the ball track (frame,p0_x,p0_y). "
                f"The keypoints belong in {video.stem}_kiki49.csv. Move this file out of the way "
                "(keep it, do not delete it), then correct the ball of this camera in getpixelvideo: "
                f"it writes the ball to {markers.name}"
            )
        keypoints = video.with_name(f"{video.stem}_kiki49.csv")
        n = int(nominal_clock(video).size)
        cams.append(
            CameraInput(
                video,
                markers,
                parsed[0],
                parsed[1],
                video_fps(video),
                n,
                keypoints if keypoints.is_file() else None,
            )
        )
    if missing:
        commands = "\n".join(f"  uv run --no-sync vaila/getpixelvideo.py -f {v}" for v in missing)
        raise KickInputError(
            "ball markers not found: "
            + ", ".join(str(v.with_name(f"{v.stem}_markers.csv")) for v in missing)
            + ". Mark the ball of each camera in getpixelvideo (Save writes <video>_markers.csv "
            "next to the video) and run again:\n" + commands
        )
    if len(cams) == 1:
        return cams
    if abs(cams[0].fps - cams[1].fps) > FPS_TOLERANCE:
        raise KickInputError(
            f"the two cameras have different frame rates ({cams[0].fps:.3f} and "
            f"{cams[1].fps:.3f} fps); the binocular clock needs the same rate"
        )
    return [cams[reference - 1], cams[2 - reference]]


def _file_key(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime": round(st.st_mtime, 3)}


def _outputs_of(outputs, result: dict) -> list[Path]:
    """The output files of a stage: a fixed list, or a function of the stage's result."""
    return list(outputs(result) if callable(outputs) else outputs)


class _State:
    """``pipeline_status.json``: the stages, their inputs, outputs and results."""

    def __init__(self, out: Path, ts: str) -> None:
        self.out = out
        self.path = out / STATUS_NAME
        self.runs = 0  # stages that really ran (not reused) in this call
        self.data: dict = {}
        if self.path.is_file():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"ts": ts, "created": datetime.now().isoformat(timespec="seconds"),
                         "stages": {}}  # fmt: skip

    @property
    def ts(self) -> str:
        return str(self.data["ts"])

    def save(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(_jsonable(self.data), indent=2), encoding="utf-8")

    def stage(
        self,
        key: str,
        inputs: dict,
        outputs: list[Path] | Callable[[dict], list[Path]],
        fn: Callable[[], dict],
        *,
        reuse: bool,
        legacy: tuple[str, ...] = (),
    ) -> dict:
        """Run ``fn`` unless this stage finished with the same inputs and its outputs exist.

        ``legacy`` names inputs this stage no longer takes: a record of an older run that still
        holds them matches when they are left out (the ball left the detect and 3D stages).
        """
        record = self.data["stages"].get(key)
        wanted = _jsonable(inputs)
        old = record.get("inputs") if record else None
        if legacy and isinstance(old, dict):
            old = {k: v for k, v in old.items() if k not in legacy}
        same = bool(record) and record.get("status") == "ok" and old == wanted
        if (
            reuse
            and same
            and all(p.exists() for p in _outputs_of(outputs, record.get("result", {})))
        ):
            _log(f"{key}: already done, reused")
            return record["result"]
        self.runs += 1
        self.data["stages"][key] = {"status": "running", "inputs": _jsonable(inputs),
                                    "started": datetime.now().isoformat(timespec="seconds")}  # fmt: skip
        self.save()
        _log(f"{key}: running")
        try:
            result = _jsonable(fn())
        except Exception as exc:  # the stage name must reach the user
            self.data["stages"][key] = {"status": "failed", "inputs": _jsonable(inputs),
                                        "error": f"{type(exc).__name__}: {exc}"}  # fmt: skip
            self.save()
            raise KickStageError(f"{key} failed: {type(exc).__name__}: {exc}") from exc
        self.data["stages"][key] = {
            "status": "ok",
            "inputs": _jsonable(inputs),
            "outputs": [str(p) for p in _outputs_of(outputs, result)],
            "result": result,
            "finished": datetime.now().isoformat(timespec="seconds"),
        }
        self.save()
        return result


def _screen_or_raise(ws, video: Path, model: str, min_camera_ok: float) -> dict:
    """The same screen as detect_videos, but a rejected camera stops the run."""
    freekiki = _module("freekiki")
    freekiki.validate_detection_workspace(ws, model)
    ws_path = Path(ws).expanduser()
    settings = freekiki.load_settings(ws_path)
    defaults = settings["detect"]
    predictor = freekiki.load_predictor(
        freekiki.resolve_model(ws_path, model),
        imgsz=None,
        fallback_imgsz=defaults["imgsz"],
        device=None,
    )
    rates = freekiki.screen_video(
        video,
        predictor,
        conf=float(defaults["conf"]),
        kp_conf=float(defaults["kp_conf"]),
        settings=settings["geometry"],
    )
    usable, reason = freekiki.screen_verdict(rates, min_camera_ok)
    if not usable:
        raise KickInputError(
            f"{video.name} is not a camera the network was trained for ({reason}). "
            "Choose another camera or lower --min-camera-ok knowing the risk."
        )
    return {"camera_ok": round(float(rates["camera_ok"]), 3), "sampled": int(rates["sampled"])}


def _keypoints_csv(work: Path) -> Path:
    found = sorted(work.rglob("*_sam3dinov3_keypoints2d.csv"))
    if not found:
        raise KickStageError(
            f"no keypoints2d csv under {work}: the SAM 3D Body stage did not finish"
        )
    return found[-1]


def _sam_boxes_csv(work: Path) -> Path:
    found = sorted(work.rglob("sam_bbox_tracks.csv"))
    if not found:
        raise KickStageError(f"no sam_bbox_tracks.csv under {work}: the SAM stage did not finish")
    return found[-1]


def _roles_sam(player_map: Path) -> dict[str, int]:
    """The SAM ids of the keeper and the kicker in ``player_map.csv`` (``{}`` before the players)."""
    out: dict[str, int] = {}
    if not Path(player_map).is_file():
        return out
    with Path(player_map).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("role") in ("keeper", "kicker") and row.get("sam_id") not in (None, ""):
                out.setdefault(row["role"], int(row["sam_id"]))
    return out


def _keeper_id(player_map: Path) -> int | None:
    """The goalkeeper's player id (ID0 of a penalty), or None when the players link found no keeper."""
    with player_map.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["role"] == "keeper":
                return int(row["player_id"])
    return None


def _kicker_id(player_map: Path) -> int:
    with player_map.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["role"] == "kicker":
                return int(row["player_id"])
    raise KickStageError(f"no kicker in {player_map}: the players link found no kicker")


def _penalty_topview(cam3d: Path, video: Path) -> None:
    """Redraw the top view of a penalty with player ids: ID0 is the keeper, ID1 the kicker, and the
    others 2, 3, ... (the SAM ids drawn at detection time are not the ids a penalty uses)."""
    if not (cam3d / "ground_tracks.csv").is_file():
        return
    names = _module("freekiki_report_players").load_camera_roles(cam3d).get("names", {})
    fps = _module("freekiki_detect3d")._video_fps(video)
    n = int(nominal_clock(video).size)
    _module("freekiki_topview").write_topview(cam3d, video.stem, fps=fps, n_frames=n, labels=names)


def _penalty_overlay(cam3d: Path, video: Path, work: Path) -> None:
    """Redraw the overlay video of a penalty with player ids: ID0 the keeper, ID1 the kicker, the others
    2, 3, ... (the track ids written at detection time are not the ids a penalty uses). A failure is
    logged and leaves the first overlay: it is a picture, and the analysis does not depend on it."""
    detect3d = _module("freekiki_detect3d")
    run_dir = detect3d._detect_run_dir(Path(work), video.stem)
    poses = cam3d / "poses_3d_on_field.csv"
    if run_dir is None or not poses.is_file():
        return
    names = _module("freekiki_report_players").load_camera_roles(cam3d).get("names", {})
    with poses.open(newline="", encoding="utf-8") as fh:
        kept = {int(row["person_id"]) for row in csv.DictReader(fh)}
    try:
        detect3d.write_3d_overlay(video, run_dir, cam3d, Path(work), kept=kept, labels=names)
    except Exception as exc:  # only the picture: the analysis goes on with the first overlay
        _log(f"overlay not redrawn with the player ids: {type(exc).__name__}: {exc}")


def _penalty_others(player_map: Path, poses: Path, id_column: str, data_dir: Path) -> Path:
    """The people a penalty does not use, written to ``other_IDs/`` for the record and read by nothing.

    A penalty reads the keeper (ID0) and the kicker (ID1) only, taken from ``player_map.csv``; every
    other person's poses go to ``other_IDs/poses_other_persons.csv``. ``id_column`` is the poses column
    that holds the person: ``person_id`` (the SAM id, monocular poses) or ``player_id`` (binocular).
    """
    key = "sam_id" if id_column == "person_id" else "player_id"
    with player_map.open(newline="", encoding="utf-8") as fh:
        keep = {str(int(row[key])) for row in csv.DictReader(fh) if row["role"] in PENALTY_PLAYERS}
    out_dir = data_dir / "other_IDs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "poses_other_persons.csv"
    with poses.open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        with out.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames or [])
            writer.writeheader()
            for row in reader:
                if str(int(float(row[id_column]))) not in keep:
                    writer.writerow(row)
    return out


def _triangulate(
    ref3d: Path,
    other3d: Path,
    work_ref: Path,
    work_other: Path,
    out_csv: Path,
    ref_video: Path,
    other_video: Path,
) -> dict:
    """E4: the keypoints of the paired people, triangulated with the DLT3D of each frame."""
    pose = _module("freekiki_pose_binocular")
    load_view = _module("freekiki_stereo").load_view

    ref_view, _ = load_view(ref3d, markers=None, timing=None, sync_frame=None, cut_start=None,
                            name="ref")  # fmt: skip
    other_view, _ = load_view(other3d, markers=None, timing=None, sync_frame=None, cut_start=None,
                              name="other")  # fmt: skip
    vit_ref = pose.load_vitruvian(work_ref / "vitruvian.dlt3d")
    vit_other = pose.load_vitruvian(work_other / "vitruvian.dlt3d")
    src_ref = dict.fromkeys(ref_view.dlt, "run") | dict.fromkeys(vit_ref, "vitruvian")
    src_other = dict.fromkeys(other_view.dlt, "run") | dict.fromkeys(vit_other, "vitruvian")
    ref_view.dlt = {**ref_view.dlt, **vit_ref}
    other_view.dlt = {**other_view.dlt, **vit_other}
    n_ref = int(ref_view.pts_cut.size)
    n_other = int(other_view.pts_cut.size)
    rows = pose.triangulate_people(
        ref_view,
        other_view,
        ref_tracks=pose.read_tracks(ref3d / "player_map.csv"),
        other_tracks=pose.read_tracks(other3d / "player_map.csv"),
        ref_kps=pose.read_keypoints2d(_keypoints_csv(work_ref), n_ref),
        other_kps=pose.read_keypoints2d(_keypoints_csv(work_other), n_other),
        dlt_source_ref=src_ref,
        dlt_source_other=src_other,
    )
    pose.write_poses(rows, out_csv)
    return {
        "rows": len(rows),
        "valid_rows": sum(1 for r in rows if r["valid"]),
        "people": pose.summarize(rows),
        "height_check_m": pose.height_check(rows),
        "vitruvian_frames": {"ref": len(vit_ref), "other": len(vit_other)},
        "csv": str(out_csv),
        "ref_video": ref_video.name,
        "other_video": other_video.name,
    }


def _overlay_views(ref3d: Path, other3d: Path, work_ref: Path, work_other: Path) -> list[dict]:
    """Both cameras with their run DLT3D and the Vitruvian DLT3D of the weak frames (for the overlay)."""
    pose = _module("freekiki_pose_binocular")
    load_view = _module("freekiki_stereo").load_view
    views = []
    for name, run_dir, work in (("ref", ref3d, work_ref), ("other", other3d, work_other)):
        view, video = load_view(run_dir, markers=None, timing=None, sync_frame=None,
                                cut_start=None, name=name)  # fmt: skip
        view.dlt = {**view.dlt, **pose.load_vitruvian(work / "vitruvian.dlt3d")}
        views.append({"name": name, "view": view, "video": video})
    return views


ANALYSES = ("binocular", "mono-ref", "mono-other")


def _layout():
    return _module("freekiki_layout")


def _fit_dir(bino_dir: Path) -> Path:
    """Where the binocular fit is: ``data/`` of the new layout, or the folder of an older run."""
    data = Path(bino_dir) / "data"
    return data if (data / "ball_flight.json").is_file() else Path(bino_dir)


def _find_poses(out: Path, bino_dir: Path) -> Path:
    """The binocular poses: ``data/`` (new layout), the binocular folder, or the run folder (older)."""
    for candidate in (Path(bino_dir) / "data" / "poses_3d_binocular.csv",
                      Path(bino_dir) / "poses_3d_binocular.csv",
                      Path(out) / "poses_3d_binocular.csv"):  # fmt: skip
        if candidate.is_file():
            return candidate
    return Path(bino_dir) / "data" / "poses_3d_binocular.csv"


def _camera_view(folder: Path, sync: tuple[int, int] | None = None):
    """The camera of a 3D folder. ``sync`` is ``(sync_frame, cut_start)``; None reads the name.

    One camera alone needs no sync (only its DLT3D is read here), so a video with no sync in its
    name nor an equalizer manifest gets ``(0, 0)`` instead of an error.
    """
    if sync is None:
        video = _module("freekiki_detect3d")._video_from_readme(folder)
        sync = (video_sync(video) if video is not None else None) or (0, 0)
    view, _ = _module("freekiki_stereo").load_view(
        folder, markers=None, timing=None,
        sync_frame=None if sync is None else sync[0],
        cut_start=None if sync is None else sync[1],
        name="cam",
    )  # fmt: skip
    return view


def _teams_binocular(
    ref3d,
    other3d,
    ref_video,
    other_video,
    ref_work,
    other_work,
    bino_dir,
    poses_csv,
    ball_csv,
    *,
    team_a: str,
    fps: float,
) -> dict:
    """E4c: the teams of the binocular analysis (both cameras, the team folders and their data)."""
    analysis = _module("freekiki_team_analysis")
    ref_view = _camera_view(ref3d)
    sources = [analysis.Source("ref", ref3d, ref_video, ref_work, dlt=ref_view.dlt),
               analysis.Source("other", other3d, other_video, other_work)]  # fmt: skip
    return analysis.run_team_analysis(sources, bino_dir, poses_csv=poses_csv, ball_csv=ball_csv,
                                      kind="binocular", rate_hz=fps, team_a_colour=team_a)  # fmt: skip


def _teams_mono(
    cam3d,
    video,
    work,
    *,
    is_ref: bool,
    key: str,
    team_a: str,
    fps: float,
    sync: tuple[int, int] | None = None,
) -> dict:
    """E4c: the teams of one camera alone (the monocular analysis of that camera)."""
    analysis = _module("freekiki_team_analysis")
    view = _camera_view(cam3d, sync)
    source = analysis.Source("ref" if is_ref else "other", cam3d, video, work, dlt=view.dlt)
    return analysis.run_team_analysis([source], cam3d, poses_csv=cam3d / "poses_3d_on_field_players.csv",
                                      ball_csv=cam3d / "ball.csv", kind=key, rate_hz=fps,
                                      team_a_colour=team_a)  # fmt: skip


def _c3d_binocular(
    ref3d, other3d, ref_work, other_work, fit_dir: Path, data: Path, ref_video
) -> dict:
    """E4b: the field points and the ball of the binocular analysis, in its data folder."""
    st = _module("freekiki_stereo")
    bc = _module("freekiki_binocular_c3d")
    ref_view, _ = st.load_view(ref3d, markers=None, timing=None, sync_frame=None, cut_start=None,
                               name="ref")  # fmt: skip
    other_view, _ = st.load_view(other3d, markers=None, timing=None, sync_frame=None,
                                 cut_start=None, name="other")  # fmt: skip
    return bc.write_binocular_c3d(
        ref_view,
        other_view,
        data_dir=data,
        ref_2d_csv=bc.field_2d_csv(ref_work),
        other_2d_csv=bc.field_2d_csv(other_work),
        field_names_csv=ref3d / "field_kps_3d.csv",
        ball_csv=fit_dir / "ball_binocular.csv",
        fps=_module("freekiki_detect3d")._video_fps(ref_video),
    )


def _ref_camera_moves(ref3d: Path) -> bool:
    """True when the reference flight found a moving (panning/zooming) camera (``ball_flight.json``
    ``camera_moves``); False for a fixed camera, an older flight or no flight at all."""
    path = Path(ref3d) / "ball_flight.json"
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("camera_moves", False))
    except (OSError, ValueError, AttributeError):
        return False


def _fit_contact(flight: dict, marker: int | None, fps: float) -> int | None:
    """The contact frame of the fitted flight when it should replace the marker's, else None.

    It replaces the marker only when the fit is reliable and the two differ by more than one frame.
    """
    kick_time, rms = flight.get("kick_time_s"), flight.get("rms_px")
    if kick_time is None or marker is None or rms is None:
        return None
    if float(rms) > _module("freekiki_auto_flight").PIXEL_REJECT_PX:
        return None
    fit = round(float(kick_time) * fps)
    return fit if abs(fit - marker) > 1 else None


def _kicker(
    ref3d: Path,
    binocular: Path,
    poses_csv: Path,
    out: Path,
    ref_video: Path,
    *,
    kick_foot: str,
    other3d: Path | None = None,
    ref_work: Path | None = None,
    other_work: Path | None = None,
    goal_x: float | None = None,
) -> dict:
    """E5: the kicker's metrics and figures. ``binocular`` is the folder with the ball fit.

    ``goal_x`` is the attacked goal line of the set piece: it sets the side of the views from behind.
    The contact is the first frame in which the hand-marked ball moves, unless the fitted flight is
    reliable (RMS up to ``PIXEL_REJECT_PX``) and puts the kick more than a frame away: a marker that
    drifts early would otherwise move every metric of the kicker to the wrong frame.
    """
    kicker = _module("freekiki_kicker")
    fps = _module("freekiki_detect3d")._video_fps(ref_video)
    n = int(nominal_clock(ref_video).size)
    flight = json.loads((binocular / "ball_flight.json").read_text(encoding="utf-8"))
    ball_pixels = kicker.read_ball_pixels(ref3d / "ball.csv")
    # a panning reference camera moves the resting ball in the image: the marker contact is then
    # read on the pitch (ball_3d.csv, each frame's own camera), never on the pixels
    ball_ground = None
    if _ref_camera_moves(ref3d) and (ref3d / "ball_3d.csv").is_file():
        ball_ground = kicker.read_ball_ground(ref3d / "ball_3d.csv")
    # a ball not at rest (open play) or a contact given by hand: the reference flight decided it
    decided = _module("freekiki_auto_flight").flight_contact(ref3d)
    incoming = None
    if decided is not None and decided.get("source") in ("manual", "kink", "first_frame"):
        contact_override, override_source = int(decided["frame"]), str(decided["source"])
        incoming = decided.get("incoming")
    else:
        contact_override = _fit_contact(flight, kicker.contact_frame(ball_pixels, ball_ground), fps)
        override_source = "fit"
    result = kicker.analyze_kicker(
        kicker.read_poses(poses_csv),
        kicker.read_ball_xyz(binocular / "ball_3d_on_field.csv", n),
        ball_pixels,
        kicker_id=_kicker_id(ref3d / "player_map.csv"),
        keeper_id=_keeper_id(ref3d / "player_map.csv"),
        fps=fps,
        kick_foot=kick_foot,
        kick_time_s=flight.get("kick_time_s"),
        fit_peak_speed_mps=flight.get("peak_speed_mps"),
        fit_spin_rps=flight.get("spin_rps"),
        contact_override=contact_override,
        goal_x=goal_x,
        override_source=override_source,
        incoming=incoming,
        ball_ground=ball_ground,
    )
    views = None
    if other3d is not None and ref_work is not None and other_work is not None:
        views = _overlay_views(ref3d, other3d, ref_work, other_work)
    paths = kicker.write_outputs(result, out, views=views)
    values = {m["metric"]: m["value"] for m in result["metrics"]}
    keep = ("contact_frame", "kick_foot", "v_foot_pre_mps", "v_ball_post_mps", "ratio_vball_vfoot",
            "d_support_ball_m", "approach_angle_deg", "knee_ext_peak_deg_s",
            "proximal_distal_order", "goal_error_m")  # fmt: skip
    if incoming is not None:  # open play: the ball that came in
        keep += ("v_ball_pre_mps", "ball_turn_deg")
    return {
        "fps": fps,
        "values": {k: values.get(k, "") for k in keep},
        "warnings": result["warnings"],
        "files": {k: str(v) for k, v in paths.items()},
    }


def _read_pose_rows(path: Path) -> list[dict]:
    """``poses_3d_on_field.csv`` as the rows the flight reads (frame and SAM person id as ints)."""
    rows: list[dict] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                rows.append({"frame": int(row["frame"]), "person_id": int(row["person_id"]),
                             "kpt_name": row["kpt_name"], "x_m": float(row["x_m"] or "nan"),
                             "y_m": float(row["y_m"] or "nan"), "z_m": float(row["z_m"] or "nan")})  # fmt: skip
            except (KeyError, ValueError):
                continue
    return rows


def _read_lifted_ball(path: Path) -> list[dict]:
    """``ball_3d.csv`` (the ball lifted to the pitch per frame) as the rows the flight reads."""
    rows: list[dict] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({"frame": int(row["frame"]), "cx": row["cx"], "cy": row["cy"],
                         "score": row["score"], "x_m": float(row["x_m"] or "nan"),
                         "y_m": float(row["y_m"] or "nan"), "z_m": float(row["z_m"] or "nan")})  # fmt: skip
    return rows


def _save_alternatives(
    ref3d: Path, ref_video: Path, flight: dict, *, save_frame: int, keeper_sam: int, sigma: float
) -> list[dict]:
    """The reference flight refitted with the keeper's anchor moved -sigma and +sigma along the
    camera ray (the depth one camera cannot see): the spread is the uncertainty of the save."""
    af = _module("freekiki_auto_flight")
    kicker = _module("freekiki_kicker")
    fps = _module("freekiki_detect3d")._video_fps(ref_video)
    n = int(nominal_clock(ref_video).size)
    cameras = _module("freekiki_stereo").find_dlt3d(ref3d, ref_video)
    ball_all = _read_lifted_ball(ref3d / "ball_3d.csv")
    pixels = kicker.read_ball_pixels(ref3d / "ball.csv")
    pose_rows = _read_pose_rows(ref3d / "poses_3d_on_field.csv")
    out = []
    for shift in (-sigma, sigma):
        fit = af.auto_flight(ball_all, pixels, cameras, pose_rows, fps, n,
                             contact_frame=int(flight["contact_frame"]), save_frame=save_frame,
                             keeper_id=keeper_sam, save_sigma_along=sigma, save_shift_m=shift)  # fmt: skip
        if fit is not None:
            out.append({"shift_m": shift, "theta": fit["fit"]["theta"], "rms_px": fit["fit"]["rms_px"],
                        "kick_time_s": fit["fit"]["kick_time_s"]})  # fmt: skip
    return out


def _penalty(
    ref3d: Path,
    ref_video: Path,
    ref_work: Path,
    folder: Path,
    *,
    goal_x: float,
    track_csv: Path | None = None,
    opts: dict | None = None,
) -> dict:
    """The pynalty metrics, the ball path and the report of a penalty (reference camera pixels and DLT3D).

    ``folder`` is the chosen analysis: its ``data/`` gets the metrics and the path, its
    ``report_tv/penalty/`` the report. ``track_csv`` is the binocular ball of a run with two cameras:
    when it is given, the ball path's verdict is that ball (:func:`freekiki_penalty.penalty_path`); without
    it the verdict is the monocular fit, which is provisional.

    ``opts`` (the penalty options of the run): ``save_frame`` and ``shot_outcome`` (a save measured
    from the 3D flight, :func:`freekiki_penalty.save_metrics`), ``statures``, ``gk_arm_span`` and
    ``gk_mass`` (the measured bodies), ``gk_onset_frame`` (an onset marked by eye) and
    ``save_sigma_along``. The keeper's dive (:func:`freekiki_keeper.keeper_kinematics`) goes to
    ``data/keeper_metrics.json``.
    """
    opts = opts or {}
    penalty = _module("freekiki_penalty")
    kicker = _module("freekiki_kicker")
    stereo = _module("freekiki_stereo")
    fps = _module("freekiki_detect3d")._video_fps(ref_video)
    n = int(nominal_clock(ref_video).size)
    keeper = penalty.keeper_sam_id(ref3d / "player_map.csv")
    boxes = penalty.read_boxes(_sam_boxes_csv(ref_work))
    try:
        matrix = _module("monocular_dlt_align").dlt_projection_matrix
        cameras = {f: matrix(L) for f, L in stereo.find_dlt3d(ref3d, ref_video).items()}
    except FileNotFoundError:
        cameras = None  # the penalty reports the missing per-frame DLT3D
    track = None
    if track_csv is not None and track_csv.is_file():
        xyz = kicker.read_ball_xyz(track_csv, n)
        track = {f: xyz[f] for f in range(n) if np.isfinite(xyz[f]).all()}
    flight_path = ref3d / "ball_flight.json"
    flight = json.loads(flight_path.read_text(encoding="utf-8")) if flight_path.is_file() else None
    ball_on = kicker.read_ball_xyz(ref3d / "ball_3d_on_field.csv", n)
    flight_track = track or {f: ball_on[f] for f in range(n) if np.isfinite(ball_on[f]).all()}
    save_frame = opts.get("save_frame")
    if save_frame is None and flight is not None:
        save_frame = flight.get("save_frame")
    outcome = opts.get("shot_outcome") or "auto"
    shot_outcome = None if outcome == "auto" else outcome
    keeper_id = _keeper_id(ref3d / "player_map.csv")
    keeper_points = None
    players_csv = ref3d / "poses_3d_on_field_players.csv"
    if keeper_id is not None and players_csv.is_file():
        keeper_points = kicker.person_points(kicker.read_poses(players_csv), keeper_id, n)
    spec = opts.get("statures") or {}
    anthro = _module("pynalty_analysis").Anthropometrics(
        gk_height_m=spec.get("keeper"), gk_arm_span_m=opts.get("gk_arm_span"),
        kicker_height_m=spec.get("kicker"),
    )  # fmt: skip
    keeper_result = None
    if keeper_points and flight is not None and flight.get("contact_frame") is not None:
        last = int(save_frame) if save_frame is not None else n - 1
        keeper_result = _module("freekiki_keeper").keeper_kinematics(
            keeper_points, fps=fps, contact_frame=int(flight["contact_frame"]), save_frame=last,
            goal_x=goal_x, kick_time_s=flight.get("kick_time_s"), ball_xyz=ball_on,
            onset_frame_given=opts.get("gk_onset_frame"),
        )  # fmt: skip
    gk_move = None
    if keeper_result is not None and keeper_result.get("available"):
        gk_move = keeper_result["metrics"]["onset_frame"]
    alternatives = None
    is_save = save_frame is not None and shot_outcome in (None, "save")
    if (
        is_save
        and flight is not None
        and keeper is not None
        and track is None
        and not opts.get("manual_arrival")
    ):
        sigma = opts.get("save_sigma_along") or _module("freekiki_auto_flight").SAVE_SIGMA_ALONG_M
        try:
            alternatives = _save_alternatives(ref3d, ref_video, flight, save_frame=int(save_frame),
                                              keeper_sam=keeper, sigma=float(sigma))  # fmt: skip
        except (ValueError, FileNotFoundError) as exc:
            _log(f"penalty: the +/- sigma refits of the save failed ({exc})")
    result = penalty.penalty_metrics(
        field_rows=penalty.read_field_rows(ref3d / "field_kps_3d.csv"),
        ball_px=kicker.read_ball_pixels(ref3d / "ball.csv"),
        cameras=cameras,
        keypoints=_module("freekiki_pose_binocular").read_keypoints2d(_keypoints_csv(ref_work), n),
        keeper_boxes=boxes.get(keeper, {}) if keeper is not None else {},
        keeper_sam=keeper,
        fps=fps,
        n_frames=n,
        goal_x=goal_x,
        track=track,
        flight=flight,
        flight_track=flight_track,
        save_frame=save_frame,
        shot_outcome=shot_outcome,
        keeper_points=keeper_points,
        gk_move_frame=gk_move,
        anthro=anthro,
        alternatives=alternatives,
    )
    result["anthropometrics"] = {
        "gk_height_m": spec.get("keeper"), "gk_arm_span_m": opts.get("gk_arm_span"),
        "gk_mass_kg": opts.get("gk_mass"), "kicker_height_m": spec.get("kicker"),
        "source": "given" if spec or opts.get("gk_arm_span") else "default",
    }  # fmt: skip
    path = result.get("path") or {}
    if keeper_result is not None:
        result["keeper"] = keeper_result.get("metrics") if keeper_result.get("available") else {
            "available": False, "reason": keeper_result.get("reason")}  # fmt: skip
    if result.get("event") != "save" and keeper_id is not None and path.get("available"):
        # the goalkeeper's move, partial (no goal line)
        points = keeper_points or kicker.person_points(kicker.read_poses(players_csv), keeper_id, n)
        # the contact of the fitted flight, not the first moved marker frame (which can drift early)
        contact = path["kick_frame"]
        if path.get("fit", {}).get("kick_time_s") is not None:
            contact = min(round(path["fit"]["kick_time_s"] * fps), path["last_frame"])
        result["gk_partial"] = penalty.gk_partial(
            points, kick=contact, last=path["last_frame"], fps=fps, goal_x=goal_x
        )
    data_dir = _layout().data_dir(folder)
    files = penalty.write(data_dir, result)
    if keeper_result is not None:
        keeper_json = data_dir / "keeper_metrics.json"
        keeper_json.write_text(json.dumps(_jsonable(keeper_result), indent=2), encoding="utf-8")
        files["keeper_json"] = keeper_json
    try:
        blender_files = _module("freekiki_blender").export_penalty_blender(folder, render=True)
        files.update({f"blender_{k}": v for k, v in blender_files.items() if k != "dir"})
    except Exception as exc:  # Blender package is media-only; metrics/report still stand
        _log(f"penalty: Blender export skipped ({type(exc).__name__}: {exc})")
    html = penalty.report(folder / "report_tv" / "penalty", ref_video, fps, result, cam3d=folder)
    summary_keys = ("event", "interception", "counterfactual", "uncertainty")
    return {
        "available": bool(result["available"]),
        "reason": result.get("reason", ""),
        "metrics": {
            k: result["metrics"][k] for k in PENALTY_KEEP if k in result.get("metrics", {})
        },
        "path": {k: v for k, v in (result.get("path") or {}).items() if k != "rows"} or None,
        "gk_partial": result.get("gk_partial"),
        "keeper": result.get("keeper"),
        **{k: result.get(k) for k in summary_keys if result.get(k) is not None},
        "files": {k: str(v) for k, v in files.items()},
        "report": html,
    }


def _mono_analysis(
    key, cam3d, video, work, is_ref, *, kick_foot, team_a, runner, sync=None, penalty=False,
    goal_x=None, report=True, defer_report=False,
):  # fmt: skip
    """One camera alone: its teams, its kicker and its report. Returns the entry and the kicker values.

    A failure is recorded in the entry; the other analyses still run. A penalty has no team split
    (``penalty``). ``sync`` is the ``(sync_frame, cut_start)`` of a video whose name has none.
    ``report=False`` skips the goal report (open play, dribble and pass have no shot at goal; the
    planar report is ``freekiki.py openplay-report``). ``defer_report`` leaves the report to the
    caller: the TV report of a penalty quotes ``data/penalty_metrics.json``, which the penalty stage
    writes after this analysis.
    """
    entry: dict = {"folder": str(cam3d), "status": "ok"}
    fps = _module("freekiki_detect3d")._video_fps(video)
    try:
        if (
            penalty
        ):  # no team or referee split: a penalty reads the keeper (ID0) and the kicker (ID1)
            _penalty_others(cam3d / "player_map.csv", cam3d / "poses_3d_on_field.csv", "person_id",
                            _layout().data_dir(cam3d))  # fmt: skip
            _penalty_topview(cam3d, video)
            _penalty_overlay(cam3d, video, work)
        else:
            teams = runner(f"teams:{key}", {"player_map": cam3d.name, "sync": sync},
                           [cam3d / "data" / "team_map.csv"],
                           lambda: _teams_mono(cam3d, video, work, is_ref=is_ref, key=key, team_a=team_a,
                                               fps=fps, sync=sync))  # fmt: skip
            entry["counts"] = teams["counts"]
        kick = runner(f"kicker:{key}", {"poses": "poses_3d_on_field_players.csv", "kick_foot": kick_foot,
                                        "goal_x": goal_x, "model": KICKER_MODEL},
                      [cam3d / "data" / "kicker_metrics.csv"],
                      lambda: _kicker(cam3d, cam3d, cam3d / "poses_3d_on_field_players.csv",
                                      _layout().data_dir(cam3d), video, kick_foot=kick_foot,
                                      goal_x=goal_x))  # fmt: skip
        if report and defer_report:
            entry["report"] = REPORT_AFTER_PENALTY
        elif report:
            runner(f"report:{key}", {"run": cam3d.name},
                   [cam3d / "report_tv" / "pt" / "relatorio_bola.html",
                    cam3d / "report_tv" / "en" / "ball_report.html"],
                   lambda: _module("freekiki_report").build_report(cam3d, lang="both", video=video))  # fmt: skip
        else:
            entry["report"] = REPORT_SKIPPED
    except Exception as exc:  # a monocular analysis may fail on its own
        entry["status"] = "failed"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        _log(f"{key} failed: {entry['error']}")
        return entry, None
    return entry, kick["values"]


def _sam_to_player(player_map: Path) -> dict[int, int]:
    """SAM track id -> player id (keeper 0, kicker 1, the others 2, ...) from the run's player map."""
    with player_map.open(newline="", encoding="utf-8") as fh:
        return {int(row["sam_id"]): int(row["player_id"]) for row in csv.DictReader(fh)}


def _setpiece(ref3d: Path) -> dict:
    """The kind of set piece of the reference camera, written to ``ref3d/data/setpiece.json``.

    The wall is given in the run's player ids (``wall_sam_ids`` keeps the SAM ids). The player map is
    written by the players stage, which runs before this one.
    """
    sp = _module("freekiki_setpiece")
    result = sp.classify_run(ref3d)
    player_map = ref3d / "player_map.csv"
    if player_map.is_file():
        result = sp.with_player_ids(result, _sam_to_player(player_map))
    sp.write(_layout().data_dir(ref3d), result)
    return result


def _chosen_setpiece(ref3d: Path, choice: str) -> dict:
    """The rule's set piece with the kind the user chose, written in place of the rule's own."""
    sp = _module("freekiki_setpiece")
    result = sp.apply_choice(_setpiece(ref3d), choice)
    sp.write(_layout().data_dir(ref3d), result)
    return result


def _stored_choice(ref3d: Path) -> str:
    """The choice a finished run stored (``auto`` when it stored none), so a figures rewrite keeps it."""
    path = _layout().data_dir(ref3d) / "setpiece.json"
    if not path.is_file():
        return "auto"
    return json.loads(path.read_text(encoding="utf-8")).get("choice", "auto")


def _penalty_report(
    key: str, entry: dict, runner, *, folder: Path | None, video: Path | None
) -> None:
    """The TV report of the chosen analysis of a penalty, made once ``penalty_metrics.json`` exists.

    The stage's inputs hold that file, so a new penalty result renders the report again. A report
    that fails is recorded in the entry; the metrics it quotes are already written.
    """
    if folder is None:
        return
    metrics = _layout().data_dir(folder) / "penalty_metrics.json"
    stage_key = "report:binocular" if key == "binocular" else f"report:{key}"
    inputs = {("binocular" if key == "binocular" else "run"): folder.name,
              "penalty": _file_key(metrics) if metrics.is_file() else None}  # fmt: skip
    try:
        runner(stage_key, inputs,
               [folder / "report_tv" / "pt" / "relatorio_bola.html",
                folder / "report_tv" / "en" / "ball_report.html"],
               lambda: _module("freekiki_report").build_report(folder, lang="both", video=video))  # fmt: skip
        entry["report"] = "ok"
        # the penalty report plays copies of the report's animations: refresh them, or they are the
        # ones of an earlier run (the penalty stage made them before this report was rendered)
        penalty_dir = folder / "report_tv" / "penalty"
        if penalty_dir.is_dir():
            _module("freekiki_penalty")._collect_report_videos(folder, penalty_dir)
    except Exception as exc:  # the metrics are already written; the report failing is not the run
        entry["report"] = f"failed: {type(exc).__name__}: {exc}"
        _log(f"{key} report failed: {entry['report']}")


def _analyses(
    out: Path,
    *,
    ref3d: Path,
    other3d: Path | None,
    ref_work: Path,
    other_work: Path | None,
    ref_video: Path,
    other_video: Path | None,
    bino_dir: Path | None,
    kick_foot: str,
    team_a: str,
    analysis: str | None,
    runner,
    setpiece: dict | None = None,
    ref_sync: tuple[int, int] | None = None,
    penalty_opts: dict | None = None,
) -> dict:
    """The analyses of the run: one monocular per camera and the binocular (two cameras).

    ``ref_sync`` is the ``(sync_frame, cut_start)`` given for a single reference video whose name
    has none (``None`` reads the name, as before).

    ``runner(key, inputs, outputs, fn)`` runs a stage (cached in ``pipeline_status.json`` when the run
    has one). A monocular analysis that fails is recorded and the others still run; the binocular one
    raises. ``analysis`` is the one the README marks as the result (binocular with two cameras).
    ``setpiece`` (``freekiki_setpiece.classify_run``) is written to each analysis's ``data/``; a penalty
    gets the pynalty metrics and report in the chosen analysis instead of the freekiki ball report.
    """
    two = other3d is not None
    available = ["binocular", "mono-ref", "mono-other"] if two else ["mono-ref"]
    chosen = analysis or ("binocular" if two else "mono-ref")
    if chosen not in available:
        raise KickInputError(
            f"--analysis {chosen} is not available here (choose: {', '.join(available)})"
        )
    kind = (setpiece or {}).get("kind")
    flight_report = kind not in PLANAR_KINDS
    if not flight_report:
        _log(
            f"set piece {kind}: no goal report; use freekiki.py openplay-report --folder <3D folder> "
            "--kind dribble|pass for the planar report"
        )
    # a penalty needs the ball at rest (the goal side): without it the run is a free kick analysis
    penalty = kind == "penalty" and (setpiece or {}).get("goal_x_m") is not None
    goal_x = (setpiece or {}).get(
        "goal_x_m"
    )  # the attacked goal line: the side of the views from behind
    # the TV report of a penalty quotes the pynalty metrics of the chosen analysis, so it is rendered
    # after the penalty stage (a report made before it read the metrics of an earlier run)
    defer = chosen if penalty and setpiece is not None and flight_report else None
    results: dict[str, dict] = {}
    kicker_values: dict[str, dict] = {}
    monos = [("mono-ref", ref3d, ref_video, ref_work, True)]
    if two:
        monos.append(("mono-other", other3d, other_video, other_work, False))
    for key, cam3d, video, work, is_ref in monos:
        entry, values = _mono_analysis(key, cam3d, video, work, is_ref, kick_foot=kick_foot,
                                       team_a=team_a, runner=runner,
                                       sync=ref_sync if is_ref else None, penalty=penalty,
                                       goal_x=goal_x, report=flight_report,
                                       defer_report=key == defer)  # fmt: skip
        if setpiece is not None:
            _module("freekiki_setpiece").write(_layout().data_dir(cam3d), setpiece)
        results[key] = entry
        if values is not None:
            kicker_values[key] = values
    if (
        two
        and other3d is not None
        and bino_dir is not None
        and other_work is not None
        and other_video is not None
    ):
        entry = {"folder": str(bino_dir), "status": "ok"}
        results["binocular"] = entry
        bdata = _layout().data_dir(bino_dir)
        poses = _find_poses(out, bino_dir)
        fit = _fit_dir(bino_dir)
        fps = _module("freekiki_detect3d")._video_fps(ref_video)
        ball_csv = ref3d / "ball.csv"
        c3d = runner("c3d:binocular", {"poses": poses.name},
                     [bdata / "field_ball.c3d"],
                     lambda: _c3d_binocular(ref3d, other3d, ref_work, other_work, fit, bdata, ref_video))  # fmt: skip
        entry["field_kept"] = c3d["summary"]["field_kept"]
        if (
            penalty
        ):  # no team or referee split: a penalty reads the keeper (ID0) and the kicker (ID1)
            _penalty_others(ref3d / "player_map.csv", poses, "player_id", bdata)
        else:
            teams = runner("teams:binocular", {"poses": poses.name}, [bdata / "team_map.csv"],
                           lambda: _teams_binocular(ref3d, other3d, ref_video, other_video, ref_work, other_work,
                                                    bino_dir, poses, ball_csv, team_a=team_a, fps=fps))  # fmt: skip
            entry["counts"] = teams["counts"]
        kick = runner("kicker:binocular", {"poses": poses.name, "kick_foot": kick_foot,
                                           "goal_x": goal_x, "model": KICKER_MODEL},
                      [bdata / "kicker_metrics.csv", bdata / "kicker_curves_pt.png"],
                      lambda: _kicker(ref3d, fit, poses, bdata, ref_video, kick_foot=kick_foot,
                                      other3d=other3d, ref_work=ref_work, other_work=other_work,
                                      goal_x=goal_x))  # fmt: skip
        kicker_values["binocular"] = kick["values"]
        if flight_report and defer == "binocular":
            entry["report"] = REPORT_AFTER_PENALTY
        elif flight_report:
            runner("report:binocular", {"binocular": bino_dir.name},
                   [bino_dir / "report_tv" / "pt" / "relatorio_bola.html",
                    bino_dir / "report_tv" / "en" / "ball_report.html"],
                   lambda: _module("freekiki_report").build_report(bino_dir, lang="both", video=ref_video))  # fmt: skip
        else:
            entry["report"] = REPORT_SKIPPED
        if setpiece is not None:
            _module("freekiki_setpiece").write(bdata, setpiece)
    if penalty and setpiece is not None and chosen in results:
        # the 3D crossing of the chosen analysis; the pixels and the keeper are the reference camera's
        if chosen == "binocular" and bino_dir is not None:
            folder, ball_csv = bino_dir, _layout().data_dir(bino_dir) / "ball_3d_on_field.csv"
        elif chosen == "mono-other" and other3d is not None:
            folder, ball_csv = other3d, other3d / "ball_3d_on_field.csv"
        else:
            folder, ball_csv = ref3d, ref3d / "ball_3d_on_field.csv"
        try:
            flight_file = ref3d / "ball_flight.json"
            results[chosen]["penalty"] = runner(
                f"penalty:{chosen}",
                _jsonable({"ball": ball_csv.name, "goal_x_m": setpiece["goal_x_m"],
                           "model": PENALTY_MODEL, **(penalty_opts or {}),
                           "flight": _file_key(flight_file) if flight_file.is_file() else None}),
                [_layout().data_dir(folder) / "penalty_metrics.csv"],
                lambda: _penalty(ref3d, ref_video, ref_work, folder, goal_x=setpiece["goal_x_m"],
                                 track_csv=ball_csv if chosen == "binocular" else None,
                                 opts=penalty_opts))  # fmt: skip
        except Exception as exc:  # a penalty that cannot be measured is reported, the run goes on
            results[chosen]["penalty"] = {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            _log(f"penalty failed: {exc}")
    elif kind == "penalty" and chosen in results:
        results[chosen]["penalty"] = {
            "available": False,
            "reason": "no ball at rest on the field: the pynalty metrics need the ball and the goal side",
        }
    if defer is not None and results.get(defer, {}).get("status") == "ok":
        _penalty_report(
            defer, results[defer], runner,
            folder={"binocular": bino_dir, "mono-other": other3d}.get(defer, ref3d),
            video=other_video if defer == "mono-other" else ref_video,
        )  # fmt: skip
    for key, entry in results.items():
        entry["chosen"] = key == chosen
    return {"analyses": results, "chosen": chosen, "kicker_values": kicker_values.get(chosen, {}),
            "warnings": []}  # fmt: skip


def _penalty_lines(penalty: dict | None) -> list[str]:
    if penalty is None:
        return []
    if penalty.get("available"):
        lines = [
            "Penalty (pynalty metrics; data/penalty_metrics.csv; report in report_tv/penalty/):",
            *[f"  {k}: {v}" for k, v in penalty.get("metrics", {}).items()],
        ]
    else:
        lines = [f"Penalty: pynalty metrics not available here ({penalty.get('reason', '')})"]
    path = penalty.get("path") or {}
    if path.get("available"):
        bend = "" if path["shape"] == "straight" else f" bowing {path['bow_side']}"
        if path["goal_frame"] is None:
            reach = f"does not reach the goal line in the clip (last frame {path['last_frame']})"
        else:
            mouth = "inside" if path["arrival_in_goal_mouth"] else "outside"
            reach = f"crosses the goal line at frame {path['goal_frame']} ({mouth} the goal mouth)"
        # the binocular ball is the verdict; the monocular fit is an estimate and says so
        origin = (
            "binocular ball" if path.get("source") == "binocular" else "monocular fit, provisional"
        )
        lines.append(
            f"Ball path (data/penalty_path.csv, {origin}): {path['shape']}{bend}, max "
            f"{path['max_dev_m']} m off the spot-arrival line on the ground, apex {path['apex_z_m']} m, "
            f"{reach}; the ball at rest sits {path['spot_fit_offset_m']} m from the spot"
        )
    elif path:
        lines.append(f"Ball path: not available ({path.get('reason', '')})")
    gk = penalty.get("gk_partial") or {}
    if gk.get("available") and gk.get("moved"):
        lines.append(
            f"Goalkeeper (partial, no goal line; data/penalty_metrics.csv): moves to the goalkeeper's {gk['direction']} "
            f"{abs(gk['toward_right_m'])} m, starts {gk['onset_ms_after_kick']} ms after the kick, "
            f"mean speed {gk['mean_speed_mps']} m/s up to frame {gk['last_frame']}"
        )
    elif gk.get("available"):
        lines.append(f"Goalkeeper (partial, no goal line): {gk['reason']}")
    elif gk:
        lines.append(f"Goalkeeper: not available ({gk.get('reason', '')})")
    lines += _save_lines(penalty)
    return [*lines, ""]


def _save_lines(penalty: dict) -> list[str]:
    """A save: the interception, the would-be crossing, their spread and the keeper's dive."""
    if penalty.get("event") != "save":
        return []
    lines: list[str] = []
    inter = penalty.get("interception") or {}
    if inter:
        lines.append(
            f"Save (frame {inter['frame']}): the hands stop the ball {inter['in_front_of_line_m']} m in "
            f"front of the goal line, {inter['height_m']} m high, y {inter['lateral_m']} m, "
            f"{inter['time_s_after_kick']} s after the kick; mean speed from the spot "
            f"{inter['mean_speed_mps']} m/s over {inter['distance_from_spot_m']} m"
        )
    would = penalty.get("counterfactual") or {}
    if would:
        mouth = "inside" if would.get("inside_goal_mouth") else "outside"
        lines.append(
            f"Without the keeper the same flight would cross the goal line at y {would['y_m']} m, "
            f"{would['z_m']} m high ({mouth} the goal mouth; {would.get('zone_label', '')}; "
            f"{would['dist_to_nearest_post_m']} m from the nearest post; HWR accuracy "
            f"{would['hwr_accuracy_pct']} %, Attaallah et al. 2021)"
        )
    spread = ((penalty.get("uncertainty") or {}).get("along_camera_ray")) or []
    for alt in spread:
        alt_would = alt.get("counterfactual") or {}
        xyz = alt.get("interception_xyz")
        lines.append(
            f"  keeper anchor moved {alt.get('shift_m')} m along the camera ray: ball at the save "
            f"{xyz}, would-be crossing y {alt_would.get('y_m')} m, z {alt_would.get('z_m')} m"
        )
    keeper = penalty.get("keeper") or {}
    if keeper.get("dive_side"):
        lines.append(
            f"Keeper (data/keeper_metrics.json): onset frame {keeper['onset_frame']} "
            f"({keeper['onset_source']}; 3D onset frame {keeper.get('onset_frame_auto')}), "
            f"{keeper['onset_ms_before_kick']} ms before the kick; dives {keeper['dive_side']} "
            f"{keeper['lateral_displacement_m']} m, peak lateral CoM speed "
            f"{keeper['peak_lateral_speed_mps']} m/s; hands-ball gap at the save "
            f"{keeper.get('hands_ball_gap_m')} m"
        )
    return lines


def _contact_lines(contact: dict | None) -> list[str]:
    """The kick contact of the reference flight, and how it was found."""
    if not contact:
        return []
    how = {
        "rest": "the still ball leaves its rest pixel",
        "kink": "break of the rolling ball's ground track (open play)",
        "first_frame": "the ball already flies in the first frame (the kick is at or before it)",
        "manual": "given by hand (--contact-frame)",
    }.get(str(contact.get("source")), str(contact.get("source")))
    line = f"contact: frame {contact.get('frame')} (0-based, reference camera): {how}"
    incoming = contact.get("incoming") or {}
    if incoming.get("speed_mps") is not None:
        line += f"; the ball came in at {incoming['speed_mps']} m/s" + (
            f", turned {incoming['turn_deg']} deg" if incoming.get("turn_deg") is not None else ""
        )
    return [line]


def _correction_lines(inputs: dict | None) -> list[str]:
    """The hand-corrected files next to the reference video, with their time (to correct again:
    freekiki.py kick-correct, then kick-analysis --resume)."""
    if not inputs:
        return []
    lines = []
    for label, path in (("ball", inputs.get("ball")), ("field keypoints", inputs.get("field"))):
        if path and Path(path).is_file():
            stamp = datetime.fromtimestamp(Path(path).stat().st_mtime).isoformat(timespec="seconds")
            lines.append(f"{label} (by hand): {path} (saved {stamp})")
    return lines


def _readme(out: Path, summary: dict) -> Path:
    analyses = summary.get("analyses", {})
    setpiece = summary.get("setpiece") or {}
    penalty = next((e["penalty"] for e in analyses.values() if "penalty" in e), None)
    lines = [
        "vailá FreeKiki kick analysis (one click)",
        f"created: {summary['created']}",
        f"reference camera: {summary['reference_video']}",
        f"other camera: {summary.get('other_video') or 'none (monocular only)'}",
        f"fps: {summary['fps']}  |  kicker id: {summary.get('kicker_id', '')}",
        f"set piece: {setpiece.get('kind', 'unknown')} ({setpiece.get('reason', '')})",
        *_contact_lines(summary.get("contact")),
        *_correction_lines(summary.get("inputs")),
        "",
        "Analyses (each one has its own folder; the chosen one is the result):",
        *[
            f"  {key}{'  <- chosen' if entry.get('chosen') else ''}: {entry['folder']}  "
            f"[{entry['status']}]{'  ' + str(entry.get('error', '')) if entry.get('error') else ''}"
            f"{'  teams ' + str(entry.get('counts')) if entry.get('counts') else ''}"
            for key, entry in analyses.items()
        ],
        "  Each analysis folder: team_A/, team_B/, referees/ (c3d per player and overlay per camera,",
        "  only the people on the pitch), report_tv/ and data/ (tables, kicker, team_map.csv).",
        "  A monocular analysis uses one camera: its poses carry the stature prior (1.83 m) in z.",
        "",
        "Stages (pipeline_status.json has the inputs, outputs and errors of each):",
        *[f"  {key}: {status}" for key, status in summary["stages"].items()],
        "",
        "Kicker of the chosen analysis (kicker_metrics.csv in its data/ folder has the source and quality):",
        *[f"  {k}: {v}" for k, v in summary.get("kicker_values", {}).items()],
        "",
        *_penalty_lines(penalty),
        f"Limits: at {float(summary['fps']):.2f} fps the contact (about 1 ms) falls between two "
        "frames, so the speed at impact is not measured;",
        "Vfoot is the mean over the frames before contact and Vball over the frames after it.",
        "The 2D keypoints are reprojections of each camera's SAM 3D Body mesh, so the triangulated points",
        "fuse two models: the residuals in data/poses_3d_binocular.csv are the quality control.",
        "The foot marker is the mean of big-toe-tip, small-toe-tip and heel (not the fifth metatarsal of",
        "Vieira et al. 2018). The knee angle is a three-point angle. Heights are estimates from the",
        "camera whose vertical scale is checked on the goal posts (see analysis.json in data/).",
        "Pelvis, plane, limb and knee-share quantities are operational definitions (see kicker_metrics.csv).",
        "",
        "Folders: the detect and SAM 3D Body outputs of each camera are in <camera>_3d_<time>/ and work/;",
        "processed_rec3d_binocular_<time>/ holds the binocular analysis (report_tv/, team_A/, team_B/,",
        "referees/ and data/).",
    ]
    path = out / "README.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def stale_flight(three_d: Path | str) -> str | None:
    """Why the ball flight of a 3D folder is older than the current fit, or None when it is current.

    ``kick-figures`` rewrites the analyses from the stored flight; it never fits it again, so a run
    made before ``FLIGHT_MODEL`` keeps its old ball path until ``ball-rebuild``.
    """
    path = Path(three_d) / "ball_flight.json"
    if not path.is_file():
        return None  # no flight (a ball kept as a smoothed track): nothing to be stale
    current = _module("freekiki_physics").FLIGHT_MODEL
    try:
        found = json.loads(path.read_text(encoding="utf-8")).get("flight_model")
    except (OSError, ValueError):
        return f"{path.name} cannot be read"
    if found == current:
        return None
    return (f"{path.name} comes from the {found or 'first'} flight fit (now {current}); "
            f"the ball path may be wrong: refit it with 'freekiki.py ball-rebuild --out {Path(three_d)} "
            f"--ball-csv <video>_markers.csv', then run the one-click in a new folder")  # fmt: skip


def regenerate_figures(
    kick_dir: Path | str, *, team_a: str = "green", analysis: str | None = None
) -> dict:
    """Rewrites the analyses of a finished kick folder (no detection and no SAM 3D Body runs again).

    The 3D folders, the binocular fit and the poses of the run are read; the teams, the kicker and the
    reports of every analysis are written again.
    """
    out = Path(kick_dir)
    status = json.loads((out / STATUS_NAME).read_text(encoding="utf-8"))
    ts = status["ts"]
    stems = [key.split(":", 1)[1] for key in status["stages"] if key.startswith("3d:")]
    if not stems:
        raise KickInputError(f"no 3D stage in {out / STATUS_NAME}")
    ref_stem = stems[0]
    ref3d = out / f"{ref_stem}_3d_{ts}"
    other3d = out / f"{stems[1]}_3d_{ts}" if len(stems) > 1 else None
    ref_video = Path(status["stages"][f"detect:{ref_stem}"]["inputs"]["video"]["path"])
    for folder in (ref3d, other3d):
        reason = stale_flight(folder) if folder is not None else None
        if reason:
            _log(f"WARNING: {reason}")
    other_video = None
    if other3d is not None:
        other_video = Path(status["stages"][f"detect:{stems[1]}"]["inputs"]["video"]["path"])
    bino_dir = out / f"processed_rec3d_binocular_{ts}" if other3d is not None else None
    # as in run_kick_analysis: the sync of a monocular clip whose name has none
    sync = status.get("sync")
    ref_sync = tuple(sync) if sync and other3d is None else None
    fps = _module("freekiki_detect3d")._video_fps(ref_video)
    found = _analyses(
        out, ref3d=ref3d, other3d=other3d, ref_work=out / "work" / ref_stem,
        other_work=out / "work" / stems[1] if other3d is not None else None,
        ref_video=ref_video, other_video=other_video, bino_dir=bino_dir,
        kick_foot="auto", team_a=team_a, analysis=analysis,
        runner=lambda key, inputs, outputs, fn: fn(),
        setpiece=_chosen_setpiece(ref3d, _stored_choice(ref3d)),
        ref_sync=ref_sync,
    )  # fmt: skip
    return {
        "out": str(out),
        "analyses": found["analyses"],
        "chosen": found["chosen"],
        "fps": fps,
        "kicker_values": found["kicker_values"],
    }


SUPERSEDED = "superseded"  # the run's folder for the outputs a new detection replaced


def _supersede(out: Path, cam_work: Path, stem: str) -> Path | None:
    """Move the old detect runs and SAM 3D Body output of one camera to ``superseded/<time>/<stem>/``.

    Nothing is deleted. Returns the folder they went to, or None when there was nothing to move
    (the first detection of the camera).
    """
    old = [p for p in sorted(cam_work.glob(f"freekiki_predict_{stem}_*")) if p.is_dir()]
    if (cam_work / "sam3dinov3").is_dir():
        old.append(cam_work / "sam3dinov3")
    if not old:
        return None
    dest = out / SUPERSEDED / datetime.now().strftime("%Y%m%d_%H%M%S") / stem
    dest.mkdir(parents=True, exist_ok=True)
    for path in old:
        shutil.move(str(path), str(dest / path.name))
    _log(f"{stem}: earlier detection moved to {dest} (a new one replaces it)")
    return dest


def _backup_derived(out: Path, folder: Path) -> None:
    """Retain the replaced trajectory, C3Ds, figures and reports before rebuilding them."""
    if folder.is_dir():
        dest = out / SUPERSEDED / datetime.now().strftime("%Y%m%d_%H%M%S_%f") / folder.name
        shutil.copytree(folder, dest)
        _log(f"earlier derived results copied to {dest}")


def resume_options(kick_dir: Path | str) -> dict:
    """The options of a finished or interrupted kick run, to run it again in the same folder.

    A run made by this version stores them (``options`` of ``pipeline_status.json``). An older run
    gives back what it recorded: the videos of its detect stages (the reference first), its sync
    and the set piece the user chose.
    """
    out = Path(kick_dir).expanduser().resolve()
    path = out / STATUS_NAME
    if not path.is_file():
        raise KickInputError(f"{out} is not a kick analysis folder (no {STATUS_NAME})")
    data = json.loads(path.read_text(encoding="utf-8"))
    options = dict(data.get("options") or {})
    if not options.get("videos"):
        stages = data.get("stages", {})
        options["videos"] = [
            stages[key]["inputs"]["video"]["path"]
            for key in stages
            if key.startswith("detect:") and "video" in stages[key].get("inputs", {})
        ]
    if not options["videos"]:
        raise KickInputError(f"no video recorded in {path}")
    if options.get("sync") is None and data.get("sync"):
        options["sync"] = data["sync"]
    if "set_piece" not in options:
        ts = data.get("ts")
        stem = Path(options["videos"][0]).stem
        ref3d = out / f"{stem}_3d_{ts}"
        options["set_piece"] = _stored_choice(ref3d) if ref3d.is_dir() else "auto"
    return options


def run_kick_analysis(
    ws,
    videos,
    *,
    reference: int = 1,
    out_dir=None,
    kick_foot: str = "auto",
    reuse: bool = True,
    model: str = "active",
    device: str | None = None,
    sam_device: str | None = None,
    min_camera_ok: float = 0.3,
    team_a: str = "green",
    analysis: str | None = None,
    sync: tuple[int, int] | None = None,
    set_piece: str = "auto",
    contact_frame: int | None = None,
    arrival_observation=None,
    save_frame: int | None = None,
    shot_outcome: str = "auto",
    statures: str | None = None,
    gk_arm_span: float | None = None,
    gk_mass: float | None = None,
    gk_onset_frame: int | None = None,
    save_sigma_along: float | None = None,
) -> dict:
    """The one-click kick analysis. ``videos``: a folder with one or two cut videos, or the files.

    ``sync`` is ``(sync_frame, cut_start)`` for one video whose name has no sync numbers.
    ``set_piece`` is ``auto`` (the ball rule decides; rolling/unseen → ``kick``) or the kind the
    user chose: ``penalty``, ``free_kick``, ``kick``, ``open_play``, ``dribble`` or ``pass``.
    ``contact_frame`` (0-based, reference camera) is a kick contact
    given by hand; without it the flight finds it (:func:`vaila.freekiki_auto_flight.find_contact`).

    The ball is a stage of its own (``ball:<camera>``): a corrected ``<video>_markers.csv`` or a new
    contact redoes the ball, the flight and the stages after them, not the detection nor SAM 3D Body.
    A corrected ``<video>_kiki49.csv`` redoes the detection (marked by hand, no network) and the 3D;
    the outputs it replaces are moved to ``superseded/<time>/`` of the run, never deleted.

    A penalty saved by the keeper's hands: ``save_frame`` (0-based, reference camera) stops the
    flight there and anchors it on the keeper's hands, and ``statures`` (``keeper=2.04,...``) scale
    the people whose stature is known; both are applied by the ``refine`` stage, after the players
    have roles. ``shot_outcome``, ``gk_arm_span``, ``gk_mass``, ``gk_onset_frame`` (an onset marked
    by eye) and ``save_sigma_along`` (the hands' uncertainty along the camera ray, m) go to the
    penalty metrics and the keeper module (``data/keeper_metrics.json``).

    Two videos give three analyses (the monocular of each camera and the binocular); one video gives
    the monocular analysis of that camera. ``analysis`` marks the result in the README. ``out_dir``
    default: ``kick_analysis_<time>/`` next to the first video. Give an existing folder to continue an
    interrupted run (its ``pipeline_status.json`` decides which stages are reused).
    """
    if kick_foot not in KICK_FOOT_CHOICES:
        raise KickInputError(f"--kick-foot must be one of {', '.join(KICK_FOOT_CHOICES)}")
    set_piece_choices = _module("freekiki_setpiece").SET_PIECE_CHOICES
    if set_piece not in set_piece_choices:
        raise KickInputError(f"--set-piece must be one of {', '.join(set_piece_choices)}")
    if contact_frame is not None and int(contact_frame) < 0:
        raise KickInputError("--contact-frame is a 0-based frame of the reference video (>= 0)")
    if save_frame is not None:
        save_frame = int(save_frame)
        if save_frame < 0 or (contact_frame is not None and save_frame <= int(contact_frame)):
            raise KickInputError(
                "--save-frame is a 0-based frame of the reference video after the contact"
            )
    if shot_outcome not in SHOT_OUTCOME_CHOICES:
        raise KickInputError(f"--shot-outcome must be one of {', '.join(SHOT_OUTCOME_CHOICES)}")
    try:
        stature_spec = _module("freekiki_stature").parse_stature_spec(statures)
    except ValueError as exc:
        raise KickInputError(f"--stature: {exc}") from exc
    sigma_along = float(save_sigma_along) if save_sigma_along is not None else None
    if sigma_along is not None and sigma_along <= 0:
        raise KickInputError("--save-sigma-along must be a positive length in metres")
    picked = pick_videos(videos) if isinstance(videos, (str, Path)) else [Path(v) for v in videos]
    cams = validate_inputs(picked, reference, sync=sync)
    ref = cams[0]
    arrivals = _module("freekiki_arrival")
    try:
        observation = arrivals.load_observation(arrival_observation, ref.n_frames)
    except (ValueError, OSError) as exc:
        raise KickInputError(f"arrival observation: {exc}") from exc
    other = cams[1] if len(cams) == 2 else None
    if out_dir is None:
        out = ref.video.parent / f"kick_analysis_{datetime.now():%Y%m%d_%H%M%S}"
    else:
        out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    observation_path = out / "corrections" / ref.stem / "arrival_observation.json"
    if observation is not None and (
        not observation_path.is_file() or arrivals.load_observation(observation_path) != observation
    ):
        arrivals.save_observation(observation_path, observation)
    state = _State(out, datetime.now().strftime("%Y%m%d_%H%M%S"))
    # the sync given for a clip whose name has none: a figures rewrite needs it again
    state.data["sync"] = list(sync) if sync is not None else None
    # what this run was asked for: kick-analysis --resume repeats it on the same folder
    state.data["options"] = {
        "videos": [str(c.video) for c in cams],  # the reference first
        "analysis": analysis,
        "set_piece": set_piece,
        "contact_frame": contact_frame,
        "arrival_observation": str(observation_path) if observation is not None else None,
        "save_frame": save_frame,
        "shot_outcome": shot_outcome,
        "statures": statures,
        "gk_arm_span": gk_arm_span,
        "gk_mass": gk_mass,
        "gk_onset_frame": gk_onset_frame,
        "save_sigma_along": sigma_along,
        "kick_foot": kick_foot,
        "team_a": team_a,
        "model": model,
        "min_camera_ok": min_camera_ok,
        "sync": list(sync) if sync is not None else None,
    }
    state.save()
    ts = state.ts
    work = out / "work"
    run: dict = {"reference_video": ref.video.name, "other_video": other.video.name if other else None,
           "fps": ref.fps, "created": datetime.now().isoformat(timespec="seconds"), "stages": {}}  # fmt: skip
    _log(f"reference {ref.video.name} | other {other.video.name if other else 'none'} | fps {ref.fps:.3f} "
         f"| out {out}")  # fmt: skip

    def camera_stages(
        cam: CameraInput, contact: int | None, arrival=None
    ) -> tuple[Path, Path, dict]:
        runs_before = state.runs
        detect_key = f"detect:{cam.stem}"
        inputs = {"video": _file_key(cam.video), "model": model}
        if arrivals.calibration_key(arrival) is not None:
            inputs["goal_calibration"] = arrivals.calibration_key(arrival)
        if cam.keypoints is not None:
            inputs["keypoints"] = _file_key(cam.keypoints)  # a new mark invalidates the stage
        cam_work = work / cam.stem

        def refit(predict):
            if arrivals.calibration_key(arrival) is not None:
                _, width, height = _module("freekiki_detect3d")._video_info(cam.video)
                arrivals.refit_goal_calibration(Path(predict), arrival, width, height)
            return Path(predict)

        def detect() -> dict:
            detect_video = _module("freekiki").detect_video
            # a new detection replaces the camera: its old detect run and SAM 3D Body output move
            # aside, so every lookup finds one of each and SAM 3D Body sees the new focal
            _supersede(out, cam_work, cam.stem)
            cam_work.mkdir(parents=True, exist_ok=True)
            if cam.keypoints is not None:
                predict = detect_video(
                    ws,
                    cam.video,
                    model=model,
                    output_dir=cam_work,
                    device=device,
                    keypoints_csv=cam.keypoints,
                )
                return {"predict": refit(predict), "screen": {"manual": cam.keypoints.name}}
            screen = _screen_or_raise(ws, cam.video, model, min_camera_ok)
            predict = detect_video(ws, cam.video, model=model, output_dir=cam_work, device=device)
            return {"predict": refit(predict), "screen": screen}

        found = state.stage(detect_key, inputs, lambda r: [Path(r["predict"]) / "quality.json"],
                            detect, reuse=reuse, legacy=("markers",))  # fmt: skip
        predict = Path(found["predict"])
        detect_ran = state.runs != runs_before
        threed_key = f"3d:{cam.stem}"
        threed_dir = out / f"{cam.stem}_3d_{ts}"
        ball_inputs = _jsonable({"markers": _file_key(cam.markers), "contact_frame": contact,
                                 "flight_model": _module("freekiki_physics").FLIGHT_MODEL})  # fmt: skip
        if arrival is not None:
            ball_inputs["arrival_observation"] = arrival
        d3 = _module("freekiki_detect3d")

        def threed() -> dict:
            _backup_derived(out, threed_dir)
            result = d3.detect_3d_run(predict, cam.video, threed_dir, work_dir=cam_work,
                                      device=sam_device, ball_csv=cam.markers, players_on_field=True,
                                      contact_frame=contact, fresh_sam=detect_ran,
                                      **({"arrival_observation": arrival} if arrival is not None else {}))  # fmt: skip
            result["ball_inputs"] = ball_inputs  # the ball this 3D run already wrote
            return result

        runs_3d = state.runs
        res = state.stage(threed_key, {"predict": str(predict), **inputs, "sam_device": sam_device, "players_on_field": True},
                          [threed_dir / "poses_3d_on_field.csv", threed_dir / "ball.csv"], threed,
                          reuse=reuse and not detect_ran, legacy=("markers", "flight_model"))  # fmt: skip
        threed_ran = state.runs != runs_3d

        def ball() -> dict:
            if threed_ran and res.get("ball_inputs") == ball_inputs:
                return {"from": "3d"}  # the 3D stage has just written this very ball
            _backup_derived(out, threed_dir)
            summary = d3.rebuild_ball_in_place(threed_dir, cam.video, cam_work, cam.markers,
                                               contact_frame=contact,
                                               **({"arrival_observation": arrival} if arrival is not None else {}))  # fmt: skip
            return {
                "from": "ball",
                "ball_flight": (summary.get("field3d") or {}).get("ball_flight"),
            }

        state.stage(f"ball:{cam.stem}", ball_inputs,
                    [threed_dir / "ball.csv", threed_dir / "ball_3d_on_field.csv"], ball,
                    reuse=reuse and not threed_ran)  # fmt: skip
        return threed_dir, cam_work, res

    runs_at_start = state.runs
    ref3d, ref_work, _ = camera_stages(ref, contact_frame, observation)  # reference frame
    other3d, other_work = None, None
    if other is not None:
        other3d, other_work, _ = camera_stages(other, None)
    if reuse and state.runs != runs_at_start:
        # The later stages only name the 3D folders, not their content: a camera stage that ran
        # again (a new fit, new ball marks, a new video) makes every stage built from it stale.
        _log("a camera stage ran again: the stages built from the cameras run again too")
        reuse = False

    def players() -> dict:
        folders = [ref3d] if other3d is None else [ref3d, other3d]
        return _module("freekiki_player_ids").run_player_ids(folders)

    player_outputs = [ref3d / "player_map.csv"] + ([other3d / "player_map.csv"] if other3d else [])
    state.stage("players", {"ref": ref3d.name, "other": other3d.name if other3d else None},
                player_outputs, players, reuse=reuse)  # fmt: skip
    if stature_spec or save_frame is not None:
        # the known statures and the save need the roles (keeper, kicker): the reference ball is
        # rebuilt with them, then the players' tables follow the rescaled poses
        def refine() -> dict:
            if ref.markers is None:
                raise KickInputError(
                    "--save-frame and --stature rebuild the ball: the reference camera needs its "
                    f"corrected ball ({ref.video.stem}_markers.csv)"
                )
            roles = _roles_sam(ref3d / "player_map.csv")
            by_person, prior = _module("freekiki_stature").resolve_statures(stature_spec, roles)
            save = None
            if save_frame is not None:
                save = {"frame": save_frame, "keeper_id": roles.get("keeper")}
                if sigma_along is not None:
                    save["sigma_along"] = sigma_along
            summary = _module("freekiki_detect3d").rebuild_ball_in_place(
                ref3d, ref.video, ref_work, ref.markers, contact_frame=contact_frame,
                statures=(by_person, prior) if stature_spec else None, save=save,
                **({"arrival_observation": observation} if observation is not None else {}),
            )  # fmt: skip
            players()
            after = _roles_sam(ref3d / "player_map.csv")
            if after != roles:
                _log(f"refine: the roles changed with the rescaled poses ({roles} -> {after})")
            return {
                "roles": roles,
                "statures": {str(k): list(v) for k, v in by_person.items()},
                "prior_m": prior,
                "save": save,
                "ball_flight": (summary.get("field3d") or {}).get("ball_flight"),
            }

        runs_before = state.runs
        state.stage(
            "refine",
            _jsonable(
                {
                    "ref": ref3d.name,
                    "statures": stature_spec,
                    "save_frame": save_frame,
                    "sigma_along": sigma_along,
                    "contact_frame": contact_frame,
                    "markers": _file_key(ref.markers) if ref.markers else None,
                    "flight_model": _module("freekiki_physics").FLIGHT_MODEL,
                    "pose_model": "gk-plane-v1",
                }
            ),  # fmt: skip
            [ref3d / "ball_flight.json", ref3d / "stature_by_person.csv"],
            refine,
            reuse=reuse,
        )
        if state.runs != runs_before:
            reuse = False  # the ball and the poses changed: every later stage reads them
    # the tolerance is an input: a run that reuses the stages classifies again when the rule changes
    setpiece = state.stage(
        "setpiece",
        {
            "ref": ref3d.name,
            "penalty_tolerance_m": _module("freekiki_setpiece").PENALTY_TOLERANCE_M,
            "set_piece": set_piece,  # a new choice classifies again
        },
        [_layout().data_dir(ref3d) / "setpiece.json"],
        lambda: _chosen_setpiece(ref3d, set_piece),
        reuse=reuse,
    )

    bino_dir = None
    if other is not None and other3d is not None:
        bino_dir = out / f"processed_rec3d_binocular_{ts}"
        assert other_work is not None  # set with other3d
        bdata = _layout().data_dir(bino_dir)
        fit = state.stage("binocular", {"ref": ref3d.name, "other": other3d.name},
                          [bdata / "ball_flight.json", bdata / "ball_3d_on_field.csv"],
                          lambda: _module("freekiki_stereo").run_binocular(ref3d, other3d, out_dir=bdata),
                          reuse=reuse)  # fmt: skip
        run["fit"] = {"peak_speed_mps": fit["fit"]["peak_speed_mps"]} if "fit" in fit else {}
        poses_csv = bdata / "poses_3d_binocular.csv"
        tri = state.stage("triangulate", {"ref": ref3d.name, "other": other3d.name}, [poses_csv],
                          lambda: _triangulate(ref3d, other3d, ref_work, other_work, poses_csv, ref.video, other.video),
                          reuse=reuse)  # fmt: skip
        run["triangulation"] = {
            k: tri[k] for k in ("rows", "valid_rows", "height_check_m", "vitruvian_frames")
        }

    def runner(key, inputs, outputs, fn):
        return state.stage(key, inputs, outputs, fn, reuse=reuse)

    found = _analyses(
        out, ref3d=ref3d, other3d=other3d, ref_work=ref_work, other_work=other_work,
        ref_video=ref.video, other_video=other.video if other else None, bino_dir=bino_dir,
        kick_foot=kick_foot, team_a=team_a, analysis=analysis, runner=runner, setpiece=setpiece,
        ref_sync=sync if other is None else None,
        penalty_opts={"save_frame": save_frame, "shot_outcome": shot_outcome,
                      "statures": stature_spec, "gk_arm_span": gk_arm_span, "gk_mass": gk_mass,
                      "gk_onset_frame": gk_onset_frame, "save_sigma_along": sigma_along,
                      # a confirmed arrival replaces the keeper's hands: no +/- sigma refits of those
                      "manual_arrival": bool(observation and observation.get("type") == "keeper_hand")},
    )  # fmt: skip
    run.update(
        {
            "analyses": found["analyses"],
            "chosen": found["chosen"],
            "kicker_values": found["kicker_values"],
            "warnings": found["warnings"],
        }
    )
    run["kicker_id"] = _kicker_id(ref3d / "player_map.csv")
    run["setpiece"] = setpiece
    run["contact"] = _module("freekiki_auto_flight").flight_contact(ref3d)
    run["inputs"] = {
        "ball": str(ref.markers),
        "field": str(ref.keypoints) if ref.keypoints else None,
        "arrival_observation": str(observation_path) if observation is not None else None,
    }
    run["stages"] = {key: rec.get("status", "") for key, rec in state.data["stages"].items()}
    _readme(out, run)
    _log(f"done: {out}")
    return {"out": str(out), "summary": run, "status_file": str(state.path)}
