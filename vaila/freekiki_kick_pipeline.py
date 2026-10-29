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
Version: 0.4.12
Created: 28 October 2026
Update Date: 29 October 2026

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
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from .freekiki_stereo import nominal_clock, parse_sync_name
except ImportError:
    from freekiki_stereo import nominal_clock, parse_sync_name  # ty: ignore[unresolved-import]

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
STATUS_NAME = "pipeline_status.json"
KICK_FOOT_CHOICES = ("auto", "left", "right")
# the pynalty metrics shown in the README of a penalty run (the rest is in data/penalty_metrics.csv)
PENALTY_KEEP = ("shot_outcome", "vel_kmh", "dist", "coord_x", "coord_z", "gk_response_time",
                "gk_dive_time_s", "gk_vel_ms")  # fmt: skip
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
    """The cut videos of a folder: the video files whose names carry a sync (``_sync_N_frames_A_to_B``)."""
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_SUFFIXES
        and parse_sync_name(p.stem) is not None
    )


def pick_videos(source) -> list[Path]:
    """One or two videos from a folder (one or two cut videos) or the files given."""
    path = Path(source).expanduser()
    if path.is_dir():
        videos = cut_videos_in(path)
        if len(videos) not in (1, 2):
            names = ", ".join(v.name for v in videos) or "none"
            raise KickInputError(
                f"{path} needs one or two cut videos (found {len(videos)}: {names}). Two videos "
                "are named '..._sync_N_frames_A_to_B...'. Give the files with --videos instead."
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
    """E0: two cut videos, each with its sync in the name and its corrected ball markers.

    ``reference`` (1 or 2) is the camera whose frames are the output frames. ``sync`` is
    ``(sync_frame, cut_start)`` for one video whose name carries no sync numbers; the name wins when
    it has them.
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
    for video in videos:
        video = Path(video).expanduser().resolve()
        if not video.is_file():
            raise KickInputError(f"video not found: {video}")
        parsed = parse_sync_name(video.stem)
        if parsed is None and len(videos) == 1:
            parsed = sync or (0, 0)  # one camera alone: no sync is needed unless it is given
        if parsed is None:
            raise KickInputError(
                f"{video.name} has no sync in its name ('_sync_N_frames_A_to_B'); "
                "rename the cut video as the cut tool wrote it"
            )
        markers = video.with_name(f"{video.stem}_markers.csv")
        if not markers.is_file():
            raise KickInputError(
                f"ball markers not found: {markers}. Correct the ball of this camera in "
                "getpixelvideo (the file is written next to the video) and run again"
            )
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
    ) -> dict:
        """Run ``fn`` unless this stage finished with the same inputs and its outputs exist."""
        record = self.data["stages"].get(key)
        same = (
            bool(record)
            and record.get("status") == "ok"
            and record.get("inputs") == _jsonable(inputs)
        )
        if (
            reuse
            and same
            and all(p.exists() for p in _outputs_of(outputs, record.get("result", {})))
        ):
            _log(f"{key}: already done, reused")
            return record["result"]
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


def _kicker_id(player_map: Path) -> int:
    with player_map.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["role"] == "kicker":
                return int(row["player_id"])
    raise KickStageError(f"no kicker in {player_map}: the players link found no kicker")


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
    """The camera of a 3D folder. ``sync`` is ``(sync_frame, cut_start)``; None reads the name."""
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
) -> dict:
    """E5: the kicker's metrics and figures. ``binocular`` is the folder with the ball fit."""
    kicker = _module("freekiki_kicker")
    fps = _module("freekiki_detect3d")._video_fps(ref_video)
    n = int(nominal_clock(ref_video).size)
    flight = json.loads((binocular / "ball_flight.json").read_text(encoding="utf-8"))
    result = kicker.analyze_kicker(
        kicker.read_poses(poses_csv),
        kicker.read_ball_xyz(binocular / "ball_3d_on_field.csv", n),
        kicker.read_ball_pixels(ref3d / "ball.csv"),
        kicker_id=_kicker_id(ref3d / "player_map.csv"),
        fps=fps,
        kick_foot=kick_foot,
        kick_time_s=flight.get("kick_time_s"),
        fit_peak_speed_mps=flight.get("peak_speed_mps"),
        fit_spin_rps=flight.get("spin_rps"),
    )
    views = None
    if other3d is not None and ref_work is not None and other_work is not None:
        views = _overlay_views(ref3d, other3d, ref_work, other_work)
    paths = kicker.write_outputs(result, out, views=views)
    values = {m["metric"]: m["value"] for m in result["metrics"]}
    keep = ("contact_frame", "kick_foot", "v_foot_pre_mps", "v_ball_post_mps", "ratio_vball_vfoot",
            "d_support_ball_m", "approach_angle_deg", "knee_ext_peak_deg_s",
            "proximal_distal_order", "goal_error_m")  # fmt: skip
    return {
        "fps": fps,
        "values": {k: values.get(k, "") for k in keep},
        "warnings": result["warnings"],
        "files": {k: str(v) for k, v in paths.items()},
    }


def _penalty(
    ref3d: Path, ref_video: Path, ref_work: Path, folder: Path, ball_csv: Path, *, goal_x: float
) -> dict:
    """The pynalty metrics and report of a penalty (reference camera pixels, the 3D crossing of ``ball_csv``).

    ``folder`` is the chosen analysis: its ``data/`` gets the metrics, its ``report_tv/penalty/`` the report.
    """
    penalty = _module("freekiki_penalty")
    kicker = _module("freekiki_kicker")
    fps = _module("freekiki_detect3d")._video_fps(ref_video)
    n = int(nominal_clock(ref_video).size)
    keeper = penalty.keeper_sam_id(ref3d / "player_map.csv")
    boxes = penalty.read_boxes(_sam_boxes_csv(ref_work))
    result = penalty.penalty_metrics(
        field_rows=penalty.read_field_rows(ref3d / "field_kps_3d.csv"),
        ball_px=kicker.read_ball_pixels(ref3d / "ball.csv"),
        ball_xyz=kicker.read_ball_xyz(ball_csv, n),
        keypoints=_module("freekiki_pose_binocular").read_keypoints2d(_keypoints_csv(ref_work), n),
        keeper_boxes=boxes.get(keeper, {}) if keeper is not None else {},
        keeper_sam=keeper,
        fps=fps,
        goal_x=goal_x,
    )
    files = penalty.write(_layout().data_dir(folder), result)
    html = penalty.report(folder / "report_tv" / "penalty", ref_video, fps, result)
    return {
        "available": bool(result["available"]),
        "reason": result.get("reason", ""),
        "metrics": {
            k: result["metrics"][k] for k in PENALTY_KEEP if k in result.get("metrics", {})
        },
        "files": {k: str(v) for k, v in files.items()},
        "report": html,
    }


def _mono_analysis(
    key, cam3d, video, work, is_ref, *, kick_foot, team_a, runner, report=True, sync=None
):
    """One camera alone: its teams, its kicker and its report. Returns the entry and the kicker values.

    A failure is recorded in the entry; the other analyses still run. ``report`` False leaves the
    freekiki ball report out (a penalty has its pynalty report instead). ``sync`` is the
    ``(sync_frame, cut_start)`` of a video whose name has none.
    """
    entry: dict = {"folder": str(cam3d), "status": "ok"}
    fps = _module("freekiki_detect3d")._video_fps(video)
    try:
        teams = runner(f"teams:{key}", {"player_map": cam3d.name, "sync": sync},
                       [cam3d / "data" / "team_map.csv"],
                       lambda: _teams_mono(cam3d, video, work, is_ref=is_ref, key=key, team_a=team_a,
                                           fps=fps, sync=sync))  # fmt: skip
        entry["counts"] = teams["counts"]
        kick = runner(f"kicker:{key}", {"poses": "poses_3d_on_field_players.csv", "kick_foot": kick_foot},
                      [cam3d / "data" / "kicker_metrics.csv"],
                      lambda: _kicker(cam3d, cam3d, cam3d / "poses_3d_on_field_players.csv",
                                      _layout().data_dir(cam3d), video, kick_foot=kick_foot))  # fmt: skip
        if report:
            runner(f"report:{key}", {"run": cam3d.name},
                   [cam3d / "report_tv" / "pt" / "relatorio_bola.html",
                    cam3d / "report_tv" / "en" / "ball_report.html"],
                   lambda: _module("freekiki_report").build_report(cam3d, lang="both", video=video))  # fmt: skip
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
    # a penalty needs the ball at rest (the goal side): without it the run is a free kick analysis
    penalty = kind == "penalty" and (setpiece or {}).get("goal_x_m") is not None
    results: dict[str, dict] = {}
    kicker_values: dict[str, dict] = {}
    monos = [("mono-ref", ref3d, ref_video, ref_work, True)]
    if two:
        monos.append(("mono-other", other3d, other_video, other_work, False))
    for key, cam3d, video, work, is_ref in monos:
        entry, values = _mono_analysis(key, cam3d, video, work, is_ref, kick_foot=kick_foot,
                                       team_a=team_a, runner=runner, report=not penalty,
                                       sync=ref_sync if is_ref else None)  # fmt: skip
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
        teams = runner("teams:binocular", {"poses": poses.name}, [bdata / "team_map.csv"],
                       lambda: _teams_binocular(ref3d, other3d, ref_video, other_video, ref_work, other_work,
                                                bino_dir, poses, ball_csv, team_a=team_a, fps=fps))  # fmt: skip
        entry["counts"] = teams["counts"]
        kick = runner("kicker:binocular", {"poses": poses.name, "kick_foot": kick_foot},
                      [bdata / "kicker_metrics.csv", bdata / "kicker_curves_pt.png"],
                      lambda: _kicker(ref3d, fit, poses, bdata, ref_video, kick_foot=kick_foot,
                                      other3d=other3d, ref_work=ref_work, other_work=other_work))  # fmt: skip
        kicker_values["binocular"] = kick["values"]
        if not penalty:
            runner("report:binocular", {"binocular": bino_dir.name},
                   [bino_dir / "report_tv" / "pt" / "relatorio_bola.html",
                    bino_dir / "report_tv" / "en" / "ball_report.html"],
                   lambda: _module("freekiki_report").build_report(bino_dir, lang="both", video=ref_video))  # fmt: skip
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
            results[chosen]["penalty"] = runner(
                f"penalty:{chosen}", {"ball": ball_csv.name, "goal_x_m": setpiece["goal_x_m"]},
                [_layout().data_dir(folder) / "penalty_metrics.csv"],
                lambda: _penalty(ref3d, ref_video, ref_work, folder, ball_csv,
                                 goal_x=setpiece["goal_x_m"]))  # fmt: skip
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
    for key, entry in results.items():
        entry["chosen"] = key == chosen
    return {"analyses": results, "chosen": chosen, "kicker_values": kicker_values.get(chosen, {}),
            "warnings": []}  # fmt: skip


def _penalty_lines(penalty: dict | None) -> list[str]:
    if penalty is None:
        return []
    if not penalty.get("available"):
        return [f"Penalty: pynalty metrics not available here ({penalty.get('reason', '')})", ""]
    return [
        "Penalty (pynalty metrics; data/penalty_metrics.csv; report in report_tv/penalty/):",
        *[f"  {k}: {v}" for k, v in penalty.get("metrics", {}).items()],
        "",
    ]


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
        "Limits: at 59.94 fps the contact lasts about 1 ms, so the speed at impact is not measured;",
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
    other_video = None
    if other3d is not None:
        other_video = Path(status["stages"][f"detect:{stems[1]}"]["inputs"]["video"]["path"])
    bino_dir = out / f"processed_rec3d_binocular_{ts}" if other3d is not None else None
    fps = _module("freekiki_detect3d")._video_fps(ref_video)
    found = _analyses(
        out, ref3d=ref3d, other3d=other3d, ref_work=out / "work" / ref_stem,
        other_work=out / "work" / stems[1] if other3d is not None else None,
        ref_video=ref_video, other_video=other_video, bino_dir=bino_dir,
        kick_foot="auto", team_a=team_a, analysis=analysis,
        runner=lambda key, inputs, outputs, fn: fn(),
        setpiece=_chosen_setpiece(ref3d, _stored_choice(ref3d)),
    )  # fmt: skip
    return {
        "out": str(out),
        "analyses": found["analyses"],
        "chosen": found["chosen"],
        "fps": fps,
        "kicker_values": found["kicker_values"],
    }


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
) -> dict:
    """The one-click kick analysis. ``videos``: a folder with one or two cut videos, or the files.

    ``sync`` is ``(sync_frame, cut_start)`` for one video whose name has no sync numbers.
    ``set_piece`` is ``auto`` (the ball rule decides) or the kind the user chose: ``penalty`` or ``free_kick``.

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
    picked = pick_videos(videos) if isinstance(videos, (str, Path)) else [Path(v) for v in videos]
    cams = validate_inputs(picked, reference, sync=sync)
    ref = cams[0]
    other = cams[1] if len(cams) == 2 else None
    if out_dir is None:
        out = ref.video.parent / f"kick_analysis_{datetime.now():%Y%m%d_%H%M%S}"
    else:
        out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    state = _State(out, datetime.now().strftime("%Y%m%d_%H%M%S"))
    state.save()
    ts = state.ts
    work = out / "work"
    run: dict = {"reference_video": ref.video.name, "other_video": other.video.name if other else None,
           "fps": ref.fps, "created": datetime.now().isoformat(timespec="seconds"), "stages": {}}  # fmt: skip
    _log(f"reference {ref.video.name} | other {other.video.name if other else 'none'} | fps {ref.fps:.3f} "
         f"| out {out}")  # fmt: skip

    def camera_stages(cam: CameraInput) -> tuple[Path, Path, dict]:
        detect_key = f"detect:{cam.stem}"
        inputs = {"video": _file_key(cam.video), "markers": _file_key(cam.markers), "model": model}
        if cam.keypoints is not None:
            inputs["keypoints"] = _file_key(cam.keypoints)  # a new mark invalidates the stage
        cam_work = work / cam.stem

        def detect() -> dict:
            detect_video = _module("freekiki").detect_video
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
                return {"predict": Path(predict), "screen": {"manual": cam.keypoints.name}}
            screen = _screen_or_raise(ws, cam.video, model, min_camera_ok)
            predict = detect_video(ws, cam.video, model=model, output_dir=cam_work, device=device)
            return {"predict": Path(predict), "screen": screen}

        found = state.stage(
            detect_key, inputs, lambda r: [Path(r["predict"]) / "quality.json"], detect, reuse=reuse
        )
        predict = Path(found["predict"])
        threed_key = f"3d:{cam.stem}"
        threed_dir = out / f"{cam.stem}_3d_{ts}"

        def threed() -> dict:
            return _module("freekiki_detect3d").detect_3d_run(predict, cam.video, threed_dir, work_dir=cam_work,
                                 device=sam_device, ball_csv=cam.markers,
                                 players_on_field=True)  # fmt: skip

        res = state.stage(threed_key, {"predict": str(predict), **inputs, "sam_device": sam_device, "players_on_field": True},
                          [threed_dir / "poses_3d_on_field.csv", threed_dir / "ball.csv"], threed,
                          reuse=reuse)  # fmt: skip
        return threed_dir, cam_work, res

    ref3d, ref_work, _ = camera_stages(ref)
    other3d, other_work = None, None
    if other is not None:
        other3d, other_work, _ = camera_stages(other)

    def players() -> dict:
        folders = [ref3d] if other3d is None else [ref3d, other3d]
        return _module("freekiki_player_ids").run_player_ids(folders)

    player_outputs = [ref3d / "player_map.csv"] + ([other3d / "player_map.csv"] if other3d else [])
    state.stage("players", {"ref": ref3d.name, "other": other3d.name if other3d else None},
                player_outputs, players, reuse=reuse)  # fmt: skip
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
    run["stages"] = {key: rec.get("status", "") for key, rec in state.data["stages"].items()}
    _readme(out, run)
    _log(f"done: {out}")
    return {"out": str(out), "summary": run, "status_file": str(state.path)}
