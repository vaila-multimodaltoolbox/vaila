"""Equalize two videos on a configurable common event grid using local CUDA RIFE.

Update Date: 29 October 2026
Version: 0.4.12

Frames are 1-based and inclusive. Generated images are estimates, not acquired
measurements. Linear spacing between events is assumed; output FPS/count can
change playback speed independently of the reference clock.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib
import inspect
import json
import math
import shlex
import signal
import subprocess
import sys
import tempfile
import types
from collections.abc import Callable, Iterator, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .ffmpeg_utils import get_ffprobe_path
    from .sync_events import (
        EventSyncError,
        _encode_samples,
        calibrate_camera,
        parse_event_frames,
        rational_rate,
    )
except ImportError:
    from ffmpeg_utils import get_ffprobe_path  # ty: ignore[unresolved-import]
    from sync_events import (  # ty: ignore[unresolved-import]
        EventSyncError,
        _encode_samples,
        calibrate_camera,
        parse_event_frames,
        rational_rate,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RIFE = ROOT / ".local/third_party/Practical-RIFE"
DEFAULT_WEIGHTS = ROOT / "vaila/models/rife/train_log"
WARNING = (
    "Generated frames are estimates; ball, feet and occlusions may show artifacts. "
    "Not every source original survives resampling. No new physical measurements "
    "are recovered. Reference FPS and uniform spacing between events are assumptions. "
    "Increasing FPS creates visual estimates. Choosing FPS and count independently "
    "changes playback speed. Outputs have no audio."
)


def probe(path: str | Path) -> dict[str, Any]:
    """Probe exact rational FPS and actually decoded frame count."""
    path = Path(path).expanduser().resolve(strict=True)
    result = subprocess.run(
        [
            get_ffprobe_path(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    try:
        stream = json.loads(result.stdout)["streams"][0]
        info = {
            "path": str(path),
            "frame_count": int(stream["nb_read_frames"]),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": str(rational_rate(stream["avg_frame_rate"], "Source FPS")),
        }
        if min(info["frame_count"], info["width"], info["height"]) < 1:
            raise ValueError
        return info
    except (KeyError, IndexError, ValueError) as exc:
        raise EventSyncError(f"Cannot probe video: {path}") from exc


def _optional_text(value: Any) -> str | None:
    return None if value is None or not str(value).strip() else str(value).strip()


def _round_half_up(value: Fraction) -> int:
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def output_grid(
    reference_events: Sequence[int],
    reference_count: int,
    rate: str,
    output_fps: Any = None,
    output_frame_count: Any = None,
) -> dict[str, Any]:
    """One exact grid policy for preview, CLI and both camera maps."""
    ref_fps = rational_rate(rate, "Reference FPS")
    if len(reference_events) != 3:
        raise EventSyncError("Enter exactly three events: start, contact, end")
    # Validate before deriving intervals or dividing by their lengths.
    calibrate_camera(
        "Reference",
        {"events": list(reference_events), "frame_count": reference_count},
        [Fraction(0), Fraction(1), Fraction(2)],
    )
    requested_fps = _optional_text(output_fps)
    requested_count = _optional_text(output_frame_count)
    try:
        fps = ref_fps if requested_fps is None else Fraction(requested_fps)
        if fps <= 0:
            raise ValueError
    except (ValueError, ZeroDivisionError) as exc:
        raise EventSyncError("Output FPS: enter a positive finite decimal or fraction") from exc
    span = reference_events[-1] - reference_events[0]
    reference_duration = Fraction(span) / ref_fps
    try:
        if requested_count is not None and not requested_count.removeprefix("+").isdecimal():
            raise ValueError
        count = (
            int(requested_count)
            if requested_count is not None
            else _round_half_up(reference_duration * fps) + 1
        )
        if count < 3:
            raise ValueError
    except ValueError as exc:
        raise EventSyncError("Output frame count: use an integer of at least 3 frames") from exc
    ideal_contact = Fraction(reference_events[1] - reference_events[0], span) * (count - 1)
    contact = min(count - 1, max(2, _round_half_up(ideal_contact) + 1))
    duration = Fraction(count - 1) / fps
    ref_contact_time = Fraction(reference_events[1] - reference_events[0]) / ref_fps
    contact_time = Fraction(contact - 1) / fps
    return {
        "requested_output_fps": requested_fps,
        "requested_output_frame_count": requested_count,
        "output_fps": str(fps),
        "output_frame_count": count,
        "output_event_frames": [1, contact, count],
        "reference_duration_rational": str(reference_duration),
        "output_duration_rational": str(duration),
        "output_file_duration_rational": str(Fraction(count) / fps),
        "duration_change_rational": str(duration - reference_duration),
        "duration_change_percent_rational": str((duration / reference_duration - 1) * 100),
        "contact_ideal_output_frame_rational": str(ideal_contact + 1),
        "contact_rounding_adjustment_rational": str((contact - 1 - ideal_contact) / fps),
        "contact_reference_time_rational": str(ref_contact_time),
        "contact_output_time_rational": str(contact_time),
        "contact_time_change_rational": str(contact_time - ref_contact_time),
    }


def _camera_map(
    reference_events: Sequence[int],
    source_events: Sequence[int],
    source_count: int,
    grid: dict[str, Any],
) -> list[dict[str, Any]]:
    if len(source_events) != 3:
        raise EventSyncError("Enter exactly three events: start, contact, end")
    calibrate_camera(
        "Source",
        {"events": list(source_events), "frame_count": source_count},
        [Fraction(0), Fraction(1), Fraction(2)],
    )
    fps = Fraction(grid["output_fps"])
    anchors = grid["output_event_frames"]
    event_names = dict(zip(anchors, ("start", "contact", "end"), strict=True))
    audit = {key: value for key, value in grid.items() if key not in ("output_event_frames",)}
    audit.update(
        zip(
            ("output_start_frame", "output_contact_frame", "output_end_frame"), anchors, strict=True
        )
    )
    rows = []
    for output in range(1, grid["output_frame_count"] + 1):
        segment = int(output > anchors[1])
        o0, o1 = anchors[segment : segment + 2]
        r0, r1 = reference_events[segment : segment + 2]
        s0, s1 = source_events[segment : segment + 2]
        alpha = Fraction(output - o0, o1 - o0)
        ref = r0 + alpha * (r1 - r0)
        position = s0 + alpha * (s1 - s0)
        lo, hi = math.floor(position), math.ceil(position)
        rows.append(
            {
                "output_frame": output,
                "reference_frame": int(ref) if ref.denominator == 1 else str(ref),
                "output_time_rational": str(Fraction(output - 1) / fps),
                "source_position": str(position),
                "source_frame_left": lo,
                "source_frame_right": hi,
                "interpolation_fraction": str(position - lo),
                "kind": "original" if lo == hi else "generated",
                "reference_position": str(ref),
                "event": event_names.get(output, ""),
                **audit,
            }
        )
    return rows


def frame_map(
    reference_events: Sequence[int],
    source_events: Sequence[int],
    reference_count: int,
    source_count: int,
    rate: str,
    *,
    output_fps: Any = None,
    output_frame_count: Any = None,
) -> list[dict[str, Any]]:
    """Map an inclusive common grid to exact fractional source positions."""
    grid = output_grid(reference_events, reference_count, rate, output_fps, output_frame_count)
    return _camera_map(reference_events, source_events, source_count, grid)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    reference, source = probe(args.reference), probe(args.source)
    if reference["path"] == source["path"]:
        raise EventSyncError("Choose two different camera videos")
    ref_events = parse_event_frames(args.reference_events, "Reference events")
    src_events = parse_event_frames(args.source_events, "Source events")
    grid = output_grid(
        ref_events,
        reference["frame_count"],
        reference["fps"],
        getattr(args, "output_fps", None),
        getattr(args, "output_frame_count", None),
    )
    cameras = {}
    for role, camera, events in (
        ("reference", reference, ref_events),
        ("source", source, src_events),
    ):
        rows = _camera_map(ref_events, events, camera["frame_count"], grid)
        originals = sum(row["kind"] == "original" for row in rows)
        cameras[role] = dict(
            camera,
            events=events,
            rows=rows,
            original_output_count=originals,
            generated_output_count=len(rows) - originals,
        )
    return {
        "vaila_event_equalizer": 1,
        "method": "rife_reference_grid"
        if grid["requested_output_fps"] is None and grid["requested_output_frame_count"] is None
        else "rife_common_event_grid",
        "frame_base": 1,
        "audio": False,
        "warning": WARNING,
        **grid,
        "cameras": cameras,
    }


def preview_text(manifest: dict[str, Any]) -> str:
    """Human-readable audit shared by the window and export terminal."""

    def seconds(key: str) -> str:
        value = Fraction(manifest[key])
        return f"{value} s ({float(value):.6f} s)"

    fps = Fraction(manifest["output_fps"])
    lines = [
        f"Output: {fps} FPS ({float(fps):.6f}); {manifest['output_frame_count']} frames per camera",
        f"First-to-last frame: {seconds('output_duration_rational')}",
        f"File duration: {seconds('output_file_duration_rational')}",
        f"Reference first-to-last: {seconds('reference_duration_rational')}",
        f"Duration change: {seconds('duration_change_rational')} ({float(Fraction(manifest['duration_change_percent_rational'])):+.6f}%)",
        f"Output start/contact/end: {manifest['output_event_frames']}",
        f"Contact rounding/clamping adjustment: {seconds('contact_rounding_adjustment_rational')}",
        f"Contact time change vs reference: {seconds('contact_time_change_rational')}",
    ]
    for role, camera in manifest["cameras"].items():
        lines.append(
            f"{role}: {camera['original_output_count']} original, {camera['generated_output_count']} generated"
        )
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RifeInterpolator:
    """Practical-RIFE 4.x adapter; imported only in the isolated export process."""

    def __init__(self, checkout: Path, weights: Path):
        import torch

        if not torch.cuda.is_available():
            raise EventSyncError(
                "RIFE requires CUDA. Check NVIDIA access and the CUDA PyTorch group."
            )
        checkout, weights = checkout.expanduser().resolve(), weights.expanduser().resolve()
        for path in (
            checkout / "model/warplayer.py",
            weights / "RIFE_HDv3.py",
            weights / "IFNet_HDv3.py",
            weights / "flownet.pkl",
        ):
            if not path.is_file():
                raise EventSyncError(f"Missing RIFE file: {path}. See Syncvid Help → RIFE setup.")
        # The official bundle imports train_log.*; bind that package to the chosen
        # model directory without copying code or modifying the third-party tree.
        sys.path.insert(0, str(checkout))
        package = types.ModuleType("train_log")
        package.__path__ = [str(weights)]
        sys.modules["train_log"] = package
        module = importlib.import_module("train_log.RIFE_HDv3")
        self.model = module.Model()
        if "timestep" not in inspect.signature(self.model.inference).parameters:
            raise EventSyncError("This RIFE model does not support arbitrary timestep")
        # Strict load prevents upstream's permissive loader silently accepting
        # mismatched weights, and weights_only avoids loading arbitrary objects.
        state = torch.load(weights / "flownet.pkl", map_location="cuda", weights_only=True)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        # Official 4.25 weights also contain training-only teacher and caltime
        # tensors, absent from its distributed inference architecture.
        training_keys = [key for key in state if key.startswith(("teacher.", "caltime."))]
        state = {key: value for key, value in state.items() if key not in training_keys}
        self.model.flownet.load_state_dict(state, strict=True)
        self.model.eval()
        self.torch = torch
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.metadata = {
            "checkout": str(checkout),
            "revision": revision.stdout.strip() or "unknown",
            "weights": str(weights),
            "weights_sha256": _sha256(weights / "flownet.pkl"),
            "model_version": str(getattr(self.model, "version", "unknown")),
            "omitted_training_tensors": training_keys,
            "model_code_sha256": {p.name: _sha256(p) for p in sorted(weights.glob("*.py"))},
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "cudnn": torch.backends.cudnn.version(),
        }

    def __call__(self, left: Any, right: Any, alpha: Fraction) -> Any:
        torch = self.torch
        height, width = left.shape[:2]
        # RIFE 4.25 uses five scales; pad to a multiple of 64, then crop.
        padding = (0, (-width) % 64, 0, (-height) % 64)

        def tensor(frame: Any) -> Any:
            rgb = np.ascontiguousarray(frame[:, :, ::-1].transpose(2, 0, 1))
            value = torch.from_numpy(rgb).to("cuda", dtype=torch.float32).unsqueeze(0) / 255
            return torch.nn.functional.pad(value, padding)

        with torch.inference_mode():
            result = self.model.inference(tensor(left), tensor(right), timestep=float(alpha))
            if not torch.isfinite(result).all():
                raise EventSyncError("RIFE returned non-finite pixels")
            rgb = (result[0, :, :height, :width].clamp(0, 1) * 255).round().byte()
            image = rgb.permute(1, 2, 0).cpu().numpy()[:, :, ::-1].copy()
        if image.shape != left.shape:
            raise EventSyncError("RIFE output dimensions differ from the source")
        return image


def render_rife(
    cap: Any,
    source: Path,
    rows: Sequence[dict[str, Any]],
    tick: Callable[[], None],
    interpolate: Callable[..., Any],
) -> Iterator[Any]:
    """Sequential decode, bounded two-frame cache, direct copy at exact instants."""
    decoded = 0
    cache: dict[int, Any] = {}
    for row in rows:
        lo, hi = row["source_frame_left"], row["source_frame_right"]
        cache = {key: value for key, value in cache.items() if key in (lo, hi)}
        while decoded < hi:
            tick()
            ok, frame = cap.read()
            decoded += 1
            if not ok:
                raise EventSyncError(f"{source.name}: cannot decode frame {decoded}")
            if decoded in (lo, hi):
                cache[decoded] = frame
        tick()
        yield (
            cache[lo]
            if lo == hi
            else interpolate(cache[lo], cache[hi], Fraction(row["interpolation_fraction"]))
        )


def export(args: argparse.Namespace) -> Path:
    manifest = prepare(args)
    print(">> Export preview\n" + preview_text(manifest), flush=True)
    print(">> " + WARNING, flush=True)
    print(">> Loading RIFE and checking CUDA inference…", flush=True)
    rife = RifeInterpolator(Path(args.rife_dir), Path(args.weights))
    # Preflight the actual dimensions and first pair needing interpolation before
    # creating an output directory. Catch weight/API/VRAM failures early.
    for role, camera in manifest["cameras"].items():
        generated = next((row for row in camera["rows"] if row["kind"] == "generated"), None)
        if generated:
            print(f">> Checking RIFE at {role} resolution…", flush=True)
            cap = cv2.VideoCapture(camera["path"])
            try:
                next(render_rife(cap, Path(camera["path"]), [generated], lambda: None, rife))
            finally:
                cap.release()
    parent = Path(args.output_dir).expanduser().resolve()
    destination = parent / (
        "processed_event_equalize_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    destination.mkdir(parents=True, exist_ok=False)
    incomplete = destination / "INCOMPLETE.txt"
    incomplete.write_text(
        "Export running, failed or cancelled. Do not use this set.\n", encoding="utf-8"
    )
    manifest["rife"] = rife.metadata
    try:
        for role, camera in manifest["cameras"].items():
            path, rows = Path(camera["path"]), camera["rows"]
            name = f"{role}_{path.stem}_equalized.mp4"
            camera["video"] = name
            completed = 0

            def frame_done(role: str = role, count: int = len(rows)) -> None:
                nonlocal completed
                completed += 1
                print(f">> {role}: {completed}/{count} frames", flush=True)

            _encode_samples(
                path,
                [],
                destination / name,
                manifest["output_fps"],
                interpolation_rows=rows,
                frame_done=frame_done,
                render_images=lambda cap, source, rows, tick: render_rife(
                    cap, source, rows, tick, rife
                ),
            )
            with (destination / f"{role}_frames.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            del camera["rows"]
        (destination / "event_equalizer.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (destination / "README.txt").write_text(
            WARNING + "\n" + preview_text(manifest) + "\nFrame indices are 1-based. "
            "reference_frame may be fractional on a custom grid; reference_position "
            "always records its exact rational value. See *_frames.csv for exact "
            "positions and original/generated labels.\n",
            encoding="utf-8",
        )
        incomplete.unlink()
    except BaseException:
        print(f">> Export incomplete: {destination}", flush=True)
        raise
    print(f">> Completed: {destination}", flush=True)
    return destination


def cli_command(args: argparse.Namespace, python: str = "python") -> list[str]:
    command = [python, "-m", "vaila.video_event_equalizer"]
    for key in (
        "reference",
        "source",
        "reference_events",
        "source_events",
        "rife_dir",
        "weights",
        "output_dir",
    ):
        command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
    for key in ("output_fps", "output_frame_count"):
        value = _optional_text(getattr(args, key, None))
        if value is not None:
            command.extend(["--" + key.replace("_", "-"), value])
    return command


def show_gui(args: argparse.Namespace) -> None:
    """Dedicated subprocess GUI keeps Tk and the Syncvid SDL loop independent."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Equalize by events (RIFE)")
    root.columnconfigure(1, weight=1)
    values: dict[str, Any] = {}
    process: subprocess.Popen[Any] | None = None
    log_file: Any = None
    log_position = 0
    cancelled = False
    previewing = False
    worker_values: dict[str, str] = {}
    validated_values: dict[str, str] | None = None
    labels = {
        "reference": "Reference video",
        "source": "Source video",
        "reference_events": "Reference: start, contact, end",
        "source_events": "Source: start, contact, end",
        "output_fps": "Output FPS (optional)",
        "output_frame_count": "Output frame count (optional)",
        "rife_dir": "Practical-RIFE checkout",
        "weights": "RIFE model directory",
        "output_dir": "Output parent directory",
    }

    def fill_events(key: str) -> None:
        try:
            info = probe(values[key].get())
            event_key = key + "_events"
            old = values[event_key].get().replace(",", " ").split()
            contact = old[1] if len(old) == 3 else ""
            values[event_key].set(f"1, {contact}, {info['frame_count']}")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            messagebox.showerror("Video", str(exc), parent=root)

    def browse(key: str) -> None:
        selected = (
            filedialog.askopenfilename(parent=root, title=labels[key])
            if key in ("reference", "source")
            else filedialog.askdirectory(parent=root, title=labels[key])
        )
        if selected:
            values[key].set(selected)
            if key in ("reference", "source"):
                fill_events(key)

    for row, (key, label) in enumerate(labels.items()):
        values[key] = tk.StringVar(root, str(getattr(args, key, None) or ""))
        ttk.Label(root, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=5)
        ttk.Entry(root, textvariable=values[key], width=64).grid(
            row=row, column=1, sticky="ew", padx=4
        )
        if key == "output_fps":
            presets = ttk.Frame(root)
            presets.grid(row=row, column=2, padx=8)
            for fps in (60, 120):
                ttk.Button(
                    presets,
                    text=str(fps),
                    width=5,
                    command=lambda fps=fps: values["output_fps"].set(str(fps)),
                ).pack(side="left")
        elif key not in ("reference_events", "source_events", "output_frame_count"):
            ttk.Button(root, text="Browse", command=lambda key=key: browse(key)).grid(
                row=row, column=2, padx=8
            )
    for key in ("reference", "source"):
        if values[key].get() and not values[key + "_events"].get():
            fill_events(key)
    offset = len(labels)
    ttk.Label(
        root,
        text=(
            "Leave both blank for the reference grid. FPS only calculates count; count only uses reference FPS. "
            "FPS accepts 29.970 or 30000/1001. Both values set playback speed independently."
        ),
        wraplength=780,
    ).grid(row=offset, column=0, columnspan=3, padx=8, pady=4)
    ttk.Label(root, text=WARNING, wraplength=780).grid(
        row=offset + 1, column=0, columnspan=3, padx=8, pady=8
    )
    preview = tk.Text(root, height=11, width=100, wrap="word", state="disabled")
    preview.grid(row=offset + 2, column=0, columnspan=3, sticky="nsew", padx=8)
    root.rowconfigure(offset + 2, weight=1)
    status = tk.StringVar(
        root, "Choose both videos and enter the contact frame for each camera (1-based)."
    )
    ttk.Label(root, textvariable=status, wraplength=780).grid(
        row=offset + 3, column=0, columnspan=3, padx=8
    )
    progress = ttk.Progressbar(root, mode="indeterminate")
    progress.grid(row=offset + 4, column=0, columnspan=3, sticky="ew", padx=8, pady=8)

    def current_values() -> dict[str, str]:
        return {key: value.get() for key, value in values.items()}

    def set_preview(text: str) -> None:
        preview.configure(state="normal")
        preview.delete("1.0", "end")
        preview.insert("1.0", text)
        preview.configure(state="disabled")

    def invalidate(*_args: Any) -> None:
        nonlocal validated_values
        validated_values = None
        run_button.configure(state="disabled")
        set_preview("Settings changed. Click Preview to calculate the output before Run.")

    def stop() -> None:
        nonlocal cancelled
        if process is not None and process.poll() is None and not cancelled:
            cancelled = True
            process.terminate()
            status.set("Cancelling… Partial exports remain marked INCOMPLETE.")

    def poll() -> None:
        nonlocal log_position, process, log_file, validated_values
        assert process is not None and log_file is not None
        returncode = process.poll()
        with Path(log_file.name).open(encoding="utf-8", errors="replace") as reader:
            reader.seek(log_position)
            tail = reader.read()
            log_position = reader.tell()
        if tail.strip() and not previewing:
            print(tail, end="", flush=True)
            status.set(tail.strip().splitlines()[-1])
        if returncode is None:
            root.after(150, poll)
            return
        progress.stop()
        preview_button.configure(state="normal")
        if cancelled:
            status.set("Cancelled. Any partial export is marked INCOMPLETE.")
        elif returncode:
            if previewing:
                error_log = (
                    Path(log_file.name).read_text(encoding="utf-8", errors="replace").strip()
                )
                print(error_log, flush=True)
                status.set(
                    error_log.splitlines()[-1] if error_log else "Preview failed; see terminal."
                )
            messagebox.showerror(
                "Equalize failed", status.get() + "\nSee terminal for details.", parent=root
            )
        elif previewing:
            try:
                manifest = json.loads(Path(log_file.name).read_text(encoding="utf-8"))
                if current_values() == worker_values:
                    validated_values = worker_values.copy()
                    summary = preview_text(manifest)
                    set_preview(summary)
                    print(">> Export preview\n" + summary, flush=True)
                    status.set("Review the preview, then click Run to export.")
                else:
                    status.set("Settings changed during preview. Click Preview again.")
            except (ValueError, KeyError) as exc:
                status.set(f"Cannot read preview: {exc}")
                messagebox.showerror("Preview failed", status.get(), parent=root)
        else:
            messagebox.showinfo("Equalize complete", status.get(), parent=root)
        log_path = Path(log_file.name)
        log_file.close()
        log_path.unlink(missing_ok=True)
        process = log_file = None
        run_button.configure(state="normal" if validated_values == current_values() else "disabled")

    def run(dry_run: bool = False) -> None:
        nonlocal process, log_file, log_position, cancelled, previewing, worker_values
        if process is not None:
            return
        worker_values = current_values()
        selected = argparse.Namespace(**worker_values)
        # The worker validates/decode-probes everything; no blocking scan on Tk.
        if any(
            not value.strip()
            for key, value in worker_values.items()
            if key not in ("output_fps", "output_frame_count")
        ):
            messagebox.showerror(
                "Missing values", "Fill all fields, including all three events.", parent=root
            )
            return
        if not dry_run and validated_values != worker_values:
            status.set("Click Preview before Run.")
            return
        command = cli_command(selected, sys.executable)
        previewing = dry_run
        if dry_run:
            command.append("--dry-run")
        else:
            print(">> Equivalent CLI\n" + shlex.join(cli_command(selected)), flush=True)
        # Lifetime spans Tk callbacks; closed by poll() after the worker exits.
        log_file = tempfile.NamedTemporaryFile(mode="w+t", encoding="utf-8", delete=False)  # noqa: SIM115
        log_position, cancelled = 0, False
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT)
        except OSError as exc:
            log_path = Path(log_file.name)
            log_file.close()
            log_path.unlink(missing_ok=True)
            messagebox.showerror("Launch failed", str(exc), parent=root)
            return
        run_button.configure(state="disabled")
        preview_button.configure(state="disabled")
        progress.start()
        status.set(
            "Probing videos and calculating output…"
            if dry_run
            else "Validating events, CUDA and model…"
        )
        root.after(150, poll)

    def close() -> None:
        if process is not None:
            if process.poll() is None:
                stop()
            root.after(150, close)
            return
        root.destroy()

    controls = ttk.Frame(root)
    controls.grid(row=offset + 5, column=0, columnspan=3, pady=10)
    preview_button = ttk.Button(controls, text="Preview", command=lambda: run(True))
    preview_button.pack(side="left", padx=8)
    run_button = ttk.Button(controls, text="Run", command=run, state="disabled")
    run_button.pack(side="left", padx=8)
    ttk.Button(controls, text="Cancel export", command=stop).pack(side="left", padx=8)
    ttk.Button(controls, text="Close", command=close).pack(side="left", padx=8)
    for value in values.values():
        value.trace_add("write", invalidate)
    set_preview("Click Preview to calculate FPS, frame counts, durations and contact timing.")
    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="Open the event equalizer window")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reference-events", default="", help="1-based start,contact,end")
    parser.add_argument("--source-events", default="", help="1-based start,contact,end")
    parser.add_argument(
        "--output-fps",
        default="",
        help="Exact decimal or rational FPS, e.g. 60, 120, 29.970 or 30000/1001",
    )
    parser.add_argument(
        "--output-frame-count",
        default="",
        help="Exact count >=3; without FPS uses reference rate. FPS alone derives count from reference duration.",
    )
    parser.add_argument("--rife-dir", type=Path, default=DEFAULT_RIFE)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate/probe and print maps without loading RIFE"
    )
    args = parser.parse_args(argv)
    if args.gui:
        show_gui(args)
        return 0
    if not args.reference or not args.source:
        parser.error("--reference and --source are required without --gui")

    def cancel(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancel)
    try:
        if args.dry_run:
            print(json.dumps(prepare(args), indent=2))
        else:
            export(args)
        return 0
    except KeyboardInterrupt:
        print(">> Equalize cancelled; partial exports remain marked INCOMPLETE.", flush=True)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
        print(f">> Equalize failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
