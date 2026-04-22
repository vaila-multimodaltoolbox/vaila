"""FIFA Skeletal Tracking Light 2026 — data-layout bootstrap.

This helper prepares a FIFA dataset root so the rest of
:mod:`vaila.fifa_skeletal_pipeline` can run without a manual setup:

- Creates symlinks (or copies, on Windows without developer mode) from a videos
  source directory into ``<data_root>/videos/*.mp4``.  This avoids duplicating
  the 10+ GB of broadcast footage on disk.
- Generates ``sequences_full.txt``, ``sequences_val.txt`` and
  ``sequences_test.txt`` files from the video stems.  Official challenge splits
  can be supplied via ``--val-sequences`` / ``--test-sequences``; otherwise
  ``val`` defaults to *all* sequences and ``test`` stays empty, with a warning.
- Copies the MIT ``pitch_points.txt`` (vendored under ``fifa_starter_lib/``)
  into the data root when missing.

Author: vailá team (17 April 2026).
"""

from __future__ import annotations

import argparse
import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


@dataclass
class BootstrapResult:
    videos_linked: list[Path]
    sequences_full: Path
    sequences_val: Path
    sequences_test: Path
    pitch_points: Path
    warnings: list[str]

    def as_summary(self) -> str:
        lines = [
            f"videos_linked: {len(self.videos_linked)}",
            f"sequences_full: {self.sequences_full}",
            f"sequences_val:  {self.sequences_val}",
            f"sequences_test: {self.sequences_test}",
            f"pitch_points:   {self.pitch_points}",
        ]
        if self.warnings:
            lines.append("warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


def _list_videos(videos_dir: Path) -> list[Path]:
    videos_dir = videos_dir.resolve()
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"videos directory does not exist: {videos_dir}")
    found: list[Path] = []
    for p in sorted(videos_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            found.append(p.resolve())
    return found


def _link_or_copy(src: Path, dst: Path, copy_fallback: bool = True) -> None:
    """Create a symlink ``dst`` → ``src``.

    Falls back to ``shutil.copy2`` on Windows (or any OS that refuses symlinks),
    when ``copy_fallback`` is true.  A ``BrokenPipeError`` or ``OSError`` from
    :func:`os.symlink` usually means we lack the privilege.
    """
    if dst.exists() or dst.is_symlink():
        # Always replace to keep the layout in sync with the source directory.
        try:
            dst.unlink()
        except IsADirectoryError:
            shutil.rmtree(dst)
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        if not copy_fallback:
            raise
        shutil.copy2(src, dst)


def _resolve_split_sequences(
    arg: list[str] | Path | None,
    all_seqs: list[str],
) -> list[str]:
    """Parse a ``--val-sequences`` / ``--test-sequences`` argument.

    Accepts either an inline list (already split) or a ``Path`` pointing to a
    newline-delimited text file. ``None`` or an empty value falls back to the
    caller's policy.
    """
    if arg is None:
        return []
    if isinstance(arg, Path):
        if not arg.exists():
            raise FileNotFoundError(f"split file not found: {arg}")
        lines = [
            s.strip()
            for s in arg.read_text(encoding="utf-8").splitlines()
            if s.strip() and not s.strip().startswith("#")
        ]
        return lines
    if isinstance(arg, list):
        return [s.strip() for s in arg if s and s.strip()]
    raise TypeError(f"unsupported split sequences argument: {type(arg)!r}")


def _vendored_pitch_points() -> Path:
    return Path(__file__).resolve().parent / "fifa_starter_lib" / "pitch_points.txt"


def prepare_fifa_data_layout(
    videos_dir: Path,
    data_root: Path,
    *,
    val_sequences: list[str] | Path | None = None,
    test_sequences: list[str] | Path | None = None,
    copy_fallback: bool = True,
    overwrite_pitch_points: bool = False,
) -> BootstrapResult:
    """Prepare ``data_root`` for the FIFA Skeletal Tracking Light pipeline.

    Args:
        videos_dir: Directory containing the raw broadcast video files
            (``.mp4``/``.mov``/…). Each file is symlinked into
            ``data_root/videos/`` using its original stem.
        data_root: Destination FIFA data root (the same path passed to
            ``fifa prepare --data-root``). Created if necessary.
        val_sequences: Optional explicit ``val`` split. Either a list of stems
            or a ``Path`` to a text file (one stem per line). If ``None``, the
            split defaults to **all** discovered sequences (with a warning).
        test_sequences: Optional explicit ``test`` split. Same format as
            ``val_sequences``. If ``None`` the test split is left empty.
        copy_fallback: When ``True`` (default) and symlinks are not allowed
            (e.g. Windows without developer mode), falls back to copying.
        overwrite_pitch_points: When ``True`` always refresh ``pitch_points.txt``
            from the vendored starter-kit copy. When ``False``, only writes it
            if missing.

    Returns:
        :class:`BootstrapResult` describing everything that was produced.
    """
    videos_dir = videos_dir.resolve()
    data_root = data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    videos = _list_videos(videos_dir)
    if not videos:
        raise RuntimeError(f"no video files found under {videos_dir}")

    # --- 1) symlink videos -------------------------------------------------
    dst_videos_dir = data_root / "videos"
    dst_videos_dir.mkdir(parents=True, exist_ok=True)
    videos_linked: list[Path] = []
    stems: list[str] = []
    for src in videos:
        stem = src.stem
        dst = dst_videos_dir / f"{stem}.mp4"
        # For non-.mp4 source files we keep the original suffix to avoid
        # silent transcoding here; ``fifa prepare`` (with ffmpeg) is the place
        # for that.
        if src.suffix.lower() != ".mp4":
            dst = dst_videos_dir / f"{stem}{src.suffix.lower()}"
        _link_or_copy(src, dst, copy_fallback=copy_fallback)
        videos_linked.append(dst)
        stems.append(stem)

    # --- 2) sequences_*.txt ------------------------------------------------
    seq_full = data_root / "sequences_full.txt"
    seq_val = data_root / "sequences_val.txt"
    seq_test = data_root / "sequences_test.txt"
    warns: list[str] = []

    seq_full.write_text("\n".join(stems) + "\n", encoding="utf-8")

    val_list = _resolve_split_sequences(val_sequences, stems)
    if not val_list:
        warns.append(
            "no --val-sequences provided; defaulting sequences_val.txt to the FULL set. "
            "Replace this with the official Codabench split before submitting."
        )
        val_list = list(stems)
    seq_val.write_text("\n".join(val_list) + "\n", encoding="utf-8")

    test_list = _resolve_split_sequences(test_sequences, stems)
    if not test_list:
        warns.append(
            "no --test-sequences provided; sequences_test.txt was written empty. "
            "Populate it with the official Codabench test split when available."
        )
    seq_test.write_text(
        ("\n".join(test_list) + "\n") if test_list else "",
        encoding="utf-8",
    )

    # --- 3) pitch_points.txt -----------------------------------------------
    pitch_dst = data_root / "pitch_points.txt"
    pitch_src = _vendored_pitch_points()
    if not pitch_src.exists():
        raise FileNotFoundError(
            f"vendored pitch_points.txt missing at {pitch_src}; re-run the vendor step"
        )
    if overwrite_pitch_points or not pitch_dst.exists():
        shutil.copy2(pitch_src, pitch_dst)

    for w in warns:
        warnings.warn(w, stacklevel=2)

    return BootstrapResult(
        videos_linked=videos_linked,
        sequences_full=seq_full,
        sequences_val=seq_val,
        sequences_test=seq_test,
        pitch_points=pitch_dst,
        warnings=warns,
    )


def build_bootstrap_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Bootstrap a FIFA Skeletal Tracking Light data layout: symlink videos, "
            "write sequences_full/val/test.txt and copy pitch_points.txt."
        )
    )
    p.add_argument("--videos-dir", type=Path, required=True, help="Source folder with *.mp4")
    p.add_argument("--data-root", type=Path, required=True, help="Destination FIFA data root")
    p.add_argument(
        "--val-sequences",
        type=Path,
        default=None,
        help="Optional text file with the official val split (one stem per line)",
    )
    p.add_argument(
        "--test-sequences",
        type=Path,
        default=None,
        help="Optional text file with the official test split (one stem per line)",
    )
    p.add_argument(
        "--no-copy-fallback",
        action="store_true",
        help="Do not fall back to copying when symlinks are denied (default: copy)",
    )
    p.add_argument(
        "--overwrite-pitch-points",
        action="store_true",
        help="Always refresh pitch_points.txt from the vendored copy",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_bootstrap_argparser().parse_args(argv)
    res = prepare_fifa_data_layout(
        videos_dir=args.videos_dir,
        data_root=args.data_root,
        val_sequences=args.val_sequences,
        test_sequences=args.test_sequences,
        copy_fallback=not args.no_copy_fallback,
        overwrite_pitch_points=args.overwrite_pitch_points,
    )
    print(res.as_summary())


if __name__ == "__main__":
    main()
