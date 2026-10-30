"""Install, copy or move the pinned Practical-RIFE 4.25 code/model and verify CUDA inference.

Update Date: 30 October 2026
Version: 0.4.13

Uses the existing vailá Python environment; never installs upstream requirements
or changes torch, CUDA groups, extras, pyproject.toml or uv.lock.

RIFE is optional and its folder is yours to choose (about 50 MB):

    setup_rife.py --home /big/disk/rife          install into a portable folder
    setup_rife.py --export /other/disk/rife      copy the installed RIFE there (no network)
    setup_rife.py                                 legacy folders inside the vailá checkout

Select a folder later with ``--rife-home`` on the equalizer or ``VAILA_RIFE_HOME``.
Without ``--home``/``--root`` the folder in ``VAILA_RIFE_HOME`` is used when it is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from vaila import rife_paths  # noqa: E402  (stdlib-only module; the layout lives in one place)

REPOSITORY = "https://github.com/hzwer/Practical-RIFE.git"
REVISION = "bbfd2ea90910789a860ea3e2b32a240cd577b75e"
MODEL_URL = "https://drive.usercontent.google.com/download?id=1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg&export=download&confirm=t"
ARCHIVE_SHA256 = "e63d481b7ae5d4a4e6ad7ac5b410ff78f3bf7be3b51b2e38ca8152747abde5b4"
FILES = {
    "IFNet_HDv3.py": "655b4c772b037967b86c2dd31c8fa3b5323b79dd9a0e0088708d89149bbc8a32",
    "RIFE_HDv3.py": "81bbd0648e499de79e44768d284005d9d57d0f6eb7c30adae407f22675055730",
    "flownet.pkl": "6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2",
    "refine.py": "0c5698b4a05b9f6ab551740575c1c35e248e5b1829bab6445186081ebe15f032",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_checkout(checkout: Path) -> None:
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        # Stage clone/check-out so a network failure cannot leave a half-installed tree.
        with tempfile.TemporaryDirectory(dir=checkout.parent, prefix="rife_clone_") as temporary:
            staged = Path(temporary) / "Practical-RIFE"
            subprocess.run(["git", "clone", REPOSITORY, str(staged)], check=True)
            subprocess.run(["git", "-C", str(staged), "checkout", "--detach", REVISION], check=True)
            staged.rename(checkout)
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if actual != REVISION or dirty:
        raise ValueError(
            f"Existing RIFE checkout is modified or uses another revision: {checkout}. "
            f"Preserve your changes and select the validated revision {REVISION} before retrying."
        )
    print(f">> RIFE checkout verified: {checkout} ({actual})", flush=True)


def download_archive(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256(destination) == ARCHIVE_SHA256:
        print(f">> Reusing verified download: {destination}", flush=True)
        return
    print(">> Downloading official RIFE 4.25 bundle (~23 MB)…", flush=True)
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "vaila-rife-setup/0.4.13"})
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix="rife_download_") as temporary:
        staged = Path(temporary) / "rife425.zip"
        with urllib.request.urlopen(request, timeout=120) as response, staged.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        if sha256(staged) != ARCHIVE_SHA256:
            raise ValueError(
                "RIFE archive checksum mismatch (possibly a Drive quota/HTML response). "
                "Download the official 4.25 ZIP from the link in Syncvid Help, then use --archive FILE."
            )
        staged.replace(destination)


def install_bundle(archive: Path, weights: Path) -> None:
    """Extract only known, verified model files. No arbitrary ZIP paths/code."""
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("RIFE archive SHA256 differs from the validated official 4.25 bundle")
    weights.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=weights.parent, prefix="rife_model_") as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as bundle:
            for name, digest in FILES.items():
                target = staging / name
                with bundle.open(f"train_log/{name}") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                if sha256(target) != digest:
                    raise ValueError(f"RIFE model file checksum mismatch: {name}")
        # Validate the entire bundle before replacing any model file.
        weights.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            (staging / name).replace(weights / name)
    print(f">> RIFE 4.25 model installed: {weights}", flush=True)


def layout(root: Path | None, home: Path | None) -> tuple[Path, Path, Path]:
    """(checkout, weights, download cache). A home is portable; a root keeps the legacy tree."""
    if home is not None:
        home = home.expanduser().resolve()
        checkout, weights = rife_paths.home_layout(home)
        return checkout, weights, home / "downloads/rife425.zip"
    root = (root or ROOT).expanduser().resolve()
    return (
        root / ".local/third_party/Practical-RIFE",
        root / "vaila/models/rife/train_log",
        root / ".local/downloads/rife425.zip",
    )


def check_cuda(checkout: Path, weights: Path) -> dict:
    """One real interpolation on CUDA; returns the runtime metadata."""
    import numpy as np

    # Import from the running vailá checkout even when the assets live elsewhere.
    from vaila.video_event_equalizer import RifeInterpolator

    print(">> Checking RIFE inference on CUDA…", flush=True)
    rife = RifeInterpolator(checkout, weights)
    left = np.zeros((64, 64, 3), dtype=np.uint8)
    right = np.full_like(left, 180)
    image = rife(left, right, Fraction(1, 3))
    if image.shape != left.shape or image.dtype != np.uint8:
        raise RuntimeError("RIFE CUDA inference returned an invalid image")
    return rife.metadata


def finish(checkout: Path, weights: Path, *, skip_check: bool, **extra: object) -> dict:
    """Checks CUDA (unless skipped) and writes setup_manifest.json next to the weights."""
    result = {
        "revision": REVISION,
        "model": "RIFE 4.25",
        "weights": str(weights),
        "weights_sha256": FILES["flownet.pkl"],
        "cuda_inference_verified": False,
        **extra,
    }
    if not skip_check:
        result.update(cuda_inference_verified=True, runtime=check_cuda(checkout, weights))
    else:
        print(">> Assets staged only (--skip-check); CUDA inference was NOT verified.", flush=True)
    (weights.parent / rife_paths.MANIFEST_NAME).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def announce(checkout: Path, weights: Path, *, skip_check: bool) -> None:
    portable = (checkout.parent == weights.parent) and checkout.parent != ROOT
    if portable:
        print(
            f">> Use it with --rife-home {checkout.parent} (equalizer CLI) or "
            f"export {rife_paths.ENV_HOME}={checkout.parent}",
            flush=True,
        )
    print(
        ">> RIFE ready. Open Make Sync file → Equalize by events (RIFE)."
        if not skip_check
        else ">> Re-run setup without --skip-check on the CUDA computer before use.",
        flush=True,
    )


def setup(
    root: Path | None,
    archive: Path | None = None,
    *,
    skip_check: bool = False,
    home: Path | None = None,
) -> dict:
    checkout, weights, cache = layout(root, home)
    ensure_checkout(checkout)
    valid = all(
        (weights / name).is_file() and sha256(weights / name) == digest
        for name, digest in FILES.items()
    )
    if valid:
        print(f">> Verified model already installed: {weights}", flush=True)
    else:
        if archive is None:
            archive = cache
            download_archive(archive)
        install_bundle(archive.expanduser().resolve(), weights)
    result = finish(checkout, weights, skip_check=skip_check)
    announce(checkout, weights, skip_check=skip_check)
    return result


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def export_install(
    root: Path | None,
    home: Path | None,
    destination: Path,
    *,
    skip_check: bool = False,
) -> dict:
    """Copies the installed, verified RIFE to ``destination`` (portable layout) with no network.

    The source is the install that ``setup`` would use for the same ``root``/``home``. Nothing
    in the destination is overwritten: a weight file with another hash stops the copy.
    """
    source_checkout, source_weights, _cache = layout(root, home)
    destination = destination.expanduser().resolve()
    if not source_checkout.is_dir():
        raise ValueError(f"No RIFE code to copy at {source_checkout}; install it first.")
    ensure_checkout(source_checkout)  # pinned revision, clean tree; never clones here
    for name, digest in FILES.items():
        if not (source_weights / name).is_file() or sha256(source_weights / name) != digest:
            raise ValueError(f"RIFE source file missing or altered: {source_weights / name}")
    target_checkout, target_weights = rife_paths.home_layout(destination)
    if _overlaps(source_checkout, destination) or _overlaps(source_weights, destination):
        raise ValueError(
            f"Destination {destination} overlaps the RIFE source; pick another folder."
        )
    destination.mkdir(parents=True, exist_ok=True)
    if target_checkout.exists():
        ensure_checkout(target_checkout)  # verify; an existing copy is never replaced
    else:
        # Stage next to the target so a full disk or Ctrl+C leaves no half-copied tree.
        with tempfile.TemporaryDirectory(dir=destination, prefix="rife_copy_") as temporary:
            staged = Path(temporary) / rife_paths.CHECKOUT_DIR
            shutil.copytree(
                source_checkout, staged, symlinks=True, ignore=shutil.ignore_patterns("__pycache__")
            )
            staged.rename(target_checkout)
        ensure_checkout(target_checkout)
    target_weights.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination, prefix="rife_model_") as temporary:
        staging = Path(temporary)
        for name, digest in FILES.items():
            existing = target_weights / name
            if existing.is_file():
                if sha256(existing) != digest:
                    raise ValueError(f"Refusing to overwrite a different file: {existing}")
                continue
            shutil.copy2(source_weights / name, staging / name)
            if sha256(staging / name) != digest:
                raise ValueError(f"RIFE file changed while copying: {name}")
        for staged_file in staging.iterdir():
            staged_file.replace(target_weights / staged_file.name)
    print(f">> RIFE copied (hashes verified): {destination}", flush=True)
    result = finish(
        target_checkout,
        target_weights,
        skip_check=skip_check,
        exported_from=str(source_weights.parent),
    )
    announce(target_checkout, target_weights, skip_check=skip_check)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    where = parser.add_mutually_exclusive_group()
    where.add_argument(
        "--home",
        type=Path,
        help=f"Portable RIFE folder to install into (or copy from with --export); "
        f"default ${rife_paths.ENV_HOME} when set",
    )
    where.add_argument(
        "--root",
        type=Path,
        help="Legacy layout (.local/third_party + vaila/models/rife) under this root; "
        "default is this vailá checkout",
    )
    parser.add_argument(
        "--export",
        type=Path,
        metavar="DEST",
        help="Copy the installed RIFE to DEST (portable layout, hashes verified, no network)",
    )
    parser.add_argument(
        "--archive", type=Path, help="Official RIFE 4.25 ZIP already downloaded (offline reinstall)"
    )
    parser.add_argument(
        "--skip-check", action="store_true", help="Stage assets only; do not claim CUDA readiness"
    )
    args = parser.parse_args()
    home = args.home or (None if args.root else rife_paths.env_home())
    try:
        if args.export:
            export_install(args.root, home, args.export, skip_check=args.skip_check)
        else:
            setup(args.root, args.archive, skip_check=args.skip_check, home=home)
    except (
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        KeyError,
        subprocess.SubprocessError,
        zipfile.BadZipFile,
    ) as exc:
        print(f">> RIFE setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
