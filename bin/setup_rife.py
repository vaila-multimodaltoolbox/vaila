"""Install the pinned Practical-RIFE 4.25 code/model and verify CUDA inference.

Update Date: 29 October 2026
Version: 0.4.12

Uses the existing vailá Python environment; never installs upstream requirements
or changes torch, CUDA groups, extras, pyproject.toml or uv.lock.
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
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "vaila-rife-setup/0.4.12"})
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


def setup(root: Path, archive: Path | None = None, *, skip_check: bool = False) -> dict:
    root = root.expanduser().resolve()
    checkout = root / ".local/third_party/Practical-RIFE"
    weights = root / "vaila/models/rife/train_log"
    ensure_checkout(checkout)
    valid = all(
        (weights / name).is_file() and sha256(weights / name) == digest
        for name, digest in FILES.items()
    )
    if valid:
        print(f">> Verified model already installed: {weights}", flush=True)
    else:
        if archive is None:
            archive = root / ".local/downloads/rife425.zip"
            download_archive(archive)
        install_bundle(archive.expanduser().resolve(), weights)
    result = {
        "revision": REVISION,
        "model": "RIFE 4.25",
        "weights": str(weights),
        "weights_sha256": FILES["flownet.pkl"],
        "cuda_inference_verified": False,
    }
    if not skip_check:
        # Import from the running vailá checkout even when --root stages assets elsewhere.
        sys.path.insert(0, str(ROOT))
        import numpy as np

        from vaila.video_event_equalizer import RifeInterpolator

        print(">> Checking RIFE inference on CUDA…", flush=True)
        rife = RifeInterpolator(checkout, weights)
        left = np.zeros((64, 64, 3), dtype=np.uint8)
        right = np.full_like(left, 180)
        image = rife(left, right, Fraction(1, 3))
        if image.shape != left.shape or image.dtype != np.uint8:
            raise RuntimeError("RIFE CUDA inference returned an invalid image")
        result.update(cuda_inference_verified=True, runtime=rife.metadata)
    else:
        print(">> Assets staged only (--skip-check); CUDA inference was NOT verified.", flush=True)
    (weights.parent / "setup_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        ">> RIFE ready. Open Make Sync file → Equalize by events (RIFE)."
        if not skip_check
        else ">> Re-run setup without --skip-check on the CUDA computer before use.",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Asset installation root (defaults to this vailá checkout)",
    )
    parser.add_argument(
        "--archive", type=Path, help="Official RIFE 4.25 ZIP already downloaded (offline reinstall)"
    )
    parser.add_argument(
        "--skip-check", action="store_true", help="Stage assets only; do not claim CUDA readiness"
    )
    args = parser.parse_args()
    try:
        setup(args.root, args.archive, skip_check=args.skip_check)
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
