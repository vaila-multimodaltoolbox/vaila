"""
vailá AI Kinematic Tracker (Normalized Cross-Correlation + Deep Feature Verification).

Implements robust kinematic tracking with 2D Gaussian spatial motion priors,
elliptical masking, parabolic sub-pixel peak refinement, adaptive running template blending,
and Deep Visual Feature embeddings (PyTorch ResNet50/ResNet152 / CUDA) for semantic
verification, distractor rejection, and occlusion recovery -- optionally also fused
into the retrained online discriminator's feature vector. Integrates bidirectional
keyframe infilling and Rauch-Tung-Striebel (RTS) zero-phase smoothing (Δϕ = 0).

Author: Prof. Dr. Paulo R. P. Santiago
Update Date: 18 September 2026
Version: 0.4.4
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.request import urlopen

import cv2
import numpy as np

from .rts_smoother import RTSSmoother
from .subspace_tracker import TrackingResult

if TYPE_CHECKING:
    pass

# Optional PyTorch & Torchvision for Deep Feature Verification
try:
    import torch
    import torchvision.models as tv_models
    import torchvision.transforms as tv_transforms

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_MAX_EXEMPLAR_TEMPLATES = 30
_MAX_TRAINING_FEATS = 200
_MAX_GAP_SEED_ANCHORS = 40
_VELOCITY_EMA = 0.35
_LOST_STREAK_STOP = 8
_ACCEL_EMA = 0.30  # Acceleration-magnitude smoothing (diagnostic only; set_reference()'s
# manual-correction velocity bootstrap uses it. The per-frame trajectory gate below no
# longer does -- it uses the Kalman filter's own covariance instead.)

# --- Kalman filter (constant-velocity model) for trajectory-plausibility gating ---
# State x = [px, py, vx, vy]. Replaces the earlier ad hoc EMA-velocity prediction +
# additive-pixel-distance gate with genuine predict/update covariance propagation and
# Mahalanobis-distance outlier rejection (see AITracker.track_frame()). P widens on its
# own through process noise Q during a real acceleration/blur/background-flip event
# (the reported failure mode), instead of a fixed or linearly-scaled pixel threshold --
# and tightens back down once tracking stabilizes.
_KF_Q_POS = 4.0  # process noise variance injected into position each frame (px^2)
_KF_Q_VEL = 12.0  # process noise variance injected into velocity each frame (px^2/frame^2)
_KF_R_POS = 4.0  # measurement noise variance assumed for an accepted appearance match (px^2)
_KF_P_INIT = 1000.0  # initial state covariance (diagonal) after a fresh anchor: "velocity unknown"
_KF_MAHALANOBIS_GATE = 9.21  # chi-square, 2 DOF, 99% confidence -- outlier rejection threshold
# Global scene-change distrust (mean abs diff of downsampled grayscale prev-vs-current
# frame, normalized [0, 1]). A large background/appearance flip (e.g. white->black)
# degrades what the NCC/deep/discriminator score is actually measuring, so a spurious
# nearby patch can score just above similarity_threshold and still pass the Mahalanobis
# gate. Only a marginal accept (score close to threshold) during a high scene-change
# frame is treated as a rejection -- a confidently-scored match is never second-guessed.
_SCENE_CHANGE_GATE = 0.35  # score above this = "large global scene change" this frame
_SCENE_CHANGE_MARGIN = 0.08  # combined_score within this margin of threshold = "marginal"
_FEATURE_DIM = 808  # extract_patch_feature() output length (768 color-grid + 40 region-stats)
_CHECKPOINT_BLEND_MAX_N = 4000.0  # Cap prior-session weight so a fresh session can still adapt


_BACKBONE_VARIANTS = ("resnet50", "resnet152", "mobilenet_v3_small", "efficientnet_b0")
# Backward-compatible alias (pre-cascade name).
_RESNET_VARIANTS = _BACKBONE_VARIANTS

# torchvision replaces the classification head at a different attribute per family.
_BACKBONE_HEAD_ATTR = {
    "resnet50": "fc",
    "resnet152": "fc",
    "mobilenet_v3_small": "classifier",
    "efficientnet_b0": "classifier",
}
# Pooled-embedding width per backbone (post head-removal). ResNet variants share 2048-d;
# the lightweight CPU backbones are narrower (fewer FLOPs, faster forward pass).
_BACKBONE_FEATURE_DIM = {
    "resnet50": 2048,
    "resnet152": 2048,
    "mobilenet_v3_small": 576,
    "efficientnet_b0": 1280,
}
# Local-checkpoint-file sanity floor: rejects empty/corrupt stray files, not an exact
# per-variant size check. Must stay below the smallest legitimate backbone checkpoint
# (MobileNetV3-Small ImageNet weights are ~9.8 MB).
_BACKBONE_MIN_FILE_BYTES = 1_000_000

# Official torchvision DEFAULT ImageNet checkpoint URLs (download into ai_tracker/, never
# leave the scientist-facing home as ~/.cache/torch/hub/checkpoints/).
_BACKBONE_WEIGHT_URLS: dict[str, str] = {
    "resnet50": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-f82ba261.pth",
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
    "efficientnet_b0": (
        "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth"
    ),
}


def _normalize_backbone_variant(variant: str | None) -> str:
    if variant not in _BACKBONE_VARIANTS:
        return "resnet50"
    return variant


def _is_valid_weight_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > _BACKBONE_MIN_FILE_BYTES
    except OSError:
        return False


def _torch_hub_checkpoints_dir() -> Path:
    """Torch hub cache path (migration source only — never the permanent home)."""
    return Path.home() / ".cache" / "torch" / "hub" / "checkpoints"


def _ai_tracker_resnet_local_path(variant: str = "resnet50") -> Path:
    """Canonical local checkpoint path for a backbone variant, under vaila/models/ai_tracker/.

    This is the ONLY local directory scanned for weights -- vaila/models/ai_tracker/ is
    the sole canonical location (see instructions_ai_tracker.txt). A legacy fallback to
    the general vaila/models/ directory used to exist here and has been removed.
    """
    variant = _normalize_backbone_variant(variant)
    return Path(__file__).resolve().parents[1] / "models" / "ai_tracker" / f"{variant}_imagenet.pth"


def _default_checkpoint_dir() -> Path:
    """Directory for persisted online-discriminator checkpoints (cross-session transfer learning)."""
    return Path(__file__).resolve().parents[1] / "models" / "ai_tracker"


def _scan_weight_files(directory: Path, variant: str) -> list[Path]:
    """Return valid .pth/.pt files under `directory` whose name contains `variant`."""
    found: list[Path] = []
    seen: set[str] = set()
    if not directory.is_dir():
        return found
    for cand in sorted(directory.glob(f"*{variant}*.pth")) + sorted(
        directory.glob(f"*{variant}*.pt")
    ):
        try:
            resolved = cand.resolve()
            key = str(resolved)
            if key not in seen and _is_valid_weight_file(resolved):
                seen.add(key)
                found.append(resolved)
        except OSError:
            pass
    return found


def get_available_resnet_checkpoints(variant: str = "resnet50") -> list[Path]:
    """Scan backbone weight checkpoints under vaila/models/ai_tracker/ only.

    `variant` is any of `_BACKBONE_VARIANTS`. Torch hub cache is intentionally NOT
    listed here — Ctrl+W / Cfg must advertise the scientist-visible directory only.
    Use :func:`ensure_backbone_weights` to migrate a one-time hub-cache hit into
    ``ai_tracker/``.
    """
    variant = _normalize_backbone_variant(variant)
    return _scan_weight_files(_default_checkpoint_dir(), variant)


def _hub_cache_checkpoints(variant: str = "resnet50") -> list[Path]:
    """Scan Torch hub cache for a variant (migration source only)."""
    return _scan_weight_files(_torch_hub_checkpoints_dir(), _normalize_backbone_variant(variant))


def get_available_resnet50_checkpoints() -> list[Path]:
    """Backward-compatible alias for get_available_resnet_checkpoints("resnet50")."""
    return get_available_resnet_checkpoints("resnet50")


def _ai_tracker_resnet50_local_path() -> Path:
    """Backward-compatible alias for _ai_tracker_resnet_local_path("resnet50")."""
    return _ai_tracker_resnet_local_path("resnet50")


def download_backbone_weights(variant: str = "resnet50", dest: Path | None = None) -> Path:
    """Download official clean ImageNet weights into ``vaila/models/ai_tracker/``.

    Never writes to ``~/.cache/torch/hub/checkpoints/``. Uses a temp file + atomic
    rename so a partial download cannot leave a corrupt canonical path.
    """
    variant = _normalize_backbone_variant(variant)
    url = _BACKBONE_WEIGHT_URLS[variant]
    dest_path = dest if dest is not None else _ai_tracker_resnet_local_path(variant)
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".download")

    print(f">> AI Track: downloading {variant} weights from {url}")
    print(f">> AI Track: destination -> {dest_path}")
    try:
        with urlopen(url, timeout=120) as response, tmp_path.open("wb") as out_f:
            shutil.copyfileobj(response, out_f)
        if not _is_valid_weight_file(tmp_path):
            raise RuntimeError(
                f"Downloaded {variant} weights look corrupt or too small "
                f"({tmp_path.stat().st_size if tmp_path.is_file() else 0} bytes)."
            )
        tmp_path.replace(dest_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    print(f">> AI Track: {variant} weights saved to {dest_path}")
    return dest_path.resolve()


def _migrate_hub_cache_to_ai_tracker(variant: str, hub_path: Path) -> Path | None:
    """Copy a Torch hub cache checkpoint into the canonical ai_tracker/ filename."""
    dest = _ai_tracker_resnet_local_path(variant)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.resolve() == hub_path.resolve():
            return dest.resolve()
        shutil.copy2(hub_path, dest)
        if not _is_valid_weight_file(dest):
            dest.unlink(missing_ok=True)
            return None
        print(
            f">> AI Track: migrated {variant} weights from Torch hub cache\n"
            f"   source: {hub_path}\n"
            f"   -> {dest}"
        )
        return dest.resolve()
    except OSError as err:
        print(f">> AI Track: could not migrate hub cache weights ({err})")
        return None


def ensure_backbone_weights(
    variant: str = "resnet50",
    *,
    weights_path: str | Path | None = None,
    prompt_fn: Callable[[str], str | None] | None = None,
    browse_fn: Callable[[], str | Path | None] | None = None,
) -> Path | None:
    """Resolve backbone weights under ``vaila/models/ai_tracker/``, prompting if needed.

    Resolution order:
      1. Explicit ``weights_path`` when valid
      2. Canonical ``ai_tracker/{variant}_imagenet.pth``
      3. Any other matching file already under ``ai_tracker/``
      4. Torch hub cache hit → copy into ``ai_tracker/`` (one-time migration)
      5. ``prompt_fn(variant)`` → ``"browse"`` | ``"download"`` | cancel/None

    Returns the absolute path to load, or ``None`` if the user cancelled / no file.
    """
    variant = _normalize_backbone_variant(variant)

    if weights_path:
        explicit = Path(weights_path)
        if _is_valid_weight_file(explicit):
            return explicit.resolve()

    canonical = _ai_tracker_resnet_local_path(variant)
    if _is_valid_weight_file(canonical):
        return canonical.resolve()

    local_matches = get_available_resnet_checkpoints(variant)
    if local_matches:
        return local_matches[0]

    hub_matches = _hub_cache_checkpoints(variant)
    if hub_matches:
        migrated = _migrate_hub_cache_to_ai_tracker(variant, hub_matches[0])
        if migrated is not None:
            return migrated

    if prompt_fn is None:
        print(
            f">> AI Track: no local {variant} weights under {canonical.parent}. "
            "Pass prompt_fn to browse or download a clean official checkpoint."
        )
        return None

    choice = (prompt_fn(variant) or "").strip().lower()
    if choice in {"1", "browse", "b", "buscar", "search"}:
        if browse_fn is None:
            print(">> AI Track: browse requested but no browse_fn provided.")
            return None
        browsed = browse_fn()
        if browsed and _is_valid_weight_file(Path(browsed)):
            return Path(browsed).resolve()
        print(">> AI Track: browse cancelled or invalid file.")
        return None
    if choice in {"2", "download", "d", "baixar"}:
        try:
            return download_backbone_weights(variant)
        except Exception as err:
            print(f">> AI Track: download failed ({err})")
            return None

    print(">> AI Track: weight selection cancelled.")
    return None


def default_checkpoint_path(name: str = "default") -> Path:
    """Path to the on-disk discriminator checkpoint for the given tracker profile name.

    Called by getpixelvideo.py on AI Track ON (load_checkpoint) and OFF (save_checkpoint)
    so the online-learned appearance model persists and incrementally improves across
    sessions instead of rebuilding from scratch every time the tool opens.
    """
    safe_name = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name) or "default"
    return _default_checkpoint_dir() / f"discriminator_{safe_name}.npz"


@dataclass
class TemplateMatchResult:
    """Result of a single-frame template match."""

    similarity: float  # Combined confidence score in [0.0, 1.0]
    ncc_score: float  # Pure NCC correlation score in [-1.0, 1.0]
    deep_score: float | None  # Deep feature cosine similarity in [-1.0, 1.0]
    location: tuple[float, float]  # Refined sub-pixel (x, y) coordinates
    raw_location: tuple[int, int]  # Discrete integer peak (x, y)
    template_updated: bool = False  # Whether the tracking template was updated this frame
    accepted: bool = True  # False when below similarity_threshold (do not advance lock)
    active_variant: str = ""  # Backbone that produced this frame's deep_score (primary or fallback)


@dataclass
class AITrackerParameters:
    """Parameters for AI block matching and deep feature verification."""

    search_window: tuple[int, int] = (100, 100)  # Width, Height of search area
    block_window: tuple[int, int] = (36, 36)  # Width, Height of template patch
    similarity_threshold: float = 0.50  # Threshold for valid match
    template_update_threshold: float = 0.70  # Threshold for adaptive template blending
    spatial_sigma: float = 22.0  # Gaussian spatial motion prior radius in pixels
    template_learning_rate: float = 0.12  # Exponential moving average factor for template update
    use_mask: bool = True  # Elliptical mask to discount rectangular corners
    use_deep_features: bool = True  # ResNet50 semantic embedding verification
    deep_weight: float = 0.30  # Weight for deep feature score: (1 - α) * ncc + α * deep
    tracking_shape: str = "point"  # Shape mode: "point" (default), "circle", "box" (or "rectangle")
    deep_weights_path: str = ""  # Optional local backbone .pth/.pt; empty = auto-resolve
    resnet_variant: str = (
        "resnet50"  # primary backbone: any of _BACKBONE_VARIANTS (default resnet50)
    )
    fallback_variant: str = ""  # "" = cascade disabled (default); else a second _BACKBONE_VARIANTS
    # entry re-scored on low-confidence frames (see AITracker.track_frame()).
    fallback_threshold: float = 0.48  # combined_score below this triggers the fallback re-score

    def __post_init__(self) -> None:
        if self.resnet_variant not in _BACKBONE_VARIANTS:
            self.resnet_variant = "resnet50"
        if self.fallback_variant and self.fallback_variant not in _BACKBONE_VARIANTS:
            # Invalid fallback disables the cascade rather than silently substituting a
            # default -- turning cascading on was not what was asked for.
            self.fallback_variant = ""

    def to_toml(self, toml_path: str | Path) -> None:
        """Serialize tracking parameters to a TOML file."""
        path = Path(toml_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        weights_line = f'deep_weights_path = "{self.deep_weights_path}"\n'
        content = (
            "# vailá AI Tracking Configuration\n"
            "# Generated automatically by getpixelvideo.py\n\n"
            "[tracking]\n"
            f"search_window_w = {int(self.search_window[0])}\n"
            f"search_window_h = {int(self.search_window[1])}\n"
            f"block_window_w = {int(self.block_window[0])}\n"
            f"block_window_h = {int(self.block_window[1])}\n"
            f"similarity_threshold = {float(self.similarity_threshold):.4f}\n"
            f"template_update_threshold = {float(self.template_update_threshold):.4f}\n"
            f"spatial_sigma = {float(self.spatial_sigma):.4f}\n"
            f"template_learning_rate = {float(self.template_learning_rate):.4f}\n"
            f"use_mask = {str(bool(self.use_mask)).lower()}\n"
            f"use_deep_features = {str(bool(self.use_deep_features)).lower()}\n"
            f"deep_weight = {float(self.deep_weight):.4f}\n"
            f'tracking_shape = "{self.tracking_shape}"\n'
            f'resnet_variant = "{self.resnet_variant}"\n'
            f'fallback_variant = "{self.fallback_variant}"\n'
            f"fallback_threshold = {float(self.fallback_threshold):.4f}\n"
            f"{weights_line}"
        )
        path.write_text(content, encoding="utf-8")

    @classmethod
    def from_toml(cls, toml_path: str | Path) -> AITrackerParameters:
        """Load tracking parameters from a TOML file."""
        import tomllib

        path = Path(toml_path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        track_cfg = data.get("tracking", data)
        sw_w = int(track_cfg.get("search_window_w", 100))
        sw_h = int(track_cfg.get("search_window_h", 100))
        bw_w = int(track_cfg.get("block_window_w", 36))
        bw_h = int(track_cfg.get("block_window_h", 36))
        sim_th = float(track_cfg.get("similarity_threshold", 0.50))
        tpl_th = float(track_cfg.get("template_update_threshold", 0.70))
        sigma = float(track_cfg.get("spatial_sigma", 22.0))
        lr = float(track_cfg.get("template_learning_rate", 0.12))
        mask = bool(track_cfg.get("use_mask", True))
        deep = bool(track_cfg.get("use_deep_features", False))
        d_wt = float(track_cfg.get("deep_weight", 0.25))
        shape = str(track_cfg.get("tracking_shape", "point")).lower()
        if shape not in ("point", "circle", "box", "rectangle"):
            shape = "point"
        if shape == "rectangle":
            shape = "box"
        weights_path = str(track_cfg.get("deep_weights_path", "") or "")
        resnet_variant = str(track_cfg.get("resnet_variant", "resnet50") or "resnet50")
        fallback_variant = str(track_cfg.get("fallback_variant", "") or "")
        fallback_threshold = float(track_cfg.get("fallback_threshold", 0.48))

        return cls(
            search_window=(sw_w, sw_h),
            block_window=(bw_w, bw_h),
            similarity_threshold=sim_th,
            template_update_threshold=tpl_th,
            spatial_sigma=sigma,
            template_learning_rate=lr,
            use_mask=mask,
            use_deep_features=deep,
            deep_weight=d_wt,
            tracking_shape=shape,
            deep_weights_path=weights_path,
            resnet_variant=resnet_variant,
            fallback_variant=fallback_variant,
            fallback_threshold=fallback_threshold,
        )


# Backward-compatible alias
KinoveaTrackerParameters = AITrackerParameters


def refine_location_parabola(
    smap: np.ndarray,
    max_loc: tuple[int, int] | Sequence[int],
    central_value: float,
) -> tuple[float, float]:
    """Fit 1D parabolas along X and Y axes in the 3x3 neighborhood of the discrete peak.

    Given 3 points (-1, y_-1), (0, y_0), (+1, y_+1),
    fits y = a + bx + cx^2 and solves dx = -b / (2c).

    Parameters
    ----------
    smap : np.ndarray
        2D similarity map from matchTemplate (float32).
    max_loc : tuple[int, int] | Sequence[int]
        Discrete (x, y) coordinates of the peak in smap.
    central_value : float
        Peak value smap[max_loc[1], max_loc[0]].

    Returns
    -------
    tuple[float, float]
        Sub-pixel refined offset (x + dx, y + dy).
    """
    h, w = smap.shape[:2]
    x0, y0 = int(max_loc[0]), int(max_loc[1])

    # Edge cases: cannot fit parabola on map boundaries
    if x0 <= 0 or x0 >= w - 1 or y0 <= 0 or y0 >= h - 1:
        return float(x0), float(y0)

    # Parabola along X axis
    xm1 = float(smap[y0, x0 - 1])
    xp1 = float(smap[y0, x0 + 1])
    denom_x = xm1 - 2.0 * central_value + xp1
    if abs(denom_x) > 1e-7:
        dx = (xm1 - xp1) / (2.0 * denom_x)
        dx = max(-1.0, min(1.0, dx))
    else:
        dx = 0.0

    # Parabola along Y axis
    ym1 = float(smap[y0 - 1, x0])
    yp1 = float(smap[y0 + 1, x0])
    denom_y = ym1 - 2.0 * central_value + yp1
    if abs(denom_y) > 1e-7:
        dy = (ym1 - yp1) / (2.0 * denom_y)
        dy = max(-1.0, min(1.0, dy))
    else:
        dy = 0.0

    return float(x0 + dx), float(y0 + dy)


def extract_region_stats_feature(patch: np.ndarray, shape: str = "point") -> np.ndarray:
    """Compact region descriptor (mean/std/hist + effective area) for circle/box cues."""
    if patch.size == 0:
        return np.zeros(40, dtype=np.float32)

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    h, w = gray.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    if shape == "circle":
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2, 255, -1)

    masked = gray[mask > 0]
    if masked.size == 0:
        return np.zeros(40, dtype=np.float32)

    mean_v = float(np.mean(masked))
    std_v = float(np.std(masked))
    area_frac = float(masked.size) / float(max(1, h * w))
    hist = cv2.calcHist([gray], [0], mask, [32], [0, 256]).flatten().astype(np.float32)
    hist_sum = float(hist.sum())
    if hist_sum > 1e-7:
        hist /= hist_sum
    feat = np.concatenate(
        [np.array([mean_v / 255.0, std_v / 255.0, area_frac], dtype=np.float32), hist[:37]]
    )
    feat = np.pad(feat, (0, 40 - feat.size)) if feat.size < 40 else feat[:40]
    norm = float(np.linalg.norm(feat))
    if norm > 1e-7:
        feat = feat / norm
    return feat.astype(np.float32)


class DeepFeatureExtractor:
    """Pre-trained CNN (ResNet50/152, MobileNetV3-Small, or EfficientNet-B0) feature
    extractor with cosine similarity.

    Extracts an L2-normalized visual semantic embedding on GPU/CUDA (or CPU) to verify
    target identity across dynamic athletic movements. Embedding width depends on the
    active variant (`self.feature_dim`, see `_BACKBONE_FEATURE_DIM`): 2048-d for the
    ResNet variants, 576-d for MobileNetV3-Small, 1280-d for EfficientNet-B0 -- callers
    that mix variants (e.g. AITracker's primary/fallback cascade) must not assume a
    fixed width.
    """

    _instances: dict[str, DeepFeatureExtractor] = {}
    _threads_configured = False

    def __init__(
        self,
        use_cuda: bool = True,
        weights_path: str | Path | None = None,
        variant: str = "resnet50",
    ) -> None:
        self.enabled = False
        self.device = "cpu"
        self.model: Any = None
        self.transform: Any = None
        self.feature_dim = 0
        self.weights_path = str(weights_path) if weights_path else ""
        self.variant = variant if variant in _BACKBONE_VARIANTS else "resnet50"

        if not TORCH_AVAILABLE:
            return

        try:
            if use_cuda and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                if not DeepFeatureExtractor._threads_configured:
                    # Avoid oversubscribing CPU threads across the lightweight
                    # cascade backbones; set once per process, not per instance.
                    torch.set_num_threads(min(4, torch.get_num_threads()))
                    DeepFeatureExtractor._threads_configured = True

            resolved = self._resolve_weights_path(weights_path, self.variant)
            if resolved is None:
                print(
                    f">> DeepFeatureExtractor: no local {self.variant} weights under "
                    f"{_default_checkpoint_dir()}. "
                    "Call ensure_backbone_weights() to browse or download into "
                    "vaila/models/ai_tracker/ first."
                )
                return

            model = tv_models.get_model(self.variant, weights=None)
            state = torch.load(resolved, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # Torchvision DEFAULT checkpoint keys may include the head's own
            # weights (e.g. "fc.*"/"classifier.*") — load then strip the head below.
            missing_unexpected = model.load_state_dict(state, strict=False)
            _ = missing_unexpected
            print(f">> DeepFeatureExtractor: loaded {self.variant} weights from {resolved}")
            self.weights_path = str(resolved)

            # Remove the classification head to output the pooled feature vector.
            # Head attribute differs by family: "fc" for ResNet, "classifier" for
            # MobileNetV3/EfficientNet.
            setattr(model, _BACKBONE_HEAD_ATTR[self.variant], torch.nn.Identity())
            self.feature_dim = _BACKBONE_FEATURE_DIM[self.variant]
            model.eval()
            model.to(self.device)

            self.model = model
            self.transform = tv_transforms.Compose(
                [
                    tv_transforms.ToPILImage(),
                    # Deliberate deviation from the official hub recipe's
                    # Resize(256)+CenterCrop(224): tracked patches are already small,
                    # near-square crops around the target, so a direct square resize
                    # keeps all edge content instead of cropping useful pixels away.
                    tv_transforms.Resize((224, 224)),
                    tv_transforms.ToTensor(),
                    tv_transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            self.enabled = True
        except Exception as err:
            print(f"DeepFeatureExtractor warning: Could not initialize {self.variant} ({err}).")
            self.enabled = False

    @staticmethod
    def _resolve_weights_path(
        weights_path: str | Path | None, variant: str = "resnet50"
    ) -> Path | None:
        """Resolve a local weight file under ai_tracker/ (never Torch hub cache)."""
        candidates: list[Path] = []
        if weights_path:
            candidates.append(Path(weights_path))
        candidates.append(_ai_tracker_resnet_local_path(variant))
        for extra in get_available_resnet_checkpoints(variant):
            candidates.append(extra)
        for cand in candidates:
            if _is_valid_weight_file(cand):
                try:
                    return cand.resolve()
                except OSError:
                    continue
        return None

    @classmethod
    def get_shared(
        cls,
        weights_path: str | Path | None = None,
        variant: str = "resnet50",
    ) -> DeepFeatureExtractor:
        """Get or initialize a feature extractor keyed by (variant, weights path)."""
        variant = variant if variant in _BACKBONE_VARIANTS else "resnet50"
        resolved = cls._resolve_weights_path(weights_path, variant)
        path_key = (
            str(resolved)
            if resolved is not None
            else (str(weights_path) if weights_path else "__default__")
        )
        key = f"{variant}:{path_key}"
        actual_path = resolved if resolved is not None else weights_path
        if key not in cls._instances:
            cls._instances[key] = cls(weights_path=actual_path, variant=variant)
        return cls._instances[key]

    def extract_embedding(self, patch_bgr: np.ndarray) -> np.ndarray | None:
        """Extract L2-normalized feature vector (width = self.feature_dim) from a BGR patch."""
        if not self.enabled or self.model is None or patch_bgr.size == 0:
            return None

        try:
            # Convert BGR to RGB
            if len(patch_bgr.shape) == 2:
                patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_GRAY2RGB)
            else:
                patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)

            tensor = self.transform(patch_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(tensor)
                feat = feat.squeeze().cpu().numpy().astype(np.float64)

            norm = np.linalg.norm(feat)
            if norm > 1e-7:
                feat = feat / norm
            return feat
        except Exception:
            return None

    @staticmethod
    def cosine_similarity(emb1: np.ndarray | None, emb2: np.ndarray | None) -> float:
        """Compute cosine similarity between two L2-normalized embeddings."""
        if emb1 is None or emb2 is None:
            return 1.0  # Neutral / no penalty if features unavailable
        dot = float(np.dot(emb1, emb2))
        return max(-1.0, min(1.0, dot))


class AITracker:
    """vailá AI Kinematic Tracker (NCC + Spatial Motion Prior + ResNet50 Verification).

    Features:
    - Normalized Cross-Correlation (cv2.TM_CCOEFF_NORMED)
    - 2D Gaussian spatial motion prior with velocity prediction
    - Elliptical mask to discard rectangular background corners
    - Parabolic sub-pixel peak refinement
    - Adaptive running template blending (exponential moving average)
    - ResNet50 visual feature cosine similarity to reject distractors
    - Online ridge discriminator that survives manual corrections
    """

    def __init__(self, parameters: AITrackerParameters | None = None) -> None:
        self.params = parameters or AITrackerParameters()
        self.template: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.anchor_template: np.ndarray | None = None
        self.anchor_embedding: np.ndarray | None = None
        # Opt-in cascade fallback backbone (Part 4): only built by set_reference() when
        # params.fallback_variant is truthy. Never feeds the discriminator -- see
        # effective_feature_dim()/_discriminator_feature().
        self.extractor_fallback: DeepFeatureExtractor | None = None
        self.anchor_embedding_fallback: np.ndarray | None = None
        self.last_point: tuple[float, float] | None = None
        self.acceleration: tuple[float, float] = (0.0, 0.0)
        # Kalman filter (constant-velocity model) state/covariance -- see module-level
        # comment above _KF_Q_POS for design rationale. `self.velocity` is a thin
        # property view over `self._kf_x[2:4]` so existing call sites (gap-seed
        # velocity bootstrap in infill_and_smooth, prediction, spatial-prior sigma)
        # keep working unchanged.
        self._kf_x: np.ndarray = np.zeros(4, dtype=np.float64)
        self._kf_P: np.ndarray = np.eye(4, dtype=np.float64) * _KF_P_INIT
        self._lost_streak: int = 0
        # Downsampled grayscale of the previous track_frame() call, used only to score
        # global scene change (see _SCENE_CHANGE_GATE) -- reset on set_reference so a
        # fresh anchor never compares against a stale/unrelated frame.
        self._prev_gray_small: np.ndarray | None = None

        # Online Appearance Model / Multi-Anchor Retraining
        self.anchors: list[dict[str, Any]] = []
        self.exemplar_templates: list[np.ndarray] = []
        self.training_feats: list[np.ndarray] = []
        self.training_labels: list[float] = []
        self.discriminator_w: np.ndarray | None = None
        self.discriminator_b: float = 0.0

        # Cross-session checkpoint (transfer learning): weights loaded from disk, kept
        # separate from the live session weights above so retrain_online_model() can
        # blend them by sample-count instead of overwriting prior knowledge.
        self._checkpoint_w: np.ndarray | None = None
        self._checkpoint_b: float = 0.0
        self._checkpoint_n: float = 0.0
        # Cumulative sample-count "confidence" behind the current discriminator_w/_b
        # (session samples plus, once blended, the checkpoint's own prior count).
        # This is what gets persisted by save_checkpoint() for the next session.
        self._live_n_samples: float = 0.0

        if self.params.use_deep_features:
            self.extractor = DeepFeatureExtractor.get_shared(
                weights_path=self.params.deep_weights_path or None,
                variant=self.params.resnet_variant,
            )
        else:
            self.extractor = None

        self._error_reported: bool = False

    @property
    def velocity(self) -> tuple[float, float]:
        """Current Kalman-filtered velocity estimate (px/frame), view over `_kf_x[2:4]`."""
        return (float(self._kf_x[2]), float(self._kf_x[3]))

    @velocity.setter
    def velocity(self, value: tuple[float, float]) -> None:
        self._kf_x[2] = float(value[0])
        self._kf_x[3] = float(value[1])

    def _create_elliptical_mask(self, width: int, height: int) -> np.ndarray:
        """Generate a binary elliptical mask matching circular marker boundaries."""
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        axes = (width // 2, height // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        return mask

    def _create_circular_mask(self, width: int, height: int) -> np.ndarray:
        """Generate a binary circular mask of maximum inscribed radius."""
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        radius = min(width, height) // 2
        cv2.circle(mask, center, radius, 255, -1)
        return mask

    def _apply_shape_mask(self) -> None:
        """Refresh self.mask from current block_window and tracking_shape."""
        bw, bh = self.params.block_window
        shape_mode = getattr(self.params, "tracking_shape", "point").lower()
        if shape_mode in ("circle",):
            self.mask = self._create_circular_mask(bw, bh)
        elif shape_mode in ("box", "rectangle"):
            self.mask = None
        elif self.params.use_mask:
            self.mask = self._create_elliptical_mask(bw, bh)
        else:
            self.mask = None

    @staticmethod
    def compute_shape_centroid(
        patch: np.ndarray,
        shape: str,
        bw: int,
        bh: int,
    ) -> tuple[float, float]:
        """Compute the 2D feature centroid (center of mass) inside the candidate patch.

        Parameters
        ----------
        patch : np.ndarray
            Cropped image patch (BGR or grayscale).
        shape : str
            Shape mode: "circle" or "box" / "rectangle".
        bw : int
            Patch width.
        bh : int
            Patch height.

        Returns
        -------
        tuple[float, float]
            Sub-pixel coordinates of the centroid relative to patch top-left.
        """
        if patch.size == 0:
            return bw / 2.0, bh / 2.0

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        h, w = gray.shape[:2]
        if shape == "circle":
            r = min(w, h) // 2
            cx_center, cy_center = w / 2.0, h / 2.0
            yy, xx = np.ogrid[:h, :w]
            circ_mask = ((xx - cx_center + 0.5) ** 2 + (yy - cy_center + 0.5) ** 2) <= (r**2)
            inside_vals = gray[circ_mask]
            if inside_vals.size == 0:
                return bw / 2.0, bh / 2.0
            w_min = float(np.min(inside_vals))
            w_max = float(np.max(inside_vals))
        else:  # "box" or "rectangle"
            circ_mask = np.ones((h, w), dtype=bool)
            w_min = float(np.min(gray))
            w_max = float(np.max(gray))

        if w_max - w_min > 12.0:
            weights = gray.astype(np.float32)
            saliency = (weights - w_min) / (w_max - w_min)
            if shape == "circle":
                saliency *= circ_mask
            moments = cv2.moments(saliency.astype(np.float32))
            if moments["m00"] > 1e-4:
                cx_res = float(moments["m10"] / moments["m00"])
                cy_res = float(moments["m01"] / moments["m00"])
                cx_res = max(1.0, min(float(w - 2), cx_res))
                cy_res = max(1.0, min(float(h - 2), cy_res))
                return cx_res, cy_res

        return bw / 2.0, bh / 2.0

    def extract_patch(
        self,
        frame: np.ndarray,
        center: tuple[float, float],
        size: tuple[int, int],
    ) -> np.ndarray:
        """Extract an integer or boundary-clamped image patch around a center coordinate."""
        h_img, w_img = frame.shape[:2]
        bw, bh = size
        cx, cy = int(round(center[0])), int(round(center[1]))

        x1 = cx - bw // 2
        y1 = cy - bh // 2
        x2 = x1 + bw
        y2 = y1 + bh

        # Clamp and handle padding if near frame boundary
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w_img)
        pad_bottom = max(0, y2 - h_img)

        crop_x1 = max(0, x1)
        crop_y1 = max(0, y1)
        crop_x2 = min(w_img, x2)
        crop_y2 = min(h_img, y2)

        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            cropped = cv2.copyMakeBorder(
                cropped,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_REPLICATE,
            )
        return cropped

    @staticmethod
    def extract_patch_feature(patch: np.ndarray, shape: str = "point") -> np.ndarray:
        """Fast L2-normalized appearance descriptor (color grid + region stats)."""
        if patch.size == 0:
            return np.zeros(808, dtype=np.float32)
        p_small = cv2.resize(patch, (16, 16)).astype(np.float32)
        color_feat = p_small.reshape(-1)
        region = extract_region_stats_feature(patch, shape=shape)
        feat = np.concatenate([color_feat, region])
        norm = float(np.linalg.norm(feat))
        if norm > 1e-7:
            feat = feat / norm
        return feat.astype(np.float32)

    def effective_feature_dim(self) -> int:
        """Discriminator feature length: 808 classical, or 808 + primary extractor's
        feature_dim (2048 for ResNet, 576 MobileNetV3-Small, 1280 EfficientNet-B0) when
        deep features are on. Only the PRIMARY extractor ever feeds the discriminator --
        a configured fallback_variant (see track_frame()) never changes this value, so
        checkpoint compatibility stays independent of which backbone wins a given frame.
        """
        if self.params.use_deep_features and self.extractor is not None and self.extractor.enabled:
            return _FEATURE_DIM + self.extractor.feature_dim
        return _FEATURE_DIM

    def _discriminator_feature(self, patch: np.ndarray, shape_mode: str) -> np.ndarray:
        """Classical 808-d appearance feature, optionally concatenated with the frozen
        primary-backbone embedding so the (retrained) discriminator benefits from deep
        semantic features without ever backpropagating through the CNN itself.
        """
        base = self.extract_patch_feature(patch, shape=shape_mode)
        if not (
            self.params.use_deep_features and self.extractor is not None and self.extractor.enabled
        ):
            return base
        deep_dim = self.extractor.feature_dim
        try:
            deep = self.extractor.extract_embedding(patch)
            if deep is None or deep.shape[0] != deep_dim:
                deep = np.zeros(deep_dim, dtype=np.float32)
        except Exception:
            deep = np.zeros(deep_dim, dtype=np.float32)
        return np.concatenate([base, deep.astype(np.float32)])

    def add_anchor(
        self,
        frame: np.ndarray,
        point: tuple[float, float],
        frame_idx: int = -1,
    ) -> None:
        """Add a keyframe anchor point, extract positive & negative exemplar patches, and stage samples."""
        bw, bh = self.params.block_window
        h_img, w_img = frame.shape[:2]
        shape_mode = getattr(self.params, "tracking_shape", "point").lower()

        pos_patch = self.extract_patch(frame, point, (bw, bh))
        if pos_patch.size == 0:
            return

        self.anchors.append({"frame": frame_idx, "point": point, "patch": pos_patch})
        if len(self.anchors) > _MAX_EXEMPLAR_TEMPLATES:
            self.anchors = self.anchors[-_MAX_EXEMPLAR_TEMPLATES:]
        self.exemplar_templates.append(pos_patch)
        if len(self.exemplar_templates) > _MAX_EXEMPLAR_TEMPLATES:
            self.exemplar_templates.pop(0)

        pos_feat = self._discriminator_feature(pos_patch, shape_mode)
        self.training_feats.append(pos_feat)
        self.training_labels.append(1.0)

        # Sample 6 negative background patches around the anchor point
        cx, cy = point
        neg_offsets = [
            (-1.5 * bw, 0),
            (1.5 * bw, 0),
            (0, -1.5 * bh),
            (0, 1.5 * bh),
            (-1.2 * bw, -1.2 * bh),
            (1.2 * bw, 1.2 * bh),
        ]
        for dx, dy in neg_offsets:
            nx = cx + dx
            ny = cy + dy
            if 0 <= nx < w_img and 0 <= ny < h_img:
                neg_patch = self.extract_patch(frame, (nx, ny), (bw, bh))
                if neg_patch.size > 0:
                    neg_feat = self._discriminator_feature(neg_patch, shape_mode)
                    self.training_feats.append(neg_feat)
                    self.training_labels.append(-1.0)

        if len(self.training_feats) > _MAX_TRAINING_FEATS:
            self.training_feats = self.training_feats[-_MAX_TRAINING_FEATS:]
            self.training_labels = self.training_labels[-_MAX_TRAINING_FEATS:]

    def retrain_online_model(self) -> float:
        """Retrain the online appearance discriminator using regularized dual Ridge regression (<5 ms).

        Returns
        -------
        float
            Elapsed training time in milliseconds.
        """
        if len(self.training_feats) < 2:
            return 0.0

        t0 = time.perf_counter()
        X = np.stack(self.training_feats, axis=0)  # (N, d)
        y = np.array(self.training_labels, dtype=np.float32)  # (N,)
        N = X.shape[0]

        # Solve dual system (N x N)
        reg_lambda = 0.05
        K = X @ X.T + reg_lambda * np.eye(N, dtype=np.float32)
        alpha = np.linalg.solve(K, y)
        w_session = X.T @ alpha
        b_session = float(np.mean(y - X @ w_session))

        # Transfer learning: blend this session's freshly-trained weights with a
        # loaded checkpoint (if any) by sample-count, so a returning session refines
        # the discriminator instead of rebuilding it from scratch every time it opens.
        if self._checkpoint_w is not None and self._checkpoint_w.shape == w_session.shape:
            n_session = float(N)
            n_checkpoint = min(float(self._checkpoint_n), _CHECKPOINT_BLEND_MAX_N)
            total_n = n_session + n_checkpoint
            self.discriminator_w = (
                n_session * w_session + n_checkpoint * self._checkpoint_w
            ) / total_n
            self.discriminator_b = (
                n_session * b_session + n_checkpoint * self._checkpoint_b
            ) / total_n
            self._live_n_samples = total_n
        else:
            self.discriminator_w = w_session
            self.discriminator_b = b_session
            self._live_n_samples = float(N)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms

    def load_checkpoint(self, path: str | Path) -> bool:
        """Load a persisted discriminator checkpoint for transfer learning.

        Returns True on success, False if the file is missing, unreadable, or has an
        incompatible feature dimension (silently ignored — tracking still works from
        scratch, just without the head start).
        """
        p = Path(path)
        if not p.is_file():
            return False
        try:
            data = np.load(p, allow_pickle=False)
            w = np.asarray(data["w"], dtype=np.float32)
            if w.shape != (self.effective_feature_dim(),):
                return False
            self._checkpoint_w = w
            self._checkpoint_b = float(data["b"])
            self._checkpoint_n = float(data["n_samples"])
        except Exception:
            return False
        return True

    def save_checkpoint(self, path: str | Path) -> bool:
        """Persist the current discriminator (weights + cumulative sample count).

        Returns True on success, False if there is no trained discriminator yet or the
        write failed (best-effort — never raises).
        """
        if self.discriminator_w is None or self.discriminator_w.shape != (
            self.effective_feature_dim(),
        ):
            return False
        p = Path(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            n_samples = self._live_n_samples if self._live_n_samples > 0 else 1.0
            np.savez(
                p,
                w=self.discriminator_w.astype(np.float32),
                b=np.float32(self.discriminator_b),
                n_samples=np.float32(n_samples),
                feat_dim=np.int32(self.effective_feature_dim()),
            )
        except Exception:
            return False
        return True

    def score_patch_discriminator(self, patch: np.ndarray) -> float:
        """Score candidate patch with the online retrained discriminator in [0.0, 1.0]."""
        if self.discriminator_w is None or patch.size == 0:
            return 1.0
        shape_mode = getattr(self.params, "tracking_shape", "point").lower()
        feat = self._discriminator_feature(patch, shape_mode)
        if feat.shape[0] != self.discriminator_w.shape[0]:
            return 1.0
        raw_val = float(np.dot(self.discriminator_w, feat) + self.discriminator_b)
        return float(1.0 / (1.0 + np.exp(-np.clip(raw_val, -10.0, 10.0))))

    def seed_anchors_from_known(
        self,
        get_frame: Callable[[int], np.ndarray | None],
        known_points: dict[int, tuple[float, float]],
        max_anchors: int = _MAX_GAP_SEED_ANCHORS,
    ) -> int:
        """Seed online model from a subsample of known keyframes (batch gap init)."""
        if not known_points:
            return 0
        frames = sorted(known_points.keys())
        if len(frames) > max_anchors:
            idx = np.linspace(0, len(frames) - 1, max_anchors).astype(int)
            frames = [frames[i] for i in idx]
        count = 0
        for f_idx in frames:
            frm = get_frame(f_idx)
            if frm is None:
                continue
            self.add_anchor(frm, known_points[f_idx], f_idx)
            count += 1
        if count > 0:
            self.retrain_online_model()
        return count

    def set_reference(
        self,
        frame: np.ndarray,
        point: tuple[float, float],
        frame_idx: int = 0,
        *,
        reset_online: bool = True,
    ) -> None:
        """Set reference template and deep feature embedding at the given keyframe point.

        Parameters
        ----------
        reset_online : bool
            When True (default), clear the online appearance model and reseed from this
            point. When False, keep existing anchors/discriminator and only refresh the
            running template + embedding (manual corrections / soft re-anchor).
        """
        bw, bh = self.params.block_window
        self.template = self.extract_patch(frame, point, (bw, bh))
        self.anchor_template = self.template.copy()
        # A (re-)anchor frame isn't guaranteed adjacent in time to the next track_frame()
        # call -- discard any stale scene-change baseline rather than comparing across a
        # jump cut.
        self._prev_gray_small = None
        if self.last_point is not None and not reset_online:
            dx = point[0] - self.last_point[0]
            dy = point[1] - self.last_point[1]
            new_vx = (1.0 - _VELOCITY_EMA) * self.velocity[0] + _VELOCITY_EMA * dx
            new_vy = (1.0 - _VELOCITY_EMA) * self.velocity[1] + _VELOCITY_EMA * dy
            self.acceleration = (
                (1.0 - _ACCEL_EMA) * self.acceleration[0]
                + _ACCEL_EMA * (new_vx - self.velocity[0]),
                (1.0 - _ACCEL_EMA) * self.acceleration[1]
                + _ACCEL_EMA * (new_vy - self.velocity[1]),
            )
            self.velocity = (new_vx, new_vy)
        elif reset_online:
            self.velocity = (0.0, 0.0)
            self.acceleration = (0.0, 0.0)
            self._kf_P = np.eye(4, dtype=np.float64) * _KF_P_INIT
        # A manual anchor/correction is exact ground truth: resync the Kalman position
        # state and shrink its position covariance to the measurement-noise floor, but
        # keep the velocity covariance (don't forget the filter's confidence on every
        # re-anchor) unless this was a full reset, handled above.
        self._kf_x[0], self._kf_x[1] = point
        self._kf_P[0, 0] = self._kf_P[1, 1] = _KF_R_POS
        self._kf_P[0, 1] = self._kf_P[1, 0] = 0.0
        self.last_point = point
        self._lost_streak = 0
        self._apply_shape_mask()

        if reset_online:
            self.anchors.clear()
            self.exemplar_templates.clear()
            self.training_feats.clear()
            self.training_labels.clear()
            self.discriminator_w = None
            self.discriminator_b = 0.0

        self.add_anchor(frame, point, frame_idx)
        self.retrain_online_model()

        # Handle deep feature extractor
        if self.params.use_deep_features:
            if self.extractor is None:
                self.extractor = DeepFeatureExtractor.get_shared(
                    weights_path=self.params.deep_weights_path or None,
                    variant=self.params.resnet_variant,
                )
            if self.extractor and self.extractor.enabled:
                self.anchor_embedding = self.extractor.extract_embedding(self.anchor_template)
            else:
                self.anchor_embedding = None

            # Fallback (cascade) extractor: opt-in, only built when configured. Used
            # solely for per-frame verification re-scoring in track_frame() -- never
            # feeds _discriminator_feature()/effective_feature_dim().
            if self.params.fallback_variant:
                if (
                    self.extractor_fallback is None
                    or self.extractor_fallback.variant != self.params.fallback_variant
                ):
                    self.extractor_fallback = DeepFeatureExtractor.get_shared(
                        variant=self.params.fallback_variant
                    )
                if self.extractor_fallback and self.extractor_fallback.enabled:
                    self.anchor_embedding_fallback = self.extractor_fallback.extract_embedding(
                        self.anchor_template
                    )
                else:
                    self.anchor_embedding_fallback = None
            else:
                self.extractor_fallback = None
                self.anchor_embedding_fallback = None
        else:
            self.anchor_embedding = None
            self.extractor_fallback = None
            self.anchor_embedding_fallback = None

    def update_from_correction(
        self,
        frame: np.ndarray,
        point: tuple[float, float],
        frame_idx: int = -1,
    ) -> None:
        """Apply a manual correction without wiping online learning."""
        self.set_reference(frame, point, frame_idx=frame_idx, reset_online=False)

    def track_frame(
        self,
        frame: np.ndarray,
        last_point: tuple[float, float],
    ) -> TemplateMatchResult:
        """Track the target in the current frame given the previous coordinate.

        Returns
        -------
        TemplateMatchResult
            Detailed matching result including refined sub-pixel coordinates and confidence.
            ``accepted`` is False when combined score is below ``similarity_threshold``.
        """
        if self.template is None:
            raise RuntimeError("Tracker has no reference template. Call set_reference first.")

        h_img, w_img = frame.shape[:2]
        sw, sh = self.params.search_window
        bw, bh = self.params.block_window

        # Global scene-change score: mean abs diff of downsampled grayscale prev-vs-
        # current frame, normalized [0, 1]. Cheap (64x64), and only ever used to
        # distrust a marginal accept below (see _SCENE_CHANGE_GATE) -- never rejects a
        # confident match. Update the baseline before any early return so it always
        # reflects the immediately-preceding frame on the next call.
        gray_small = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame,
            (64, 64),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        scene_change_score = 0.0
        if self._prev_gray_small is not None:
            scene_change_score = float(np.mean(np.abs(gray_small - self._prev_gray_small)) / 255.0)
        self._prev_gray_small = gray_small

        # Kalman predict (constant-velocity model). `last_point` is authoritative (the
        # previous frame's accepted location, or a manual anchor via set_reference) --
        # resync the filter's position state to it before propagating one frame ahead,
        # so any external re-anchor never leaves the filter's belief stale.
        self._kf_x[0], self._kf_x[1] = last_point
        kf_f = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        kf_q = np.diag([_KF_Q_POS, _KF_Q_POS, _KF_Q_VEL, _KF_Q_VEL])
        kf_x_pred = kf_f @ self._kf_x
        kf_p_pred = kf_f @ self._kf_P @ kf_f.T + kf_q
        pred_x = float(kf_x_pred[0])
        pred_y = float(kf_x_pred[1])
        lx, ly = int(round(pred_x)), int(round(pred_y))

        # Search window bounding box centered on prediction
        sx1 = lx - sw // 2
        sy1 = ly - sh // 2
        sx2 = sx1 + sw
        sy2 = sy1 + sh

        # Intersect with frame dimensions
        crop_sx1 = max(0, sx1)
        crop_sy1 = max(0, sy1)
        crop_sx2 = min(w_img, sx2)
        crop_sy2 = min(h_img, sy2)

        roi = frame[crop_sy1:crop_sy2, crop_sx1:crop_sx2]
        if roi.shape[0] < bh or roi.shape[1] < bw:
            # No measurement this frame: propagate-only (predicted state becomes the new
            # belief, uncertainty keeps growing under Q).
            self._kf_x = kf_x_pred
            self._kf_P = kf_p_pred
            self._lost_streak += 1
            return TemplateMatchResult(
                similarity=0.0,
                ncc_score=0.0,
                deep_score=0.0,
                location=last_point,
                raw_location=(int(round(last_point[0])), int(round(last_point[1]))),
                template_updated=False,
                accepted=False,
            )

        # Match template using NCC (cv2.TM_CCOEFF_NORMED)
        mask_to_use = self.mask if (self.params.use_mask and self.mask is not None) else None
        try:
            if mask_to_use is not None:
                if len(roi.shape) == 3 and mask_to_use.ndim == 2:
                    cv_mask = cv2.merge([mask_to_use, mask_to_use, mask_to_use])
                else:
                    cv_mask = mask_to_use
                smap = cv2.matchTemplate(roi, self.template, cv2.TM_CCOEFF_NORMED, mask=cv_mask)
            else:
                smap = cv2.matchTemplate(roi, self.template, cv2.TM_CCOEFF_NORMED)
        except Exception:
            smap = cv2.matchTemplate(roi, self.template, cv2.TM_CCOEFF_NORMED)

        # Handle NaNs or Infs
        if np.isnan(smap).any() or np.isinf(smap).any():
            smap = np.nan_to_num(smap, nan=-1.0, posinf=1.0, neginf=-1.0)

        # 2D Gaussian Spatial Motion Prior around predicted position
        center_map_x = lx - crop_sx1 - bw // 2
        center_map_y = ly - crop_sy1 - bh // 2
        sh_map, sw_map = smap.shape[:2]
        yy, xx = np.mgrid[:sh_map, :sw_map]
        dist2 = (xx - center_map_x) ** 2 + (yy - center_map_y) ** 2
        speed = float(np.hypot(self.velocity[0], self.velocity[1]))
        spatial_sigma = float(self.params.spatial_sigma) + 0.5 * speed
        spatial_prior = np.exp(-dist2 / (2.0 * max(1.0, spatial_sigma) ** 2))

        # Weight positive correlation scores by distance from predicted position
        weighted_smap = np.maximum(0.0, smap) * spatial_prior
        _min_val, _max_val, _min_loc, max_loc = cv2.minMaxLoc(weighted_smap)
        ncc_score = float(smap[max_loc[1], max_loc[0]])

        # Refine sub-pixel location with parabolic fit
        ref_x, ref_y = refine_location_parabola(smap, max_loc, ncc_score)

        # Compute candidate point in full image coordinates (center of template)
        cand_x = float(crop_sx1 + ref_x + bw / 2.0)
        cand_y = float(crop_sy1 + ref_y + bh / 2.0)

        # Extract candidate patch for visual / discriminator evaluation
        cand_patch = self.extract_patch(frame, (cand_x, cand_y), (bw, bh))

        # Shape-aware centroid refinement: compute feature centroid for circle or box
        shape_mode = getattr(self.params, "tracking_shape", "point").lower()
        if shape_mode in ("circle", "box", "rectangle") and cand_patch.size > 0:
            cx_c, cy_c = self.compute_shape_centroid(cand_patch, shape_mode, bw, bh)
            cand_x = float(crop_sx1 + ref_x + cx_c)
            cand_y = float(crop_sy1 + ref_y + cy_c)
            cand_patch = self.extract_patch(frame, (cand_x, cand_y), (bw, bh))

        # Deep Feature Cosine Similarity Verification
        deep_score = None
        if self.params.use_deep_features:
            if self.extractor is None:
                self.extractor = DeepFeatureExtractor.get_shared(
                    weights_path=self.params.deep_weights_path or None,
                    variant=self.params.resnet_variant,
                )
            if (
                self.anchor_embedding is None
                and self.anchor_template is not None
                and self.extractor.enabled
            ):
                self.anchor_embedding = self.extractor.extract_embedding(self.anchor_template)
            if self.extractor and self.extractor.enabled and self.anchor_embedding is not None:
                cand_emb = self.extractor.extract_embedding(cand_patch)
                deep_score = self.extractor.cosine_similarity(self.anchor_embedding, cand_emb)

        # Online Retrained Appearance Model Verification
        disc_score = (
            self.score_patch_discriminator(cand_patch) if self.discriminator_w is not None else None
        )

        # Score fusion
        norm_ncc = max(0.0, ncc_score)
        if deep_score is not None:
            norm_deep = max(0.0, (deep_score + 1.0) / 2.0)
            alpha = self.params.deep_weight
            if disc_score is not None:
                combined_score = (
                    (1.0 - alpha) * 0.85 * norm_ncc
                    + alpha * norm_deep
                    + (1.0 - alpha) * 0.15 * disc_score
                )
            else:
                combined_score = (1.0 - alpha) * norm_ncc + alpha * norm_deep
        else:
            if disc_score is not None:
                combined_score = 0.85 * norm_ncc + 0.15 * disc_score
            else:
                combined_score = norm_ncc

        # Opt-in cascade fallback (Part 4): only when a second backbone is configured
        # and the primary-backbone score is low. Re-scores with the SAME NCC/discriminator
        # fusion formula, using the fallback embedding only for this frame's verification --
        # never feeds _discriminator_feature()/effective_feature_dim() (primary-only, by design).
        active_variant = self.params.resnet_variant
        if (
            self.extractor_fallback is not None
            and self.extractor_fallback.enabled
            and self.anchor_embedding_fallback is not None
            and deep_score is not None
            and combined_score < self.params.fallback_threshold
        ):
            cand_emb_fb = self.extractor_fallback.extract_embedding(cand_patch)
            deep_score_fb = self.extractor_fallback.cosine_similarity(
                self.anchor_embedding_fallback, cand_emb_fb
            )
            norm_deep_fb = max(0.0, (deep_score_fb + 1.0) / 2.0)
            alpha = self.params.deep_weight
            if disc_score is not None:
                combined_score_fb = (
                    (1.0 - alpha) * 0.85 * norm_ncc
                    + alpha * norm_deep_fb
                    + (1.0 - alpha) * 0.15 * disc_score
                )
            else:
                combined_score_fb = (1.0 - alpha) * norm_ncc + alpha * norm_deep_fb
            if combined_score_fb > combined_score:
                combined_score = combined_score_fb
                deep_score = deep_score_fb
                active_variant = self.params.fallback_variant

        accepted = float(combined_score) >= float(self.params.similarity_threshold)

        # Trajectory-plausibility gate: genuine Kalman-filter innovation covariance +
        # Mahalanobis-distance outlier rejection. A candidate can pass the soft
        # similarity score yet still land far from where the covariance-propagated
        # prediction says the target should be — this is exactly how drift happens
        # during acceleration + motion blur + background-polarity-flip (a spurious
        # high-scoring patch on the far side of the flip hijacks the match). Reject such
        # jumps even though the raw score passed; freeze at last_point like any other
        # rejection. Unlike a fixed/additive pixel threshold, the gate widens on its own
        # during a real acceleration event (P grows through process noise Q every frame
        # a measurement is rejected) and tightens back down once tracking stabilizes.
        kf_h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        kf_r = np.eye(2) * _KF_R_POS
        innovation = np.array([cand_x - pred_x, cand_y - pred_y])
        kf_s = kf_h @ kf_p_pred @ kf_h.T + kf_r
        try:
            kf_s_inv = np.linalg.inv(kf_s)
            mahalanobis_sq = float(innovation @ kf_s_inv @ innovation)
        except np.linalg.LinAlgError:
            mahalanobis_sq = 0.0
        if accepted and mahalanobis_sq > _KF_MAHALANOBIS_GATE:
            accepted = False

        # Scene-change distrust: a marginal accept (score only just above threshold)
        # during a large global scene change (e.g. background polarity flip) is the
        # signature of a spurious patch on the far side of the flip scoring just high
        # enough to pass while still landing inside the (possibly already-widened)
        # Kalman gate. Only downgrade a marginal accept -- a confidently-scored match
        # (well above threshold) is never second-guessed by this gate.
        if (
            accepted
            and scene_change_score > _SCENE_CHANGE_GATE
            and float(combined_score)
            < float(self.params.similarity_threshold) + _SCENE_CHANGE_MARGIN
        ):
            accepted = False

        template_updated = False
        raw_loc = (
            int(crop_sx1 + max_loc[0] + bw // 2),
            int(crop_sy1 + max_loc[1] + bh // 2),
        )

        if not accepted:
            # No measurement update: propagate-only, so uncertainty keeps growing under
            # Q and next frame's gate widens automatically -- exactly the behavior
            # needed to ride out an acceleration/blur event without drifting onto a
            # spurious match.
            self._kf_x = kf_x_pred
            self._kf_P = kf_p_pred
            self._lost_streak += 1
            return TemplateMatchResult(
                similarity=float(combined_score),
                ncc_score=ncc_score,
                deep_score=deep_score,
                location=last_point,
                raw_location=raw_loc,
                template_updated=False,
                accepted=False,
                active_variant=active_variant,
            )

        self._lost_streak = 0
        final_loc = (cand_x, cand_y)

        # Adaptive Running Template Blending only on confident accepted matches
        if ncc_score >= self.params.template_update_threshold:
            new_patch = cand_patch
            if new_patch.shape == self.template.shape:
                gamma = float(self.params.template_learning_rate)
                self.template = np.clip(
                    (1.0 - gamma) * self.template.astype(np.float32)
                    + gamma * new_patch.astype(np.float32),
                    0,
                    255,
                ).astype(np.uint8)
                template_updated = True

        # Kalman update. Position is still reported as the raw appearance-matched
        # `final_loc` (cand_x, cand_y) -- an accepted match is, by construction, the
        # most trustworthy appearance evidence available, and blending it toward the
        # prediction would only soften the sub-pixel accuracy the NCC/deep/discriminator
        # fusion already achieved. The Kalman gain/update instead governs what the
        # filter *believes* (velocity + covariance) going into next frame's predict --
        # `_kf_x`/`_kf_P` are resynced in position at the top of the next call anyway,
        # so only the velocity posterior and the shrunk covariance carry forward.
        old_vx, old_vy = self.velocity
        kf_gain = kf_p_pred @ kf_h.T @ kf_s_inv
        kf_x_upd = kf_x_pred + kf_gain @ innovation
        kf_p_upd = (np.eye(4) - kf_gain @ kf_h) @ kf_p_pred
        self._kf_x = kf_x_upd
        self._kf_P = kf_p_upd
        new_vx, new_vy = self.velocity
        self.acceleration = (
            (1.0 - _ACCEL_EMA) * self.acceleration[0] + _ACCEL_EMA * (new_vx - old_vx),
            (1.0 - _ACCEL_EMA) * self.acceleration[1] + _ACCEL_EMA * (new_vy - old_vy),
        )
        self.last_point = final_loc
        return TemplateMatchResult(
            similarity=float(combined_score),
            ncc_score=ncc_score,
            deep_score=deep_score,
            location=final_loc,
            raw_location=raw_loc,
            template_updated=template_updated,
            accepted=True,
            active_variant=active_variant,
        )


# Backward-compatible alias
KinoveaTracker = AITracker


def _track_segment(
    tracker: AITracker,
    get_frame: Callable[[int], np.ndarray | None],
    start_pt: tuple[float, float],
    frame_range: range,
    progress_callback: Callable[[int, int, str], bool] | None,
    total_frames: int,
    msg_prefix: str,
) -> tuple[dict[int, tuple[float, float]], dict[int, float]]:
    """Track along frame_range; freeze position on rejected matches; stop after lost streak."""
    pts: dict[int, tuple[float, float]] = {}
    scores: dict[int, float] = {}
    curr_pt = start_pt
    for f in frame_range:
        if progress_callback and not progress_callback(
            f, total_frames, f"{msg_prefix} frame {f + 1}..."
        ):
            break
        frm = get_frame(f)
        if frm is None:
            break
        res = tracker.track_frame(frm, curr_pt)
        scores[f] = res.similarity
        if res.accepted:
            pts[f] = res.location
            curr_pt = res.location
        else:
            # Keep last good lock but do not invent a new peak; leave NaN for interp later
            if tracker._lost_streak >= _LOST_STREAK_STOP:
                break
    return pts, scores


def infill_and_smooth(
    cap: Any,
    total_frames: int,
    known_points: dict[int, tuple[float, float]],
    fps: float = 60.0,
    parameters: AITrackerParameters | None = None,
    progress_callback: Callable[[int, int, str], bool] | None = None,
) -> TrackingResult:
    """Infill gaps between user-annotated keyframe markers using bidirectional tracking and RTS smoothing.

    Parameters
    ----------
    cap : Any
        Video source implementing set(CAP_PROP_POS_FRAMES) and read().
    total_frames : int
        Total number of frames in the video.
    known_points : dict[int, tuple[float, float]]
        Map of frame_index -> (x, y) coordinates manually placed by the user.
    fps : float
        Video acquisition frame rate in Hz.
    parameters : AITrackerParameters | None
        Tracker configuration parameters.
    progress_callback : Callable[[current_step, total_steps, message], bool] | None
        Callback for GUI progress pumping and cancellation. Return False to cancel.

    Returns
    -------
    TrackingResult
        Complete trajectory with zero-phase smoothed positions, velocities, accelerations,
        and confidence scores.
    """
    if not known_points:
        raise ValueError("At least 1 anchor point is required for tracking.")

    params = parameters or AITrackerParameters()
    sorted_frames = sorted(known_points.keys())

    # Raw tracked positions and confidence scores
    raw_positions = np.full((total_frames, 2), np.nan, dtype=np.float64)
    confidences = np.zeros(total_frames, dtype=np.float64)

    # Set user annotations as immutable ground truth
    for f, pt in known_points.items():
        raw_positions[f] = pt
        confidences[f] = 1.0

    # Frame cache to eliminate the catastrophic codec seeking penalty on backward passes
    frame_cache: dict[int, np.ndarray] = {}

    def fetch_frames(start_idx: int, end_idx: int) -> None:
        """Prefetch frames sequentially into cache to eliminate backward seeking penalty."""
        nonlocal frame_cache
        start_c = max(0, min(start_idx, end_idx))
        end_c = min(total_frames - 1, max(start_idx, end_idx))
        needed = [f for f in range(start_c, end_c + 1) if f not in frame_cache]
        if not needed:
            return
        min_f = min(needed)
        max_f = max(needed)
        if hasattr(cap, "set"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min_f)
        for f in range(min_f, max_f + 1):
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_cache[f] = frame
            else:
                break

    def get_frame(f_idx: int) -> np.ndarray | None:
        if f_idx in frame_cache:
            return frame_cache[f_idx]
        if hasattr(cap, "set"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_cache[f_idx] = frame
            return frame
        return None

    def _init_tracker_at(
        f_idx: int,
        seed_known: dict[int, tuple[float, float]] | None = None,
    ) -> AITracker | None:
        frm = get_frame(f_idx)
        if frm is None:
            return None
        trk = AITracker(params)
        trk.set_reference(frm, known_points[f_idx], frame_idx=f_idx, reset_online=True)
        if seed_known:
            # Re-add other anchors without wiping primary template (reset already done)
            other = {k: v for k, v in seed_known.items() if k != f_idx}
            if other:
                trk.seed_anchors_from_known(get_frame, other)
        return trk

    # Determine gaps to fill
    # Single anchor case: track forward to end and backward to start
    if len(sorted_frames) == 1:
        f0 = sorted_frames[0]
        fetch_frames(0, total_frames - 1)
        tracker = _init_tracker_at(f0)
        if tracker is None:
            raise RuntimeError(f"Failed to read anchor frame {f0}")

        fwd_pts, fwd_scores = _track_segment(
            tracker,
            get_frame,
            known_points[f0],
            range(f0 + 1, total_frames),
            progress_callback,
            total_frames,
            "Tracking forward",
        )
        for f, pt in fwd_pts.items():
            raw_positions[f] = pt
            confidences[f] = fwd_scores.get(f, 0.0)

        tracker_bwd = _init_tracker_at(f0)
        if tracker_bwd is not None:
            bwd_pts, bwd_scores = _track_segment(
                tracker_bwd,
                get_frame,
                known_points[f0],
                range(f0 - 1, -1, -1),
                progress_callback,
                total_frames,
                "Tracking backward",
            )
            for f, pt in bwd_pts.items():
                raw_positions[f] = pt
                confidences[f] = bwd_scores.get(f, 0.0)

    else:
        # Multiple anchors: identify gaps between keyframe segments
        # 1. Fill gaps between consecutive segments with bidirectional tracking
        for i in range(len(sorted_frames) - 1):
            f_start = sorted_frames[i]
            f_end = sorted_frames[i + 1]

            if f_end - f_start <= 1:
                continue  # Contiguous, no gap

            # Prefetch gap frames sequentially in one quick pass
            fetch_frames(f_start, f_end)

            # Seed with all known anchors up to and including gap endpoints
            seed_slice = {
                f: known_points[f] for f in sorted_frames if f_start <= f <= f_end or f <= f_start
            }

            trk_fwd = _init_tracker_at(f_start, seed_known=seed_slice)
            fwd_pts: dict[int, tuple[float, float]] = {}
            fwd_scores: dict[int, float] = {}
            if trk_fwd is not None:
                # Bootstrap velocity from next known if available
                if f_end - f_start > 1:
                    dx = (known_points[f_end][0] - known_points[f_start][0]) / float(
                        f_end - f_start
                    )
                    dy = (known_points[f_end][1] - known_points[f_start][1]) / float(
                        f_end - f_start
                    )
                    trk_fwd.velocity = (dx, dy)
                fwd_pts, fwd_scores = _track_segment(
                    trk_fwd,
                    get_frame,
                    known_points[f_start],
                    range(f_start + 1, f_end),
                    progress_callback,
                    total_frames,
                    f"Tracking gap {f_start}→{f_end}: forward",
                )

            trk_bwd = _init_tracker_at(f_end, seed_known=seed_slice)
            bwd_pts: dict[int, tuple[float, float]] = {}
            bwd_scores: dict[int, float] = {}
            if trk_bwd is not None:
                if f_end - f_start > 1:
                    dx = (known_points[f_start][0] - known_points[f_end][0]) / float(
                        f_end - f_start
                    )
                    dy = (known_points[f_start][1] - known_points[f_end][1]) / float(
                        f_end - f_start
                    )
                    trk_bwd.velocity = (dx, dy)
                bwd_pts, bwd_scores = _track_segment(
                    trk_bwd,
                    get_frame,
                    known_points[f_end],
                    range(f_end - 1, f_start, -1),
                    progress_callback,
                    total_frames,
                    f"Tracking gap {f_start}→{f_end}: backward",
                )

            # Fuse forward and backward tracks in the gap with distance & squared confidence weighting
            for f in range(f_start + 1, f_end):
                p_f = fwd_pts.get(f)
                p_b = bwd_pts.get(f)
                s_f = fwd_scores.get(f, 0.0)
                s_b = bwd_scores.get(f, 0.0)
                thr = float(params.similarity_threshold)

                if p_f is not None and p_b is not None:
                    # Temporal distance weights: 1.0 at f_start -> 0.0 at f_end
                    w_dist_f = float(f_end - f) / float(f_end - f_start)
                    w_dist_b = float(f - f_start) / float(f_end - f_start)

                    w_f = w_dist_f * (max(0.1, s_f) ** 2)
                    w_b = w_dist_b * (max(0.1, s_b) ** 2)
                    total_w = w_f + w_b
                    if total_w > 0:
                        w_f /= total_w
                        w_b /= total_w

                    fused_x = w_f * p_f[0] + w_b * p_b[0]
                    fused_y = w_f * p_f[1] + w_b * p_b[1]
                    raw_positions[f] = (fused_x, fused_y)
                    confidences[f] = w_f * s_f + w_b * s_b
                elif p_f is not None and s_f >= thr:
                    raw_positions[f] = p_f
                    confidences[f] = s_f
                elif p_b is not None and s_b >= thr:
                    raw_positions[f] = p_b
                    confidences[f] = s_b
                # else: leave NaN → linear interp between anchors

            # Evict frames older than f_start from cache to bound RAM usage
            old_keys = [k for k in list(frame_cache.keys()) if k < f_start]
            for k in old_keys:
                del frame_cache[k]

        # 2. Track backward before the first anchor (0 ... sorted_frames[0] - 1)
        first_f = sorted_frames[0]
        if first_f > 0:
            fetch_frames(0, first_f)
            trk_head = _init_tracker_at(first_f, seed_known=known_points)
            if trk_head is not None:
                head_pts, head_scores = _track_segment(
                    trk_head,
                    get_frame,
                    known_points[first_f],
                    range(first_f - 1, -1, -1),
                    progress_callback,
                    total_frames,
                    "Tracking head",
                )
                for f, pt in head_pts.items():
                    raw_positions[f] = pt
                    confidences[f] = head_scores.get(f, 0.0)

        # 3. Track forward after the last anchor (sorted_frames[-1] + 1 ... total_frames - 1)
        last_f = sorted_frames[-1]
        if last_f < total_frames - 1:
            fetch_frames(last_f, total_frames - 1)
            trk_tail = _init_tracker_at(last_f, seed_known=known_points)
            if trk_tail is not None:
                tail_pts, tail_scores = _track_segment(
                    trk_tail,
                    get_frame,
                    known_points[last_f],
                    range(last_f + 1, total_frames),
                    progress_callback,
                    total_frames,
                    "Tracking tail",
                )
                for f, pt in tail_pts.items():
                    raw_positions[f] = pt
                    confidences[f] = tail_scores.get(f, 0.0)

    # Interpolate any lingering NaNs if frames couldn't be decoded or tracking lost
    valid_mask = ~np.isnan(raw_positions[:, 0])
    if not np.all(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            raw_positions[:, 0] = np.interp(
                np.arange(total_frames),
                valid_indices,
                raw_positions[valid_indices, 0],
            )
            raw_positions[:, 1] = np.interp(
                np.arange(total_frames),
                valid_indices,
                raw_positions[valid_indices, 1],
            )
            confidences[~valid_mask] = 0.2

    # Run Rauch-Tung-Striebel (RTS) Zero-Phase Smoother over full trajectory
    smoother = RTSSmoother(fps=fps, sigma_a=50.0)
    smoothed_states, smoothed_covariances = smoother.smooth(raw_positions, confidences)

    trajectory = smoothed_states[:, 0:2].copy()
    velocities = smoothed_states[:, 2:4].copy()
    accelerations = smoothed_states[:, 4:6].copy()

    # Ensure user keyframes match original inputs exactly
    for f, pt in known_points.items():
        if 0 <= f < total_frames:
            trajectory[f] = pt

    return TrackingResult(
        trajectory=trajectory,
        velocities=velocities,
        accelerations=accelerations,
        covariances=smoothed_covariances,
        confidence_scores=confidences,
    )
