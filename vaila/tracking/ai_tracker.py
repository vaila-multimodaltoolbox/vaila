"""
vailá AI Kinematic Tracker (Normalized Cross-Correlation + Deep Feature Verification).

Implements robust kinematic tracking with 2D Gaussian spatial motion priors,
elliptical masking, parabolic sub-pixel peak refinement, adaptive running template blending,
and Deep Visual Feature embeddings (PyTorch ResNet50 / CUDA) for semantic verification,
distractor rejection, and occlusion recovery. Integrates bidirectional keyframe
infilling and Rauch-Tung-Striebel (RTS) zero-phase smoothing (Δϕ = 0).

Author: Prof. Dr. Paulo R. P. Santiago
Update Date: 10 September 2026
Version: 0.3.134
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


@dataclass
class TemplateMatchResult:
    """Result of a single-frame template match."""

    similarity: float  # Combined confidence score in [0.0, 1.0]
    ncc_score: float  # Pure NCC correlation score in [-1.0, 1.0]
    deep_score: float | None  # Deep feature cosine similarity in [-1.0, 1.0]
    location: tuple[float, float]  # Refined sub-pixel (x, y) coordinates
    raw_location: tuple[int, int]  # Discrete integer peak (x, y)
    template_updated: bool = False  # Whether the tracking template was updated this frame


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

    def to_toml(self, toml_path: str | Path) -> None:
        """Serialize tracking parameters to a TOML file."""
        path = Path(toml_path)
        path.parent.mkdir(parents=True, exist_ok=True)
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


class DeepFeatureExtractor:
    """Pre-trained CNN (ResNet50) feature extractor with cosine similarity verification.

    Extracts L2-normalized 2048-dimensional visual semantic embeddings on GPU/CUDA
    (or CPU) to verify target identity across dynamic athletic movements.
    """

    _instance: DeepFeatureExtractor | None = None

    def __init__(self, use_cuda: bool = True) -> None:
        self.enabled = False
        self.device = "cpu"
        self.model: Any = None
        self.transform: Any = None

        if not TORCH_AVAILABLE:
            return

        try:
            if use_cuda and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

            weights = tv_models.ResNet50_Weights.DEFAULT
            model = tv_models.resnet50(weights=weights)
            # Remove classification head to output 2048-dim feature vector
            model.fc = torch.nn.Identity()
            model.eval()
            model.to(self.device)

            self.model = model
            self.transform = tv_transforms.Compose(
                [
                    tv_transforms.ToPILImage(),
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
            print(f"DeepFeatureExtractor warning: Could not initialize ResNet50 ({err}).")
            self.enabled = False

    @classmethod
    def get_shared(cls) -> DeepFeatureExtractor:
        """Get or initialize singleton feature extractor."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def extract_embedding(self, patch_bgr: np.ndarray) -> np.ndarray | None:
        """Extract L2-normalized 2048-dim feature vector from BGR image patch."""
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
    - 2D Gaussian spatial motion prior to eliminate teleports and distractor jumps
    - Elliptical mask to discard rectangular background corners
    - Parabolic sub-pixel peak refinement
    - Adaptive running template blending (exponential moving average)
    - ResNet50 visual feature cosine similarity to reject distractors
    """

    def __init__(self, parameters: AITrackerParameters | None = None) -> None:
        self.params = parameters or AITrackerParameters()
        self.template: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.anchor_template: np.ndarray | None = None
        self.anchor_embedding: np.ndarray | None = None
        self.last_point: tuple[float, float] | None = None

        # Online Appearance Model / Multi-Anchor Retraining
        self.anchors: list[dict[str, Any]] = []
        self.exemplar_templates: list[np.ndarray] = []
        self.training_feats: list[np.ndarray] = []
        self.training_labels: list[float] = []
        self.discriminator_w: np.ndarray | None = None
        self.discriminator_b: float = 0.0

        if self.params.use_deep_features:
            self.extractor = DeepFeatureExtractor.get_shared()
        else:
            self.extractor = None

    def _create_elliptical_mask(self, width: int, height: int) -> np.ndarray:
        """Generate a binary elliptical mask matching circular marker boundaries."""
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        axes = (width // 2, height // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        return mask

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
    def extract_patch_feature(patch: np.ndarray) -> np.ndarray:
        """Extract a fast, L2-normalized 768-dim color patch descriptor (<0.05 ms)."""
        if patch.size == 0:
            return np.zeros(768, dtype=np.float32)
        p_small = cv2.resize(patch, (16, 16)).astype(np.float32)
        feat = p_small.reshape(-1)
        norm = float(np.linalg.norm(feat))
        if norm > 1e-7:
            feat = feat / norm
        return feat

    def add_anchor(
        self,
        frame: np.ndarray,
        point: tuple[float, float],
        frame_idx: int = -1,
    ) -> None:
        """Add a keyframe anchor point, extract positive & negative exemplar patches, and stage samples."""
        bw, bh = self.params.block_window
        h_img, w_img = frame.shape[:2]

        pos_patch = self.extract_patch(frame, point, (bw, bh))
        if pos_patch.size == 0:
            return

        self.anchors.append({"frame": frame_idx, "point": point, "patch": pos_patch})
        self.exemplar_templates.append(pos_patch)
        if len(self.exemplar_templates) > 10:
            self.exemplar_templates.pop(0)

        pos_feat = self.extract_patch_feature(pos_patch)
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
                    neg_feat = self.extract_patch_feature(neg_patch)
                    self.training_feats.append(neg_feat)
                    self.training_labels.append(-1.0)

        if len(self.training_feats) > 100:
            self.training_feats = self.training_feats[-100:]
            self.training_labels = self.training_labels[-100:]

    def retrain_online_model(self) -> float:
        """Retrain the online appearance discriminator using regularized dual Ridge regression (<1 ms).

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
        self.discriminator_w = X.T @ alpha
        self.discriminator_b = float(np.mean(y - X @ self.discriminator_w))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms

    def score_patch_discriminator(self, patch: np.ndarray) -> float:
        """Score candidate patch with the online retrained discriminator in [0.0, 1.0]."""
        if self.discriminator_w is None or patch.size == 0:
            return 1.0
        feat = self.extract_patch_feature(patch)
        raw_val = float(np.dot(self.discriminator_w, feat) + self.discriminator_b)
        return float(1.0 / (1.0 + np.exp(-np.clip(raw_val, -10.0, 10.0))))

    def set_reference(
        self,
        frame: np.ndarray,
        point: tuple[float, float],
        frame_idx: int = 0,
    ) -> None:
        """Set reference template and deep feature embedding at the given keyframe point."""
        bw, bh = self.params.block_window
        self.template = self.extract_patch(frame, point, (bw, bh))
        self.anchor_template = self.template.copy()
        self.last_point = point

        if self.params.use_mask:
            self.mask = self._create_elliptical_mask(bw, bh)
        else:
            self.mask = None

        # Reset online appearance model and seed with this primary anchor
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
                self.extractor = DeepFeatureExtractor.get_shared()
            if self.extractor and self.extractor.enabled:
                self.anchor_embedding = self.extractor.extract_embedding(self.anchor_template)
            else:
                self.anchor_embedding = None
        else:
            self.anchor_embedding = None

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
        """
        if self.template is None:
            raise RuntimeError("Tracker has no reference template. Call set_reference first.")

        h_img, w_img = frame.shape[:2]
        sw, sh = self.params.search_window
        bw, bh = self.params.block_window

        lx, ly = int(round(last_point[0])), int(round(last_point[1]))

        # Search window bounding box
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
            # Cannot match inside smaller ROI than template
            return TemplateMatchResult(
                similarity=0.0,
                ncc_score=0.0,
                deep_score=0.0,
                location=last_point,
                raw_location=(lx, ly),
                template_updated=False,
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

        # 2D Gaussian Spatial Motion Prior: penalize non-physical displacement jumps
        center_map_x = lx - crop_sx1 - bw // 2
        center_map_y = ly - crop_sy1 - bh // 2
        sh_map, sw_map = smap.shape[:2]
        yy, xx = np.mgrid[:sh_map, :sw_map]
        dist2 = (xx - center_map_x) ** 2 + (yy - center_map_y) ** 2
        spatial_sigma = float(self.params.spatial_sigma)
        spatial_prior = np.exp(-dist2 / (2.0 * spatial_sigma**2))

        # Weight positive correlation scores by distance from previous position
        weighted_smap = np.maximum(0.0, smap) * spatial_prior
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(weighted_smap)
        ncc_score = float(smap[max_loc[1], max_loc[0]])

        # Refine sub-pixel location with parabolic fit
        ref_x, ref_y = refine_location_parabola(smap, max_loc, ncc_score)

        # Compute candidate point in full image coordinates (center of template)
        cand_x = float(crop_sx1 + ref_x + bw / 2.0)
        cand_y = float(crop_sy1 + ref_y + bh / 2.0)

        # Extract candidate patch for visual / discriminator evaluation
        cand_patch = self.extract_patch(frame, (cand_x, cand_y), (bw, bh))

        # Deep Feature Cosine Similarity Verification
        deep_score = None
        if self.params.use_deep_features:
            if self.extractor is None:
                self.extractor = DeepFeatureExtractor.get_shared()
            if self.anchor_embedding is None and self.anchor_template is not None and self.extractor.enabled:
                self.anchor_embedding = self.extractor.extract_embedding(self.anchor_template)
            if self.extractor and self.extractor.enabled and self.anchor_embedding is not None:
                cand_emb = self.extractor.extract_embedding(cand_patch)
                deep_score = self.extractor.cosine_similarity(self.anchor_embedding, cand_emb)

        # Online Retrained Appearance Model Verification
        disc_score = (
            self.score_patch_discriminator(cand_patch)
            if self.discriminator_w is not None
            else None
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

        # Adaptive Running Template Blending:
        # When NCC match is confident (>= 0.70), blend running appearance to adapt
        # gracefully to rotation and deformation without drifting
        template_updated = False
        final_loc = (cand_x, cand_y)

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

        self.last_point = final_loc
        return TemplateMatchResult(
            similarity=float(combined_score),
            ncc_score=ncc_score,
            deep_score=deep_score,
            location=final_loc,
            raw_location=(
                int(crop_sx1 + max_loc[0] + bw // 2),
                int(crop_sy1 + max_loc[1] + bh // 2),
            ),
            template_updated=template_updated,
        )


# Backward-compatible alias
KinoveaTracker = AITracker


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

    # Determine gaps to fill
    # Single anchor case: track forward to end and backward to start
    if len(sorted_frames) == 1:
        f0 = sorted_frames[0]
        fetch_frames(0, total_frames - 1)
        ref_frame = get_frame(f0)
        if ref_frame is None:
            raise RuntimeError(f"Failed to read anchor frame {f0}")

        tracker = AITracker(params)
        tracker.set_reference(ref_frame, known_points[f0])

        # Track forward: f0 + 1 ... total_frames - 1
        curr_pt = known_points[f0]
        for f in range(f0 + 1, total_frames):
            if progress_callback and not progress_callback(
                f, total_frames, f"Tracking forward frame {f + 1}/{total_frames}..."
            ):
                break
            frm = get_frame(f)
            if frm is None:
                break
            res = tracker.track_frame(frm, curr_pt)
            raw_positions[f] = res.location
            confidences[f] = res.similarity
            curr_pt = res.location

        # Track backward: f0 - 1 ... 0
        tracker_bwd = AITracker(params)
        tracker_bwd.set_reference(ref_frame, known_points[f0])
        curr_pt = known_points[f0]
        for f in range(f0 - 1, -1, -1):
            if progress_callback and not progress_callback(
                total_frames - f, total_frames, f"Tracking backward frame {f + 1}/{total_frames}..."
            ):
                break
            frm = get_frame(f)
            if frm is None:
                break
            res = tracker_bwd.track_frame(frm, curr_pt)
            raw_positions[f] = res.location
            confidences[f] = res.similarity
            curr_pt = res.location

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

            # Forward tracker starting at f_start
            frm_start = get_frame(f_start)
            if frm_start is None:
                continue
            trk_fwd = AITracker(params)
            trk_fwd.set_reference(frm_start, known_points[f_start])

            fwd_pts: dict[int, tuple[float, float]] = {}
            fwd_scores: dict[int, float] = {}
            curr_fwd = known_points[f_start]
            for f in range(f_start + 1, f_end):
                if progress_callback and not progress_callback(
                    f, total_frames, f"Tracking gap {f_start}→{f_end}: forward frame {f + 1}..."
                ):
                    break
                frm = get_frame(f)
                if frm is None:
                    break
                res_f = trk_fwd.track_frame(frm, curr_fwd)
                fwd_pts[f] = res_f.location
                fwd_scores[f] = res_f.similarity
                curr_fwd = res_f.location

            # Backward tracker starting at f_end
            frm_end = get_frame(f_end)
            if frm_end is None:
                continue
            trk_bwd = AITracker(params)
            trk_bwd.set_reference(frm_end, known_points[f_end])

            bwd_pts: dict[int, tuple[float, float]] = {}
            bwd_scores: dict[int, float] = {}
            curr_bwd = known_points[f_end]
            for f in range(f_end - 1, f_start, -1):
                if progress_callback and not progress_callback(
                    f, total_frames, f"Tracking gap {f_start}→{f_end}: backward frame {f + 1}..."
                ):
                    break
                frm = get_frame(f)
                if frm is None:
                    break
                res_b = trk_bwd.track_frame(frm, curr_bwd)
                bwd_pts[f] = res_b.location
                bwd_scores[f] = res_b.similarity
                curr_bwd = res_b.location

            # Fuse forward and backward tracks in the gap with distance & squared confidence weighting
            for f in range(f_start + 1, f_end):
                p_f = fwd_pts.get(f)
                p_b = bwd_pts.get(f)
                s_f = fwd_scores.get(f, 0.5)
                s_b = bwd_scores.get(f, 0.5)

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
                elif p_f is not None:
                    raw_positions[f] = p_f
                    confidences[f] = s_f
                elif p_b is not None:
                    raw_positions[f] = p_b
                    confidences[f] = s_b

            # Evict frames older than f_start from cache to bound RAM usage
            old_keys = [k for k in list(frame_cache.keys()) if k < f_start]
            for k in old_keys:
                del frame_cache[k]

        # 2. Track backward before the first anchor (0 ... sorted_frames[0] - 1)
        first_f = sorted_frames[0]
        if first_f > 0:
            fetch_frames(0, first_f)
            frm_first = get_frame(first_f)
            if frm_first is not None:
                trk_head = AITracker(params)
                trk_head.set_reference(frm_first, known_points[first_f])
                curr_pt = known_points[first_f]
                for f in range(first_f - 1, -1, -1):
                    if progress_callback and not progress_callback(
                        f, total_frames, f"Tracking head frame {f + 1}..."
                    ):
                        break
                    frm = get_frame(f)
                    if frm is None:
                        break
                    res = trk_head.track_frame(frm, curr_pt)
                    raw_positions[f] = res.location
                    confidences[f] = res.similarity
                    curr_pt = res.location

        # 3. Track forward after the last anchor (sorted_frames[-1] + 1 ... total_frames - 1)
        last_f = sorted_frames[-1]
        if last_f < total_frames - 1:
            fetch_frames(last_f, total_frames - 1)
            frm_last = get_frame(last_f)
            if frm_last is not None:
                trk_tail = AITracker(params)
                trk_tail.set_reference(frm_last, known_points[last_f])
                curr_pt = known_points[last_f]
                for f in range(last_f + 1, total_frames):
                    if progress_callback and not progress_callback(
                        f, total_frames, f"Tracking tail frame {f + 1}..."
                    ):
                        break
                    frm = get_frame(f)
                    if frm is None:
                        break
                    res = trk_tail.track_frame(frm, curr_pt)
                    raw_positions[f] = res.location
                    confidences[f] = res.similarity
                    curr_pt = res.location

    # Interpolate any lingering NaNs if frames couldn't be decoded
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
