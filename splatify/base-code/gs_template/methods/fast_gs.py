"""
FastGS — Training 3D Gaussian Splatting in 100 Seconds
========================================================
Paper: "FastGS: Training 3D Gaussian Splatting in 100 Seconds"
       https://arxiv.org/abs/2511.04283
Complexity: ★★★☆☆ (custom densification filter + pruning + optimizer scheduling)

Core idea: Keep Gaussian count low via multi-view consistent densification
(VCD) and pruning (VCP). Only densify/prune Gaussians where multiple views
agree there is a reconstruction problem.

Key components:
    1. View-Consistent Densification (VCD): Accumulate per-Gaussian importance
       scores from sampled training views. At densification time, zero out
       gradient accumulation for low-importance Gaussians, preventing
       unnecessary densification and keeping total count small.
    2. View-Consistent Pruning (VCP): After densification ends (15K), prune
       Gaussians with high multi-view pruning scores or low opacity.
    3. AbsGrad: Use absolute gradients for split decisions (from AbsGS).
    4. Opacity cap: Clamp opacity to 0.8 after each densification round.

Train:
    ns-train fast-gs --data <path>           # Base: densify every 500 steps
    ns-train fast-gs-big --data <path>       # Big: densify every 100 steps
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.rich_utils import CONSOLE

from gs_template.model import GSTemplateModel, GSTemplateModelConfig


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class FastGSModelConfig(GSTemplateModelConfig):
    """FastGS configuration.

    Two variants:
        fast-gs:     refine_every=500  (fewer Gaussians, fastest training)
        fast-gs-big: refine_every=100  (more Gaussians, higher PSNR)
    """

    _target: Type = field(default_factory=lambda: FastGSModel)

    # ── AbsGrad for split decisions (AbsGS backbone) ──────────────
    absgrad: bool = True

    # ── VCD: View-Consistent Densification ────────────────────────
    vcd_importance_thresh: float = 0.3
    """Minimum average importance for densification. Range [0, 1].
    Fraction of sampled views where the Gaussian center is in a
    high-error region."""

    vcd_error_thresh: float = 0.1
    """Threshold on min-max normalized per-pixel error map."""

    vcd_accumulate_every: int = 50
    """Accumulate scores every N steps (not every step).
    With refine_every=500, this gives ~10 views per cycle."""

    # ── VCP: View-Consistent Pruning ──────────────────────────────
    vcp_score_thresh: float = 0.9
    """Normalized pruning score threshold for late-stage pruning."""

    vcp_opacity_thresh: float = 0.1
    """Opacity threshold for late-stage pruning."""

    vcp_late_prune_every: int = 3000
    """Pruning interval after densification ends (15K-30K)."""

    # ── Opacity cap ───────────────────────────────────────────────
    opacity_cap: float = 0.8
    """Cap opacity after each densification round."""


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════


class FastGSModel(GSTemplateModel):
    """FastGS: fast training via multi-view consistent densification/pruning."""

    config: FastGSModelConfig

    def populate_modules(self):
        super().populate_modules()
        self._mv_importance: Optional[Tensor] = None
        self._mv_pruning: Optional[Tensor] = None
        self._mv_count: int = 0
        self._last_camera: Optional[Cameras] = None
        self._opacity_cap_logit: float = torch.logit(
            torch.tensor(self.config.opacity_cap)
        ).item()

    def get_outputs(self, camera: Cameras):
        if self.training:
            self._last_camera = camera
        return super().get_outputs(camera)

    # ── Score accumulation (runs every vcd_accumulate_every steps) ──

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        if not self.training or self._last_camera is None:
            return {}

        step = self.step
        # Only accumulate periodically, and only during active phases
        if step < self.strategy.refine_start_iter:
            return {}
        if step % self.config.vcd_accumulate_every != 0:
            return {}

        self._accumulate_view_scores(outputs, batch)
        return {}

    @torch.no_grad()
    def _accumulate_view_scores(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]
    ):
        """Accumulate importance/pruning scores from the current view.

        Designed to avoid CPU-GPU synchronization:
        - All operations stay on GPU tensors
        - No .item() calls
        - No early returns based on GPU values
        - grid_sample with padding_mode='zeros' handles out-of-bounds
        """
        N = self.num_points
        camera = self._last_camera

        # Reset accumulators if Gaussian count changed
        if self._mv_importance is None or self._mv_importance.shape[0] != N:
            self._mv_importance = torch.zeros(N, device=self.device)
            self._mv_pruning = torch.zeros(N, device=self.device)
            self._mv_count = 0

        rendered = outputs["rgb"].detach()  # (H, W, 3)
        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        ).detach()

        H, W = rendered.shape[:2]
        gt_img = gt_img[:H, :W]

        # Per-pixel L1 error, min-max normalized
        error = (rendered - gt_img).abs().mean(dim=-1)  # (H, W)
        emin = error.min()
        emax = error.max()
        denom = (emax - emin).clamp(min=1e-8)
        error_norm = (error - emin) / denom

        # High-error binary mask
        high_error = (error_norm > self.config.vcd_error_thresh).float()

        # Project Gaussian centers to 2D
        means = self.means.detach()  # (N, 3)
        c2w = camera.camera_to_worlds.to(self.device)
        if c2w.dim() == 3:
            c2w = c2w.squeeze(0)

        R = c2w[:3, :3]
        t = c2w[:3, 3]

        # World-to-camera: P_cam = (P_world - t) @ R
        points_cam = (means - t.unsqueeze(0)) @ R  # (N, 3)
        z = points_cam[:, 2].clamp(min=1e-4)  # avoid div-by-zero

        # Camera intrinsics (already downscaled by get_outputs)
        fx = camera.fx.squeeze().float()
        fy = camera.fy.squeeze().float()
        cx = camera.cx.squeeze().float()
        cy = camera.cy.squeeze().float()

        # Project to normalized [-1, 1] coordinates for grid_sample
        u = (points_cam[:, 0] / z * fx + cx)
        v = (points_cam[:, 1] / z * fy + cy)
        u_norm = 2.0 * u / W - 1.0
        v_norm = 2.0 * v / H - 1.0

        grid = torch.stack([u_norm, v_norm], dim=-1).view(1, 1, -1, 2)

        # grid_sample with padding_mode='zeros' returns 0 for out-of-bounds
        sampled = F.grid_sample(
            high_error.view(1, 1, H, W),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).view(-1)  # (N,)

        # Mask out points behind camera (z was clamped, so check original)
        behind = points_cam[:, 2] < 0.01
        sampled = sampled.masked_fill(behind, 0.0)

        self._mv_importance += sampled

        # Photometric loss as GPU tensor (no .item() sync)
        photo_loss = (rendered - gt_img).abs().mean()
        self._mv_pruning += sampled * photo_loss

        self._mv_count += 1

    # ── step_post_backward (VCD filter + VCP prune) ────────────────

    def step_post_backward(self, step):
        assert step == self.step

        is_refine_step = (
            step > self.strategy.refine_start_iter
            and step < self.strategy.refine_stop_iter
            and step % self.strategy.refine_every == 0
            and step % self.strategy.reset_every
            >= self.strategy.pause_refine_after_reset
        )

        # ── VCD: zero grad2d for low-importance Gaussians ─────────
        if is_refine_step and self._mv_importance is not None and self._mv_count > 0:
            grad2d = self.strategy_state.get("grad2d")
            if grad2d is not None and grad2d.shape[0] == self._mv_importance.shape[0]:
                avg_importance = self._mv_importance / self._mv_count
                low_imp = avg_importance < self.config.vcd_importance_thresh
                grad2d[low_imp] = 0.0

        n_before = self.num_points

        # Parent handles strategy step
        super().step_post_backward(step)

        # Reset accumulators if Gaussian count changed
        if self.num_points != n_before:
            self._mv_importance = None
            self._mv_pruning = None
            self._mv_count = 0

        # ── Cap opacity after densification ───────────────────────
        if is_refine_step:
            self.opacities.data.clamp_(max=self._opacity_cap_logit)

        # ── VCP: late-stage pruning ───────────────────────────────
        if (
            step >= self.strategy.refine_stop_iter
            and step % self.config.vcp_late_prune_every == 0
            and self._mv_pruning is not None
            and self._mv_count > 0
        ):
            self._apply_vcp_pruning()

    # ── VCP pruning ────────────────────────────────────────────────

    @torch.no_grad()
    def _apply_vcp_pruning(self):
        from gsplat.strategy.ops import remove

        # Normalize pruning scores
        pmin = self._mv_pruning.min()
        pmax = self._mv_pruning.max()
        denom = (pmax - pmin).clamp(min=1e-8)
        norm_pruning = (self._mv_pruning - pmin) / denom

        # Prune by low opacity OR high VCP score
        opa = torch.sigmoid(self.opacities.flatten())
        prune_mask = (opa < self.config.vcp_opacity_thresh) | (
            norm_pruning > self.config.vcp_score_thresh
        )

        n_prune = prune_mask.sum().item()
        if n_prune > 0:
            remove(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                mask=prune_mask,
            )
            CONSOLE.print(
                f"  [VCP step {self.step}] Pruned {n_prune:,} | "
                f"Remaining: {self.num_points:,}"
            )

        # Reset accumulators
        self._mv_importance = None
        self._mv_pruning = None
        self._mv_count = 0

    # ── Metrics ────────────────────────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metrics = super().get_metrics_dict(outputs, batch)
        if self._mv_importance is not None and self._mv_count > 0:
            avg_imp = self._mv_importance / self._mv_count
            metrics["vcd_avg_importance"] = avg_imp.mean()
        return metrics
