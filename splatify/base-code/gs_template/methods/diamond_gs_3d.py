"""
DiamondGS3D — Diamond-aware Gaussian Splatting for Nerfstudio
=============================================================

A practical diamond-specialized method built on top of GSTemplateModel.

Core idea
---------
Vanilla splatfacto struggles on diamonds because SH colors cannot represent:
1) strong Fresnel reflection,
2) total internal reflection,
3) refractive transport through a high-index medium, and
4) sparkle / fire-like high-frequency glints.

This method keeps splatfacto geometry and densification, but replaces the
appearance model in diamond regions with a two-pass, deferred-shading style
approximation:

    pass A: render non-diamond background
    pass B: render diamond attribute buffers and shade them using
            screen-space reflection / refraction queries

The implementation is intentionally lightweight:
- no external ray tracer
- no spectral renderer
- RGB-only dispersion surrogate
- can train with user-provided masks, alpha masks, or fallback full-frame masks

What is learned
---------------
Per Gaussian:
    diamond_logit        : probability the Gaussian belongs to diamond
    roughness_logit      : small roughness for highlight sharpness
    transmission_logit   : how transmissive the Gaussian is
    absorption_log_rgb   : Beer-Lambert style absorption
    eta_offset           : small residual around eta_init
    dispersion_logit     : RGB dispersion spread
    sparkle_feat         : latent feature used by a residual sparkle MLP

What is supervised
------------------
- global RGB loss (inherited from splatfacto)
- extra weighted RGB loss on diamond pixels
- stronger loss on highlight pixels
- diamond mask BCE
- optional depth loss (inherited / from datamanager)
- eta prior, dispersion ordering prior, facet flattening prior
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import opacity_entropy_loss
from gs_template.rendering import prepare_camera, gather_base_params, rasterize_gaussians


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class DiamondGS3DModelConfig(GSTemplateModelConfig):
    """Diamond-aware Gaussian Splatting configuration."""

    _target: Type = field(default_factory=lambda: DiamondGS3DModel)

    # Use degree-0 by default so the learned appearance is driven mostly by the
    # explicit diamond branch instead of SH trying to memorize glints.
    sh_degree: int = 0
    absgrad: bool = True
    output_depth_during_training: bool = True

    # Diamond transport priors
    eta_init: float = 2.42
    """Base refractive index for diamond."""

    eta_offset_max: float = 0.08
    """Small learnable residual around eta_init."""

    dispersion_max: float = 0.05
    """Maximum RGB spread surrogate for dispersion."""

    reflect_step_scale: float = 0.08
    """Screen-space offset scale for reflection query."""

    refract_step_scale: float = 0.12
    """Screen-space offset scale for refraction query."""

    # Appearance branch sizes
    sparkle_feat_dim: int = 8
    sparkle_hidden_dim: int = 32

    # Loss weights
    diamond_rgb_weight: float = 3.0
    highlight_weight: float = 6.0
    mask_weight: float = 0.5
    eta_prior_weight: float = 0.02
    dispersion_order_weight: float = 0.02
    facet_flatten_weight: float = 0.01
    opacity_entropy_weight: float = 0.005
    roughness_prior_weight: float = 0.001
    sparkle_weight: float = 0.1

    # Optional training convenience
    freeze_diamond_branch_until: int = 1000
    """Keep transport residuals weak during geometry bootstrap."""

    diamond_fg_thresh: float = 0.5
    """Threshold used when splitting background vs diamond Gaussians."""


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def _quat_to_rotmat(quat: Tensor) -> Tensor:
    """Convert normalized quaternions (N,4) to rotation matrices (N,3,3)."""
    q = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(dim=-1)

    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    return torch.stack(
        [
            torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )


def _gaussian_normals_from_quat_scale(quats: Tensor, scales_log: Tensor) -> Tensor:
    """Use the shortest ellipsoid axis as an approximate facet normal."""
    scales = torch.exp(scales_log)
    rot = _quat_to_rotmat(quats)  # (N,3,3)
    min_axis = torch.argmin(scales, dim=-1)  # (N,)
    gather_idx = min_axis[:, None, None].expand(-1, 3, 1)
    normals = torch.gather(rot, 2, gather_idx).squeeze(-1)
    normals = F.normalize(normals, dim=-1)
    return normals


def _sigmoid_range(x: Tensor, lo: float, hi: float) -> Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


def _safe_normalize(x: Tensor, dim: int = -1) -> Tensor:
    return F.normalize(x, dim=dim, eps=1e-8)


def _reflect(v: Tensor, n: Tensor) -> Tensor:
    return _safe_normalize(v - 2.0 * (v * n).sum(dim=-1, keepdim=True) * n, dim=-1)


def _refract(v: Tensor, n: Tensor, eta_ratio: Tensor) -> Tuple[Tensor, Tensor]:
    """Approximate Snell refraction. Returns (dir, tir_mask)."""
    cosi = (-v * n).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    k = 1.0 - (eta_ratio ** 2) * (1.0 - cosi ** 2)
    tir = k < 0.0
    refr = eta_ratio * v + (eta_ratio * cosi - torch.sqrt(torch.clamp(k, min=0.0))) * n
    refr = _safe_normalize(refr, dim=-1)
    return refr, tir


def _schlick_fresnel(cos_theta: Tensor, eta: Tensor) -> Tensor:
    """Schlick approximation with eta-dependent F0."""
    eta = torch.clamp(eta, min=1.01)
    f0 = ((eta - 1.0) / (eta + 1.0)) ** 2
    return f0 + (1.0 - f0) * torch.pow(1.0 - cos_theta.clamp(0.0, 1.0), 5.0)


def _make_uv_grid(h: int, w: int, device: torch.device) -> Tensor:
    ys = torch.linspace(-1.0, 1.0, h, device=device)
    xs = torch.linspace(-1.0, 1.0, w, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1)  # (H,W,2)


def _sample_screen(rgb: Tensor, uv: Tensor) -> Tensor:
    """
    rgb : (H,W,3)
    uv  : (H,W,2) in [-1,1]
    """
    img = rgb.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    grid = uv.unsqueeze(0)  # (1,H,W,2)
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.squeeze(0).permute(1, 2, 0)  # (H,W,3)


def _camera_ray_dirs(K: Tensor, h: int, w: int, device: torch.device) -> Tensor:
    """
    Build normalized camera-space ray directions.
    K is (1,3,3).
    """
    fx = K[0, 0, 0]
    fy = K[0, 1, 1]
    cx = K[0, 0, 2]
    cy = K[0, 1, 2]

    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    x = (gx - cx) / (fx + 1e-8)
    y = (gy - cy) / (fy + 1e-8)
    z = torch.ones_like(x)
    rays = torch.stack([x, y, z], dim=-1)
    return _safe_normalize(rays, dim=-1)


def _render_raw(
    model,
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    cam: Dict,
    *,
    sh_degree: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> Tuple[Tensor, Tensor, Dict]:
    return rasterize_gaussians(
        model,
        means,
        quats,
        scales,
        opacities,
        colors,
        cam["viewmat"],
        cam["K"],
        cam["W"],
        cam["H"],
        sh_degree=sh_degree,
        render_mode=render_mode,
    )


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class DiamondGS3DModel(GSTemplateModel):
    """Diamond-aware Gaussian Splatting model."""

    config: DiamondGS3DModelConfig

    def populate_modules(self):
        super().populate_modules()

        n = self.num_points
        device = self.means.device

        self.gauss_params["diamond_logit"] = Parameter(torch.full((n, 1), -4.0, device=device))
        self.gauss_params["roughness_logit"] = Parameter(torch.full((n, 1), -3.5, device=device))
        self.gauss_params["transmission_logit"] = Parameter(torch.zeros((n, 1), device=device))
        self.gauss_params["absorption_log_rgb"] = Parameter(torch.full((n, 3), -4.0, device=device))
        self.gauss_params["eta_offset"] = Parameter(torch.zeros((n, 1), device=device))
        self.gauss_params["dispersion_logit"] = Parameter(torch.full((n, 1), -2.5, device=device))
        self.gauss_params["sparkle_feat"] = Parameter(
            torch.zeros((n, self.config.sparkle_feat_dim), device=device)
        )

        self.sparkle_mlp = nn.Sequential(
            nn.Linear(self.config.sparkle_feat_dim + 2, self.config.sparkle_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.sparkle_hidden_dim, 3),
        )

    # -------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        cam, crop_ids = prepare_camera(self, camera)
        if cam is None:
            return crop_ids

        means, quats, scales, opacities, colors = gather_base_params(self, crop_ids)

        if crop_ids is None:
            diamond_logit = self.gauss_params["diamond_logit"]
            roughness_logit = self.gauss_params["roughness_logit"]
            transmission_logit = self.gauss_params["transmission_logit"]
            absorption_log_rgb = self.gauss_params["absorption_log_rgb"]
            eta_offset = self.gauss_params["eta_offset"]
            dispersion_logit = self.gauss_params["dispersion_logit"]
            sparkle_feat = self.gauss_params["sparkle_feat"]
        else:
            diamond_logit = self.gauss_params["diamond_logit"][crop_ids]
            roughness_logit = self.gauss_params["roughness_logit"][crop_ids]
            transmission_logit = self.gauss_params["transmission_logit"][crop_ids]
            absorption_log_rgb = self.gauss_params["absorption_log_rgb"][crop_ids]
            eta_offset = self.gauss_params["eta_offset"][crop_ids]
            dispersion_logit = self.gauss_params["dispersion_logit"][crop_ids]
            sparkle_feat = self.gauss_params["sparkle_feat"][crop_ids]

        # ---- Full base render of ALL gaussians ----
        # Provides: (a) correct self.info for densification strategy,
        #           (b) background RGB for diamond screen-space sampling.
        base_render, base_alpha, base_info = _render_raw(
            self, means, quats, scales, opacities, colors, cam,
            render_mode="RGB+ED",
        )

        # Store info for densification (step_post_backward uses self.info)
        self.info = base_info
        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state,
                self.step, self.info,
            )
        if self.training and base_info["means2d"].requires_grad:
            base_info["means2d"].retain_grad()
        self.xys = base_info["means2d"]
        self.radii = base_info["radii"]

        # Composite base render
        bg_color = self._get_background_color()
        base_rgb = torch.clamp(
            base_render[:, ..., :3] + (1 - base_alpha) * bg_color, 0.0, 1.0
        )
        base_depth = torch.where(
            base_alpha > 0,
            base_render[:, ..., -1:],
            base_render[:, ..., -1:].detach().max(),
        )

        diamond_prob = torch.sigmoid(diamond_logit).squeeze(-1)
        diamond_mask = diamond_prob > self.config.diamond_fg_thresh

        # ---- No diamond gaussians: return base render directly ----
        if not diamond_mask.any():
            return {
                "rgb": base_rgb.squeeze(0),
                "depth": base_depth.squeeze(0),
                "accumulation": base_alpha.squeeze(0),
                "background": bg_color,
                "diamond_mask_pred": torch.zeros_like(base_alpha.squeeze(0)),
                "highlight_strength": torch.zeros_like(base_alpha.squeeze(0)),
                "reflection_rgb": torch.zeros_like(base_rgb.squeeze(0)),
                "refraction_rgb": torch.zeros_like(base_rgb.squeeze(0)),
            }

        # ---- Diamond attribute buffer render (diamond gaussians only) ----
        d_means = means[diamond_mask]
        d_quats = quats[diamond_mask]
        d_scales = scales[diamond_mask]
        d_opacities = opacities[diamond_mask]
        d_colors = colors[diamond_mask]

        normals_world = _gaussian_normals_from_quat_scale(d_quats, d_scales)
        view_rot = cam["viewmat"][0, :3, :3]
        normals_cam = torch.matmul(normals_world, view_rot.T)
        normals_cam = _safe_normalize(normals_cam, dim=-1)

        roughness = _sigmoid_range(roughness_logit[diamond_mask], 0.005, 0.2)
        transmission = torch.sigmoid(transmission_logit[diamond_mask])
        absorption = torch.exp(absorption_log_rgb[diamond_mask])  # positive
        eta = self.config.eta_init + self.config.eta_offset_max * torch.tanh(eta_offset[diamond_mask])
        dispersion = self.config.dispersion_max * torch.sigmoid(dispersion_logit[diamond_mask])

        # Approximate thickness from shortest-axis scale
        thickness = torch.exp(d_scales).min(dim=-1, keepdim=True).values

        # Fused attribute: color(3) + normal(3) + roughness(1) + transmission(1)
        # + absorption(3) + eta(1) + dispersion(1) + diamond_prob(1) + thickness(1)
        # + sparkle_feat(K)
        attr = torch.cat(
            [
                d_colors if d_colors.dim() == 2 else torch.sigmoid(d_colors[:, 0, :]),
                normals_cam,
                roughness,
                transmission,
                absorption,
                eta,
                dispersion,
                torch.sigmoid(diamond_logit[diamond_mask]),
                thickness,
                sparkle_feat[diamond_mask],
            ],
            dim=-1,
        )

        diamond_render, diamond_alpha, _diamond_info = _render_raw(
            self,
            d_means,
            d_quats,
            d_scales,
            d_opacities,
            attr,
            cam,
            sh_degree=None,
            render_mode="RGB",
        )

        # ---- Screen-space diamond shading ----
        pix = diamond_render.squeeze(0)
        base_rgb_pix = pix[..., 0:3]
        normal_cam = _safe_normalize(pix[..., 3:6], dim=-1)
        roughness_pix = pix[..., 6:7].clamp(0.005, 0.25)
        transmission_pix = pix[..., 7:8].clamp(0.0, 1.0)
        absorption_pix = pix[..., 8:11].clamp(1e-4, 10.0)
        eta_pix = pix[..., 11:12].clamp(1.5, 3.0)
        dispersion_pix = pix[..., 12:13].clamp(0.0, self.config.dispersion_max)
        diamond_prob_pix = pix[..., 13:14].clamp(0.0, 1.0)
        thickness_pix = pix[..., 14:15].clamp(1e-4, 1e3)
        sparkle_feat_pix = pix[..., 15:15 + self.config.sparkle_feat_dim]

        H, W = cam["H"], cam["W"]
        rays_cam = _camera_ray_dirs(cam["K"], H, W, means.device)
        view_dir = _safe_normalize(-rays_cam, dim=-1)
        cos_theta = (normal_cam * view_dir).sum(dim=-1, keepdim=True).abs()
        fresnel = _schlick_fresnel(cos_theta, eta_pix)

        # Reflection query
        reflect_dir = _reflect(view_dir, normal_cam)
        uv = _make_uv_grid(H, W, means.device)
        reflect_offset = self.config.reflect_step_scale * reflect_dir[..., :2] / (
            reflect_dir[..., 2:3].abs() + 1e-3
        )
        uv_reflect = (uv + reflect_offset).clamp(-1.0, 1.0)

        bg_rgb_img = base_rgb.squeeze(0)  # full base render as background
        reflection_rgb = _sample_screen(bg_rgb_img, uv_reflect)

        # Refraction query with RGB dispersion surrogate
        eta_r = 1.0 / (eta_pix + 0.5 * dispersion_pix)
        eta_g = 1.0 / eta_pix
        eta_b = 1.0 / (eta_pix - 0.5 * dispersion_pix + 1e-3)

        refr_r, tir_r = _refract(view_dir, normal_cam, eta_r)
        refr_g, tir_g = _refract(view_dir, normal_cam, eta_g)
        refr_b, tir_b = _refract(view_dir, normal_cam, eta_b)

        def _uv_from_dir(d: Tensor) -> Tensor:
            off = self.config.refract_step_scale * d[..., :2] / (d[..., 2:3].abs() + 1e-3)
            return (uv + off).clamp(-1.0, 1.0)

        samp_r = _sample_screen(bg_rgb_img, _uv_from_dir(refr_r))[..., 0:1]
        samp_g = _sample_screen(bg_rgb_img, _uv_from_dir(refr_g))[..., 1:2]
        samp_b = _sample_screen(bg_rgb_img, _uv_from_dir(refr_b))[..., 2:3]
        refraction_rgb = torch.cat([samp_r, samp_g, samp_b], dim=-1)

        tir_mask = (tir_r | tir_g | tir_b).float()
        refraction_rgb = torch.exp(-absorption_pix * thickness_pix) * refraction_rgb
        refraction_rgb = (1.0 - tir_mask) * refraction_rgb + tir_mask * reflection_rgb

        # Sparkle residual
        sparkle_in = torch.cat(
            [
                sparkle_feat_pix,
                cos_theta,
                (1.0 - roughness_pix),
            ],
            dim=-1,
        )
        sparkle_rgb = torch.tanh(self.sparkle_mlp(sparkle_in))
        sparkle_rgb = self.config.sparkle_weight * sparkle_rgb

        # Blend
        if self.step < self.config.freeze_diamond_branch_until:
            transport_mix = 0.1
        else:
            transport_mix = 1.0

        diamond_rgb = (
            base_rgb_pix * (1.0 - transmission_pix)
            + transport_mix * fresnel * reflection_rgb
            + transport_mix * (1.0 - fresnel) * transmission_pix * refraction_rgb
            + transport_mix * sparkle_rgb
        )
        diamond_rgb = torch.clamp(diamond_rgb, 0.0, 1.0)

        diamond_alpha_img = diamond_alpha.squeeze(0)
        final_rgb = torch.clamp(
            diamond_rgb * diamond_alpha_img + (1.0 - diamond_alpha_img) * bg_rgb_img,
            0.0,
            1.0,
        )

        # Highlight proxy for weighting and visualization
        highlight_strength = torch.clamp(fresnel * (1.0 - roughness_pix), 0.0, 1.0)

        return {
            "rgb": final_rgb,
            "depth": base_depth.squeeze(0),
            "accumulation": torch.clamp(base_alpha.squeeze(0) + diamond_alpha_img, 0.0, 1.0),
            "background": bg_color,
            "diamond_mask_pred": diamond_prob_pix * diamond_alpha_img,
            "highlight_strength": highlight_strength,
            "reflection_rgb": reflection_rgb,
            "refraction_rgb": refraction_rgb,
        }

    # -------------------------------------------------------------
    # Losses
    # -------------------------------------------------------------
    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        losses: Dict[str, Tensor] = {}

        pred = outputs["rgb"]
        target_hw = (int(pred.shape[0]), int(pred.shape[1]))

        def resize_hwc(x: Tensor, hw, mode="bilinear") -> Tensor:
            """Resize HWC tensor to target HW."""
            if x.shape[0] == hw[0] and x.shape[1] == hw[1]:
                return x
            x = x.permute(2, 0, 1).unsqueeze(0).float()
            if mode in ("bilinear", "bicubic"):
                x = F.interpolate(x, size=hw, mode=mode, align_corners=False)
            else:
                x = F.interpolate(x, size=hw, mode=mode)
            return x.squeeze(0).permute(1, 2, 0)

        # ---------------------------
        # Ground-truth image handling
        # ---------------------------
        gt = batch["image"].to(pred.device).float()

        if gt.dim() == 2:
            gt = gt.unsqueeze(-1)

        # Resize GT first to match prediction resolution
        gt = resize_hwc(gt, target_hw, mode="bilinear")

        # Normalize if image came in as 0..255
        if gt.max() > 1.5:
            gt = gt / 255.0

        # Handle RGBA by compositing onto the training background
        if gt.shape[-1] == 4:
            alpha = gt[..., 3:4]
            gt_rgb = gt[..., :3]

            bg = self._get_background_color().to(pred.device).float()
            if bg.dim() == 1:
                bg = bg.view(1, 1, 3).expand(target_hw[0], target_hw[1], 3)
            elif bg.dim() == 3 and bg.shape[:2] != target_hw:
                bg = resize_hwc(bg, target_hw, mode="bilinear")

            gt = gt_rgb * alpha + bg * (1.0 - alpha)
        elif gt.shape[-1] > 3:
            gt = gt[..., :3]

        # ---------------------------
        # Diamond-region RGB loss
        # ---------------------------
        if "diamond_mask" in batch:
            mask = batch["diamond_mask"].to(pred.device).float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            elif mask.shape[-1] > 1:
                mask = mask[..., :1]

            mask = resize_hwc(mask, target_hw, mode="nearest")
            mask = mask.clamp(0.0, 1.0)

            losses["diamond_rgb_loss"] = self.config.diamond_rgb_weight * (
                (pred - gt).abs() * mask
            ).mean()

            pred_mask = outputs["diamond_mask_pred"]
            if pred_mask.shape[:2] != target_hw:
                pred_mask = resize_hwc(pred_mask, target_hw, mode="bilinear")
            pred_mask = pred_mask.clamp(1e-5, 1 - 1e-5)

            losses["diamond_mask_bce"] = self.config.mask_weight * F.binary_cross_entropy(
                pred_mask,
                mask,
            )

        # ---------------------------
        # Highlight-region RGB loss
        # ---------------------------
        if "highlight_mask" in batch:
            hmask = batch["highlight_mask"].to(pred.device).float()
            if hmask.dim() == 2:
                hmask = hmask.unsqueeze(-1)
            elif hmask.shape[-1] > 1:
                hmask = hmask[..., :1]

            hmask = resize_hwc(hmask, target_hw, mode="nearest")
            hmask = hmask.clamp(0.0, 1.0)

            losses["highlight_rgb_loss"] = self.config.highlight_weight * (
                (pred - gt).abs() * hmask
            ).mean()

        # ---------------------------
        # Existing physics priors
        # ---------------------------
        # if self.config.eta_prior_weight > 0:
        #     eta = self.get_eta()
        #     losses["eta_prior"] = self.config.eta_prior_weight * ((eta - 2.42) ** 2).mean()

        # if self.config.dispersion_order_weight > 0:
        #     eta_rgb = self.get_eta_rgb()
        #     order_penalty = F.relu(eta_rgb[..., 0] - eta_rgb[..., 1]) + F.relu(
        #         eta_rgb[..., 1] - eta_rgb[..., 2]
        #     )
        #     losses["dispersion_order"] = (
        #         self.config.dispersion_order_weight * order_penalty.mean()
        #     )

        if self.config.facet_flatten_weight > 0:
            diamond_prob = torch.sigmoid(self.gauss_params["diamond_logit"])
            scales = torch.exp(self.scales)
            sorted_scales, _ = torch.sort(scales, dim=-1)
            flatness = sorted_scales[:, 0] / (sorted_scales[:, 2] + 1e-8)
            losses["facet_flatten"] = (
                self.config.facet_flatten_weight * (diamond_prob.squeeze(-1) * flatness).mean()
            )

        return losses
    # -------------------------------------------------------------
    # Densification / optimizer wiring / checkpointing
    # -------------------------------------------------------------

    def split_gaussians(self, split_mask, samps):
        out = super().split_gaussians(split_mask, samps)
        for name in [
            "diamond_logit",
            "roughness_logit",
            "transmission_logit",
            "absorption_log_rgb",
            "eta_offset",
            "dispersion_logit",
            "sparkle_feat",
        ]:
            out[name] = self.gauss_params[name][split_mask].repeat(samps, 1)
        return out

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {name: [self.gauss_params[name]] for name in self.gauss_params.keys()}

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = super().get_param_groups()
        groups["sparkle_mlp"] = list(self.sparkle_mlp.parameters())
        return groups

    @torch.no_grad()
    def get_outputs_for_camera(
        self, camera: Cameras, obb_box: Optional[OrientedBox] = None
    ) -> Dict[str, Tensor]:
        outs = super().get_outputs_for_camera(camera, obb_box)
        # Keep base viewer outputs, but surface useful debugging channels if available.
        if "diamond_mask_pred" in outs:
            outs["diamond_overlay"] = outs["diamond_mask_pred"].expand_as(outs["rgb"])
        return outs

    def load_state_dict(self, state_dict, **kwargs):
        self.step = 30000
        if "means" in state_dict:
            for p in self.gauss_params.keys():
                state_dict[f"gauss_params.{p}"] = state_dict[p]

        newp = state_dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            new_shape = (newp,) + param.shape[1:]
            self.gauss_params[name] = Parameter(
                torch.zeros(new_shape, device=self.device)
            )
        super().load_state_dict(state_dict, **kwargs)

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["diamond_fraction"] = torch.sigmoid(self.gauss_params["diamond_logit"]).mean()
        metrics["avg_eta"] = (
            self.config.eta_init + self.config.eta_offset_max * torch.tanh(self.gauss_params["eta_offset"])
        ).mean()
        metrics["avg_roughness"] = _sigmoid_range(self.gauss_params["roughness_logit"], 0.005, 0.2).mean()
        return metrics
