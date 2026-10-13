"""
DiamondGS3D — Neural Appearance Gaussian Splatting for Gemstone Scenes
========================================================================

Replace SH with a per-Gaussian neural appearance model that can represent
sharp specular highlights, chromatic dispersion, and scintillation.

    color(x, d) = diffuse(x) + tint(x) * specular(embed(x), PE(reflect(d, n(x)), roughness(x)))

Key features over vanilla splatfacto:
    - Reflection-direction PE: encodes the physically correct reflection vector
      (from geometric normal) instead of raw view direction.
    - Roughness-gated PE: smooth surfaces get all frequency bands, rough surfaces
      get only low frequencies.
    - Per-channel specular: the MLP outputs 3 independent specular values,
      naturally learning chromatic dispersion.
    - Softplus specular output: allows highlights brighter than 1 (HDR-like),
      critical for diamond sparkle.
"""

from __future__ import annotations

import math
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
from gs_template.losses import opacity_entropy_loss, scale_flatten_loss
from gs_template.rendering import (
    assemble_outputs,
    crop_custom_param,
    prepare_camera,
    rasterize_gaussians,
)


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════

def _gaussian_normals(quats: Tensor, scales_log: Tensor) -> Tensor:
    """Compute approximate surface normal from Gaussian ellipsoid shortest axis."""
    q = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(dim=-1)

    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)  # (N, 3, 3)

    scales = torch.exp(scales_log)
    min_axis = torch.argmin(scales, dim=-1)  # (N,)
    idx = min_axis[:, None, None].expand(-1, 3, 1)
    normals = torch.gather(R, 2, idx).squeeze(-1)  # (N, 3)
    return F.normalize(normals, dim=-1)


# ═══════════════════════════════════════════════════════════════════
# Positional Encoding
# ═══════════════════════════════════════════════════════════════════

class RoughnessGatedPE(nn.Module):
    """Positional encoding with per-Gaussian roughness gating.

    For smooth surfaces (low roughness), all frequency bands are active.
    For rough surfaces, high-frequency bands are suppressed.
    """

    def __init__(self, num_levels: int = 12, beta: float = 10.0):
        super().__init__()
        self.num_levels = num_levels
        self.beta = beta
        self.out_dim = 3 + 3 * 2 * num_levels

    def forward(self, dirs: Tensor, roughness: Tensor) -> Tensor:
        encoded = [dirs]
        for level in range(self.num_levels):
            freq = 2.0 ** level * math.pi
            threshold = 1.0 - level / self.num_levels
            gate = torch.sigmoid(self.beta * (threshold - roughness))
            encoded.append(torch.sin(freq * dirs) * gate)
            encoded.append(torch.cos(freq * dirs) * gate)
        return torch.cat(encoded, dim=-1)


# ═══════════════════════════════════════════════════════════════════
# Specular Appearance MLP
# ═══════════════════════════════════════════════════════════════════

class SpecularMLP(nn.Module):
    """Maps appearance embedding + encoded reflection direction to specular color.

    Architecture: input → 256 → 256(+skip) → 128 → 3
    Output uses softplus to allow bright highlights > 1.
    """

    def __init__(
        self,
        appearance_dim: int = 48,
        pe_dim: int = 75,
        hidden_dim: int = 256,
    ):
        super().__init__()
        input_dim = appearance_dim + pe_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, 3)

        # Skip connection projection
        self.skip = nn.Linear(input_dim, hidden_dim, bias=False)

        # Initialize output near zero for stable diffuse-only start
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, appearance: Tensor, encoded_refldir: Tensor) -> Tensor:
        x = torch.cat([appearance, encoded_refldir], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h) + self.skip(x))  # residual
        h = F.relu(self.fc3(h))
        return F.softplus(self.fc_out(h))  # positive, can be > 1


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DiamondGS3DModelConfig(GSTemplateModelConfig):
    """DiamondGS3D: Neural appearance Gaussian Splatting for gemstones."""

    _target: Type = field(default_factory=lambda: DiamondGS3DModel)

    sh_degree: int = 0
    ssim_lambda: float = 0.4
    absgrad: bool = True

    # Neural Appearance
    appearance_dim: int = 48
    pe_levels: int = 12
    specular_hidden_dim: int = 256

    # Per-Gaussian Material
    init_roughness: float = 0.3
    init_specular_tint: float = 0.15

    # Loss Weights
    scale_flatten_weight: float = 0.005
    opacity_entropy_weight: float = 0.003
    specular_sparsity_weight: float = 0.0003

    # Training Strategy
    specular_warmup_steps: int = 500
    output_depth_during_training: bool = True


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class DiamondGS3DModel(GSTemplateModel):
    """DiamondGS3D — Neural appearance field for specular/dispersive scenes."""

    config: DiamondGS3DModelConfig

    def populate_modules(self):
        super().populate_modules()

        N = self.num_points

        self.gauss_params["appearance"] = Parameter(
            torch.randn(N, self.config.appearance_dim) * 0.01
        )

        init_logit = math.log(
            self.config.init_roughness / (1 - self.config.init_roughness + 1e-8)
        )
        self.gauss_params["roughness"] = Parameter(
            torch.full((N, 1), init_logit)
        )

        init_spec_logit = math.log(
            self.config.init_specular_tint
            / (1 - self.config.init_specular_tint + 1e-8)
        )
        self.gauss_params["specular_tint"] = Parameter(
            torch.full((N, 3), init_spec_logit)
        )

        self.pe_encoder = RoughnessGatedPE(num_levels=self.config.pe_levels)

        self.specular_mlp = SpecularMLP(
            appearance_dim=self.config.appearance_dim,
            pe_dim=self.pe_encoder.out_dim,
            hidden_dim=self.config.specular_hidden_dim,
        )

    # ── get_outputs ─────────────────────────────────────────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        cam, crop_ids = prepare_camera(self, camera)
        if cam is None:
            return crop_ids

        from gs_template.rendering import gather_base_params
        means, quats, scales, opacities, colors_diffuse = gather_base_params(
            self, crop_ids
        )

        appearance = crop_custom_param(self.gauss_params["appearance"], crop_ids)
        roughness_logit = crop_custom_param(self.gauss_params["roughness"], crop_ids)
        specular_tint_logit = crop_custom_param(self.gauss_params["specular_tint"], crop_ids)

        roughness = torch.sigmoid(roughness_logit)
        specular_tint = torch.sigmoid(specular_tint_logit)

        # Camera position
        viewmat = cam["viewmat"]
        R = viewmat[0, :3, :3]
        t = viewmat[0, :3, 3]
        cam_pos = -R.T @ t

        viewdirs = F.normalize(cam_pos.unsqueeze(0) - means, dim=-1)

        # Geometric normals from Gaussian shortest axis
        normals = _gaussian_normals(quats, scales)
        # Flip to face camera
        dot_vn = (viewdirs * normals).sum(dim=-1, keepdim=True)
        normals = normals * torch.sign(dot_vn + 1e-8)

        # Reflection direction — physically correct for specular
        reflect_dir = 2.0 * dot_vn * normals - viewdirs
        reflect_dir = F.normalize(reflect_dir, dim=-1)

        # Encode reflection direction with roughness gating
        encoded_refldir = self.pe_encoder(reflect_dir, roughness)

        # Specular color from MLP (softplus output, can be > 1)
        specular_color = self.specular_mlp(appearance, encoded_refldir)

        # Warmup
        if self.training and self.step < self.config.specular_warmup_steps:
            warmup_factor = self.step / max(self.config.specular_warmup_steps, 1)
        else:
            warmup_factor = 1.0

        # Compose: diffuse + tint * specular
        # NO clamp — allow > 1 for bright highlights; compositing clamps later
        final_color = colors_diffuse + warmup_factor * specular_tint * specular_color

        # Rasterize
        render, alpha, info = rasterize_gaussians(
            self, means, quats, scales, opacities, final_color,
            cam["viewmat"], cam["K"], cam["W"], cam["H"],
            sh_degree=None,
        )
        self.info = info

        outputs = assemble_outputs(self, render, alpha, info, cam)

        if self.training:
            outputs["_roughness"] = roughness
            outputs["_specular_tint"] = specular_tint

        return outputs

    # ── compute_custom_losses ───────────────────────────────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        losses = {}

        if self.config.scale_flatten_weight > 0:
            losses["scale_flatten"] = (
                self.config.scale_flatten_weight * scale_flatten_loss(self.scales)
            )

        if self.config.opacity_entropy_weight > 0:
            losses["opacity_entropy"] = (
                self.config.opacity_entropy_weight * opacity_entropy_loss(self.opacities)
            )

        if self.config.specular_sparsity_weight > 0 and "_specular_tint" in outputs:
            losses["specular_sparsity"] = (
                self.config.specular_sparsity_weight
                * outputs["_specular_tint"].abs().mean()
            )

        return losses

    # ── split_gaussians ─────────────────────────────────────────

    def split_gaussians(self, split_mask, samps):
        out = super().split_gaussians(split_mask, samps)
        out["appearance"] = self.gauss_params["appearance"][split_mask].repeat(samps, 1)
        out["roughness"] = self.gauss_params["roughness"][split_mask].repeat(samps, 1)
        out["specular_tint"] = self.gauss_params["specular_tint"][split_mask].repeat(samps, 1)
        return out

    # ── param groups ────────────────────────────────────────────

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {name: [self.gauss_params[name]] for name in self.gauss_params.keys()}

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = super().get_param_groups()
        groups["specular_mlp"] = list(self.specular_mlp.parameters())
        return groups

    # ── viewer outputs ──────────────────────────────────────────

    @torch.no_grad()
    def get_outputs_for_camera(
        self, camera: Cameras, obb_box: Optional[OrientedBox] = None
    ) -> Dict[str, Tensor]:
        return super().get_outputs_for_camera(camera, obb_box)

    # ── checkpoint compat ───────────────────────────────────────

    def load_state_dict(self, dict, **kwargs):
        self.step = 30000

        if "means" in dict and "gauss_params.means" not in dict:
            for p in self.gauss_params.keys():
                if p in dict:
                    dict[f"gauss_params.{p}"] = dict.pop(p)

        if "gauss_params.means" in dict:
            newp = dict["gauss_params.means"].shape[0]
            for name, param in self.gauss_params.items():
                key = f"gauss_params.{name}"
                if key not in dict:
                    new_shape = (newp,) + param.shape[1:]
                    dict[key] = torch.zeros(new_shape)
                new_shape = (newp,) + param.shape[1:]
                self.gauss_params[name] = Parameter(
                    torch.zeros(new_shape, device=self.device)
                )

        super().load_state_dict(dict, **kwargs)

    # ── metrics ─────────────────────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["num_gaussians"] = torch.tensor(self.num_points, dtype=torch.float32)
        if "roughness" in self.gauss_params:
            rough = torch.sigmoid(self.gauss_params["roughness"])
            metrics["mean_roughness"] = rough.mean()
            metrics["pct_specular"] = (rough < 0.3).float().mean()
        if "specular_tint" in self.gauss_params:
            metrics["mean_specular_tint"] = torch.sigmoid(
                self.gauss_params["specular_tint"]
            ).mean()
        return metrics
