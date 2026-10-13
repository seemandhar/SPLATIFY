"""
gs_template/configs.py — Method Registrations
===============================================

All MethodSpecification entries for nerfstudio's entry-point discovery.
Each config wires together: Trainer → Pipeline → DataManager → Model → Optimizers.

Registered methods:
    gs-template       — vanilla base (identical to splatfacto)
    absgs             — AbsGS (absgrad densification)
    depth-reg-gs      — depth-regularized GS
    feature-gs        — feature distillation GS
    micro-splatting   — Micro-Splatting (covariance reg + pruning/merging)
"""

from __future__ import annotations

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanagerConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
    NerfstudioDataParserConfig,
)
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from gs_template.model import GSTemplateModelConfig
from gs_template.methods.absgs import AbsGSModelConfig
from gs_template.methods.depth_reg import DepthRegGSModelConfig
from gs_template.methods.feature_gs import FeatureGSModelConfig
from gs_template.methods.micro_splatting import MicroSplattingModelConfig
from gs_template.datamanagers import DepthDataManagerConfig
from gs_template.methods.compact_gs import CompactGSModelConfig
from gs_template.methods.msaa_gs import MSAAGSModelConfig
from gs_template.methods.opti3dgs import Opti3DGSModelConfig
from gs_template.methods.decomp_gs import DecompGSModelConfig
from gs_template.methods.bogauss import BOGausSModelConfig
from gs_template.methods.light_gs import LightGSModelConfig
from gs_template.methods.stablegs_datamanager import StableGSDataManagerConfig
from gs_template.methods.stablegs import StableGSModelConfig
from gs_template.methods.vod3dgs import VoD3DGSModelConfig
from gs_template.methods.rain_gs import RAINGSModelConfig
from gs_template.methods.fast_gs import FastGSModelConfig


# ═══════════════════════════════════════════════════════════════════
# Shared optimizer / datamanager / viewer building blocks
# ═══════════════════════════════════════════════════════════════════

def _base_datamanager() -> FullImageDatamanagerConfig:
    """Standard full-image datamanager for GS methods."""
    return FullImageDatamanagerConfig(
        dataparser=NerfstudioDataParserConfig(load_3D_points=True),
        cache_images_type="uint8",
    )


def _base_optimizers(max_steps: int = 30000) -> dict:
    """Standard 3DGS optimizer set (from the original paper)."""
    return {
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6, max_steps=max_steps,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None,
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=max_steps,
                warmup_steps=1000, lr_pre_warmup=0,
            ),
        },
    }


def _base_viewer() -> ViewerConfig:
    return ViewerConfig(num_rays_per_chunk=1 << 15)


# ═══════════════════════════════════════════════════════════════════
# Method 1: gs-template (vanilla base)
# ═══════════════════════════════════════════════════════════════════

gs_template_method = MethodSpecification(
    config=TrainerConfig(
        method_name="gs-template",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=GSTemplateModelConfig(),
        ),
        optimizers=_base_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description="Gaussian Splatting base template (equivalent to splatfacto).",
)


# ═══════════════════════════════════════════════════════════════════
# Method 2: absgs (absolute gradient densification)
# ═══════════════════════════════════════════════════════════════════

absgs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="absgs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=AbsGSModelConfig(),  # absgrad=True by default
        ),
        optimizers=_base_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description="AbsGS: absolute gradient densification for fine detail recovery.",
)


# ═══════════════════════════════════════════════════════════════════
# Method 3: depth-reg-gs (depth-regularized)
# ═══════════════════════════════════════════════════════════════════

def _depth_datamanager() -> DepthDataManagerConfig:
    """DataManager that runs monocular depth estimation + caching."""
    return DepthDataManagerConfig(
        dataparser=NerfstudioDataParserConfig(load_3D_points=True),
        cache_images_type="uint8",
        depth_model="depth_anything_v2",
        depth_model_size="vitb",
        enable_cache=True,
    )


depth_reg_gs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="depth-reg-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_depth_datamanager(),  # ← runs depth estimation
            model=DepthRegGSModelConfig(),
        ),
        optimizers=_base_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description="Depth-regularized GS with Pearson depth loss + opacity entropy.",
)


# ═══════════════════════════════════════════════════════════════════
# Method 4: feature-gs (feature distillation)
# ═══════════════════════════════════════════════════════════════════

def _feature_gs_optimizers() -> dict:
    """Base optimizers + distill_features + feature_mlp."""
    opts = _base_optimizers()
    opts["distill_features"] = {
        "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(
            lr_final=5e-4, max_steps=10000,
        ),
    }
    opts["feature_mlp"] = {
        "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
        "scheduler": None,
    }
    return opts


feature_gs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="feature-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=FeatureGSModelConfig(sh_degree=0),
        ),
        optimizers=_feature_gs_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description="Feature GS: distill CLIP/DINO features into per-Gaussian latents.",
)


micro_splatting_method = MethodSpecification(
    config=TrainerConfig(
        method_name="micro-splatting",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=MicroSplattingModelConfig(
                # Stage transition
                refine_transition_iter=15000,
                # L2 photometric (additive to splatfacto's L1+SSIM)
                l2_weight=0.2,
                # Covariance regularization (conservative)
                cov_loss_weight=0.001,
                trace_cap=0.1,
                use_isotropy_loss=True,
                isotropy_loss_weight=0.0005,
                # Stage II pruning/merging (conservative)
                enable_pruning=True,
                prune_percent=0.05,
                prune_interval=2000,
                enable_merging=True,
                merge_interval=3000,
                merge_spatial_thresh=0.01,
                merge_color_thresh=0.03,
                merge_scale_thresh=0.01,
                # Absgrad for better densification
                absgrad=True,
                output_depth_during_training=True,
            ),
        ),
        optimizers=_base_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "Micro-Splatting: two-stage isotropy-informed covariance regularization. "
        "Stage I grows compact Gaussians with trace-based regularization; "
        "Stage II prunes low-importance splats and merges redundant neighbors. "
        "Reduces splat count ~60% while matching or surpassing baseline PSNR."
    ),
)


def _compact_gs_optimizers(max_steps: int = 30000) -> dict:
    """Optimizers for Compact GS — standard LRs.

    Paper Section A.6 specifies 4× opacity LR *only during NS*
    (15K–23K), then restore to 1×. Since nerfstudio doesn't expose
    optimizer LR to the model at runtime, we use standard LR (0.05)
    throughout and compensate via reg_loss_weight instead.

    This is safer than static 4× because:
        - Phase 1 (0–15K): standard densification, proven stable
        - Phase 2 (15K–23K): reg_loss_weight controls pressure strength
        - Phase 3 (23K–30K): standard fine-tuning

    The paper's 4× effectively multiplies *both* rendering and reg
    gradients for opacity. Using standard LR with calibrated
    reg_loss_weight achieves the same differential pressure without
    disrupting densification.
    """
    return {
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6, max_steps=max_steps,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None,
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=max_steps,
                warmup_steps=1000, lr_pre_warmup=0,
            ),
        },
    }


compact_gs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="compact-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=CompactGSModelConfig(
                # ── Matched to paper Table 1 for T&T scenes ──
                gaussian_budget=600_000,    # Paper: 0.6M for truck
                absgrad=True,               # Paper uses Improved-GS (= absgrad)

                # ── Natural selection schedule ──
                ns_start_iter=15_000,
                ns_max_iter=23_000,
                ns_reg_every=50,
                ns_soft_prune_every=500,
                ns_warmup_iters=3000,
                reg_target_T=-20.0,
                reg_loss_weight=0.01,
                survival_threshold=0.001,
                use_finite_prior=True,
                early_ns_iters=0,
            ),
        ),
        optimizers=_compact_gs_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "Compact 3DGS via gradient-driven natural selection. "
        "Prunes Gaussians to a target budget using opacity-based "
        "gradient competition. Achieves >0.6dB PSNR gain over 3DGS "
        "with only 15% of Gaussians."
    ),
)





def _opti3dgs_optimizers(max_steps: int = 30000) -> dict:
    """Opti3DGS optimizer set with paper-specified modifications.

    From paper Section "Baselines / integration protocol":
        scaling_lr = 0.01      (vs default 0.005)
        densification_interval = 500  (handled in model config)
        densify_grad_threshold = 0.0001  (handled in model config)
    """
    opts = _base_optimizers(max_steps)
    # Paper: scaling_lr = 0.01 (doubled from default 0.005)
    opts["scales"] = {
        "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
        "scheduler": None,
    }
    return opts



opti3dgs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="opti3dgs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=Opti3DGSModelConfig(
                # Paper: densification_interval = 500 (vs splatfacto default 100)
                refine_every=500,
                # Paper: densify_grad_threshold = 0.0001 (vs default 0.0002)
                densify_grad_thresh=0.0001,
                # Paper: filter settings
                initial_kernel_size=15,
                filter_stop_iter=12000,
                num_filter_levels=5,
                blur_type="mean",
            ),
        ),
        # Paper: scaling_lr = 0.01 (modified from default)
        optimizers=_opti3dgs_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "Opti3DGS: Optimized 3DGS via coarse-to-fine image frequency modulation. "
        "Reduces Gaussian count by up to 62% with minimal quality loss."
    ),
)



decomp_gs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="decomp-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=DecompGSModelConfig(
                # Paper defaults
                enable_g2l=True,
                enable_c2f=True,
                enable_adaptive_pruning=True,
                max_scale_factor=4,       # Paper=8, 4 is safer with splatfacto
                densify_until_iter=25000,  # Paper: 25K
                pruning_ratio=0.03,        # Paper: 0.03
                pruning_upper_limit=0.05,  # Paper: 0.05
            ),
        ),
        optimizers=_base_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "DecompGS: G2L densification + C2F multi-resolution + "
        "adaptive opacity pruning. ~2x speedup, comparable quality."
    ),
)
def _nebula_gs_optimizers() -> dict:
    """Base optimizers + emission parameter.

    Key differences from standard:
    - Higher learning rate for means (nebulae need more spatial exploration)
    - emission parameter with moderate LR and decay
    """
    from nerfstudio.engine.optimizers import AdamOptimizerConfig
    from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

    opts = _base_optimizers()

    # Slightly boost means LR for volumetric exploration
    opts["means"] = {
        "optimizer": AdamOptimizerConfig(lr=2.0e-4, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(
            lr_final=2.0e-6, max_steps=30000,
        ),
    }

    # Per-Gaussian emission intensity
    opts["emission"] = {
        "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(
            lr_final=0.001, max_steps=30000,
        ),
    }

    return opts


from gs_template.methods.nebula_gs import NebulaGSModelConfig

nebula_gs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="nebula-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=NebulaGSModelConfig(),
        ),
        optimizers=_nebula_gs_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "NebulaGS v2: physics-informed Gaussian Splatting for volumetric "
        "nebula scenes. Gentle spectral, opacity-ceiling, scale-isotropy, "
        "and emission-decoupled priors with linear warmup scheduling."
    ),
)




bogauss_method = MethodSpecification(
    config=TrainerConfig(
        method_name="bogauss",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=BOGausSModelConfig(
                refine_every=500,
                warmup_length=500,
            ),
        ),
        optimizers=_base_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description="BOGausS: Better Optimized Gaussian Splatting with precision-aware "
                "optimizer, effective opacity pruning, and distortion-based splitting.",
)


def _light_gs_optimizers() -> dict:
    """Optimizers for LightGaussian.

    Total training: 30k (base) + 5k (recovery) + 5k (distill) + 5k (VQ) = 45k
    LR schedules extend to cover the full training run.
    """
    max_steps = 45000
    return {
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6, max_steps=max_steps,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None,
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=max_steps,
                warmup_steps=1000, lr_pre_warmup=0,
            ),
        },
    }


light_gs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="light-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=5000,
        steps_per_eval_all_images=1000,
        max_num_iterations=45000,  # 30k train + 5k recovery + 5k distill + 5k VQ
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=LightGSModelConfig(
                # Paper defaults
                prune_step=30000,
                recovery_steps=5000,
                distill_steps=5000,
                vq_finetune_steps=5000,
                prune_ratio=0.5,
                significance_beta=0.1,
                target_sh_degree=2,
                distill_pseudo_sigma=0.1,
                vq_codebook_size=8192,
                vq_ratio=0.6,
            ),
        ),
        optimizers=_light_gs_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "LightGaussian: 3DGS compression via significance pruning, "
        "SH distillation, and vector quantization. "
        "Achieves 15x reduction with minimal quality loss."
    ),
)


def _stablegs_datamanager() -> StableGSDataManagerConfig:
    """DataManager with DUSt3R depth priors + pair selection."""
    return StableGSDataManagerConfig(
        dataparser=NerfstudioDataParserConfig(load_3D_points=True),
        cache_images_type="uint8",
        use_dust3r=True,
        enable_cache=True,
        min_covisible_points=30,
        min_rotation_deg=16.0,
        max_rotation_deg=60.0,
        max_homography_inlier_ratio=0.8,
        num_warp_pairs=4,
    )


def _stablegs_optimizers() -> dict:
    """Base optimizers + aux_opacities for dual opacity."""
    opts = _base_optimizers()
    opts["aux_opacities"] = {
        "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
        "scheduler": None,
    }
    return opts


stablegs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="stablegs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_stablegs_datamanager(),
            model=StableGSModelConfig(
                output_depth_during_training=True,
                lambda_geo=0.1,
                lambda_s=0.2,
                lambda_consis=0.05,
                lambda_prior=0.005,
                use_dust3r_prior=True,
            ),
        ),
        optimizers=_stablegs_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "StableGS: floater-free 3DGS via dual opacity with "
        "self-supervised depth consistency + DUSt3R priors."
    ),
)


def _vod3dgs_optimizers() -> dict:
    """Base optimizers + opacity_S at 1/4 of opacity learning rate.

    Per the VoD-3DGS paper: lr(Ŝ) = (1/4) * lr(γ).
    Base opacity lr = 0.05  →  opacity_S lr = 0.0125.
    """
    opts = _base_optimizers()
    opts["opacity_S"] = {
        "optimizer": AdamOptimizerConfig(lr=0.0125, eps=1e-15),
        "scheduler": None,
    }
    return opts


vod3dgs_s_method = MethodSpecification(
    config=TrainerConfig(
        method_name="vod3dgs-s",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=VoD3DGSModelConfig(
                vod_variant="S",
                vc_loss_weight=0.01,
                use_vc_loss=True,
            ),
        ),
        optimizers=_vod3dgs_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "VoD-3DGS[S]: view-dependent opacity via SGGX-inspired symmetric "
        "matrix. Smallest-eigenvalue reset variant (more compact)."
    ),
)


# ═══════════════════════════════════════════════════════════════════
# Method 6: vod3dgs-l (VoD-3DGS, largest eigenvalue reset)
# ═══════════════════════════════════════════════════════════════════

vod3dgs_l_method = MethodSpecification(
    config=TrainerConfig(
        method_name="vod3dgs-l",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=VoD3DGSModelConfig(
                vod_variant="L",
                vc_loss_weight=0.01,
                use_vc_loss=True,
            ),
        ),
        optimizers=_vod3dgs_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "VoD-3DGS[L]: view-dependent opacity via SGGX-inspired symmetric "
        "matrix. Largest-eigenvalue reset variant (higher quality)."
    ),
)



def _rain_gs_optimizers(max_steps: int = 40000) -> dict:
    """RAIN-GS optimizer set.

    Key differences from vanilla 3DGS:
    - Total training extended to 40k steps (10k warm-up + 25k densif + 5k opt)
    - Mean learning rate held constant during the 10k warm-up phase by
      setting warmup_steps = 10000 with lr_pre_warmup equal to the initial lr.
      After warm-up, exponential decay proceeds normally over the remaining steps.
    - Scale division ratio is lowered from 1.6 to 1.4 (handled in model config,
      not here; splatfacto's split logic reads this from the strategy config).
    """
    return {
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=max_steps,
                # Hold lr constant at 1.6e-4 for first 10k steps.
                # After warm-up, decay kicks in over the remaining 30k.
                warmup_steps=10000,
                lr_pre_warmup=1.6e-4,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None,
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=max_steps,
                warmup_steps=1000, lr_pre_warmup=0,
            ),
        },
    }


rain_gs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="rain-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=5000,
        steps_per_eval_all_images=1000,
        # 40k total: 10k warm-up + 25k densification + 5k optimization
        max_num_iterations=40000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=RAINGSModelConfig(
                # SLV initialization
                use_random_init=True,
                num_random_points=10,
                # ABE-Split during first 10k steps
                use_abe_split=True,
                abe_end_step=10000,
                # Densification must start early so the 10 initial
                # Gaussians can grow.  warmup_length controls when
                # splatfacto's densification begins, NOT the RAIN-GS
                # warm-up phase (which is handled by abe_end_step).
                warmup_length=500,
                sh_degree_interval=1000,
                # Depth output for monitoring
                output_depth_during_training=True,
            ),
        ),
        optimizers=_rain_gs_optimizers(max_steps=40000),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "RAIN-GS: Relaxing Accurate Initialization Constraint. "
        "Enables 3DGS training from random or sparse point clouds via "
        "SLV initialization, progressive low-pass filter, and ABE-Split."
    ),
)




from gs_template.methods.bubble_gs import BubbleGSModelConfig

# ===================================================================
# Method 5: bubble-gs (thin-film interference physics)
# ===================================================================

def _bubble_gs_optimizers() -> dict:
    """Base optimizers + film_thickness_raw."""
    opts = _base_optimizers()
    opts["film_thickness_raw"] = {
        "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(
            lr_final=0.001, max_steps=30000,
        ),
    }
    return opts


bubble_gs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="bubble-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=BubbleGSModelConfig(
                sh_degree=3,                      # full SH for view-dependent effects
                output_depth_during_training=True, # needed for incidence angle
            ),
        ),
        optimizers=_bubble_gs_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "BubbleGS: physics-informed Gaussian Splatting for scenes with "
        "soap bubbles and other thin-film iridescent structures. Adds "
        "per-Gaussian film thickness and six physics losses based on "
        "Fabry-Perot thin-film interference."
    ),
)


from gs_template.methods.trick_gs import TrickGSModelConfig


def _trick_gs_optimizers(max_steps: int = 30000) -> dict:
    """Base optimizers + learned mask optimizers.

    Mask learning rates from the paper:
      gaussian_mask  lr = 0.5
      sh_masks       lr = 0.05
    """
    opts = _base_optimizers(max_steps)

    opts["gaussian_mask"] = {
        "optimizer": AdamOptimizerConfig(lr=0.5, eps=1e-15),
        "scheduler": None,
    }

    opts["sh_masks"] = {
        "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
        "scheduler": None,
    }

    return opts


trick_gs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="trick-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=TrickGSModelConfig(
                # Progressive resolution via splatfacto's warmup mechanism.
                # Paper starts at 8x downsample; resolution_schedule=300
                # ramps up over early training.
                resolution_schedule=300,
                # Absolute gradients for densification (AbsGS)
                absgrad=True,
                # Output depth during training for geometry monitoring
                output_depth_during_training=True,
            ),
        ),
        optimizers=_trick_gs_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "Trick-GS: learned masks, significance pruning, SH masking, "
        "and progressive training for efficient Gaussian Splatting."
    ),
)


from gs_template.methods.clod_gs import CLoDGSModelConfig

def _clod_gs_optimizers(max_steps: int = 30000) -> dict:
    """Base optimizers + sigma_d with its own learning rate.

    The paper specifies lr = 1e-2 for sigma_d, which is considerably
    higher than the position lr (1.6e-4).  This makes sense because
    sigma_d controls a broad monotonic decay and converges quickly.
    """
    opts = _base_optimizers(max_steps=max_steps)
    opts["sigma_d"] = {
        "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
        "scheduler": None,
    }
    return opts


clod_gs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="clod-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=CLoDGSModelConfig(),
        ),
        optimizers=_clod_gs_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "CLoD-GS: continuous level-of-detail through distance-adaptive "
        "opacity decay.  Adds one scalar per Gaussian (~1.6%% overhead) "
        "and enables quality-performance trade-off at inference via s_v."
    ),
)

from gs_template.methods.twodgs import TwoDGSModelConfig


twodgs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="twodgs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=TwoDGSModelConfig(),
        ),
        optimizers=_base_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "2D Gaussian Splatting (Huang et al.): planar Gaussians with "
        "depth distortion and normal consistency losses for improved "
        "surface geometry."
    ),
)


from gs_template.methods.pixel_gs import PixelGSModelConfig

pixel_gs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="pixel-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=PixelGSModelConfig(),
        ),
        optimizers=_base_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description="Pixel-GS: pixel-aware gradient + depth scaling for density control.",
)


from gs_template.methods.gaussian_shader import GaussianShaderModelConfig


def _gaussian_shader_optimizers(max_steps: int = 30000) -> dict:
    """Optimizer groups for GaussianShader.

    The paper specifies Adam and 30k iters, but not exact per-parameter learning
    rates for all new shading parameters. These defaults follow the base template's
    color-like / geometry-like split and are intended as a strong starting point.
    """
    opts = _base_optimizers(max_steps=max_steps)
    opts["diffuse_color"] = {
        "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
        "scheduler": None,
    }
    opts["specular_tint"] = {
        "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
        "scheduler": None,
    }
    opts["roughness"] = {
        "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
        "scheduler": None,
    }
    opts["normal_delta_1"] = {
        "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
        "scheduler": None,
    }
    opts["normal_delta_2"] = {
        "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
        "scheduler": None,
    }
    opts["lighting"] = {
        "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=max_steps),
    }
    return opts


gaussian_shader_method = MethodSpecification(
    config=TrainerConfig(
        method_name="gaussian-shader",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=GaussianShaderModelConfig(),
        ),
        optimizers=_gaussian_shader_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description="GaussianShader: reflective Gaussian Splatting with shading attributes and env-lighting.",
)

from gs_template.methods.scaffold_gs import ScaffoldGSModelConfig

def _scaffold_gs_optimizers(max_steps: int = 30000) -> dict:
    """Scaffold-GS optimizers: anchor params + 5 MLPs + camera.

    Optimizer groups must match the keys returned by:
        get_gaussian_param_groups() → anchor_positions, anchor_features,
                                       anchor_offsets, anchor_scales
        get_param_groups()          → mlp_feature_bank, mlp_opacity,
                                       mlp_color, mlp_quat, mlp_scale,
                                       camera_opt

    NOTE: The base splatfacto param groups (means, features_dc, etc.)
    still need entries in the optimizer dict even though Scaffold-GS
    doesn't use them — nerfstudio's optimizer builder iterates over
    ALL param_groups returned by the model and looks them up.
    We include them with tiny LR as no-ops.
    """
    opts = {}

    # ── Anchor params ─────────────────────────────────────────────
    opts["anchor_positions"] = {
        "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(
            lr_final=1.6e-6, max_steps=max_steps,
        ),
    }
    opts["anchor_features"] = {
        "optimizer": AdamOptimizerConfig(lr=0.0075, eps=1e-15),
        "scheduler": None,
    }
    opts["anchor_offsets"] = {
        "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(
            lr_final=0.001, max_steps=max_steps,
        ),
    }
    opts["anchor_scales"] = {
        "optimizer": AdamOptimizerConfig(lr=0.007, eps=1e-15),
        "scheduler": None,
    }

    # ── MLP params ────────────────────────────────────────────────
    mlp_lr = 0.001
    for mlp_name in [
        "mlp_feature_bank", "mlp_opacity", "mlp_color",
        "mlp_quat", "mlp_scale",
    ]:
        opts[mlp_name] = {
            "optimizer": AdamOptimizerConfig(lr=mlp_lr, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-5, max_steps=max_steps,
            ),
        }

    # ── Camera optimizer ──────────────────────────────────────────
    opts["camera_opt"] = {
        "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(
            lr_final=5e-7, max_steps=max_steps,
            warmup_steps=1000, lr_pre_warmup=0,
        ),
    }

    return opts


scaffold_gs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="scaffold-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=ScaffoldGSModelConfig(),
        ),
        optimizers=_scaffold_gs_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "Scaffold-GS: Structured 3D Gaussians with anchor-based "
        "representation and MLP-decoded view-adaptive neural Gaussians."
    ),
)

from gs_template.methods.dwtgs import DWTGSModelConfig


dwtgs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="dwtgs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=DWTGSModelConfig(
                # Paper defaults
                lf_loss_weight=0.5,       # λ_LL = 0.5
                dwt_levels=2,             # N = 2
                use_hf_sparsity=True,     # Enable HF regularization
                hf_loss_weight=0.1,       # Weight for L1(HH, 0)
                hf_novel_view_interval=10,  # Every 10 iters
                hf_stop_step=5000,        # Stop after 5K iters
            ),
        ),
        optimizers=_base_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "DWTGS: wavelet-space frequency regularization for sparse-view 3DGS. "
        "Haar DWT LF supervision + HF sparsity on novel views."
    ),
)

from gs_template.methods.lp3dgs import LP3DGSModelConfig

def _lp3dgs_optimizers(max_steps: int = 30000) -> dict:
    """Base optimizers + mask_logits optimizer.

    Mask optimizer uses a moderate learning rate — too high causes
    oscillation, too low prevents convergence in the 500-iter window.
    """
    opts = _base_optimizers(max_steps=max_steps)
    opts["mask_logits"] = {
        "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
        "scheduler": None,
    }
    return opts


lp3dgs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="lp3dgs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=LP3DGSModelConfig(
                # Paper defaults
                mask_learning_start=19500,
                mask_learning_end=20000,
                importance_update_freq=20,
                gumbel_tau=0.5,
                importance_type="radsplat",
                mask_reg_weight=5e-4,
                prune_threshold=0.5,
            ),
        ),
        optimizers=_lp3dgs_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "LP-3DGS: Learning to Prune 3D Gaussian Splatting. "
        "Learns per-Gaussian masks via Gumbel-Sigmoid to automatically "
        "determine the optimal pruning ratio per scene."
    ),
)


from gs_template.methods.compgs import CompGSModelConfig
from gs_template.methods.compgs_enhanced import CompGSEnhancedModelConfig

compgs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="compgs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=CompGSModelConfig(
                # CompGS uses full SH degree 3 (default in splatfacto)
                sh_degree=3,
                # VQ schedule (paper defaults)
                vq_start_iter=20000,
                vq_freeze_iter=25000,
                vq_assign_freq=100,
                # Codebook sizes (paper: "CompGS 16K" variant)
                dc_codebook_size=4096,
                sh_codebook_size=4096,
                scale_codebook_size=16384,
                quat_codebook_size=16384,
                # Opacity regularization (paper: λ=1e-7, applied 15K-20K)
                opacity_reg_weight=1e-7,
                opacity_reg_start=15000,
                opacity_reg_end=20000,
                # Lower cull threshold to work with opacity L1 reg
                cull_alpha_thresh=0.005,
            ),
        ),
        optimizers=_base_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "CompGS: Compact Gaussian Splatting with vector quantization "
        "(K-means + STE) and opacity L1 regularization for smaller, "
        "faster models."
    ),
)


def _compgs_enhanced_optimizers(max_steps: int = 45000) -> dict:
    """Base optimizers + bilateral_grid for CompGS Enhanced."""
    opts = _base_optimizers(max_steps=max_steps)
    opts["bilateral_grid"] = {
        "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
        "scheduler": None,
    }
    return opts


compgs_enhanced_method = MethodSpecification(
    config=TrainerConfig(
        method_name="compgs-enhanced",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=45000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=CompGSEnhancedModelConfig(
                sh_degree=3,
                # Larger codebooks (2× CompGS)
                dc_codebook_size=8192,
                sh_codebook_size=8192,
                scale_codebook_size=32768,
                quat_codebook_size=32768,
                # Extended VQ schedule for 45K training
                vq_start_iter=25000,
                vq_freeze_iter=35000,
                vq_assign_freq=100,
                # Opacity entropy reg (replaces L1)
                opacity_reg_start=15000,
                opacity_reg_end=25000,
                use_opacity_entropy=True,
                opacity_entropy_weight=0.01,
                # Residual compensation
                use_residuals=True,
                residual_weight=0.1,
                # Soft VQ annealing
                use_soft_vq=True,
                soft_vq_temperature_start=1.0,
                soft_vq_temperature_end=0.01,
                soft_vq_anneal_iters=5000,
                # Quality boosters
                use_bilateral_grid=True,
                rasterize_mode="antialiased",
                use_scale_regularization=True,
                cull_alpha_thresh=0.005,
                continue_cull_post_densification=False,
            ),
        ),
        optimizers=_compgs_enhanced_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "CompGS Enhanced: CompGS + 2× codebooks + residual compensation + "
        "soft VQ annealing + opacity entropy + bilateral grid + antialiased "
        "rasterization. Expected +2 dB PSNR over CompGS."
    ),
)


def _cor_gs_optimizers(max_steps: int = 10000) -> dict:
    """Base optimizers for field 1 + identical optimizers for field 2.

    CoR-GS trains two fields simultaneously. Both use the same
    optimizer configuration (standard 3DGS settings).

    Paper training schedules:
        LLFF / DTU / Blender:   10k iterations (max_steps=10000)
        Mip-NeRF360:            30k iterations (max_steps=30000)
    """
    # Field 1 optimizers (standard)
    opts = {
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6, max_steps=max_steps,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None,
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=max_steps,
                warmup_steps=1000, lr_pre_warmup=0,
            ),
        },
    }

    # Field 2 optimizers — same learning rates as field 1
    # (registered as field2_<param_name> to match get_param_groups())
    opts["field2_means"] = {
        "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(
            lr_final=1.6e-6, max_steps=max_steps,
        ),
    }
    opts["field2_features_dc"] = {
        "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
        "scheduler": None,
    }
    opts["field2_features_rest"] = {
        "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
        "scheduler": None,
    }
    opts["field2_opacities"] = {
        "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
        "scheduler": None,
    }
    opts["field2_scales"] = {
        "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
        "scheduler": None,
    }
    opts["field2_quats"] = {
        "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
        "scheduler": None,
    }

    return opts


from gs_template.methods.cor_gs import CoRGSModelConfig

cor_gs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="cor-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,  # LLFF/DTU/Blender: 10k
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=CoRGSModelConfig(
                # ── Co-pruning settings (from paper) ───────────
                co_prune_tau=5.0,
                co_prune_every_n_refine=5,
                co_prune_start_step=500,
                # ── Pseudo-view co-regularization ──────────────
                pseudo_view_weight=1.0,      # λ_p = 1.0
                dssim_weight=0.2,            # λ = 0.2
                pseudo_view_noise_std=0.05,
                pseudo_view_start_step=200,
                # ── Field 2 densification ──────────────────────
                field2_densify=True,
                field2_grad_thresh=0.0002,
            ),
        ),
        optimizers=_cor_gs_optimizers(max_steps=10000),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "CoR-GS: Sparse-View 3DGS via Co-Regularization. "
        "Trains two fields with co-pruning + pseudo-view co-regularization. "
        "Default: 10k iters (LLFF/DTU/Blender). "
        "For Mip-NeRF360: set max_num_iterations=30000."
    ),
)



from gs_template.methods.fsgs import FSGSModelConfig

def _fsgs_datamanager() -> DepthDataManagerConfig:
    """DataManager with monocular depth for FSGS training-view depth loss.

    Uses Depth Anything V2 (paper showed robustness across DPT, MiDaS,
    DepthAnything estimators). Caching is essential for fast reload on
    few-shot datasets.

    num_training_views defaults to -1 (use all). Override at CLI:
        --pipeline.datamanager.num_training_views 3    # LLFF few-shot
        --pipeline.datamanager.num_training_views 12   # custom
        --pipeline.datamanager.num_training_views 24   # Mip-NeRF360
    """
    return DepthDataManagerConfig(
        dataparser=NerfstudioDataParserConfig(load_3D_points=True),
        cache_images_type="uint8",
        depth_model="depth_anything_v2",
        depth_model_size="vitb",
        enable_cache=True,
        num_training_views=-1,  # -1 = use all; set via CLI for few-shot
    )


def _fsgs_optimizers() -> dict:
    """FSGS optimizer set matching the paper's learning rates.

    From Section 4 / training schedule:
        Position:  0.00016 (with exponential decay)
        SH DC:     0.0025
        SH rest:   0.0025 / 20 = 0.000125
        Opacity:   0.05
        Scaling:   0.005
        Rotation:  0.001

    NOTE: max_steps=10000 (paper trains for 10k iterations, not 30k).
    """
    max_steps = 10000
    return {
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6, max_steps=max_steps,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None,
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=max_steps,
                warmup_steps=1000, lr_pre_warmup=0,
            ),
        },
    }


fsgs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="fsgs",
        steps_per_eval_image=500,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=5000,
        max_num_iterations=10000,       # ← Paper: 10k iterations (not 30k)
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_fsgs_datamanager(),
            model=FSGSModelConfig(
                # ── Loss weights (paper Section 4) ─────────────────
                # lambda_1=0.8, lambda_2=0.2 → handled by splatfacto defaults
                depth_loss_weight=0.05,          # lambda_3
                pseudo_depth_loss_weight=0.05,   # lambda_4

                # ── Proximity unpooling ────────────────────────────
                proximity_k=3,
                proximity_threshold=10.0,
                unpooling_start_iter=500,
                unpooling_every=100,

                # ── Pseudo-view synthesis ──────────────────────────
                pseudo_view_start_iter=2000,
                pseudo_view_noise_std=0.01,

                # ── SH schedule (degree 0→4, increase every 500) ──
                sh_degree=4,
                sh_degree_interval=500,

                # ── Opacity reset at {2000, 5000, 7000} ───────────
                opacity_reset_iters=(2000, 5000, 7000),
                opacity_reset_value=0.05,

                # ── Must render depth during training ──────────────
                output_depth_during_training=True,

                # ── Densification schedule ─────────────────────────
                # Standard 3DGS densification also runs alongside
                # proximity unpooling (every 100 iters, start at 500)
                refine_every=100,
                warmup_length=500,
            ),
        ),
        optimizers=_fsgs_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting. "
        "Proximity-guided Gaussian unpooling + pseudo-view synthesis + "
        "depth regularization. Optimized for 3-24 training views."
    ),
)




from gs_template.methods.msaa_gs import MSAAGSModelConfig

msaa_gs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="msaa-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,  # Paper: 30k iterations
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=MSAAGSModelConfig(
                # Paper defaults
                msaa_samples=4,           # 4× MSAA
                aws_weight=0.1,           # additive AWS loss weight
                grad_loss_weight=0.1,     # λ₃ GDC
                aws_alpha=0.5,            # α for adaptive weighting
            ),
        ),
        optimizers=_base_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description="MSAA-GS: Multi-Sample Anti-Aliasing + Adaptive Weighting + Gradient Constraints.",
)


# ═══════════════════════════════════════════════════════════════════
# Method 6: msaa-gs-plus (MSAA + bilateral grid + longer training)
# ═══════════════════════════════════════════════════════════════════

def _bilagrid_optimizers(max_steps: int = 45000) -> dict:
    """Base optimizers + bilateral_grid."""
    opts = _base_optimizers(max_steps=max_steps)
    opts["bilateral_grid"] = {
        "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
        "scheduler": None,
    }
    return opts


msaa_gs_plus_method = MethodSpecification(
    config=TrainerConfig(
        method_name="msaa-gs-plus",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=45000,       # 50% more training
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=MSAAGSModelConfig(
                # Paper defaults (unchanged)
                msaa_samples=4,
                aws_weight=0.1,
                grad_loss_weight=0.1,
                aws_alpha=0.5,
                # Only addition: bilateral grid
                use_bilateral_grid=True,
            ),
        ),
        optimizers=_bilagrid_optimizers(max_steps=45000),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description="MSAA-GS+: MSAA paper + bilateral grid color correction + 45k iters.",
)


from gs_template.methods.diamond_gs import DiamondGS3DModelConfig





def _diamond_gs_optimizers() -> dict:
    """Base optimizers + appearance, roughness, specular_tint, specular_mlp."""
    opts = _base_optimizers(max_steps=50000)

    # Per-Gaussian appearance embedding — higher LR for faster feature learning
    opts["appearance"] = {
        "optimizer": AdamOptimizerConfig(lr=0.008, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(
            lr_final=0.0008, max_steps=50000,
        ),
    }

    # Per-Gaussian roughness
    opts["roughness"] = {
        "optimizer": AdamOptimizerConfig(lr=0.002, eps=1e-15),
        "scheduler": None,
    }

    # Per-Gaussian specular tint
    opts["specular_tint"] = {
        "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(
            lr_final=0.0005, max_steps=50000,
        ),
    }

    # Specular MLP — higher LR for faster convergence
    opts["specular_mlp"] = {
        "optimizer": AdamOptimizerConfig(lr=0.002, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(
            lr_final=2e-4, max_steps=50000,
        ),
    }

    return opts


diamond_gs_3d_method = MethodSpecification(
    config=TrainerConfig(
        method_name="diamond-gs-3d",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=50000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=DiamondGS3DModelConfig(),
        ),
        optimizers=_diamond_gs_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "DiamondGS3D: Neural appearance Gaussian Splatting for diamond/gemstone "
        "scenes. Reflection-direction PE + roughness gating + neural specular "
        "MLP with softplus output for HDR highlights."
    ),
)


# ═══════════════════════════════════════════════════════════════════
# FastGS — Training 3D Gaussian Splatting in 100 Seconds
# ═══════════════════════════════════════════════════════════════════


def _fastgs_optimizers(max_steps: int = 30000) -> dict:
    """FastGS optimizer set — lower opacity LR (0.025 vs 0.05)."""
    opts = _base_optimizers(max_steps)
    opts["opacities"] = {
        "optimizer": AdamOptimizerConfig(lr=0.025, eps=1e-15),
        "scheduler": None,
    }
    return opts


fast_gs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="fast-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=FastGSModelConfig(
                refine_every=500,
            ),
        ),
        optimizers=_fastgs_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "FastGS: Training 3D Gaussian Splatting in 100 seconds via "
        "multi-view consistent densification (VCD) and pruning (VCP). "
        "Base variant: densifies every 500 steps for fastest training."
    ),
)


fast_gs_big_method = MethodSpecification(
    config=TrainerConfig(
        method_name="fast-gs-big",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=FastGSModelConfig(
                refine_every=100,
            ),
        ),
        optimizers=_fastgs_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description=(
        "FastGS-Big: Higher quality variant with densification every 100 steps. "
        "More Gaussians (~1.15M vs ~0.4M), better PSNR, but slower than base."
    ),
)
