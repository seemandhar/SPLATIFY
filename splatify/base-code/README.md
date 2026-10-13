# GS Method Template v2

A minimal, modular template for implementing Gaussian Splatting research papers as nerfstudio plugins.

## Architecture

Extends `SplatfactoModel` — inherits all battle-tested densification, rendering, and optimizer logic.
Paper authors override only what their contribution changes.

```
gs_template/
├── __init__.py              # Package exports
├── model.py                 # GSTemplateModel base class (9 extension points)
├── losses.py                # Reusable loss library (cosine, depth, entropy, TV, ...)
├── rendering.py             # Rendering helpers (prepare_camera, rasterize, assemble)
├── datamanagers.py          # Custom DataManagers (depth estimation, caching)
├── configs.py               # All MethodSpecification entries + optimizer builders
├── shading.py               # SH evaluation, normals, cubemap sampling
├── optimizers.py            # Custom optimizers (BOGausSAdam)
├── methods/                 # 33 registered methods, e.g.
│   ├── absgs.py             # ★☆☆☆☆  Config-only (absgrad=True)
│   ├── depth_reg.py         # ★★☆☆☆  Custom loss + DepthDataManager
│   ├── feature_gs.py        # ★★★★☆  Full: params, rendering, loss, densification
│   └── ...                  # scaffold_gs, twodgs, fast_gs, nebula_gs, ...
CFG.md                       # Context-Free Grammar for paper-to-code mapping
eval.py / read_tb.py         # Train+eval across scenes / PSNR curves
pyproject.toml               # nerfstudio entry points
```

## Quick Start

```bash
pip install -e .
ns-install-cli

# Train any registered method (see pyproject.toml for the full list):
ns-train gs-template  --data data/nerfstudio/poster    # vanilla base
ns-train absgs        --data data/nerfstudio/poster    # AbsGS
ns-train depth-reg-gs --data data/nerfstudio/poster    # depth-regularized
ns-train feature-gs   --data data/nerfstudio/poster    # feature distillation
```

## Modular Components

### `losses.py` — Reusable Loss Library

| Function | Used By | Description |
|----------|---------|-------------|
| `cosine_loss()` | Feature Splatting, LangSplat | 1 - cos_sim, averaged |
| `l2_feature_loss()` | DINO-Splatting | MSE feature loss |
| `pearson_depth_loss()` | DepthRegGS, MonoGS | Scale-invariant depth |
| `l1_depth_loss()` | DepthRegGS | Absolute depth with mask |
| `opacity_entropy_loss()` | Anti-artifact papers | Binary opacity regularizer |
| `scale_flatten_loss()` | 2DGS, SuGaR | Penalize spikey Gaussians |
| `distortion_loss()` | Mip-Splatting | Concentrate ray weights |
| `total_variation_loss()` | Feature smoothness | Spatial regularization |

```python
from gs_template.losses import cosine_loss, pearson_depth_loss

class MyModel(GSTemplateModel):
    def compute_custom_losses(self, outputs, batch, metrics_dict=None):
        return {"my_loss": 0.01 * cosine_loss(pred, target)}
```

### `rendering.py` — Rendering Helpers

For papers that override `get_outputs()`, three helpers eliminate 80+ lines of boilerplate:

```python
from gs_template.rendering import (
    prepare_camera,        # Phase 1: camera setup, cropping
    gather_base_params,    # Gather means/quats/scales/opacities/colors
    crop_custom_param,     # Crop a custom param with same crop_ids
    rasterize_gaussians,   # Phase 2: gsplat rasterization call
    assemble_outputs,      # Phase 3: background, depth, output dict
)

class MyModel(GSTemplateModel):
    def get_outputs(self, camera):
        cam, crop_ids = prepare_camera(self, camera)
        if cam is None:
            return crop_ids
        means, quats, scales, opacities, colors = gather_base_params(self, crop_ids)
        my_feats = crop_custom_param(self.gauss_params["my_feats"], crop_ids)
        fused = torch.cat([colors, my_feats], dim=-1)
        render, alpha, info = rasterize_gaussians(
            self, means, quats, scales, opacities, fused,
            cam["viewmat"], cam["K"], cam["W"], cam["H"])
        return assemble_outputs(self, render, alpha, info, cam,
                                extra_channels={"feature": (3, 16)})
```

### `datamanagers.py` — Custom DataManagers

Heavy preprocessing (depth estimation, feature extraction) lives in the DataManager, not the Model. This follows the feature-splatting pattern:

| DataManager | Preprocessing | Injects Into Batch |
|-------------|--------------|-------------------|
| `DepthDataManager` | Monocular depth (Depth Anything V2, ZoeDepth, MiDAS) | `batch["depth_image"]` |

```python
# depth-reg-gs uses DepthDataManager automatically (configured in configs.py)
ns-train depth-reg-gs --data <path>
ns-train depth-reg-gs --data <path> --pipeline.datamanager.depth_model zoedepth
```

Architecture: DataManager runs at startup → caches to disk → injects per-batch. Model just consumes `batch["depth_image"]`.

## Extension Points

| # | Method | Override When | Example |
|---|--------|-------------|---------|
| 1 | `populate_modules()` | New Gaussian params or MLPs | Feature GS |
| 2 | `get_outputs()` | Rendering changes (rare) | Feature GS |
| 3 | `compute_custom_losses()` | Novel loss terms | Depth-Reg GS |
| 4 | `split_gaussians()` | New params need splitting | Feature GS |
| 5 | `get_gaussian_param_groups()` | New Gaussian optimizers | Feature GS |
| 6 | `get_param_groups()` | Non-Gaussian params (MLPs) | Feature GS |
| 7 | `get_outputs_for_camera()` | Viewer visualizations | Feature GS |
| 8 | `load_state_dict()` | Checkpoint with new params | Feature GS |
| 9 | `get_metrics_dict()` | Custom metrics | Feature GS |

## Adding a New Method

1. Create `gs_template/methods/my_method.py` with your `Config` + `Model`
2. Add a `MethodSpecification` in `gs_template/configs.py`
3. Add entry point in `pyproject.toml`
4. `pip install -e . && ns-install-cli`

## Design Principle

> **Don't reimplement what splatfacto already does.**
> Extend and override. Use `losses.py` and `rendering.py` as building blocks.
