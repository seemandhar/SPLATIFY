"""
gs_template/diamond_datamanagers.py — Diamond-specific DataManagers
===================================================================

Builds on top of DepthDataManager and injects:
    batch["diamond_mask"]
    batch["highlight_mask"]

Mask loading priority:
1) user-provided grayscale masks in diamond_mask_dir
2) alpha channel in images (if enabled)
3) full-frame mask fallback

Highlight mask:
- computed inside the diamond mask
- uses luminance percentile threshold
- cached to disk
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Tuple, Type

import torch
from PIL import Image

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.rich_utils import CONSOLE

from gs_template.datamanagers import DepthDataManager, DepthDataManagerConfig


@dataclass
class DiamondDataManagerConfig(DepthDataManagerConfig):
    """Depth + mask datamanager for diamond scenes."""

    _target: Type = field(default_factory=lambda: DiamondDataManager)

    diamond_mask_dirname: str = "diamond_masks"
    """Folder under the dataset root with one grayscale mask per image."""

    mask_suffix: str = ".png"
    """Expected suffix for mask files."""

    use_alpha_as_mask: bool = True
    """If no explicit mask exists, use the image alpha channel if present."""

    fallback_to_full_frame_mask: bool = True
    """If no mask exists, use an all-ones mask."""

    highlight_percentile: float = 97.5
    """Highlight threshold percentile over luminance inside the mask."""

    enable_mask_cache: bool = True
    """Cache loaded / computed masks."""


class DiamondDataManager(DepthDataManager):
    """DepthDataManager + cached diamond / highlight masks."""

    config: DiamondDataManagerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        all_masks, all_highlights = self._load_or_build_masks()

        n_train = len(self.train_dataset)
        self.train_diamond_masks = all_masks[:n_train]
        self.eval_diamond_masks = all_masks[n_train:]
        self.train_highlight_masks = all_highlights[:n_train]
        self.eval_highlight_masks = all_highlights[n_train:]

    def _load_or_build_masks(self):
        image_fnames = self.train_dataset.image_filenames + self.eval_dataset.image_filenames
        cache_dir = self.config.dataparser.data
        cache_path = cache_dir / "diamond_masks_cache.pt"

        if self.config.enable_mask_cache and cache_path.exists():
            CONSOLE.print(f"Loading cached diamond masks from {cache_path}")
            cache = torch.load(cache_path, map_location="cpu")
            if cache.get("image_fnames") == image_fnames:
                return cache["masks"], cache["highlights"]

        masks = []
        highlights = []

        for i, path in enumerate(image_fnames):
            if i % 50 == 0:
                CONSOLE.print(f"Preparing masks {i+1}/{len(image_fnames)}...")

            mask = self._load_single_mask(path)
            highlight = self._compute_highlight_mask(path, mask)

            masks.append(mask.unsqueeze(-1))
            highlights.append(highlight.unsqueeze(-1))

        masks = torch.stack(masks)          # (N,H,W,1)
        highlights = torch.stack(highlights)

        if self.config.enable_mask_cache:
            torch.save(
                {
                    "image_fnames": image_fnames,
                    "masks": masks,
                    "highlights": highlights,
                },
                cache_path,
            )
            CONSOLE.print(f"  ✓ Cached diamond masks to {cache_path}")

        return masks, highlights

    def _candidate_mask_paths(self, img_path: Path):
        root = self.config.dataparser.data
        mask_root = root / self.config.diamond_mask_dirname
        stem = Path(img_path).stem
        return [
            mask_root / f"{stem}{self.config.mask_suffix}",
            mask_root / f"{Path(img_path).name}{self.config.mask_suffix}",
        ]

    def _load_single_mask(self, img_path: Path) -> torch.Tensor:
        for p in self._candidate_mask_paths(img_path):
            if p.exists():
                mask = Image.open(p).convert("L")
                mask_t = torch.from_numpy(
                    __import__("numpy").array(mask, dtype="float32") / 255.0
                )
                return (mask_t > 0.5).float()

        img = Image.open(img_path)
        if self.config.use_alpha_as_mask and "A" in img.getbands():
            alpha = img.getchannel("A")
            alpha_t = torch.from_numpy(
                __import__("numpy").array(alpha, dtype="float32") / 255.0
            )
            return (alpha_t > 0.5).float()

        if self.config.fallback_to_full_frame_mask:
            w, h = img.size
            return torch.ones((h, w), dtype=torch.float32)

        raise FileNotFoundError(
            f"No diamond mask found for {img_path}. Provide masks under "
            f"{self.config.diamond_mask_dirname}/ or enable alpha/full-frame fallback."
        )

    def _compute_highlight_mask(self, img_path: Path, mask: torch.Tensor) -> torch.Tensor:
        img = Image.open(img_path).convert("RGB")
        arr = __import__("numpy").array(img, dtype="float32") / 255.0
        arr_t = torch.from_numpy(arr)

        # luminance in [0,1]
        lum = 0.2126 * arr_t[..., 0] + 0.7152 * arr_t[..., 1] + 0.0722 * arr_t[..., 2]
        masked = lum[mask > 0.5]

        if masked.numel() < 20:
            return torch.zeros_like(mask)

        thresh = torch.quantile(masked, self.config.highlight_percentile / 100.0)
        highlight = ((lum >= thresh) & (mask > 0.5)).float()
        return highlight

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        camera, data = super().next_train(step)
        cam_idx = camera.metadata["cam_idx"]
        data["diamond_mask"] = self.train_diamond_masks[cam_idx]
        data["highlight_mask"] = self.train_highlight_masks[cam_idx]
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        camera, data = super().next_eval(step)
        cam_idx = camera.metadata["cam_idx"]
        data["diamond_mask"] = self.eval_diamond_masks[cam_idx]
        data["highlight_mask"] = self.eval_highlight_masks[cam_idx]
        return camera, data
