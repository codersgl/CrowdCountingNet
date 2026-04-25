"""Shared utilities for GCN diagnostics.

Loads a *baseline* DSGCNet (vgg16_bn, fusion_mode=gcn, gcn_mode=fixed)
with weights/SHTechA.pth and exposes a minimal forward up to PA-FPN
plus density head.  All graph reasoning is then done offline by the
diagnostic scripts on top of `(features_pa, density_pred, gt_points)`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import torch
import torchvision.transforms as T
from PIL import Image

from crowdcount.models.backbone import Backbone_VGG
from crowdcount.models.head import Density_pred
from crowdcount.models.neck import Decoder_SPD_PAFPN

CACHE_DIR = Path("outputs/diag_cache")
DEFAULT_WEIGHTS = Path("weights/SHTechA.pth")
DEFAULT_TEST_DIR = Path("data/shanghaitech/part_A_final/test_data")


# ---------------------------------------------------------------------------
# Minimal "PA-FPN + density" extractor (no GCN, no plugins)
# ---------------------------------------------------------------------------


class BaselineExtractor(torch.nn.Module):
    """Backbone -> PA-FPN -> Density_pred only.

    Reuses the exact module classes/weights that the published checkpoint
    contains, so loaded weights map cleanly via ``strict=False``.
    """

    def __init__(self) -> None:
        super().__init__()
        # backbone returns dict with c2,c3,c4,c5
        self.backbone = Backbone_VGG("vgg16_bn", return_interm_layers=True)
        self.pa = Decoder_SPD_PAFPN(256, 512, 512, use_dcn=False, fpn_attention=False)
        self.density_pred = Density_pred()

    @torch.no_grad()
    def forward(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(samples)
        feats_list = [feats[0], feats[1], feats[2], feats[3]]
        c3, c4, c5 = feats_list[1], feats_list[2], feats_list[3]
        features_pa = self.pa([c3, c4, c5])  # [B, 256, H/8, W/8]
        density = self.density_pred(features_pa)  # [B, 1, H/8, W/8]
        return features_pa, density


def load_extractor(
    weights_path: Path = DEFAULT_WEIGHTS, device: str = "cuda"
) -> BaselineExtractor:
    """Load BaselineExtractor with SHTechA weights (strict=False on plugin keys)."""
    model = BaselineExtractor()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    # Filter to keys matching extractor; report stats
    model_keys = set(model.state_dict().keys())
    matched = {k: v for k, v in state.items() if k in model_keys}
    missing, unexpected = model.load_state_dict(matched, strict=False)
    print(
        f"[load_extractor] matched={len(matched)}/{len(model_keys)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    model.eval().to(device)
    return model


# ---------------------------------------------------------------------------
# Test-set iteration
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    name: str
    image: torch.Tensor  # [1, 3, H, W] normalised
    gt_points: np.ndarray  # [N, 2] in original-image (x, y) coords
    H: int  # padded height (mult of 128)
    W: int  # padded width
    H_feat: int  # H // 8
    W_feat: int  # W // 8


_TRANSFORM = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _load_gt_points(gt_path: Path) -> np.ndarray:
    """Load Nx2 (x, y) point array from SHA .mat file."""
    m = sio.loadmat(str(gt_path))
    return np.asarray(m["image_info"][0, 0][0, 0][0], dtype=np.float32)


def iter_test_samples(
    test_dir: Path = DEFAULT_TEST_DIR,
    num_samples: int | None = None,
    seed: int = 0,
):
    """Yield Sample objects for SHA test images.

    Image is resized so H, W are multiples of 128 (as in scripts/predict.py).
    GT points are *not* rescaled — pass them unchanged; downstream code
    multiplies by (new_W/orig_W, new_H/orig_H) when needed.
    """
    img_dir = test_dir / "images"
    gt_dir = test_dir / "ground_truth"
    image_paths = sorted(img_dir.glob("*.jpg"), key=lambda p: int(p.stem.split("_")[1]))

    if num_samples is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(
            len(image_paths), size=min(num_samples, len(image_paths)), replace=False
        )
        image_paths = [image_paths[i] for i in sorted(idx)]

    for p in image_paths:
        gt_path = gt_dir / f"GT_{p.stem}.mat"
        gt_xy = _load_gt_points(gt_path)
        img_raw = Image.open(p).convert("RGB")
        orig_w, orig_h = img_raw.size
        new_w = max(128, orig_w // 128 * 128)
        new_h = max(128, orig_h // 128 * 128)
        img_raw = img_raw.resize((new_w, new_h), Image.BICUBIC)
        # Rescale GT points to padded resolution
        sx, sy = new_w / orig_w, new_h / orig_h
        gt_xy_scaled = gt_xy.copy()
        gt_xy_scaled[:, 0] *= sx
        gt_xy_scaled[:, 1] *= sy

        img = _TRANSFORM(img_raw).unsqueeze(0)
        yield Sample(
            name=p.stem,
            image=img,
            gt_points=gt_xy_scaled,
            H=new_h,
            W=new_w,
            H_feat=new_h // 8,
            W_feat=new_w // 8,
        )


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.pt"


def save_cache(
    sample: Sample, features_pa: torch.Tensor, density: torch.Tensor
) -> Path:
    """Persist (features_pa, density, gt_points, meta) for later reuse."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = cache_path(sample.name)
    torch.save(
        {
            "name": sample.name,
            "features_pa": features_pa.detach()
            .cpu()
            .half(),  # save FP16 to shrink disk
            "density": density.detach().cpu().float(),
            "gt_points": sample.gt_points.astype(np.float32),
            "H": sample.H,
            "W": sample.W,
            "H_feat": sample.H_feat,
            "W_feat": sample.W_feat,
        },
        out,
    )
    return out


def load_cache(name: str) -> dict:
    return torch.load(cache_path(name), map_location="cpu", weights_only=False)


def list_cache() -> list[str]:
    if not CACHE_DIR.exists():
        return []
    return sorted(p.stem for p in CACHE_DIR.glob("*.pt"))


# ---------------------------------------------------------------------------
# Helpers shared by diag scripts
# ---------------------------------------------------------------------------


def gt_to_feat_grid(
    gt_xy: np.ndarray, H_feat: int, W_feat: int, stride: int = 8
) -> np.ndarray:
    """Project GT (x, y) to feature-map flat indices in [0, H_feat*W_feat).

    Returns array of int64 indices for points falling inside the feature grid.
    """
    fx = np.clip(gt_xy[:, 0] / stride, 0, W_feat - 1).astype(np.int64)
    fy = np.clip(gt_xy[:, 1] / stride, 0, H_feat - 1).astype(np.int64)
    return fy * W_feat + fx


def density_bins(density_flat: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """Quantile-bin density values into n_bins (low/med/high) → [0..n_bins-1]."""
    qs = np.quantile(density_flat, np.linspace(0, 1, n_bins + 1))
    qs[0] -= 1e-9
    return np.clip(np.digitize(density_flat, qs[1:-1]), 0, n_bins - 1)


def homophily_ratio(edge_index: torch.Tensor, node_labels: torch.Tensor) -> float:
    """Fraction of edges (i, j) with node_labels[i] == node_labels[j].

    Args:
        edge_index: [2, E] long tensor.
        node_labels: [N] long tensor.

    Returns:
        float in [0, 1].
    """
    src, dst = edge_index[0], edge_index[1]
    same = (node_labels[src] == node_labels[dst]).float().mean().item()
    return float(same)
