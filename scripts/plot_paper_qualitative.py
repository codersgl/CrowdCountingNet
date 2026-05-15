"""Generate qualitative SHHA visualisations for the paper draft."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import torch
import torch.nn.functional as F
import torchvision.transforms as standard_transforms
from omegaconf import OmegaConf
from PIL import Image

from crowdcount.models import build_model


def load_shha_points(gt_path: Path) -> torch.Tensor:
    """Load ShanghaiTech point annotations as an [N, 2] tensor."""
    mat = scipy.io.loadmat(str(gt_path))
    points = mat["image_info"][0, 0][0, 0][0]
    return torch.as_tensor(points, dtype=torch.float32).reshape(-1, 2)


def pad_to_divisor(image_tensor: torch.Tensor, divisor: int = 128) -> torch.Tensor:
    """Pad a CHW tensor to the next multiple of divisor."""
    channels, height, width = image_tensor.shape
    padded_height = ((height + divisor - 1) // divisor) * divisor
    padded_width = ((width + divisor - 1) // divisor) * divisor
    padded = torch.zeros(
        (1, channels, padded_height, padded_width), dtype=image_tensor.dtype
    )
    padded[0, :, :height, :width] = image_tensor
    return padded


def normalise_density_for_display(density: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Upsample and min-max normalise density for heatmap display."""
    density_up = F.interpolate(
        density,
        size=(density.shape[-2] * 8, density.shape[-1] * 8),
        mode="bilinear",
        align_corners=False,
    )[0, 0, :height, :width]
    density_min = density_up.min()
    density_max = density_up.max()
    return (density_up - density_min) / (density_max - density_min + 1e-8)


def infer_one(
    model: torch.nn.Module,
    image_path: Path,
    gt_path: Path,
    threshold: float,
    device: torch.device,
) -> dict[str, object]:
    """Run one-image inference and collect visualisation tensors."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    gt_points = load_shha_points(gt_path)

    transform = standard_transforms.Compose(
        [
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    sample = pad_to_divisor(transform(image)).to(device)

    with torch.no_grad():
        outputs = model(sample)

    scores = torch.softmax(outputs["pred_logits"], dim=-1)[0, :, 1].detach().cpu()
    pred_points = outputs["pred_points"][0].detach().cpu()
    valid_mask = (
        (scores > threshold)
        & (pred_points[:, 0] >= 0)
        & (pred_points[:, 0] < width)
        & (pred_points[:, 1] >= 0)
        & (pred_points[:, 1] < height)
    )
    pred_points = pred_points[valid_mask]
    pred_scores = scores[valid_mask]
    density = outputs["density_out"].detach().cpu()

    return {
        "image_name": image_path.name,
        "image": image,
        "width": width,
        "height": height,
        "gt_points": gt_points,
        "pred_points": pred_points,
        "pred_scores": pred_scores,
        "gt_count": int(gt_points.shape[0]),
        "pred_count": int(pred_points.shape[0]),
        "density_sum": float(density.sum().item()),
        "density_display": normalise_density_for_display(density, height, width),
    }


def draw_points(axis: plt.Axes, points: torch.Tensor, color: str, size: float) -> None:
    """Draw point annotations on an axis."""
    if points.numel() == 0:
        return
    axis.scatter(
        points[:, 0].numpy(),
        points[:, 1].numpy(),
        s=size,
        c=color,
        linewidths=0,
        alpha=0.78,
    )


def save_composite(results: list[dict[str, object]], output_path: Path) -> None:
    """Save a 3x4 qualitative comparison figure."""
    figure, axes = plt.subplots(
        nrows=len(results), ncols=4, figsize=(12.5, 3.35 * len(results)), dpi=180
    )
    if len(results) == 1:
        axes = axes.reshape(1, -1)

    column_titles = ["Input", "GT points", "Predicted points", "Density head"]
    for axis, title in zip(axes[0], column_titles):
        axis.set_title(title, fontsize=11, pad=6)

    for row_index, item in enumerate(results):
        image = item["image"]
        gt_points = item["gt_points"]
        pred_points = item["pred_points"]
        density_display = item["density_display"]
        image_name = str(item["image_name"])
        gt_count = int(item["gt_count"])
        pred_count = int(item["pred_count"])
        density_sum = float(item["density_sum"])

        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_ylabel(image_name, fontsize=9)

        axes[row_index, 1].imshow(image)
        draw_points(axes[row_index, 1], gt_points, color="#10b981", size=6.0)
        axes[row_index, 1].text(
            0.02,
            0.96,
            f"GT={gt_count}",
            transform=axes[row_index, 1].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="white",
            bbox={"facecolor": "#065f46", "alpha": 0.82, "pad": 2, "edgecolor": "none"},
        )

        axes[row_index, 2].imshow(image)
        draw_points(axes[row_index, 2], pred_points, color="#ef4444", size=6.0)
        axes[row_index, 2].text(
            0.02,
            0.96,
            f"Pred={pred_count}",
            transform=axes[row_index, 2].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="white",
            bbox={"facecolor": "#7f1d1d", "alpha": 0.82, "pad": 2, "edgecolor": "none"},
        )

        axes[row_index, 3].imshow(image)
        axes[row_index, 3].imshow(density_display.numpy(), cmap="magma", alpha=0.66)
        axes[row_index, 3].text(
            0.02,
            0.96,
            f"Density={density_sum:.1f}",
            transform=axes[row_index, 3].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="white",
            bbox={"facecolor": "#3b0764", "alpha": 0.82, "pad": 2, "edgecolor": "none"},
        )

        for col_index in range(4):
            axes[row_index, col_index].set_xticks([])
            axes[row_index, col_index].set_yticks([])

    figure.tight_layout(w_pad=0.5, h_pad=0.8)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("outputs/2026-04-25/22-51-51/.hydra/config.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/2026-04-25/22-51-51/checkpoints/best_mae.pth"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/shanghaitech/part_A_final/test_data"),
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=["IMG_136.jpg", "IMG_56.jpg", "IMG_150.jpg"],
        help="SHHA test image names to visualise.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures/paper"),
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cfg = OmegaConf.load(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(checkpoint.get("best_threshold", cfg.eval_counting.threshold))
    )

    model = build_model(cfg, training=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    results = []
    for image_name in args.images:
        image_path = args.data_root / "images" / image_name
        gt_path = args.data_root / "ground_truth" / f"GT_{Path(image_name).stem}.mat"
        results.append(infer_one(model, image_path, gt_path, threshold, device))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "fig_qualitative_shha_best.png"
    save_composite(results, figure_path)

    rows = [
        {
            "image": item["image_name"],
            "gt_count": item["gt_count"],
            "pred_count": item["pred_count"],
            "abs_error": abs(int(item["pred_count"]) - int(item["gt_count"])),
            "density_sum": round(float(item["density_sum"]), 4),
            "threshold": threshold,
        }
        for item in results
    ]
    pd.DataFrame(rows).to_csv(args.output_dir / "fig_qualitative_shha_best.csv", index=False)
    print(figure_path)


if __name__ == "__main__":
    main()