"""Batch prediction script — visualise crowd counts on scene images.

Adapted from run_test.py — logic unchanged.

Usage::

    python scripts/predict.py \\
        model.backbone=vgg16_bn \\
        predict.weight_path=checkpoints/SHTechA.pth \\
        predict.root_dir=./sha_a/test \\
        predict.output_dir=./pred_result \\
        predict.threshold=0.5
"""

from __future__ import annotations

import os

import cv2
import hydra
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from crowdcount.models import build_model
from crowdcount.utils.logging import logger, setup_logger


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logger(log_dir=".", log_file="predict.log")

    # predict-specific overrides can be passed as predict.xxx on CLI
    predict_cfg = OmegaConf.to_container(cfg, resolve=True)
    weight_path = predict_cfg.get("predict", {}).get(
        "weight_path", "weights/SHTechA.pth"
    )
    root_dir = predict_cfg.get("predict", {}).get("root_dir", "./sha_a/test")
    output_dir = predict_cfg.get("predict", {}).get("output_dir", "./pred_result")
    threshold = predict_cfg.get("predict", {}).get("threshold", 0.5)
    gpu_id = cfg.gpu_id

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, training=False)
    model.to(device)

    if weight_path and os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        logger.info(f"Loaded weights from {weight_path}")
    else:
        logger.warning(
            f"Weight file not found: {weight_path}. Using random initialisation."
        )

    model.eval()
    transform = standard_transforms.Compose(
        [
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    os.makedirs(output_dir, exist_ok=True)

    scene_folders = sorted(
        [
            f
            for f in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, f)) and f.startswith("scene_")
        ],
        key=lambda x: int(x.split("_")[-1]),
    )

    for scene_name in scene_folders:
        scene_path = os.path.join(root_dir, scene_name)
        jpg_file = next((f for f in os.listdir(scene_path) if f.endswith(".jpg")), None)
        if not jpg_file:
            logger.warning(f"No .jpg files found in {scene_path}, skipping.")
            continue

        img_path = os.path.join(scene_path, jpg_file)
        img_raw = Image.open(img_path).convert("RGB")
        width, height = img_raw.size
        new_width = width // 128 * 128
        new_height = height // 128 * 128
        img_raw = img_raw.resize((new_width, new_height), cv2.INTER_CUBIC)

        img = transform(img_raw)
        samples = torch.Tensor(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(samples)

        outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], dim=-1)[
            :, :, 1
        ][0]
        outputs_points = outputs["pred_points"][0]
        valid_mask = outputs_scores > threshold
        points = outputs_points[valid_mask].detach().cpu().numpy().tolist()
        predict_cnt = int(valid_mask.sum().item())

        img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        for p in points:
            cv2.circle(img_to_draw, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        text = str(predict_cnt)
        font_face = cv2.FONT_HERSHEY_TRIPLEX
        font_scale, thickness = 2.0, 3
        H, W, _ = img_to_draw.shape
        (text_w, _), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
        cv2.putText(
            img_to_draw,
            text,
            (W - text_w - 10, H - 10),
            font_face,
            font_scale,
            (255, 255, 255),
            thickness,
        )

        base_name = os.path.splitext(jpg_file)[0]
        out_path = os.path.join(output_dir, f"{base_name}_pred.jpg")
        cv2.imwrite(out_path, img_to_draw)
        logger.info(f"Processed {img_path} → count={predict_cnt}  saved to {out_path}")


if __name__ == "__main__":
    main()
