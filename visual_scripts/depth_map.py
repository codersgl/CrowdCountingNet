import cv2
import torch
# import numpy as np

from crowdcount.plugins.depth_anything_v2.dpt import DepthAnythingV2


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}

encoder = "vitl"  # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(
    torch.load(f"checkpoints/depth_anything_v2_{encoder}.pth", map_location="cpu")
)
model = model.to(DEVICE).eval()

raw_img = cv2.imread("data/shanghaitech/part_A_final/test_data/images/IMG_2.jpg")


# cv2.imshow("raw_img", raw_img)

depth = model.infer_image(raw_img)  # HxW raw depth map in numpy
# depth_min = depth.min()
# depth_max = depth.max()
# if depth_max - depth_min > 0:
#     depth_norm = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
# else:
#     depth_norm = np.zeros_like(depth, dtype=np.uint8)
#
# print(depth_norm.shape)
#
#
# cv2.imshow("depth_norm", depth_norm)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
