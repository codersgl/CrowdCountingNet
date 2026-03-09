"""VGG backbone variants.

Adapted from the original torchvision VGG source.
Pretrained weights are loaded via the torchvision weights API instead of
local file paths.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]


class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True
    ):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm: bool = False, sync: bool = False) -> nn.Sequential:
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                if sync:
                    layers += [conv2d, nn.SyncBatchNorm(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(
    arch: str, cfg: str, batch_norm: bool, pretrained: bool, sync: bool = False
) -> "VGG":
    """Build VGG model, loading pretrained weights via torchvision when requested."""
    if pretrained:
        model = VGG(
            make_layers(cfgs[cfg], batch_norm=batch_norm, sync=sync), init_weights=False
        )
        # Load corresponding torchvision pretrained weights into just the features component
        _weights_map = {
            "vgg11": tv_models.VGG11_Weights.IMAGENET1K_V1,
            "vgg11_bn": tv_models.VGG11_BN_Weights.IMAGENET1K_V1,
            "vgg13": tv_models.VGG13_Weights.IMAGENET1K_V1,
            "vgg13_bn": tv_models.VGG13_BN_Weights.IMAGENET1K_V1,
            "vgg16": tv_models.VGG16_Weights.IMAGENET1K_V1,
            "vgg16_bn": tv_models.VGG16_BN_Weights.IMAGENET1K_V1,
            "vgg19": tv_models.VGG19_Weights.IMAGENET1K_V1,
            "vgg19_bn": tv_models.VGG19_BN_Weights.IMAGENET1K_V1,
        }
        weights = _weights_map[arch]
        tv_model = getattr(tv_models, arch)(weights=weights)
        model.features.load_state_dict(tv_model.features.state_dict())
    else:
        model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, sync=sync))
    return model


def vgg11(pretrained: bool = False, **kwargs) -> VGG:
    return _vgg("vgg11", "A", False, pretrained)


def vgg11_bn(pretrained: bool = False, **kwargs) -> VGG:
    return _vgg("vgg11_bn", "A", True, pretrained)


def vgg13(pretrained: bool = False, **kwargs) -> VGG:
    return _vgg("vgg13", "B", False, pretrained)


def vgg13_bn(pretrained: bool = False, **kwargs) -> VGG:
    return _vgg("vgg13_bn", "B", True, pretrained)


def vgg16(pretrained: bool = False, **kwargs) -> VGG:
    return _vgg("vgg16", "D", False, pretrained)


def vgg16_bn(pretrained: bool = False, sync: bool = False, **kwargs) -> VGG:
    return _vgg("vgg16_bn", "D", True, pretrained, sync=sync)


def vgg19(pretrained: bool = False, **kwargs) -> VGG:
    return _vgg("vgg19", "E", False, pretrained)


def vgg19_bn(pretrained: bool = False, **kwargs) -> VGG:
    return _vgg("vgg19_bn", "E", True, pretrained)
