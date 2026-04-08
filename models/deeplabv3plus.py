import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

from models.attention import ASPP, ConvBNReLU


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 6, pretrained: bool = False):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)

        if in_channels != 3:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.aspp = ASPP(512, 256)
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            ConvBNReLU(256 + 48, 256, kernel_size=3, padding=1),
            ConvBNReLU(256, 256, kernel_size=3, padding=1),
        )
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]

        x = self.stem(x)
        x = self.maxpool(x)
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)
        x = F.interpolate(x, size=low_level.shape[-2:], mode="bilinear", align_corners=False)

        low_level = self.low_level_proj(low_level)
        x = torch.cat([x, low_level], dim=1)
        x = self.decoder(x)
        x = self.classifier(x)

        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
