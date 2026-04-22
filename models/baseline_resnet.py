import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class BaselineResNet34UNet(nn.Module):
    """
    基线 + ResNet34 编码器。
    仅输出主分割分支，兼容现有 loss 接口：
    {
        "main": logits,
        "ds": [],
        "boundary": None,
    }
    """

    def __init__(self, num_classes: int = 6, in_channels: int = 3, pretrained: bool = True):
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

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2, 64
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # /4, 64
        self.layer2 = backbone.layer2  # /8, 128
        self.layer3 = backbone.layer3  # /16, 256
        self.layer4 = backbone.layer4  # /32, 512

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, 64),
        )
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        s1 = self.stem(x)
        s2 = self.layer1(self.maxpool(s1))
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        x = self.layer4(s4)

        d4 = self.dec4(x, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        out = self.final_up(d1)
        logits = self.head(out)

        return {
            "main": logits,
            "ds": [],
            "boundary": None,
        }
