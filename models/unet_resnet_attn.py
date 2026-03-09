import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

from models.attention import ASPP, ConvBNReLU, scSE


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_scse: bool = True):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)
        self.attn = scSE(out_channels) if use_scse else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.attn(x)


class UNetResNet34Attn(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 3,
        pretrained: bool = True,
        use_scse: bool = True,
        use_aspp: bool = True,
    ):
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

        self.skip1 = scSE(64) if use_scse else nn.Identity()
        self.skip2 = scSE(64) if use_scse else nn.Identity()
        self.skip3 = scSE(128) if use_scse else nn.Identity()
        self.skip4 = scSE(256) if use_scse else nn.Identity()

        self.bottleneck = ASPP(512, 512) if use_aspp else ConvBNReLU(512, 512)

        self.dec4 = DecoderBlock(512, 256, 256, use_scse=use_scse)
        self.dec3 = DecoderBlock(256, 128, 128, use_scse=use_scse)
        self.dec2 = DecoderBlock(128, 64, 64, use_scse=use_scse)
        self.dec1 = DecoderBlock(64, 64, 64, use_scse=use_scse)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, 64),
        )
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.stem(x)
        s2 = self.layer1(self.maxpool(s1))
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        x = self.layer4(s4)

        x = self.bottleneck(x)
        x = self.dec4(x, self.skip4(s4))
        x = self.dec3(x, self.skip3(s3))
        x = self.dec2(x, self.skip2(s2))
        x = self.dec1(x, self.skip1(s1))
        x = self.final_up(x)
        return self.head(x)
