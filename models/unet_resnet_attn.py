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
        use_deep_supervision: bool = True,
        use_boundary_branch: bool = True,
    ):
        super().__init__()
        self.use_deep_supervision = use_deep_supervision
        self.use_boundary_branch = use_boundary_branch

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

        # Encoder
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2, 64
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # /4, 64
        self.layer2 = backbone.layer2  # /8, 128
        self.layer3 = backbone.layer3  # /16, 256
        self.layer4 = backbone.layer4  # /32, 512

        # Skip attention
        self.skip1 = scSE(64) if use_scse else nn.Identity()
        self.skip2 = scSE(64) if use_scse else nn.Identity()
        self.skip3 = scSE(128) if use_scse else nn.Identity()
        self.skip4 = scSE(256) if use_scse else nn.Identity()

        # Bottleneck
        self.bottleneck = ASPP(512, 512) if use_aspp else ConvBNReLU(512, 512)

        # Decoder
        self.dec4 = DecoderBlock(512, 256, 256, use_scse=use_scse)  # /16
        self.dec3 = DecoderBlock(256, 128, 128, use_scse=use_scse)  # /8
        self.dec2 = DecoderBlock(128, 64, 64, use_scse=use_scse)    # /4
        self.dec1 = DecoderBlock(64, 64, 64, use_scse=use_scse)     # /2

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, 64),
        )
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

        # Deep Supervision heads
        if self.use_deep_supervision:
            self.ds4_head = nn.Conv2d(256, num_classes, kernel_size=1)  # d4
            self.ds3_head = nn.Conv2d(128, num_classes, kernel_size=1)  # d3
            self.ds2_head = nn.Conv2d(64, num_classes, kernel_size=1)   # d2
            self.ds1_head = nn.Conv2d(64, num_classes, kernel_size=1)   # d1

        # Boundary Branch
        if self.use_boundary_branch:
            self.boundary_head = nn.Sequential(
                ConvBNReLU(64, 32),
                nn.Conv2d(32, 1, kernel_size=1)
            )

    def forward(self, x: torch.Tensor):
        # ===== Encoder =====
        s1 = self.stem(x)                  # /2, 64
        s2 = self.layer1(self.maxpool(s1)) # /4, 64
        s3 = self.layer2(s2)               # /8, 128
        s4 = self.layer3(s3)               # /16, 256
        x = self.layer4(s4)                # /32, 512

        # ===== Bottleneck =====
        x = self.bottleneck(x)

        # ===== Decoder =====
        d4 = self.dec4(x, self.skip4(s4))  # /16, 256
        d3 = self.dec3(d4, self.skip3(s3)) # /8, 128
        d2 = self.dec2(d3, self.skip2(s2)) # /4, 64
        d1 = self.dec1(d2, self.skip1(s1)) # /2, 64

        out = self.final_up(d1)            # /1, 64
        logits = self.head(out)            # /1, num_classes

        outputs = {"main": logits}

        # ===== Deep Supervision =====
        if self.use_deep_supervision:
            ds4 = self.ds4_head(d4)
            ds3 = self.ds3_head(d3)
            ds2 = self.ds2_head(d2)
            ds1 = self.ds1_head(d1)

            target_size = logits.shape[-2:]
            ds4 = F.interpolate(ds4, size=target_size, mode="bilinear", align_corners=False)
            ds3 = F.interpolate(ds3, size=target_size, mode="bilinear", align_corners=False)
            ds2 = F.interpolate(ds2, size=target_size, mode="bilinear", align_corners=False)
            ds1 = F.interpolate(ds1, size=target_size, mode="bilinear", align_corners=False)

            outputs["ds"] = [ds1, ds2, ds3, ds4]
        else:
            outputs["ds"] = []

        # ===== Boundary Branch =====
        if self.use_boundary_branch:
            boundary = self.boundary_head(d1)  # /2, 1
            boundary = F.interpolate(boundary, size=logits.shape[-2:], mode="bilinear", align_corners=False)
            outputs["boundary"] = boundary
        else:
            outputs["boundary"] = None

        return outputs
