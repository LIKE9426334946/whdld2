from models.baseline_resnet import BaselineResNet34UNet
from models.unet_resnet_attn import UNetResNet34Attn


def build_model(model_cfg: dict, num_classes: int):
    name = model_cfg.get("name", "baseline_resnet34")

    if name == "baseline_resnet34":
        return BaselineResNet34UNet(
            num_classes=num_classes,
            in_channels=model_cfg.get("in_channels", 3),
            pretrained=model_cfg.get("pretrained", True),
        )

    if name == "resnet34_attn":
        return UNetResNet34Attn(
            num_classes=num_classes,
            in_channels=model_cfg.get("in_channels", 3),
            pretrained=model_cfg.get("pretrained", True),
            use_scse=True,
            use_aspp=False,
            use_deep_supervision=model_cfg.get("use_deep_supervision", True),
            use_boundary_branch=model_cfg.get("use_boundary_branch", True),
        )

    if name == "resnet34_attn_aspp":
        return UNetResNet34Attn(
            num_classes=num_classes,
            in_channels=model_cfg.get("in_channels", 3),
            pretrained=model_cfg.get("pretrained", True),
            use_scse=True,
            use_aspp=True,
            use_deep_supervision=model_cfg.get("use_deep_supervision", True),
            use_boundary_branch=model_cfg.get("use_boundary_branch", True),
        )

    raise ValueError(f"Unsupported model.name: {name}")
