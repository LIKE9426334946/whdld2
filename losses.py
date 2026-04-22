import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C, H, W]
        targets: [B, H, W]
        """
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        union = torch.sum(probs + one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class SegLoss(nn.Module):
    """
    单个分割输出的 CE + Dice
    """
    def __init__(self, num_classes: int, ce_weight: float = 1.0, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes=num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice


class BoundaryLoss(nn.Module):
    """
    边界分支损失
    输入:
        pred:   [B, 1, H, W]  (logits)
        target: [B, H, W]     (类别标签)
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    @staticmethod
    def mask_to_boundary(target: torch.Tensor) -> torch.Tensor:
        """
        用邻域差异生成简单边界图
        target: [B, H, W]
        return: [B, 1, H, W] float
        """
        target = target.unsqueeze(1).float()  # [B,1,H,W]

        # 通过局部最大/最小池化判断邻域内是否存在类别变化
        max_pool = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
        min_pool = -F.max_pool2d(-target, kernel_size=3, stride=1, padding=1)
        boundary = (max_pool != min_pool).float()

        return boundary

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        boundary_target = self.mask_to_boundary(target)
        return self.bce(pred, boundary_target)


class CEDiceBoundaryDeepSupervisionLoss(nn.Module):
    """
    总损失：
        main loss
      + ds_weight * deep supervision loss
      + boundary_weight * boundary loss

    适配模型输出:
    outputs = {
        "main": main_logits,
        "ds": [ds1, ds2, ds3, ds4],
        "boundary": boundary_logits
    }
    """
    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 1.0,
        dice_weight: float = 0.5,
        ds_weight: float = 0.4,
        boundary_weight: float = 0.2,
    ):
        super().__init__()
        self.seg_loss = SegLoss(
            num_classes=num_classes,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
        )
        self.boundary_loss = BoundaryLoss()
        self.ds_weight = ds_weight
        self.boundary_weight = boundary_weight

    def forward(self, outputs, targets: torch.Tensor):
        """
        outputs:
            {
                "main": Tensor [B,C,H,W],
                "ds": list[Tensor [B,C,H,W], ...],
                "boundary": Tensor [B,1,H,W] or None
            }
        targets:
            [B,H,W]
        """
        main_logits = outputs["main"]
        ds_logits = outputs.get("ds", [])
        boundary_logits = outputs.get("boundary", None)

        # 主输出损失
        loss_main = self.seg_loss(main_logits, targets)

        # Deep supervision 损失
        loss_ds = torch.tensor(0.0, device=targets.device)
        if ds_logits is not None and len(ds_logits) > 0:
            ds_total = 0.0
            for ds_pred in ds_logits:
                ds_total = ds_total + self.seg_loss(ds_pred, targets)
            loss_ds = ds_total / len(ds_logits)

        # Boundary loss
        loss_boundary = torch.tensor(0.0, device=targets.device)
        if boundary_logits is not None:
            loss_boundary = self.boundary_loss(boundary_logits, targets)

        total_loss = loss_main + self.ds_weight * loss_ds + self.boundary_weight * loss_boundary

        loss_dict = {
            "loss_total": total_loss,
            "loss_main": loss_main.detach(),
            "loss_ds": loss_ds.detach(),
            "loss_boundary": loss_boundary.detach(),
        }
        return total_loss, loss_dict


class CEDiceLoss(nn.Module):
    def __init__(self, num_classes: int, ce_weight: float = 1.0, dice_weight: float = 0.5):
        super().__init__()
        self.seg_loss = SegLoss(
            num_classes=num_classes,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.seg_loss(logits, targets)
