import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from datasets.transforms import get_transforms
from datasets.whdld_dataset import CLASS_NAMES, WHDLDataset
from losses import CEDiceBoundaryDeepSupervisionLoss
from models.unet_resnet_attn import UNetResNet34Attn
from utils.metrics import SegmentationMetric
from utils.seed import set_seed
from utils.split import make_split
from utils.visualize import save_visualizations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def validate(model, loader, criterion, metric, device, save_dir=None, vis_samples=4):
    model.eval()
    metric.reset()
    total_loss = 0.0
    total_main_loss = 0.0
    total_ds_loss = 0.0
    total_boundary_loss = 0.0
    saved = False

    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        outputs = model(images)
        loss, loss_dict = criterion(outputs, masks)

        logits = outputs["main"]
        preds = torch.argmax(logits, dim=1)

        metric.update(preds, masks)
        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_main_loss += float(loss_dict["loss_main"]) * batch_size
        total_ds_loss += float(loss_dict["loss_ds"]) * batch_size
        total_boundary_loss += float(loss_dict["loss_boundary"]) * batch_size

        if save_dir is not None and not saved:
            save_visualizations(batch, preds, save_dir=save_dir, max_items=vis_samples)
            saved = True

    results = metric.compute()
    results["loss"] = total_loss / len(loader.dataset)
    results["loss_main"] = total_main_loss / len(loader.dataset)
    results["loss_ds"] = total_ds_loss / len(loader.dataset)
    results["loss_boundary"] = total_boundary_loss / len(loader.dataset)
    return results


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, amp):
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_ds_loss = 0.0
    total_boundary_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=amp):
            outputs = model(images)
            loss, loss_dict = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_main_loss += float(loss_dict["loss_main"]) * batch_size
        total_ds_loss += float(loss_dict["loss_ds"]) * batch_size
        total_boundary_loss += float(loss_dict["loss_boundary"]) * batch_size

    num_samples = len(loader.dataset)
    return {
        "loss": total_loss / num_samples,
        "loss_main": total_main_loss / num_samples,
        "loss_ds": total_ds_loss / num_samples,
        "loss_boundary": total_boundary_loss / num_samples,
    }


def to_serializable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
        return v.item()
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().tolist()
    return v


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    runs_root = Path(cfg["runs"]["root"])
    runs_root.mkdir(parents=True, exist_ok=True)

    split_dir = Path(cfg["data"]["split_dir"])
    split_dir.mkdir(parents=True, exist_ok=True)

    if not (split_dir / "train.txt").exists():
        make_split(cfg["data"]["root"], str(split_dir), seed=cfg["seed"])

    exp_dir = runs_root / "exp"
    exp_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = exp_dir / "checkpoints"
    vis_dir = exp_dir / "visualizations"
    log_dir = exp_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "used_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = get_transforms(tuple(cfg["data"]["image_size"]))

    train_ds = WHDLDataset(
        cfg["data"]["root"],
        str(split_dir / "train.txt"),
        transform=transforms["train"]
    )
    val_ds = WHDLDataset(
        cfg["data"]["root"],
        str(split_dir / "val.txt"),
        transform=transforms["eval"]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    model = UNetResNet34Attn(
        num_classes=cfg["num_classes"],
        in_channels=cfg["model"]["in_channels"],
        pretrained=cfg["model"]["pretrained"],
        use_scse=cfg["model"]["use_scse"],
        use_aspp=cfg["model"]["use_aspp"],
        use_deep_supervision=cfg["model"].get("use_deep_supervision", True),
        use_boundary_branch=cfg["model"].get("use_boundary_branch", True),
    ).to(device)

    criterion = CEDiceBoundaryDeepSupervisionLoss(
        num_classes=cfg["num_classes"],
        ce_weight=cfg["loss"]["ce_weight"],
        dice_weight=cfg["loss"]["dice_weight"],
        ds_weight=cfg["loss"].get("ds_weight", 0.4),
        boundary_weight=cfg["loss"].get("boundary_weight", 0.2),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["train"]["epochs"],
        eta_min=cfg["scheduler"]["min_lr"]
    )
    scaler = GradScaler("cuda", enabled=cfg["train"]["amp"])
    metric = SegmentationMetric(cfg["num_classes"])

    history = []
    best_miou = -1.0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, cfg["train"]["amp"]
        )

        val_metrics = validate(
            model,
            val_loader,
            criterion,
            metric,
            device,
            save_dir=str(vis_dir / f"epoch_{epoch:03d}"),
            vis_samples=cfg["train"]["vis_samples"],
        )

        scheduler.step()

        val_metrics = {k: to_serializable(v) for k, v in val_metrics.items()}
        train_stats = {k: to_serializable(v) for k, v in train_stats.items()}

        record = {
            "epoch": int(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_stats["loss"]),
            "train_loss_main": float(train_stats["loss_main"]),
            "train_loss_ds": float(train_stats["loss_ds"]),
            "train_loss_boundary": float(train_stats["loss_boundary"]),
            **val_metrics,
        }

        history.append(record)

        with open(log_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        print(
            f"Epoch [{epoch:03d}/{cfg['train']['epochs']:03d}] "
            f"train_loss={train_stats['loss']:.4f} "
            f"(main={train_stats['loss_main']:.4f}, ds={train_stats['loss_ds']:.4f}, boundary={train_stats['loss_boundary']:.4f}) "
            f"val_loss={val_metrics['loss']:.4f} "
            f"(main={val_metrics['loss_main']:.4f}, ds={val_metrics['loss_ds']:.4f}, boundary={val_metrics['loss_boundary']:.4f}) "
            f"mIoU={val_metrics['mIoU']:.4f} "
            f"mPA={val_metrics['mPA']:.4f} "
            f"Precision={val_metrics['Precision']:.4f} "
            f"Recall={val_metrics['Recall']:.4f}"
        )

        current_miou = float(val_metrics["mIoU"])
        if current_miou > best_miou:
            best_miou = current_miou

        checkpoint = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if cfg["train"]["amp"] else None,
            "best_miou": float(best_miou),
            "config": cfg,
            "class_names": CLASS_NAMES,
        }

        torch.save(checkpoint, ckpt_dir / "last.pth")

        if current_miou == best_miou:
            torch.save(checkpoint, ckpt_dir / "best.pth")
            print(f"[*] Best model updated at epoch {epoch}, mIoU={best_miou:.4f}")

    print(f"Training finished. Best mIoU: {best_miou:.4f}")
    print(f"All outputs saved under: {exp_dir}")


if __name__ == "__main__":
    main()
