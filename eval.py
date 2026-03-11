import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from datasets.transforms import get_transforms
from datasets.whdld_dataset import WHDLDataset
from losses import CEDiceLoss
from models.unet_resnet_attn import UNetResNet34Attn
from utils.metrics import SegmentationMetric
from utils.visualize import save_visualizations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="runs/exp/checkpoints/best.pth")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    return parser.parse_args()


def to_json_serializable(obj):
    """
    把 numpy / torch 类型递归转成可被 json 序列化的 python 原生类型
    """
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [to_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    else:
        return obj


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_dir = Path(cfg["data"]["split_dir"])
    split_file = split_dir / f"{args.split}.txt"

    transforms = get_transforms(tuple(cfg["data"]["image_size"]))
    dataset = WHDLDataset(cfg["data"]["root"], str(split_file), transform=transforms["eval"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    model = UNetResNet34Attn(
        num_classes=cfg["num_classes"],
        in_channels=cfg["model"]["in_channels"],
        pretrained=False,
        use_scse=cfg["model"]["use_scse"],
        use_aspp=cfg["model"]["use_aspp"],
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = CEDiceLoss(
        num_classes=cfg["num_classes"],
        ce_weight=cfg["loss"]["ce_weight"],
        dice_weight=cfg["loss"]["dice_weight"],
    )
    metric = SegmentationMetric(cfg["num_classes"])

    total_loss = 0.0
    vis_dir = Path(cfg["runs"]["root"]) / "exp" / f"eval_{args.split}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    saved = False

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval-{args.split}"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)
            preds = torch.argmax(logits, dim=1)

            metric.update(preds, masks)
            total_loss += loss.item() * images.size(0)

            if not saved:
                save_visualizations(batch, preds, str(vis_dir), max_items=8)
                saved = True

    results = metric.compute()
    results["loss"] = total_loss / len(loader.dataset)

    # 转成可 JSON 序列化格式
    results = to_json_serializable(results)

    print(json.dumps(results, indent=2, ensure_ascii=False))

    with open(vis_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
