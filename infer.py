import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import yaml
from torchvision.transforms import functional as TF
import torchvision.transforms as T

from datasets.whdld_dataset import ID2COLOR
from models.deeplabv3plus import DeepLabV3Plus
from utils.visualize import overlay


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="runs/exp/checkpoints/best.pth")
    parser.add_argument("--input", type=str, required=True, help="single image path or a directory of jpg images")
    parser.add_argument("--output_dir", type=str, default="runs/infer")
    return parser.parse_args()


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, rgb in ID2COLOR.items():
        color[mask == class_id] = np.array(rgb, dtype=np.uint8)
    return color


def preprocess(image: Image.Image, size):
    resized = TF.resize(image, size, interpolation=T.InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, MEAN, STD)
    return tensor.unsqueeze(0)


def predict(model, image: Image.Image, size, device):
    x = preprocess(image, size).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    return pred


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3Plus(
        num_classes=cfg["num_classes"],
        in_channels=cfg["model"]["in_channels"],
        pretrained=False,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        image_paths = sorted(list(input_path.glob("*.jpg")))
    else:
        image_paths = [input_path]

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        pred = predict(model, image, tuple(cfg["data"]["image_size"]), device)
        pred_color = mask_to_color(pred)
        base = image.resize(tuple(cfg["data"]["image_size"])[::-1])
        image_np = np.array(base, dtype=np.uint8)
        over = overlay(image_np, pred_color)

        stem = image_path.stem
        Image.fromarray(pred).save(output_dir / f"{stem}_mask.png")
        Image.fromarray(pred_color).save(output_dir / f"{stem}_color.png")
        Image.fromarray(over).save(output_dir / f"{stem}_overlay.png")

    print(f"Inference outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
