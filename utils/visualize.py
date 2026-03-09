from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image
import torch

from datasets.whdld_dataset import CLASS_NAMES, ID2COLOR


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, rgb in ID2COLOR.items():
        color[mask == class_id] = np.array(rgb, dtype=np.uint8)
    return color


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image * STD + MEAN, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def overlay(image: np.ndarray, color_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = image.astype(np.float32) * (1 - alpha) + color_mask.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def save_visualizations(batch, preds: torch.Tensor, save_dir: str, max_items: int = 4):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    images = batch["image"]
    masks = batch["mask"]
    names: List[str] = batch["name"]
    preds = preds.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()

    for i in range(min(max_items, len(names))):
        image = denormalize_image(images[i])
        gt_color = mask_to_color(masks[i])
        pred_color = mask_to_color(preds[i])

        Image.fromarray(image).save(save_path / f"{names[i]}_image.png")
        Image.fromarray(gt_color).save(save_path / f"{names[i]}_gt.png")
        Image.fromarray(pred_color).save(save_path / f"{names[i]}_pred.png")
        Image.fromarray(overlay(image, pred_color)).save(save_path / f"{names[i]}_overlay.png")
