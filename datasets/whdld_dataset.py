from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import yaml

CLASS_NAMES = [
    "vegetation",
    "water",
    "road",
    "building",
    "pavement",
    "bare_soil",
]

PALETTE = {
    (0, 255, 0): 0,
    (0, 0, 255): 1,
    (255, 255, 0): 2,
    (255, 0, 0): 3,
    (192, 192, 0): 4,
    (128, 128, 128): 5,
}

ID2COLOR = {v: k for k, v in PALETTE.items()}

def load_cfg(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
cfg = load_cfg()
class WHDLDataset(Dataset):
    def __init__(self, root: str, split_file: str, transform=None):
        self.root = Path(root)
        self.image_dir = self.root / cfg["images_dir"]
        self.mask_dir = self.root / cfg["masks_dir"]
        self.transform = transform
        with open(split_file, "r", encoding="utf-8") as f:
            self.names = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.names)

    def _rgb_to_mask(self, mask_rgb: np.ndarray) -> np.ndarray:
        mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
        for color, class_id in PALETTE.items():
            color_arr = np.array(color, dtype=np.uint8)
            match = np.all(mask_rgb == color_arr, axis=-1)
            mask[match] = class_id
        return mask

    def __getitem__(self, idx: int):
        name = self.names[idx]
        image_path = self.image_dir / f"{name}.jpg"
        mask_path = self.mask_dir / f"{name}.png"

        image = Image.open(image_path).convert("RGB")
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
        mask = Image.fromarray(self._rgb_to_mask(mask_rgb))

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return {
            "image": image,
            "mask": mask.long(),
            "name": name,
        }
