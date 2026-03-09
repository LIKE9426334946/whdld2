from typing import Callable, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
import torchvision.transforms as T


class SegCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: Image.Image, mask: Image.Image):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class Resize:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.resize(image, self.size, interpolation=T.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=T.InterpolationMode.NEAREST)
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, mask: Image.Image):
        if np.random.rand() < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, mask: Image.Image):
        if np.random.rand() < self.p:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask


class RandomRotate90:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, mask: Image.Image):
        if np.random.rand() < self.p:
            k = int(np.random.choice([1, 2, 3]))
            image = TF.rotate(image, 90 * k)
            mask = TF.rotate(mask, 90 * k)
        return image, mask


class ColorJitterOnlyImage:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02):
        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, image: Image.Image, mask: Image.Image):
        return self.jitter(image), mask


class ToTensorAndNormalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        return image, mask


def get_transforms(image_size=(256, 256)) -> Dict[str, Callable]:
    train_tf = SegCompose(
        [
            Resize(image_size),
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.1),
            RandomRotate90(0.3),
            ColorJitterOnlyImage(),
            ToTensorAndNormalize(),
        ]
    )
    eval_tf = SegCompose([Resize(image_size), ToTensorAndNormalize()])
    return {"train": train_tf, "eval": eval_tf}
