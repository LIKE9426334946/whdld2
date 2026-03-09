import argparse
import random
from pathlib import Path


def make_split(data_root: str, output_dir: str, train_ratio=0.8, val_ratio=0.1, seed=42):
    data_root = Path(data_root)
    image_dir = data_root / "image/Images"
    mask_dir = data_root / "imagesPNG/ImagesPNG"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_names = sorted([p.stem for p in image_dir.glob("*.jpg")])
    mask_names = {p.stem for p in mask_dir.glob("*.png")}
    names = [name for name in image_names if name in mask_names]

    if len(names) == 0:
        raise RuntimeError("No matched .jpg/.png pairs found under dataset root.")

    random.Random(seed).shuffle(names)
    n = len(names)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_names = names[:n_train]
    val_names = names[n_train:n_train + n_val]
    test_names = names[n_train + n_val:]

    for split_name, split_names in {
        "train": train_names,
        "val": val_names,
        "test": test_names,
    }.items():
        with open(output_dir / f"{split_name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(split_names))

    print(f"Total: {n}, train: {len(train_names)}, val: {len(val_names)}, test: {len(test_names)}")
    print(f"Split files saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/whdld")
    parser.add_argument("--output_dir", type=str, default="runs/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    make_split(args.data_root, args.output_dir, seed=args.seed)
