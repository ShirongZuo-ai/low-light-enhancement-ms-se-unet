from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import UNet
from utils.dataset import LOLDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check LOL dataset loading and U-Net input/output tensor sizes.",
    )
    parser.add_argument(
        "--data-root",
        default=Path("data/LOL"),
        type=Path,
        help="Path to LOL dataset root directory.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "test"),
        help="Dataset split to check.",
    )
    parser.add_argument(
        "--batch-size",
        default=2,
        type=int,
        help="Batch size used for the shape check.",
    )
    parser.add_argument(
        "--image-size",
        default=256,
        type=int,
        help="Square resize size for both low and high images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = LOLDataset(
        root=args.data_root,
        split=args.split,
        image_size=(args.image_size, args.image_size),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    low_images, high_images, filenames = next(iter(loader))
    model = UNet(in_channels=3, out_channels=3)
    model.eval()

    with torch.no_grad():
        outputs = model(low_images)

    print(f"Dataset split: {args.split}")
    print(f"Number of pairs: {len(dataset)}")
    print(f"Example filenames: {list(filenames)[:5]}")
    print(f"Low image batch shape: {tuple(low_images.shape)}")
    print(f"High image batch shape: {tuple(high_images.shape)}")
    print(f"Model output shape: {tuple(outputs.shape)}")
    print(f"Output value range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")

    expected_shape = low_images.shape
    if high_images.shape != expected_shape:
        raise RuntimeError(
            f"High image shape mismatch: expected {tuple(expected_shape)}, "
            f"got {tuple(high_images.shape)}"
        )
    if outputs.shape != expected_shape:
        raise RuntimeError(
            f"Model output shape mismatch: expected {tuple(expected_shape)}, "
            f"got {tuple(outputs.shape)}"
        )

    print("Dataset and model shape check passed.")


if __name__ == "__main__":
    main()
