from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from methods import clahe_enhance, gamma_enhance, retinex_enhance


EnhanceFunction = Callable[[np.ndarray], np.ndarray]


METHODS: dict[str, EnhanceFunction] = {
    "gamma": gamma_enhance,
    "clahe": clahe_enhance,
    "retinex": retinex_enhance,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run traditional low-light image enhancement methods.",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=sorted(METHODS.keys()),
        help="Enhancement method to run.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("results/traditional"),
        type=Path,
        help="Directory for enhanced output images.",
    )
    return parser.parse_args()


def build_output_path(input_path: Path, method: str, output_dir: Path) -> Path:
    suffix = input_path.suffix if input_path.suffix else ".png"
    return output_dir / f"{input_path.stem}_{method}{suffix}"


def load_image(image_path: Path) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Input path is not a file: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image with OpenCV: {image_path}")
    return image


def save_image(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise ValueError(f"Failed to save image: {output_path}")


def run_inference(method: str, input_path: Path, output_dir: Path) -> Path:
    image = load_image(input_path)
    enhanced = METHODS[method](image)
    output_path = build_output_path(input_path, method, output_dir)
    save_image(enhanced, output_path)
    return output_path


def main() -> None:
    args = parse_args()
    output_path = run_inference(args.method, args.input, args.output_dir)
    print(f"Saved enhanced image to: {output_path}")


if __name__ == "__main__":
    main()
