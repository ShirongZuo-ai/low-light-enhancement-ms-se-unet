from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
COLUMNS = [
    ("Low", Path("data/LOL/test/low")),
    ("Gamma", Path("results/traditional_eval/gamma")),
    ("CLAHE", Path("results/traditional_eval/clahe")),
    ("Retinex", Path("results/traditional_eval/retinex")),
    ("U-Net", Path("results/unet")),
    ("MS-SE-U-Net", Path("results/ms_se_unet")),
    ("Ground Truth", Path("data/LOL/test/high")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report-ready comparison figures.")
    parser.add_argument("--data-root", type=Path, default=Path("data/LOL"), help="Root directory of the LOL dataset.")
    parser.add_argument("--unet-dir", type=Path, default=Path("results/unet"), help="Directory of U-Net output images.")
    parser.add_argument(
        "--ms-se-unet-dir",
        type=Path,
        default=Path("results/ms_se_unet"),
        help="Directory of MS-SE-U-Net output images.",
    )
    parser.add_argument(
        "--traditional-root",
        type=Path,
        default=Path("results/traditional_eval"),
        help="Root directory of traditional method output images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/comparison_figures"),
        help="Directory for generated comparison figures.",
    )
    parser.add_argument("--num-images", type=int, default=5, help="Number of test images to include.")
    parser.add_argument("--filenames", nargs="*", default=None, help="Optional explicit filenames to plot.")
    return parser.parse_args()


def collect_filenames(low_dir: Path, num_images: int, filenames: list[str] | None) -> list[str]:
    if filenames:
        return filenames
    paths = [
        path
        for path in sorted(low_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return [path.name for path in paths[:num_images]]


def read_bgr(path: Path, display_size: tuple[int, int]) -> object:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image with OpenCV: {path}")
    return cv2.resize(image_bgr, display_size, interpolation=cv2.INTER_AREA)


def build_columns(
    data_root: Path,
    traditional_root: Path,
    unet_dir: Path,
    ms_se_unet_dir: Path,
) -> list[tuple[str, Path]]:
    return [
        ("Low", data_root / "test" / "low"),
        ("Gamma", traditional_root / "gamma"),
        ("CLAHE", traditional_root / "clahe"),
        ("Retinex", traditional_root / "retinex"),
        ("U-Net", unet_dir),
        ("MS-SE-U-Net", ms_se_unet_dir),
        ("Ground Truth", data_root / "test" / "high"),
    ]


def make_figure(filename: str, columns: list[tuple[str, Path]], output_dir: Path) -> Path:
    low_image = cv2.imread(str(columns[0][1] / filename), cv2.IMREAD_COLOR)
    if low_image is None:
        raise ValueError(f"Failed to read low image: {columns[0][1] / filename}")
    height, width = low_image.shape[:2]
    display_width = 256
    display_height = max(1, round(height * display_width / width))
    display_size = (display_width, display_height)

    margin = 24
    gap = 12
    title_height = 48
    header_height = 36
    canvas_width = margin * 2 + len(columns) * display_width + (len(columns) - 1) * gap
    canvas_height = margin * 2 + title_height + header_height + display_height
    canvas = 255 * np.ones((canvas_height, canvas_width, 3), dtype="uint8")

    cv2.putText(
        canvas,
        f"LOL Test Image: {filename}",
        (margin, margin + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    y_header = margin + title_height
    y_image = y_header + header_height
    for index, (title, directory) in enumerate(columns):
        x = margin + index * (display_width + gap)
        image = read_bgr(directory / filename, display_size)
        cv2.putText(
            canvas,
            title,
            (x, y_header + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        canvas[y_image : y_image + display_height, x : x + display_width] = image

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(filename).stem}_comparison.png"
    if not cv2.imwrite(str(output_path), canvas):
        raise IOError(f"Failed to write comparison figure: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    low_dir = args.data_root / "test" / "low"
    filenames = collect_filenames(low_dir, args.num_images, args.filenames)
    columns = build_columns(args.data_root, args.traditional_root, args.unet_dir, args.ms_se_unet_dir)

    for filename in filenames:
        output_path = make_figure(filename, columns, args.output_dir)
        print(f"Saved comparison figure: {output_path}")


if __name__ == "__main__":
    main()
