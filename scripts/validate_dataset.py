from __future__ import annotations

import argparse
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "test")
DOMAINS = ("low", "high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LOL dataset image readability and low/high pairs.")
    parser.add_argument("--data-root", type=str, default="data/LOL", help="Root directory of the LOL dataset.")
    return parser.parse_args()


def collect_images(directory: Path) -> dict[str, Path]:
    if not directory.exists():
        print(f"Missing directory: {directory}")
        return {}
    if not directory.is_dir():
        print(f"Not a directory: {directory}")
        return {}

    images: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images[path.name] = path
    return images


def validate_readability(paths: list[Path]) -> list[Path]:
    failed_paths: list[Path] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            failed_paths.append(path)
    return failed_paths


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)

    all_paths: list[Path] = []
    has_error = False

    for split in SPLITS:
        split_images: dict[str, dict[str, Path]] = {}
        for domain in DOMAINS:
            directory = data_root / split / domain
            images = collect_images(directory)
            split_images[domain] = images
            all_paths.extend(images.values())
            print(f"{directory}: {len(images)} images")

        low_names = set(split_images["low"].keys())
        high_names = set(split_images["high"].keys())
        missing_high = sorted(low_names - high_names)
        missing_low = sorted(high_names - low_names)

        if missing_high or missing_low:
            has_error = True
            print(f"Pair mismatch in {split}:")
            if missing_high:
                print(f"  Missing in {split}/high: {missing_high[:20]}")
            if missing_low:
                print(f"  Missing in {split}/low: {missing_low[:20]}")
        else:
            print(f"{split}: low/high filenames match ({len(low_names)} pairs)")

    failed_paths = validate_readability(all_paths)
    if failed_paths:
        has_error = True
        print("Unreadable images:")
        for path in failed_paths:
            print(path)
    else:
        print("All scanned images are readable by OpenCV.")

    if has_error:
        raise SystemExit(1)

    print("Dataset validation passed.")


if __name__ == "__main__":
    main()
