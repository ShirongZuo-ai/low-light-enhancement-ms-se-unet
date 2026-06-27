from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class LOLDataset(Dataset):
    """Paired LOL dataset reader.

    Expected directory layout:
        data/LOL/train/low/
        data/LOL/train/high/
        data/LOL/test/low/
        data/LOL/test/high/
    """

    def __init__(
        self,
        root: str | Path = Path("data/LOL"),
        split: str = "train",
        image_size: tuple[int, int] = (256, 256),
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size

        self.low_dir = self.root / split / "low"
        self.high_dir = self.root / split / "high"

        self.low_paths = self._collect_images(self.low_dir)
        self.high_paths = self._collect_images(self.high_dir)
        self.filenames = self._validate_pairs(self.low_paths, self.high_paths)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        filename = self.filenames[index]
        low_image = self._read_image(self.low_paths[filename])
        high_image = self._read_image(self.high_paths[filename])
        return low_image, high_image, filename

    @staticmethod
    def _collect_images(directory: Path) -> dict[str, Path]:
        if not directory.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {directory}")

        image_paths: dict[str, Path] = {}
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if path.name in image_paths:
                raise ValueError(f"Duplicate image filename found: {path.name}")
            image_paths[path.name] = path

        if not image_paths:
            raise ValueError(f"No jpg/jpeg/png images found in: {directory}")
        return image_paths

    @staticmethod
    def _validate_pairs(
        low_paths: dict[str, Path],
        high_paths: dict[str, Path],
    ) -> list[str]:
        low_names = set(low_paths.keys())
        high_names = set(high_paths.keys())

        missing_high = sorted(low_names - high_names)
        missing_low = sorted(high_names - low_names)

        if missing_high or missing_low:
            messages = [
                "LOL low/high image filenames do not match.",
                f"low count: {len(low_names)}, high count: {len(high_names)}",
            ]
            if missing_high:
                messages.append(f"Missing in high/: {missing_high[:10]}")
            if missing_low:
                messages.append(f"Missing in low/: {missing_low[:10]}")
            raise ValueError(" ".join(messages))

        return sorted(low_names)

    def _read_image(self, image_path: Path) -> torch.Tensor:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is not None:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            try:
                with Image.open(image_path) as image:
                    image_rgb = np.array(image.convert("RGB"))
            except Exception as exc:
                raise ValueError(f"Failed to read image with OpenCV and PIL: {image_path}") from exc

        width, height = self.image_size
        image_rgb = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
        image_float = image_rgb.astype(np.float32) / 255.0
        image_chw = np.transpose(image_float, (2, 0, 1))
        return torch.from_numpy(image_chw).float()
