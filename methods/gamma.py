from __future__ import annotations

import cv2
import numpy as np


def gamma_enhance(image: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """Enhance a low-light BGR image with gamma correction.

    gamma < 1 brightens the image, while gamma > 1 darkens it.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    if gamma <= 0:
        raise ValueError("gamma must be greater than 0.")

    table = np.array(
        [((value / 255.0) ** gamma) * 255 for value in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)
