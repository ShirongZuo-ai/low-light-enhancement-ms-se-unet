from __future__ import annotations

import cv2
import numpy as np


def _single_scale_retinex(channel: np.ndarray, sigma: float) -> np.ndarray:
    channel_float = channel.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(channel_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.log(channel_float) - np.log(blur + 1.0)


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def retinex_enhance(
    image: np.ndarray,
    sigmas: tuple[float, ...] = (15.0, 80.0, 250.0),
) -> np.ndarray:
    """Enhance a low-light BGR image with multi-scale Retinex."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    if not sigmas:
        raise ValueError("sigmas must contain at least one value.")

    image_float = image.astype(np.float32)
    retinex = np.zeros_like(image_float, dtype=np.float32)

    for sigma in sigmas:
        if sigma <= 0:
            raise ValueError("All sigma values must be greater than 0.")
        for channel_index in range(image.shape[2]):
            retinex[:, :, channel_index] += _single_scale_retinex(
                image_float[:, :, channel_index],
                sigma,
            )

    retinex /= float(len(sigmas))

    result_channels = [
        _normalize_to_uint8(retinex[:, :, channel_index])
        for channel_index in range(image.shape[2])
    ]
    result = cv2.merge(result_channels)

    # A light contrast stretch improves visual usability after Retinex normalization.
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.equalizeHist(l_channel)
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
