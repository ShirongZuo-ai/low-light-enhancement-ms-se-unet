from __future__ import annotations

import math

import cv2
import numpy as np
import torch


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_value: float = 1.0) -> float:
    """Calculate PSNR for image tensors in [0, 1]."""
    pred = pred.detach().float().clamp(0.0, max_value)
    target = target.detach().float().clamp(0.0, max_value)
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(max_value) - 10.0 * math.log10(mse)


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate RGB SSIM for CHW image tensors in [0, 1]."""
    pred_np = _tensor_to_hwc_numpy(pred)
    target_np = _tensor_to_hwc_numpy(target)
    channel_scores = [_calculate_single_channel_ssim(pred_np[:, :, i], target_np[:, :, i]) for i in range(3)]
    return float(np.mean(channel_scores))


def _tensor_to_hwc_numpy(image: torch.Tensor) -> np.ndarray:
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB tensor with 3 channels, got shape {tuple(image.shape)}")
    return image.permute(1, 2, 0).numpy()


def _calculate_single_channel_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2

    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    mu_pred = cv2.GaussianBlur(pred, (11, 11), 1.5)
    mu_target = cv2.GaussianBlur(target, (11, 11), 1.5)

    mu_pred_sq = mu_pred * mu_pred
    mu_target_sq = mu_target * mu_target
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = cv2.GaussianBlur(pred * pred, (11, 11), 1.5) - mu_pred_sq
    sigma_target_sq = cv2.GaussianBlur(target * target, (11, 11), 1.5) - mu_target_sq
    sigma_pred_target = cv2.GaussianBlur(pred * target, (11, 11), 1.5) - mu_pred_target

    numerator = (2.0 * mu_pred_target + c1) * (2.0 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    return float(np.mean(numerator / denominator))
