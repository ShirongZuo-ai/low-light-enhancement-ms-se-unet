from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def build_l1_loss() -> nn.Module:
    """Return the baseline reconstruction loss for U-Net training."""
    return nn.L1Loss()


class CombinedLoss(nn.Module):
    """L1 + SSIM + color constancy + total variation loss."""

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.2,
        color_weight: float = 0.05,
        tv_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.color_weight = color_weight
        self.tv_weight = tv_weight
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1_loss = self.l1(pred, target)
        ssim_loss = 1.0 - _ssim(pred, target)
        color_loss = _color_loss(pred, target)
        tv_loss = _tv_loss(pred)
        return (
            self.l1_weight * l1_loss
            + self.ssim_weight * ssim_loss
            + self.color_weight * color_loss
            + self.tv_weight * tv_loss
        )


def build_loss(name: str) -> nn.Module:
    if name == "l1":
        return build_l1_loss()
    if name == "combined":
        return CombinedLoss()
    raise ValueError(f"Unsupported loss: {name}")


def _ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    c1 = 0.01**2
    c2 = 0.03**2

    mu_pred = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_target = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    mu_pred_sq = mu_pred * mu_pred
    mu_target_sq = mu_target * mu_target
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.avg_pool2d(pred * pred, kernel_size=3, stride=1, padding=1) - mu_pred_sq
    sigma_target_sq = F.avg_pool2d(target * target, kernel_size=3, stride=1, padding=1) - mu_target_sq
    sigma_pred_target = F.avg_pool2d(pred * target, kernel_size=3, stride=1, padding=1) - mu_pred_target

    numerator = (2.0 * mu_pred_target + c1) * (2.0 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    return (numerator / (denominator + 1e-8)).mean()


def _color_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mean = pred.mean(dim=(2, 3))
    target_mean = target.mean(dim=(2, 3))
    return torch.mean(torch.abs(pred_mean - target_mean))


def _tv_loss(image: torch.Tensor) -> torch.Tensor:
    horizontal = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    vertical = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    return horizontal + vertical
