"""Differentiable objectives aligned with the DoseRAD Level-1 beam metrics."""

from __future__ import annotations

import torch
import torch.nn as nn


def _volumes(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 5 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 4:
        raise ValueError(f"Expected (B,D,H,W) or (B,1,D,H,W), got {tensor.shape}")
    return tensor.float()


def masked_beam_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.10,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    prediction, target = _volumes(prediction), _volumes(target)
    target_max = target.amax(dim=(1, 2, 3), keepdim=True)
    mask = target >= threshold * target_max
    valid = target_max.flatten() > eps
    count = mask.sum(dim=(1, 2, 3)).clamp_min(1)
    error = (torch.abs(prediction - target) * mask).sum(dim=(1, 2, 3)) / count
    error = error / target_max.flatten().clamp_min(eps)
    return error[valid].mean() if valid.any() else prediction.sum() * 0.0


def idd_curve_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    prediction, target = _volumes(prediction), _volumes(target)
    prediction_idd = prediction.sum(dim=(2, 3))
    target_idd = target.sum(dim=(2, 3))
    target_max = target_idd.amax(dim=1, keepdim=True)
    valid = target_max.flatten() > eps
    difference = (prediction_idd - target_idd) / target_max.clamp_min(eps)
    rms = torch.sqrt(difference.square().mean(dim=1) + eps) - eps**0.5
    return rms[valid].mean() if valid.any() else prediction.sum() * 0.0


class Level1Loss(nn.Module):
    """Equal-weight masked beam MAE plus IDD RMSE."""

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        mae = masked_beam_mae(prediction, target)
        idd = idd_curve_loss(prediction, target)
        return {"loss": mae + idd, "masked_mae": mae, "idd": idd}
