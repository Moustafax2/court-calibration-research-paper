"""Contrastive loss for siamese retrieval training."""

from __future__ import annotations

import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """
    Standard contrastive loss:
    y=1 (similar): d^2
    y=0 (dissimilar): max(0, margin - d)^2
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d = torch.norm(emb1 - emb2, p=2, dim=1)
        pos = target * d.pow(2)
        neg = (1.0 - target) * torch.clamp(self.margin - d, min=0.0).pow(2)
        return (pos + neg).mean()
