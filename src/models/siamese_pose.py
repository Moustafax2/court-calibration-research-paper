"""Siamese encoder for template retrieval."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseEncoder(nn.Module):
    """Lightweight CNN encoder producing normalized embeddings."""

    def __init__(self, in_channels: int = 1, embedding_dim: int = 128, base_channels: int = 32):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(c3, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x).flatten(1)
        z = self.proj(z)
        return F.normalize(z, p=2, dim=1)


class SiameseRetrievalModel(nn.Module):
    """Shared-weight siamese network."""

    def __init__(self, in_channels: int = 1, embedding_dim: int = 128, base_channels: int = 32):
        super().__init__()
        self.encoder = SiameseEncoder(
            in_channels=in_channels, embedding_dim=embedding_dim, base_channels=base_channels
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x1), self.encoder(x2)
