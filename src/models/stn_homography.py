"""STN-style homography regression model."""

from __future__ import annotations

import torch
import torch.nn as nn


class STNHomographyRegressor(nn.Module):
    """Regresses 8 homography parameters from stacked masks."""

    def __init__(self, in_channels: int = 2, base_channels: int = 32):
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(c4, 8)

        # Identity initialization (flattened 3x3 minus last element).
        nn.init.zeros_(self.head.weight)
        identity_8 = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        with torch.no_grad():
            self.head.bias.copy_(identity_8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x).flatten(1)
        return self.head(z)
