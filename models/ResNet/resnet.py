from .block import ResBlock

import torch
from torch import nn

def norm(dim):
    return nn.BatchNorm2d(dim)

class ResNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.downsampling = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        )

        self.feature = nn.Sequential(
            *[ResBlock(64, 64) for _ in range(layers)]
        )

        self.norm = norm(64)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.downsampling(x)
        x = self.feature(x)
        x = self.norm(x)
        x = self.avg_pool(x)
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        x = x.view(-1, shape)
        out = self.fc(x)
        return out

    @classmethod
    def from_config(cls, cfg):
        return cls(layers=cfg.get('layers', 6))