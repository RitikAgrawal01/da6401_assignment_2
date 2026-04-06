"""Classification head on top of VGG11."""

import torch
import torch.nn as nn
from .vgg11 import VGG11
from .layers import CustomDropout


class ClassificationHead(nn.Module):
    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11 encoder + classification head."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11(in_channels=in_channels)
        self.pool    = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()
        self.head    = ClassificationHead(num_classes, dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.flatten(self.pool(self.encoder(x))))