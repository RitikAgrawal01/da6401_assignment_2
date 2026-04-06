"""Bounding box localization model.

Loss during training: MSE + IoULoss  (per additional instructions).
Output: [x_center, y_center, width, height] in pixel space (not normalised).
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11
from .layers import CustomDropout


class LocalizationHead(nn.Module):
    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        # Input image size fixed at 224 per VGG11 paper
        self.img_size = 224
        self.head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(1024, 4),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sigmoid -> scale to pixel space [0, 224]
        return torch.sigmoid(self.head(x)) * self.img_size


class VGG11Localizer(nn.Module):
    """VGG11 encoder + localization regression head.

    Args:
        in_channels        : Input channels (default 3).
        dropout_p          : Dropout probability in the head.
        pretrained_encoder : Path to classifier checkpoint to init encoder.
        freeze_encoder     : If True, freeze encoder weights.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        pretrained_encoder: str = None,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = VGG11(in_channels=in_channels)
        self.pool    = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()
        self.head    = LocalizationHead(dropout_p=dropout_p)

        if pretrained_encoder is not None:
            self._load_encoder(pretrained_encoder)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _load_encoder(self, path: str):
        ckpt  = torch.load(path, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        enc_s = {k.replace('encoder.', ''): v
                 for k, v in state.items() if k.startswith('encoder.')}
        if enc_s:
            self.encoder.load_state_dict(enc_s, strict=False)
            print(f'[Localizer] Encoder loaded from {path}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, 4] bbox in pixel space."""
        return self.head(self.flatten(self.pool(self.encoder(x))))