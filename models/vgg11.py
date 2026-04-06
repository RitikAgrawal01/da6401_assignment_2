"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv2d(3×3, same padding) → BatchNorm2d → ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()
 
        # ------------------------------------------------------------------- #
        # Convolutional blocks — each block ends with a MaxPool2d(2, 2)
        # ------------------------------------------------------------------- #
 
        # Block 1: 224 → 112
        self.block1 = nn.Sequential(
            _conv_bn_relu(in_channels, 64),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
 
        # Block 2: 112 → 56
        self.block2 = nn.Sequential(
            _conv_bn_relu(64, 128),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
 
        # Block 3: 56 → 28  (two conv layers)
        self.block3 = nn.Sequential(
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
 
        # Block 4: 28 → 14  (two conv layers)
        self.block4 = nn.Sequential(
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
 
        # Block 5: 14 → 7  (two conv layers)
        self.block5 = nn.Sequential(
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
 
        # Weight initialisation (Xavier for Conv, constant for BN)
        self._init_weights()

    # ----------------------------------------------------------------------- #
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
 
    # ----------------------------------------------------------------------- #

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        # Block 1
        f1 = self.block1(x)          # [B, 64,  224, 224]  ← before pool
        x  = self.pool1(f1)          # [B, 64,  112, 112]
 
        # Block 2
        f2 = self.block2(x)          # [B, 128, 112, 112]
        x  = self.pool2(f2)          # [B, 128,  56,  56]
 
        # Block 3
        f3 = self.block3(x)          # [B, 256,  56,  56]
        x  = self.pool3(f3)          # [B, 256,  28,  28]
 
        # Block 4
        f4 = self.block4(x)          # [B, 512,  28,  28]
        x  = self.pool4(f4)          # [B, 512,  14,  14]
 
        # Block 5
        f5 = self.block5(x)          # [B, 512,  14,  14]
        bottleneck = self.pool5(f5)  # [B, 512,   7,   7]
 
        if return_features:
            features = {
                "block1": f1,          # [B,  64, 224, 224]
                "block2": f2,          # [B, 128, 112, 112]
                "block3": f3,          # [B, 256,  56,  56]
                "block4": f4,          # [B, 512,  28,  28]
                "block5": f5,          # [B, 512,  14,  14]
            }
            return bottleneck, features
 
        return bottleneck
 
 
# --------------------------------------------------------------------------- #
# Quick sanity-check
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    model = VGG11Encoder()
    x = torch.randn(2, 3, 224, 224)
 
    bottleneck = model(x)
    print("bottleneck:", bottleneck.shape)   # [2, 512, 7, 7]
 
    bottleneck, feats = model(x, return_features=True)
    for k, v in feats.items():
        print(f"  {k}: {v.shape}")
 