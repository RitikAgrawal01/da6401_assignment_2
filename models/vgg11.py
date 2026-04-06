"""VGG11 encoder — implemented per the official paper (Simonyan & Zisserman 2014).

Autograder import:
    from models.vgg11 import VGG11
"""

from typing import Dict, Tuple, Union
import torch
import torch.nn as nn


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11(nn.Module):
    """VGG11 with BatchNorm from scratch. Input: [B, 3, 224, 224].

    Design justification:
      - BatchNorm2d after every Conv2d and before ReLU: stabilises activations,
        accelerates convergence, allows higher stable learning rates.
      - Dropout placed only in FC task heads (not here) so conv feature maps
        remain clean for U-Net skip connections.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # Block 1: 224 -> 112
        self.block1 = nn.Sequential(_conv_bn_relu(in_channels, 64))
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 112 -> 56
        self.block2 = nn.Sequential(_conv_bn_relu(64, 128))
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 56 -> 28  (2 convs)
        self.block3 = nn.Sequential(
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
        )
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 28 -> 14  (2 convs)
        self.block4 = nn.Sequential(
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
        )
        self.pool4  = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5: 14 -> 7  (2 convs)
        self.block5 = nn.Sequential(
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
        )
        self.pool5  = nn.MaxPool2d(kernel_size=2, stride=2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        f1 = self.block1(x);           x = self.pool1(f1)
        f2 = self.block2(x);           x = self.pool2(f2)
        f3 = self.block3(x);           x = self.pool3(f3)
        f4 = self.block4(x);           x = self.pool4(f4)
        f5 = self.block5(x); bottleneck = self.pool5(f5)

        if return_features:
            return bottleneck, {
                'block1': f1,
                'block2': f2,
                'block3': f3,
                'block4': f4,
                'block5': f5,
            }
        return bottleneck


# Backwards-compat alias
VGG11Encoder = VGG11