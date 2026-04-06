"""Classification components
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class ClassificationHead(nn.Module):
    """FC head that goes from flattened VGG11 bottleneck to class logits."""
 
    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5) -> None:
        super().__init__()
 
        self.head = nn.Sequential(
            # 512 * 7 * 7 = 25088
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
 
        self._init_weights()
 
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
    

class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        super().__init__()
 
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.pool    = nn.AdaptiveAvgPool2d((7, 7))   # ensures 7×7 regardless of input
        self.flatten = nn.Flatten()
        self.head    = ClassificationHead(num_classes=num_classes, dropout_p=dropout_p)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        features = self.encoder(x)           # [B, 512, 7, 7]
        pooled   = self.pool(features)        # [B, 512, 7, 7]  (already 7×7 for 224px input)
        flat     = self.flatten(pooled)       # [B, 25088]
        logits   = self.head(flat)            # [B, num_classes]
        return logits


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    model = VGG11Classifier(num_classes=37)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("logits:", out.shape)   # [2, 37]