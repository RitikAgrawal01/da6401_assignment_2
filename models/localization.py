"""Localization modules
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class LocalizationHead(nn.Module):
    """FC regression head: 25088 → 4 bounding box coordinates."""
 
    def __init__(self, dropout_p: float = 0.5, img_size: int = 224) -> None:
        super().__init__()
        self.img_size = img_size
 
        self.head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
 
            nn.Linear(4096, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
 
            nn.Linear(1024, 4),       # raw outputs
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
        """
        Args:
            x: [B, 25088] flattened bottleneck features.
        Returns:
            [B, 4] in pixel space — values in [0, img_size].
        """
        raw  = self.head(x)                          # [B, 4]  unbounded
        # Sigmoid maps to (0, 1); scaling gives values in pixel space
        out  = torch.sigmoid(raw) * self.img_size    # [B, 4]
        return out

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(
        self,
        in_channels:        int   = 3,
        dropout_p:          float = 0.5,
        img_size:           int   = 224,
        pretrained_encoder: str   = None,
        freeze_encoder:     bool  = False,
    ) -> None:
        super().__init__()
 
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.pool    = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()
        self.head    = LocalizationHead(dropout_p=dropout_p, img_size=img_size)
 
        # Load pre-trained encoder weights if provided
        if pretrained_encoder is not None:
            self._load_encoder(pretrained_encoder)
 
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

 # ----------------------------------------------------------------------- #
    def _load_encoder(self, path: str) -> None:
        """Load encoder weights from a VGG11Classifier checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        # Checkpoint may be the full model state_dict or just the encoder
        state = checkpoint.get("model_state_dict", checkpoint)
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in state.items()
            if k.startswith("encoder.")
        }
        if encoder_state:
            self.encoder.load_state_dict(encoder_state, strict=False)
            print(f"[Localizer] Loaded encoder weights from '{path}'")
        else:
            print(f"[Localizer] Warning: no 'encoder.*' keys found in '{path}'")
 
    # ----------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        features = self.encoder(x)          # [B, 512, 7, 7]
        pooled   = self.pool(features)      # [B, 512, 7, 7]
        flat     = self.flatten(pooled)     # [B, 25088]
        bbox     = self.head(flat)          # [B, 4]
        return bbox

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    model = VGG11Localizer()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("bbox:", out.shape)    # [2, 4]
    print("range:", out.min().item(), out.max().item())  # should be in [0, 224]