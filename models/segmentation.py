"""Segmentation model
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


# --------------------------------------------------------------------------- #
# Decoder building blocks
# --------------------------------------------------------------------------- #
 
def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two Conv2d-BN-ReLU layers (standard U-Net decoder block)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )
 
 
class DecoderBlock(nn.Module):
    """One step of the U-Net decoder:
       1. TransposedConv → upsample spatial dims × 2, halve channels.
       2. Concatenate with skip feature (channel dim).
       3. Double conv to fuse.
    """
 
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        # Learnable upsampling — NOT bilinear interpolation
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        # After concat: (in_ch//2 + skip_ch) input channels
        self.conv = _double_conv(in_ch // 2 + skip_ch, out_ch)
 
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : [B, in_ch, H, W]           — upsampled feature map
            skip: [B, skip_ch, H*2, W*2]      — skip connection from encoder
        Returns:
            [B, out_ch, H*2, W*2]
        """
        x = self.up(x)                          # [B, in_ch//2, H*2, W*2]
 
        # Handle potential size mismatch (e.g. odd input dimensions)
        if x.shape != skip.shape:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
 
        x = torch.cat([x, skip], dim=1)         # concat along channel axis
        x = self.conv(x)
        return x

 
# --------------------------------------------------------------------------- #
# Full U-Net model
# --------------------------------------------------------------------------- #
 

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self,
        num_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        pretrained_encoder: str   = None,
        freeze_encoder:     bool  = False,):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()
 
        # ---- Encoder (VGG11 backbone) -------------------------------------- #
        self.encoder = VGG11Encoder(in_channels=in_channels)
 
        # ---- Bottleneck dropout -------------------------------------------- #
        self.bottleneck_drop = CustomDropout(p=dropout_p)
 
        # ---- Decoder blocks ------------------------------------------------ #
        # Input to decoder: bottleneck [B, 512, 7, 7]
        # up5: 7→14  in_ch=512 skip=block5[512] out=512
        self.dec5 = DecoderBlock(in_ch=512, skip_ch=512, out_ch=512)
        # up4: 14→28 in_ch=512 skip=block4[512] out=256
        self.dec4 = DecoderBlock(in_ch=512, skip_ch=512, out_ch=256)
        # up3: 28→56 in_ch=256 skip=block3[256] out=128
        self.dec3 = DecoderBlock(in_ch=256, skip_ch=256, out_ch=128)
        # up2: 56→112 in_ch=128 skip=block2[128] out=64
        self.dec2 = DecoderBlock(in_ch=128, skip_ch=128, out_ch=64)
        # up1:112→224 in_ch=64  skip=block1[64]  out=32
        self.dec1 = DecoderBlock(in_ch=64,  skip_ch=64,  out_ch=32)
 
        # ---- Final 1×1 conv → class logits --------------------------------- #
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
 
        # ---- Optional pre-trained encoder ---------------------------------- #
        if pretrained_encoder is not None:
            self._load_encoder(pretrained_encoder)
 
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
 
        self._init_decoder_weights()


    # ----------------------------------------------------------------------- #
    def _init_decoder_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
 
    def _load_encoder(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in state.items()
            if k.startswith("encoder.")
        }
        if encoder_state:
            self.encoder.load_state_dict(encoder_state, strict=False)
            print(f"[UNet] Loaded encoder weights from '{path}'")
 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # ---- Encoder with skip connections --------------------------------- #
        bottleneck, feats = self.encoder(x, return_features=True)
        # bottleneck: [B, 512, 7, 7]
        # feats['block1']: [B,  64, 224, 224]
        # feats['block2']: [B, 128, 112, 112]
        # feats['block3']: [B, 256,  56,  56]
        # feats['block4']: [B, 512,  28,  28]
        # feats['block5']: [B, 512,  14,  14]
 
        # Apply dropout at bottleneck
        bottleneck = self.bottleneck_drop(bottleneck)
 
        # ---- Decoder ------------------------------------------------------- #
        x = self.dec5(bottleneck,     feats["block5"])   # [B, 512, 14, 14]
        x = self.dec4(x,              feats["block4"])   # [B, 256, 28, 28]
        x = self.dec3(x,              feats["block3"])   # [B, 128, 56, 56]
        x = self.dec2(x,              feats["block2"])   # [B,  64,112,112]
        x = self.dec1(x,              feats["block1"])   # [B,  32,224,224]
 
        # ---- Output -------------------------------------------------------- #
        logits = self.final_conv(x)                      # [B, num_classes, 224, 224]
        return logits
    

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    model = VGG11UNet(num_classes=3)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("seg logits:", out.shape)   # [2, 3, 224, 224]
 