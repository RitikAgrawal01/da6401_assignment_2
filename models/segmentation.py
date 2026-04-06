"""U-Net style segmentation with VGG11 encoder and transposed-conv decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg11 import VGG11
from .layers import CustomDropout


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DecoderBlock(nn.Module):
    """TransposedConv upsample + skip concat + double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # Learnable upsampling — NOT bilinear interpolation (per assignment)
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _double_conv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class VGG11UNet(nn.Module):
    """U-Net with VGG11 encoder.

    Loss justification (CE + Dice):
      CrossEntropyLoss calibrates per-pixel class probabilities correctly.
      Dice loss directly optimises the overlap metric used at evaluation time
      and handles class imbalance better than CE alone.

    Args:
        num_classes        : Segmentation classes (3 for trimap).
        in_channels        : Input image channels.
        dropout_p          : Dropout on bottleneck.
        pretrained_encoder : Path to classifier checkpoint.
        freeze_encoder     : Freeze encoder if True.
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        pretrained_encoder: str = None,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder         = VGG11(in_channels=in_channels)
        self.bottleneck_drop = CustomDropout(p=dropout_p)

        # Decoder mirrors the encoder blocks
        self.dec5 = DecoderBlock(512, 512, 512)
        self.dec4 = DecoderBlock(512, 512, 256)
        self.dec3 = DecoderBlock(256, 256, 128)
        self.dec2 = DecoderBlock(128, 128,  64)
        self.dec1 = DecoderBlock( 64,  64,  32)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        if pretrained_encoder is not None:
            self._load_encoder(pretrained_encoder)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self._init_decoder()

    def _init_decoder(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _load_encoder(self, path: str):
        ckpt  = torch.load(path, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        enc_s = {k.replace('encoder.', ''): v
                 for k, v in state.items() if k.startswith('encoder.')}
        if enc_s:
            self.encoder.load_state_dict(enc_s, strict=False)
            print(f'[UNet] Encoder loaded from {path}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns segmentation logits [B, num_classes, H, W]."""
        bn, feats = self.encoder(x, return_features=True)
        bn = self.bottleneck_drop(bn)
        x  = self.dec5(bn,         feats['block5'])
        x  = self.dec4(x,          feats['block4'])
        x  = self.dec3(x,          feats['block3'])
        x  = self.dec2(x,          feats['block2'])
        x  = self.dec1(x,          feats['block1'])
        return self.final_conv(x)