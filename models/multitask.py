"""Unified multi-task model
"""

from typing import Dict

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout
from .classification import ClassificationHead
from .localization import LocalizationHead
from .segmentation import DecoderBlock

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()
 
        # ---- Shared encoder ------------------------------------------------ #
        self.encoder = VGG11Encoder(in_channels=in_channels)
 
        # ---- Task heads ---------------------------------------------------- #
        # Classification
        self.cls_pool    = nn.AdaptiveAvgPool2d((7, 7))
        self.cls_flatten = nn.Flatten()
        self.cls_head    = ClassificationHead(num_classes=num_breeds, dropout_p=0.5)
 
        # Localization
        self.loc_pool    = nn.AdaptiveAvgPool2d((7, 7))
        self.loc_flatten = nn.Flatten()
        self.loc_head    = LocalizationHead(dropout_p=0.5, img_size=224)
 
        # Segmentation decoder (mirrors VGG11UNet structure)
        self.bottleneck_drop = CustomDropout(p=0.5)
        self.dec5 = DecoderBlock(in_ch=512, skip_ch=512, out_ch=512)
        self.dec4 = DecoderBlock(in_ch=512, skip_ch=512, out_ch=256)
        self.dec3 = DecoderBlock(in_ch=256, skip_ch=256, out_ch=128)
        self.dec2 = DecoderBlock(in_ch=128, skip_ch=128, out_ch=64)
        self.dec1 = DecoderBlock(in_ch=64,  skip_ch=64,  out_ch=32)
        self.seg_final = nn.Conv2d(32, seg_classes, kernel_size=1)
 
        # ---- Load pre-trained weights -------------------------------------- #
        self._load_weights(classifier_path, localizer_path, unet_path)
 
    # ----------------------------------------------------------------------- #
    def _load_weights(
        self,
        classifier_path: str,
        localizer_path:  str,
        unet_path:       str,
    ) -> None:
        """Load trained weights from individual task checkpoints."""
 
        def _load(path: str) -> dict:
            try:
                ckpt = torch.load(path, map_location="cpu")
                return ckpt.get("model_state_dict", ckpt)
            except FileNotFoundError:
                print(f"[MultiTask] Warning: checkpoint not found: '{path}'")
                return {}
 
        # --- Encoder: load from classifier checkpoint ----------------------- #
        cls_state = _load(classifier_path)
        enc_state  = {k.replace("encoder.", ""): v
                      for k, v in cls_state.items()
                      if k.startswith("encoder.")}
        if enc_state:
            self.encoder.load_state_dict(enc_state, strict=False)
            print("[MultiTask] Encoder loaded from classifier checkpoint.")
 
        # --- Classification head ------------------------------------------- #
        head_state = {k.replace("head.", ""): v
                      for k, v in cls_state.items()
                      if k.startswith("head.")}
        if head_state:
            self.cls_head.load_state_dict(head_state, strict=False)
            print("[MultiTask] Classification head loaded.")
 
        # --- Localization head --------------------------------------------- #
        loc_state  = _load(localizer_path)
        loc_hstate = {k.replace("head.", ""): v
                      for k, v in loc_state.items()
                      if k.startswith("head.")}
        if loc_hstate:
            self.loc_head.load_state_dict(loc_hstate, strict=False)
            print("[MultiTask] Localization head loaded.")
 
        # --- Segmentation decoder ------------------------------------------ #
        unet_state = _load(unet_path)
        # Map decoder keys from VGG11UNet checkpoint
        dec_keys = {
            "dec5": self.dec5, "dec4": self.dec4, "dec3": self.dec3,
            "dec2": self.dec2, "dec1": self.dec1,
        }
        for name, module in dec_keys.items():
            sub = {k.replace(f"{name}.", ""): v
                   for k, v in unet_state.items()
                   if k.startswith(f"{name}.")}
            if sub:
                module.load_state_dict(sub, strict=False)
        seg_final_state = {k.replace("final_conv.", ""): v
                           for k, v in unet_state.items()
                           if k.startswith("final_conv.")}
        if seg_final_state:
            self.seg_final.load_state_dict(seg_final_state, strict=False)
        if unet_state:
            print("[MultiTask] Segmentation decoder loaded.")

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        # ---- Shared encoder pass ------------------------------------------ #
        bottleneck, feats = self.encoder(x, return_features=True)
        # bottleneck : [B, 512, 7, 7]
 
        # ---- Classification branch ---------------------------------------- #
        cls_feat  = self.cls_pool(bottleneck)       # [B, 512, 7, 7]
        cls_flat  = self.cls_flatten(cls_feat)      # [B, 25088]
        cls_out   = self.cls_head(cls_flat)          # [B, 37]
 
        # ---- Localization branch ------------------------------------------ #
        loc_feat  = self.loc_pool(bottleneck)       # [B, 512, 7, 7]
        loc_flat  = self.loc_flatten(loc_feat)      # [B, 25088]
        loc_out   = self.loc_head(loc_flat)          # [B, 4]
 
        # ---- Segmentation branch ------------------------------------------ #
        bn = self.bottleneck_drop(bottleneck)
        s  = self.dec5(bn,           feats["block5"])
        s  = self.dec4(s,            feats["block4"])
        s  = self.dec3(s,            feats["block3"])
        s  = self.dec2(s,            feats["block2"])
        s  = self.dec1(s,            feats["block1"])
        seg_out = self.seg_final(s)                  # [B, 3, H, W]
 
        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }
 
 
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os
    # Without checkpoints — just test forward shape
    model = MultiTaskPerceptionModel(
        classifier_path="", localizer_path="", unet_path=""
    )
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print("cls :", out["classification"].shape)  # [2, 37]
    print("loc :", out["localization"].shape)     # [2, 4]
    print("seg :", out["segmentation"].shape)     # [2, 3, 224, 224]
 
