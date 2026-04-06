"""Unified multi-task model — root level as required by autograder.

Autograder import:
    from multitask import MultiTaskPerceptionModel

Downloads checkpoints from Google Drive via gdown in __init__.
Replace the placeholder IDs with your actual Drive file IDs.
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.layers import CustomDropout
from models.classification import ClassificationHead
from models.localization import LocalizationHead
from models.segmentation import DecoderBlock


"""
links:-
https://drive.google.com/file/d/1WF8cWBxjOZy7Shg9zsu1-GWwtX_icWmg/view?usp=sharing
https://drive.google.com/file/d/1JOETyBwkM2gzNJRzoeOq3aRiLWbVh168/view?usp=sharing
https://drive.google.com/file/d/1j0_UAv7PgrXJwHF5-zwFQBh2l4AKF-Po/view?usp=sharing
"""


class MultiTaskPerceptionModel(nn.Module):
    """Shared VGG11 backbone with three task heads.

    Args:
        num_breeds      : Number of breed classes (37).
        seg_classes     : Number of segmentation classes (3).
        in_channels     : Input image channels (3).
        classifier_path : Relative path to classifier checkpoint.
        localizer_path  : Relative path to localizer checkpoint.
        unet_path       : Relative path to unet checkpoint.
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = 'checkpoints/classifier.pth',
        localizer_path: str  = 'checkpoints/localizer.pth',
        unet_path: str       = 'checkpoints/unet.pth',
    ):
        super().__init__()

        # ------------------------------------------------------------------ #
        # Download checkpoints from Google Drive
        # Replace these IDs with your actual Drive file IDs from Step 2
        # ------------------------------------------------------------------ #
        import gdown
        gdown.download(id='1WF8cWBxjOZy7Shg9zsu1-GWwtX_icWmg', output=classifier_path, quiet=False)
        gdown.download(id='1JOETyBwkM2gzNJRzoeOq3aRiLWbVh168', output=localizer_path, quiet=False)
        gdown.download(id='1j0_UAv7PgrXJwHF5-zwFQBh2l4AKF-Po', output=unet_path, quiet=False)

        # ------------------------------------------------------------------ #
        # Shared encoder
        # ------------------------------------------------------------------ #
        self.encoder = VGG11(in_channels=in_channels)

        # ------------------------------------------------------------------ #
        # Classification head
        # ------------------------------------------------------------------ #
        self.cls_pool    = nn.AdaptiveAvgPool2d((7, 7))
        self.cls_flatten = nn.Flatten()
        self.cls_head    = ClassificationHead(num_classes=num_breeds, dropout_p=0.5)

        # ------------------------------------------------------------------ #
        # Localization head
        # ------------------------------------------------------------------ #
        self.loc_pool    = nn.AdaptiveAvgPool2d((7, 7))
        self.loc_flatten = nn.Flatten()
        self.loc_head    = LocalizationHead(dropout_p=0.5)

        # ------------------------------------------------------------------ #
        # Segmentation decoder
        # ------------------------------------------------------------------ #
        self.bottleneck_drop = CustomDropout(p=0.5)
        self.dec5 = DecoderBlock(512, 512, 512)
        self.dec4 = DecoderBlock(512, 512, 256)
        self.dec3 = DecoderBlock(256, 256, 128)
        self.dec2 = DecoderBlock(128, 128,  64)
        self.dec1 = DecoderBlock( 64,  64,  32)
        self.seg_final = nn.Conv2d(32, seg_classes, kernel_size=1)

        # ------------------------------------------------------------------ #
        # Load trained weights
        # ------------------------------------------------------------------ #
        self._load_weights(classifier_path, localizer_path, unet_path)

    # ---------------------------------------------------------------------- #
    def _load_weights(self, cls_p: str, loc_p: str, unet_p: str):
        def _load(path):
            try:
                c = torch.load(path, map_location='cpu')
                return c.get('model_state_dict', c)
            except Exception as e:
                print(f'[MultiTask] Warning loading {path}: {e}')
                return {}

        # Encoder + classification head from classifier checkpoint
        cls_s = _load(cls_p)
        enc_s = {k.replace('encoder.', ''): v
                 for k, v in cls_s.items() if k.startswith('encoder.')}
        if enc_s:
            self.encoder.load_state_dict(enc_s, strict=False)
            print('[MultiTask] Encoder loaded.')
        head_s = {k[len('head.'):]: v
                  for k, v in cls_s.items() if k.startswith('head.')}
        if head_s:
            self.cls_head.load_state_dict(head_s, strict=False)
            print('[MultiTask] Classification head loaded.')

        # Localization head
        loc_s  = _load(loc_p)
        loc_hs = {k[len('head.'):]: v
                  for k, v in loc_s.items() if k.startswith('head.')}
        if loc_hs:
            self.loc_head.load_state_dict(loc_hs, strict=False)
            print('[MultiTask] Localization head loaded.')

        # Segmentation decoder
        unet_s = _load(unet_p)
        for name, mod in [('dec5', self.dec5), ('dec4', self.dec4),
                          ('dec3', self.dec3), ('dec2', self.dec2),
                          ('dec1', self.dec1)]:
            sub = {k.replace(f'{name}.', ''): v
                   for k, v in unet_s.items() if k.startswith(f'{name}.')}
            if sub:
                mod.load_state_dict(sub, strict=False)
        fin = {k.replace('final_conv.', ''): v
               for k, v in unet_s.items() if k.startswith('final_conv.')}
        if fin:
            self.seg_final.load_state_dict(fin, strict=False)
        if unet_s:
            print('[MultiTask] Segmentation decoder loaded.')

    # ---------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> dict:
        """Single forward pass — three simultaneous outputs.

        Args:
            x: [B, 3, 224, 224]

        Returns:
            dict:
              'classification': [B, num_breeds]        logits
              'localization'  : [B, 4]                 bbox pixel coords
              'segmentation'  : [B, seg_classes, H, W] logits
        """
        # Shared encoder
        bottleneck, feats = self.encoder(x, return_features=True)

        # Classification branch
        cls_out = self.cls_head(
            self.cls_flatten(self.cls_pool(bottleneck))
        )

        # Localization branch
        loc_out = self.loc_head(
            self.loc_flatten(self.loc_pool(bottleneck))
        )

        # Segmentation branch
        s = self.bottleneck_drop(bottleneck)
        s = self.dec5(s, feats['block5'])
        s = self.dec4(s, feats['block4'])
        s = self.dec3(s, feats['block3'])
        s = self.dec2(s, feats['block2'])
        s = self.dec1(s, feats['block1'])
        seg_out = self.seg_final(s)

        return {
            'classification': cls_out,
            'localization':   loc_out,
            'segmentation':   seg_out,
        }