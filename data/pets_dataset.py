"""Dataset loader for Oxford-IIIT Pet dataset.

Handles:
  - 37-class breed classification labels
  - Bounding box annotations (head region) in [x_center, y_center, w, h] format
  - Pixel-level trimap segmentation masks (foreground / background / uncertain)
"""

import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Helper: parse the annotation files shipped with the dataset
# --------------------------------------------------------------------------- #

def _parse_annotations(list_file: str) -> Dict[str, dict]:
    """Parse trainval.txt / test.txt shipped with the dataset.

    Each row:  Image CLASS-ID SPECIES BREED-ID
      - CLASS-ID  : 1-37  (breed index, 1-based)
      - SPECIES   : 1=cat, 2=dog
      - BREED-ID  : 1-based within species

    Returns a dict keyed by image stem (e.g. 'Abyssinian_1').
    """
    info = {}
    with open(list_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            stem = parts[0]
            class_id = int(parts[1]) - 1      # 0-based for CrossEntropyLoss
            species  = int(parts[2])           # 1=cat 2=dog
            breed_id = int(parts[3])
            info[stem] = {
                "class_id": class_id,
                "species":  species,
                "breed_id": breed_id,
            }
    return info


def _parse_bboxes(xml_dir: str) -> Dict[str, Tuple[float, float, float, float]]:
    """Parse Pascal VOC XML annotations for bounding boxes.

    Returns a dict: stem -> (x_center, y_center, width, height) in **pixel** coords.
    The XML stores xmin/ymin/xmax/ymax; we convert here.
    """
    import xml.etree.ElementTree as ET

    bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    xml_path = Path(xml_dir)

    if not xml_path.exists():
        return bboxes

    for xml_file in xml_path.glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            size = root.find("size")
            # Some XMLs are missing size; skip gracefully
            if size is None:
                continue

            obj = root.find("object")
            if obj is None:
                continue

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            x_center = (xmin + xmax) / 2.0
            y_center  = (ymin + ymax) / 2.0
            width     = xmax - xmin
            height    = ymax - ymin

            stem = xml_file.stem          # filename without extension
            bboxes[stem] = (x_center, y_center, width, height)

        except Exception:
            # Malformed XML — skip silently
            continue

    return bboxes


# --------------------------------------------------------------------------- #
# Main Dataset class
# --------------------------------------------------------------------------- #

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.

    Provides per-sample:
      - image      : FloatTensor [3, H, W]  (after transforms)
      - class_id   : int (0-36)             breed label
      - bbox       : FloatTensor [4]        [x_center, y_center, w, h] in pixels
      - mask       : LongTensor  [H, W]     trimap classes 0/1/2
                       0 = foreground pet
                       1 = background
                       2 = uncertain / boundary

    Args:
        root:       Path to the dataset root (contains 'images/', 'annotations/').
        split:      'trainval' or 'test'.
        img_size:   Resize images (and masks) to this square size. Default 224.
        transform:  Optional image transform (albumentations or torchvision).
        target_transform: Optional mask transform.
    """

    # Map raw trimap pixel values → class indices
    #   trimap pixel 1 → class 0 (foreground)
    #   trimap pixel 2 → class 1 (background)
    #   trimap pixel 3 → class 2 (uncertain)
    TRIMAP_TO_CLASS = {1: 0, 2: 1, 3: 2}

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        img_size: int = 224,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        assert split in ("trainval", "test"), \
            f"split must be 'trainval' or 'test', got '{split}'"

        self.root   = Path(root)
        self.split  = split
        self.img_size = img_size
        self.transform = transform
        self.target_transform = target_transform

        # ---- file paths --------------------------------------------------- #
        self.img_dir   = self.root / "images"
        self.mask_dir  = self.root / "annotations" / "trimaps"
        self.xml_dir   = self.root / "annotations" / "xmls"
        self.list_file = self.root / "annotations" / f"{split}.txt"

        # ---- load metadata ------------------------------------------------- #
        self._info   = _parse_annotations(str(self.list_file))
        self._bboxes = _parse_bboxes(str(self.xml_dir))

        # Only keep stems that have both an image AND a list entry
        all_stems = sorted(self._info.keys())
        self._stems = [
            s for s in all_stems
            if (self.img_dir / f"{s}.jpg").exists()
        ]

        if len(self._stems) == 0:
            raise RuntimeError(
                f"No valid images found under '{self.img_dir}'. "
                "Did you download the Oxford-IIIT Pet dataset?"
            )

        print(f"[PetDataset] split={split} | samples={len(self._stems)}")

    # ----------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._stems)

    # ----------------------------------------------------------------------- #
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stem = self._stems[idx]

        # ---- image --------------------------------------------------------- #
        img_path = self.img_dir / f"{stem}.jpg"
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size          # PIL gives (W, H)

        # ---- bounding box (pixel coords in original image) ----------------- #
        if stem in self._bboxes:
            xc, yc, bw, bh = self._bboxes[stem]
        else:
            # Fallback: no annotation → use whole image
            xc, yc  = orig_w / 2.0, orig_h / 2.0
            bw, bh  = float(orig_w), float(orig_h)

        # ---- trimap mask --------------------------------------------------- #
        mask_path = self.mask_dir / f"{stem}.png"
        if mask_path.exists():
            mask_pil = Image.open(mask_path)
        else:
            # Fallback: all-uncertain mask
            mask_pil = Image.fromarray(
                np.full((orig_h, orig_w), 3, dtype=np.uint8)
            )

        # ---- resize image & mask to img_size ------------------------------ #
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h

        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask_pil = mask_pil.resize(
            (self.img_size, self.img_size), Image.NEAREST
        )

        # Scale bbox coordinates accordingly
        xc = xc * scale_x
        yc = yc * scale_y
        bw = bw * scale_x
        bh = bh * scale_y

        # ---- optional transforms ------------------------------------------ #
        img_np   = np.array(image)          # H×W×3  uint8
        mask_np  = np.array(mask_pil)       # H×W    uint8

        if self.transform is not None:
            # albumentations-style: pass image + mask together so spatial
            # augmentations (flip, crop) are applied consistently.
            augmented = self.transform(image=img_np, mask=mask_np)
            img_np   = augmented["image"]
            mask_np  = augmented.get("mask", mask_np)

        if self.target_transform is not None:
            mask_np = self.target_transform(mask_np)

        # ---- convert to tensors ------------------------------------------- #
        # Image: HWC uint8 → CHW float [0, 1]
        image_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        # Normalise with ImageNet stats (standard for VGG)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        # Bbox tensor
        bbox_tensor = torch.tensor(
            [xc, yc, bw, bh], dtype=torch.float32
        )

        # Mask: trimap pixel values are 1, 2, 3 → remap to 0, 1, 2
        mask_remapped = np.vectorize(self.TRIMAP_TO_CLASS.get)(mask_np, 2)
        mask_tensor = torch.from_numpy(
            mask_remapped.astype(np.int64)
        )                                   # H×W  LongTensor

        class_id = self._info[stem]["class_id"]

        return {
            "image":    image_tensor,               # [3, H, W]  float
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "bbox":     bbox_tensor,                # [4]        float  (pixel space)
            "mask":     mask_tensor,                # [H, W]     long
            "stem":     stem,                       # for debugging
        }

    # ----------------------------------------------------------------------- #
    def get_class_names(self):
        """Return sorted list of 37 breed names derived from image stems."""
        names_set = set()
        for stem in self._stems:
            breed = "_".join(stem.split("_")[:-1])  # remove trailing _N
            names_set.add(breed)
        return sorted(names_set)


# --------------------------------------------------------------------------- #
# Quick sanity check
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "./data"

    ds = OxfordIIITPetDataset(root=root, split="trainval", img_size=224)
    sample = ds[0]

    print("image  :", sample["image"].shape,  sample["image"].dtype)
    print("class  :", sample["class_id"].item())
    print("bbox   :", sample["bbox"])
    print("mask   :", sample["mask"].shape,   sample["mask"].unique())
    print("stem   :", sample["stem"])