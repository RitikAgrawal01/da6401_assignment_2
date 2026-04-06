"""
Inference and evaluation

Usage:
  # Evaluate multi-task model on test set
  python inference.py --mode eval \
      --data_root ./data \
      --classifier_path checkpoints/classifier_best.pth \
      --localizer_path  checkpoints/localizer_best.pth \
      --unet_path       checkpoints/unet_best.pth

  # Run on a single image (for W&B showcase)
  python inference.py --mode single --image_path my_cat.jpg \
      --classifier_path ... --localizer_path ... --unet_path ...
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models import MultiTaskPerceptionModel


# --------------------------------------------------------------------------- #
# Preprocessing (must match training pipeline)
# --------------------------------------------------------------------------- #
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_image(path: str, img_size: int = 224) -> torch.Tensor:
    """Load, resize, normalise → [1, 3, H, W] float tensor."""
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    t   = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    t   = (t - MEAN) / STD
    return t.unsqueeze(0)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalised CHW tensor back to HWC uint8 numpy array."""
    img = tensor.cpu().clone() * STD + MEAN
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Visualisation helpers
# --------------------------------------------------------------------------- #

def draw_bbox(image_np: np.ndarray, bbox: np.ndarray, color="red") -> Image.Image:
    """Draw [x_center, y_center, w, h] bbox on image."""
    img_pil = Image.fromarray(image_np)
    draw    = ImageDraw.Draw(img_pil)
    xc, yc, bw, bh = bbox
    x1, y1 = xc - bw/2, yc - bh/2
    x2, y2 = xc + bw/2, yc + bh/2
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    return img_pil


TRIMAP_COLORS = np.array([
    [0,   200,   0],    # class 0 = foreground  → green
    [200,  0,    0],    # class 1 = background  → red
    [200, 200,   0],    # class 2 = uncertain   → yellow
], dtype=np.uint8)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert HW int mask (0/1/2) to HW3 RGB colour image."""
    return TRIMAP_COLORS[mask]


# --------------------------------------------------------------------------- #
# Evaluation on test set
# --------------------------------------------------------------------------- #

def evaluate(args, device):
    model = MultiTaskPerceptionModel(
        classifier_path=args.classifier_path,
        localizer_path=args.localizer_path,
        unet_path=args.unet_path,
    ).to(device)
    model.eval()

    test_ds = OxfordIIITPetDataset(args.data_root, split="test", img_size=224)
    loader  = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    cls_correct = 0
    iou_sum     = 0.0
    dice_sum    = 0.0
    total       = 0
    eps         = 1e-6

    # W&B table for detection showcase (Section 2.5)
    table = wandb.Table(columns=[
        "image", "gt_bbox_image", "pred_bbox_image",
        "pred_class", "gt_class", "iou", "dice"
    ])
    table_count = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["class_id"].to(device)
            bboxes = batch["bbox"].to(device)
            masks  = batch["mask"].to(device)

            out = model(images)
            cls_logits = out["classification"]
            pred_bbox  = out["localization"]
            seg_logits = out["segmentation"]

            # Classification accuracy
            cls_correct += (cls_logits.argmax(1) == labels).sum().item()

            # IoU
            pw, ph = pred_bbox[:, 2].clamp(0), pred_bbox[:, 3].clamp(0)
            px1 = pred_bbox[:, 0] - pw/2; px2 = pred_bbox[:, 0] + pw/2
            py1 = pred_bbox[:, 1] - ph/2; py2 = pred_bbox[:, 1] + ph/2
            tw, th = bboxes[:, 2], bboxes[:, 3]
            tx1 = bboxes[:, 0] - tw/2; tx2 = bboxes[:, 0] + tw/2
            ty1 = bboxes[:, 1] - th/2; ty2 = bboxes[:, 1] + th/2
            iw = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0)
            ih = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0)
            inter = iw * ih
            union = pw*ph + tw*th - inter
            batch_iou = (inter / (union + eps)).mean().item()
            iou_sum  += batch_iou * images.size(0)

            # Dice (foreground class)
            preds  = seg_logits.argmax(1)
            fg_p   = (preds  == 0).float()
            fg_t   = (masks  == 0).float()
            d_inter = (fg_p * fg_t).sum()
            batch_dice = ((2*d_inter + eps) / (fg_p.sum() + fg_t.sum() + eps)).item()
            dice_sum += batch_dice * images.size(0)

            total += images.size(0)

            # Log a few samples to W&B table
            if table_count < 15:
                for i in range(min(images.size(0), 15 - table_count)):
                    img_np = denormalize(images[i])
                    # GT bbox (green) and predicted bbox (red)
                    gt_img   = draw_bbox(img_np, bboxes[i].cpu().numpy(),      "green")
                    pred_img = draw_bbox(img_np, pred_bbox[i].cpu().numpy(),   "red")
                    # Overlay both on one image for side-by-side
                    combined = draw_bbox(np.array(gt_img),
                                         pred_bbox[i].cpu().numpy(), "red")

                    pred_cls = cls_logits[i].argmax().item()
                    gt_cls   = labels[i].item()
                    b_iou    = (inter[i] / (union[i] + eps)).item()
                    b_dice   = ((2*(fg_p[i]*fg_t[i]).sum() + eps)
                                / (fg_p[i].sum() + fg_t[i].sum() + eps)).item()

                    table.add_data(
                        wandb.Image(img_np),
                        wandb.Image(np.array(combined)),
                        wandb.Image(pred_img),
                        pred_cls, gt_cls,
                        round(b_iou, 4),
                        round(b_dice, 4),
                    )
                    table_count += 1

    cls_acc  = cls_correct / total
    mean_iou = iou_sum / total
    mean_dice = dice_sum / total

    print(f"\n=== Test Results ===")
    print(f"Classification Accuracy : {cls_acc:.4f}")
    print(f"Mean IoU                : {mean_iou:.4f}")
    print(f"Mean Dice               : {mean_dice:.4f}")

    wandb.log({
        "test/cls_accuracy": cls_acc,
        "test/mean_iou":     mean_iou,
        "test/mean_dice":    mean_dice,
        "detection_table":   table,
    })


# --------------------------------------------------------------------------- #
# Single-image inference (for W&B Section 2.7 — wild images)
# --------------------------------------------------------------------------- #

def infer_single(args, device):
    model = MultiTaskPerceptionModel(
        classifier_path=args.classifier_path,
        localizer_path=args.localizer_path,
        unet_path=args.unet_path,
    ).to(device)
    model.eval()

    img_tensor = preprocess_image(args.image_path).to(device)

    with torch.no_grad():
        out = model(img_tensor)

    pred_class = out["classification"][0].argmax().item()
    pred_bbox  = out["localization"][0].cpu().numpy()
    seg_mask   = out["segmentation"][0].argmax(0).cpu().numpy()    # [H, W]

    # Original image for visualisation
    orig = Image.open(args.image_path).convert("RGB").resize((224, 224))
    orig_np = np.array(orig)

    # Overlay bbox
    bbox_img = draw_bbox(orig_np, pred_bbox, "red")

    # Overlay segmentation mask (semi-transparent)
    mask_rgb   = mask_to_rgb(seg_mask)                             # [H, W, 3]
    mask_pil   = Image.fromarray(mask_rgb).convert("RGBA")
    mask_pil.putalpha(120)                                          # semi-transparent
    orig_rgba  = Image.fromarray(orig_np).convert("RGBA")
    overlay    = Image.alpha_composite(orig_rgba, mask_pil).convert("RGB")

    print(f"\nPredicted breed class index: {pred_class}")
    print(f"Predicted bbox [xc,yc,w,h] : {pred_bbox}")

    wandb.log({
        "wild/original":     wandb.Image(orig_np),
        "wild/bbox_pred":    wandb.Image(np.array(bbox_img)),
        "wild/seg_overlay":  wandb.Image(np.array(overlay)),
        "wild/pred_class":   pred_class,
    })


# --------------------------------------------------------------------------- #
# Segmentation visualisation logger (for W&B Section 2.6)
# --------------------------------------------------------------------------- #

def log_seg_samples(args, device):
    model = MultiTaskPerceptionModel(
        classifier_path=args.classifier_path,
        localizer_path=args.localizer_path,
        unet_path=args.unet_path,
    ).to(device)
    model.eval()

    ds     = OxfordIIITPetDataset(args.data_root, split="test", img_size=224)
    loader = torch.utils.data.DataLoader(ds, batch_size=5, shuffle=True)
    batch  = next(iter(loader))

    images = batch["image"].to(device)
    masks  = batch["mask"]

    with torch.no_grad():
        seg_logits = model(images)["segmentation"]
    preds = seg_logits.argmax(1).cpu().numpy()

    seg_table = wandb.Table(columns=["original", "gt_mask", "pred_mask"])
    for i in range(images.size(0)):
        orig_np = denormalize(images[i])
        gt_rgb  = mask_to_rgb(masks[i].numpy())
        pr_rgb  = mask_to_rgb(preds[i])
        seg_table.add_data(
            wandb.Image(orig_np),
            wandb.Image(gt_rgb),
            wandb.Image(pr_rgb),
        )

    wandb.log({"seg_samples": seg_table})
    print("Segmentation samples logged to W&B.")


# --------------------------------------------------------------------------- #
# Argument parser & main
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",             choices=["eval", "single", "seg_viz"],
                   default="eval")
    p.add_argument("--data_root",        default="./data")
    p.add_argument("--image_path",       default=None,
                   help="Path to a single image (for --mode single)")
    p.add_argument("--classifier_path",  default="checkpoints/classifier_best.pth")
    p.add_argument("--localizer_path",   default="checkpoints/localizer_best.pth")
    p.add_argument("--unet_path",        default="checkpoints/unet_best.pth")
    p.add_argument("--batch_size",       type=int, default=16)
    p.add_argument("--wandb_project",    default="da6401-assignment2")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=args.wandb_project, name=f"inference-{args.mode}")

    if args.mode == "eval":
        evaluate(args, device)
    elif args.mode == "single":
        assert args.image_path, "--image_path required for mode=single"
        infer_single(args, device)
    elif args.mode == "seg_viz":
        log_seg_samples(args, device)

    wandb.finish()


if __name__ == "__main__":
    main()