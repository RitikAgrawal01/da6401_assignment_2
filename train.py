"""Training entry point for DA6401 Assignment-2.

Trains three models sequentially (Task 1 → 2 → 3).
Run examples:
  python train.py --task cls  --data_root ./data --epochs 30
  python train.py --task loc  --data_root ./data --epochs 20 --pretrained checkpoints/classifier_best.pth
  python train.py --task seg  --data_root ./data --epochs 20 --pretrained checkpoints/classifier_best.pth
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models import VGG11Classifier, VGG11Localizer, VGG11UNet
from losses import IoULoss


# --------------------------------------------------------------------------- #
# Dice loss helper (used for segmentation alongside CrossEntropy)
# --------------------------------------------------------------------------- #
class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation."""
    def __init__(self, num_classes: int = 3, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C, H, W];  targets: [B, H, W] long
        probs = torch.softmax(logits, dim=1)                # [B, C, H, W]
        B, C, H, W = probs.shape

        # One-hot encode targets
        targets_oh = torch.zeros_like(probs)                # [B, C, H, W]
        targets_oh.scatter_(1, targets.unsqueeze(1), 1)

        intersection = (probs * targets_oh).sum(dim=(2, 3)) # [B, C]
        cardinality  = (probs + targets_oh).sum(dim=(2, 3)) # [B, C]
        dice_per_class = (2 * intersection + self.eps) / (cardinality + self.eps)
        return 1 - dice_per_class.mean()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_iou_metric(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> float:
    """Compute mean IoU over a batch (for logging, not backprop)."""
    eps = 1e-6
    pw, ph = pred_boxes[:, 2].clamp(0), pred_boxes[:, 3].clamp(0)
    px1, py1 = pred_boxes[:, 0] - pw/2, pred_boxes[:, 1] - ph/2
    px2, py2 = pred_boxes[:, 0] + pw/2, pred_boxes[:, 1] + ph/2

    tw, th = target_boxes[:, 2], target_boxes[:, 3]
    tx1, ty1 = target_boxes[:, 0] - tw/2, target_boxes[:, 1] - th/2
    tx2, ty2 = target_boxes[:, 0] + tw/2, target_boxes[:, 1] + th/2

    iw = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0)
    ih = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0)
    inter = iw * ih
    union = pw*ph + tw*th - inter
    return (inter / (union + eps)).mean().item()


def compute_dice(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute mean Dice score (foreground class only) for logging."""
    preds = logits.argmax(dim=1)               # [B, H, W]
    fg_pred = (preds == 0).float()
    fg_true = (targets == 0).float()
    eps = 1e-6
    inter = (fg_pred * fg_true).sum()
    return ((2 * inter + eps) / (fg_pred.sum() + fg_true.sum() + eps)).item()


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def get_transforms(train: bool):
    """Return albumentations transforms."""
    try:
        import albumentations as A
        if train:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            return A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
    except ImportError:
        return None   # dataset already normalises without albumentations


# --------------------------------------------------------------------------- #
# Task-specific training loops
# --------------------------------------------------------------------------- #

def train_classification(args, device):
    print("\n=== Task 1: Classification ===")

    train_ds = OxfordIIITPetDataset(args.data_root, split="trainval",
                                     img_size=224,
                                     transform=get_transforms(train=True))
    val_ds   = OxfordIIITPetDataset(args.data_root, split="test",
                                     img_size=224,
                                     transform=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model     = VGG11Classifier(num_classes=37, dropout_p=0.5).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ---- Train --------------------------------------------------------- #
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["class_id"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += images.size(0)

        scheduler.step()
        train_loss /= total
        train_acc   = correct / total

        # ---- Validate ------------------------------------------------------- #
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["class_id"].to(device)
                logits = model(images)
                loss   = criterion(logits, labels)
                val_loss    += loss.item() * images.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += images.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"Val loss={val_loss:.4f} acc={val_acc:.3f}")

        wandb.log({
            "cls/train_loss": train_loss, "cls/train_acc": train_acc,
            "cls/val_loss":   val_loss,   "cls/val_acc":   val_acc,
            "epoch": epoch,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
            }, os.path.join(args.checkpoint_dir, "classifier_best.pth"))

    print(f"Best val accuracy: {best_val_acc:.4f}")


def train_localization(args, device):
    print("\n=== Task 2: Localization ===")

    train_ds = OxfordIIITPetDataset(args.data_root, split="trainval", img_size=224,
                                     transform=get_transforms(train=True))
    val_ds   = OxfordIIITPetDataset(args.data_root, split="test",    img_size=224,
                                     transform=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = VGG11Localizer(
        pretrained_encoder=args.pretrained,
        freeze_encoder=False,            # fine-tune
        img_size=224,
        dropout_p=0.5,
    ).to(device)

    iou_loss  = IoULoss(reduction="mean")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_iou_sum, total = 0.0, 0.0, 0

        for batch in train_loader:
            images = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)

            optimizer.zero_grad()
            pred = model(images)
            loss = iou_loss(pred, bboxes)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = images.size(0)
            train_loss    += loss.item() * bs
            train_iou_sum += compute_iou_metric(pred.detach(), bboxes) * bs
            total         += bs

        scheduler.step()
        train_loss /= total
        train_iou   = train_iou_sum / total

        model.eval()
        val_loss, val_iou_sum, val_total = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)
                pred   = model(images)
                loss   = iou_loss(pred, bboxes)
                bs = images.size(0)
                val_loss    += loss.item() * bs
                val_iou_sum += compute_iou_metric(pred, bboxes) * bs
                val_total   += bs

        val_loss /= val_total
        val_iou   = val_iou_sum / val_total

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train loss={train_loss:.4f} IoU={train_iou:.3f} | "
              f"Val loss={val_loss:.4f} IoU={val_iou:.3f}")

        wandb.log({
            "loc/train_loss": train_loss, "loc/train_iou": train_iou,
            "loc/val_loss":   val_loss,   "loc/val_iou":   val_iou,
            "epoch": epoch,
        })

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_iou": val_iou,
            }, os.path.join(args.checkpoint_dir, "localizer_best.pth"))

    print(f"Best val IoU: {best_val_iou:.4f}")


def train_segmentation(args, device):
    print("\n=== Task 3: Segmentation ===")

    train_ds = OxfordIIITPetDataset(args.data_root, split="trainval", img_size=224,
                                     transform=get_transforms(train=True))
    val_ds   = OxfordIIITPetDataset(args.data_root, split="test",    img_size=224,
                                     transform=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = VGG11UNet(
        num_classes=3,
        pretrained_encoder=args.pretrained,
        freeze_encoder=False,
        dropout_p=0.5,
    ).to(device)

    # Combined loss: CrossEntropy + Dice
    # Justification: CE ensures correct per-pixel class probability calibration;
    # Dice directly optimises the overlap metric used at evaluation time.
    ce_loss   = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes=3)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_dice_sum, total = 0.0, 0.0, 0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss   = ce_loss(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = images.size(0)
            train_loss      += loss.item() * bs
            train_dice_sum  += compute_dice(logits.detach(), masks) * bs
            total           += bs

        scheduler.step()
        train_loss /= total
        train_dice  = train_dice_sum / total

        model.eval()
        val_loss, val_dice_sum, val_total = 0.0, 0.0, 0
        val_px_correct, val_px_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks  = batch["mask"].to(device)
                logits = model(images)
                loss   = ce_loss(logits, masks) + dice_loss(logits, masks)
                bs     = images.size(0)
                val_loss      += loss.item() * bs
                val_dice_sum  += compute_dice(logits, masks) * bs
                val_total     += bs
                preds = logits.argmax(1)
                val_px_correct += (preds == masks).sum().item()
                val_px_total   += masks.numel()

        val_loss  /= val_total
        val_dice   = val_dice_sum / val_total
        val_px_acc = val_px_correct / val_px_total

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train loss={train_loss:.4f} Dice={train_dice:.3f} | "
              f"Val loss={val_loss:.4f} Dice={val_dice:.3f} PixAcc={val_px_acc:.3f}")

        wandb.log({
            "seg/train_loss": train_loss, "seg/train_dice": train_dice,
            "seg/val_loss":   val_loss,   "seg/val_dice":   val_dice,
            "seg/val_pixel_acc": val_px_acc,
            "epoch": epoch,
        })

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": val_dice,
            }, os.path.join(args.checkpoint_dir, "unet_best.pth"))

    print(f"Best val Dice: {best_val_dice:.4f}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="DA6401 Assignment-2 Training")
    p.add_argument("--task",           choices=["cls", "loc", "seg", "all"],
                   default="cls",      help="Which task to train")
    p.add_argument("--data_root",      default="./data",
                   help="Path to Oxford-IIIT Pet dataset root")
    p.add_argument("--checkpoint_dir", default="./checkpoints",
                   help="Directory to save model checkpoints")
    p.add_argument("--pretrained",     default=None,
                   help="Path to pre-trained classifier checkpoint (for loc/seg)")
    p.add_argument("--epochs",         type=int, default=30)
    p.add_argument("--batch_size",     type=int, default=32)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--wandb_project",  default="da6401-assignment2")
    p.add_argument("--wandb_name",     default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"task-{args.task}-{time.strftime('%m%d-%H%M')}",
        config=vars(args),
    )

    if args.task in ("cls", "all"):
        train_classification(args, device)

    if args.task in ("loc", "all"):
        train_localization(args, device)

    if args.task in ("seg", "all"):
        train_segmentation(args, device)

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()