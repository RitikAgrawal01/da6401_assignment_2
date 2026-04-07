"""Training entry point for DA6401 Assignment-2.

Checkpoint names per submission guidelines:
  classifier.pth  (Task 1)
  localizer.pth   (Task 2)
  unet.pth        (Task 3)

Localization loss = MSE + IoULoss  (per additional instructions).
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models import VGG11Classifier, VGG11Localizer, VGG11UNet
from losses import IoULoss


# --------------------------------------------------------------------------- #
# Dice loss (for segmentation)
# --------------------------------------------------------------------------- #
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs  = torch.softmax(logits, dim=1)
        tgt_oh = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1)
        inter  = (probs * tgt_oh).sum(dim=(2, 3))
        card   = (probs + tgt_oh).sum(dim=(2, 3))
        return 1 - ((2 * inter + self.eps) / (card + self.eps)).mean()


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def batch_iou(pred, tgt, eps=1e-6):
    pw, ph = pred[:, 2].clamp(0), pred[:, 3].clamp(0)
    px1, px2 = pred[:, 0] - pw/2, pred[:, 0] + pw/2
    py1, py2 = pred[:, 1] - ph/2, pred[:, 1] + ph/2
    tw, th = tgt[:, 2], tgt[:, 3]
    tx1, tx2 = tgt[:, 0] - tw/2, tgt[:, 0] + tw/2
    ty1, ty2 = tgt[:, 1] - th/2, tgt[:, 1] + th/2
    iw = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0)
    ih = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0)
    inter = iw * ih
    union = pw*ph + tw*th - inter
    return (inter / (union + eps)).mean().item()


def dice_score(logits, masks, eps=1e-6):
    preds = logits.argmax(1)
    fg_p = (preds == 0).float()
    fg_t = (masks == 0).float()
    return ((2*(fg_p*fg_t).sum() + eps) / (fg_p.sum() + fg_t.sum() + eps)).item()


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def get_transforms(train: bool):
    try:
        import albumentations as A
        if train:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.5),
                A.GaussNoise(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.3),
            ])
        else:
            return None
    except ImportError:
        return None


# --------------------------------------------------------------------------- #
# Task 1: Classification
# --------------------------------------------------------------------------- #
def train_classification(args, device):
    print('\n=== Task 1: Classification ===')

    train_ds = OxfordIIITPetDataset(args.data_root, 'trainval', 224, transform=get_transforms(True))
    val_ds   = OxfordIIITPetDataset(args.data_root, 'test',     224)
    pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)

    model     = VGG11Classifier(num_classes=37, dropout_p=0.5).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    def lr_lambda(epoch):
        if epoch < 5: return (epoch + 1) / 5
        import math
        return 0.5 * (1 + math.cos(math.pi * (epoch - 5) / (args.epochs - 5)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    wandb.init(project=args.wandb_project, name=args.wandb_name or f'task1-cls',
               config=vars(args), reinit=True)

    best_val_acc = 0.0
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for batch in train_loader:
            imgs   = batch['image'].to(device)
            labels = batch['class_id'].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            t_loss    += loss.item() * imgs.size(0)
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total   += imgs.size(0)
        scheduler.step()
        t_loss /= t_total; t_acc = t_correct / t_total

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch['image'].to(device)
                labels = batch['class_id'].to(device)
                logits = model(imgs)
                v_loss    += criterion(logits, labels).item() * imgs.size(0)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total   += imgs.size(0)
                all_preds.extend(logits.argmax(1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        v_loss /= v_total; v_acc = v_correct / v_total

        from sklearn.metrics import f1_score
        v_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f'Epoch {epoch:3d}/{args.epochs} | Train loss={t_loss:.4f} acc={t_acc:.3f} | Val loss={v_loss:.4f} acc={v_acc:.3f} F1={v_f1:.3f}')
        wandb.log({'cls/train_loss': t_loss, 'cls/train_acc': t_acc,
                   'cls/val_loss': v_loss,   'cls/val_acc': v_acc,
                   'cls/val_f1': v_f1,       'epoch': epoch})

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            # Checkpoint name: classifier.pth (per submission guidelines)
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_acc': v_acc},
                       os.path.join(args.checkpoint_dir, 'classifier.pth'))
            print(f'  ✓ Saved classifier.pth (val_acc={v_acc:.4f})')

    wandb.finish()
    print(f'Task 1 done. Best val accuracy: {best_val_acc:.4f}')


# --------------------------------------------------------------------------- #
# Task 2: Localization  — Loss = MSE + IoULoss
# --------------------------------------------------------------------------- #
def train_localization(args, device):
    print('\n=== Task 2: Localization ===')

    train_ds = OxfordIIITPetDataset(args.data_root, 'trainval', 224)
    val_ds   = OxfordIIITPetDataset(args.data_root, 'test',     224)
    pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)

    model = VGG11Localizer(
        pretrained_encoder=os.path.join(args.checkpoint_dir, 'classifier.pth'),
        freeze_encoder=True,
        dropout_p=0.5,
    ).to(device)

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction='mean')

    # Phase 1: only head
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-3
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                      patience=3, factor=0.5)

    wandb.init(project=args.wandb_project, name=args.wandb_name or f'task2-loc',
               config=vars(args), reinit=True)

    best_val_iou = 0.0
    UNFREEZE_EPOCH = 16

    for epoch in range(1, args.epochs + 1):
        if epoch == UNFREEZE_EPOCH:
            print('--- Unfreezing encoder ---')
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                              patience=3, factor=0.5)

        model.train()
        t_loss, t_iou, t_total = 0.0, 0.0, 0
        for batch in train_loader:
            imgs   = batch['image'].to(device)
            bboxes = batch['bbox'].to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            # MSE + IoU loss (per additional instructions)
            loss = mse_loss(pred, bboxes) + iou_loss(pred, bboxes)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            bs = imgs.size(0)
            t_loss += loss.item() * bs
            t_iou  += batch_iou(pred.detach(), bboxes) * bs
            t_total += bs
        t_loss /= t_total; t_iou /= t_total

        model.eval()
        v_loss, v_iou, v_total = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch['image'].to(device)
                bboxes = batch['bbox'].to(device)
                pred   = model(imgs)
                loss   = mse_loss(pred, bboxes) + iou_loss(pred, bboxes)
                bs = imgs.size(0)
                v_loss += loss.item() * bs
                v_iou  += batch_iou(pred, bboxes) * bs
                v_total += bs
        v_loss /= v_total; v_iou /= v_total
        scheduler.step(v_iou)

        phase = 'frozen' if epoch < UNFREEZE_EPOCH else 'finetune'
        print(f'Epoch {epoch:3d}/{args.epochs} [{phase}] | Train loss={t_loss:.4f} IoU={t_iou:.3f} | Val loss={v_loss:.4f} IoU={v_iou:.3f}')
        wandb.log({'loc/train_loss': t_loss, 'loc/train_iou': t_iou,
                   'loc/val_loss': v_loss,   'loc/val_iou': v_iou, 'epoch': epoch})

        if v_iou > best_val_iou:
            best_val_iou = v_iou
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_iou': v_iou},
                       os.path.join(args.checkpoint_dir, 'localizer.pth'))
            print(f'  ✓ Saved localizer.pth (val_iou={v_iou:.4f})')

    wandb.finish()
    print(f'Task 2 done. Best val IoU: {best_val_iou:.4f}')


# --------------------------------------------------------------------------- #
# Task 3: Segmentation
# --------------------------------------------------------------------------- #
def train_segmentation(args, device):
    print('\n=== Task 3: Segmentation ===')

    train_ds = OxfordIIITPetDataset(args.data_root, 'trainval', 224, transform=get_transforms(True))
    val_ds   = OxfordIIITPetDataset(args.data_root, 'test',     224)
    pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)

    model = VGG11UNet(
        num_classes=3,
        pretrained_encoder=os.path.join(args.checkpoint_dir, 'classifier.pth'),
        freeze_encoder=False,
        dropout_p=0.5,
    ).to(device)

    ce_loss   = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    wandb.init(project=args.wandb_project, name=args.wandb_name or f'task3-seg',
               config=vars(args), reinit=True)

    best_val_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss, t_dice, t_total = 0.0, 0.0, 0
        for batch in train_loader:
            imgs  = batch['image'].to(device)
            masks = batch['mask'].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = ce_loss(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            bs = imgs.size(0)
            t_loss  += loss.item() * bs
            t_dice  += dice_score(logits.detach(), masks) * bs
            t_total += bs
        scheduler.step()
        t_loss /= t_total; t_dice /= t_total

        model.eval()
        v_loss, v_dice, v_pxc, v_pxt, v_total = 0.0, 0.0, 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs  = batch['image'].to(device)
                masks = batch['mask'].to(device)
                logits = model(imgs)
                loss   = ce_loss(logits, masks) + dice_loss(logits, masks)
                bs = imgs.size(0)
                v_loss += loss.item() * bs
                v_dice += dice_score(logits, masks) * bs
                preds   = logits.argmax(1)
                v_pxc  += (preds == masks).sum().item()
                v_pxt  += masks.numel()
                v_total += bs
        v_loss /= v_total; v_dice /= v_total
        v_pxacc = v_pxc / v_pxt

        print(f'Epoch {epoch:3d}/{args.epochs} | Train loss={t_loss:.4f} Dice={t_dice:.3f} | Val loss={v_loss:.4f} Dice={v_dice:.3f} PixAcc={v_pxacc:.3f}')
        wandb.log({'seg/train_loss': t_loss, 'seg/train_dice': t_dice,
                   'seg/val_loss': v_loss,   'seg/val_dice': v_dice,
                   'seg/val_pixel_acc': v_pxacc, 'epoch': epoch})

        if v_dice > best_val_dice:
            best_val_dice = v_dice
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_dice': v_dice},
                       os.path.join(args.checkpoint_dir, 'unet.pth'))
            print(f'  ✓ Saved unet.pth (val_dice={v_dice:.4f})')

    wandb.finish()
    print(f'Task 3 done. Best val Dice: {best_val_dice:.4f}')


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task',           choices=['cls', 'loc', 'seg', 'all'], default='cls')
    p.add_argument('--data_root',      default='./data')
    p.add_argument('--checkpoint_dir', default='./checkpoints')
    p.add_argument('--epochs',         type=int,   default=50)
    p.add_argument('--batch_size',     type=int,   default=32)
    p.add_argument('--lr',             type=float, default=3e-4)
    p.add_argument('--num_workers',    type=int,   default=0)
    p.add_argument('--wandb_project',  default='da6401-assignment2')
    p.add_argument('--wandb_name',     default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.task in ('cls', 'all'): train_classification(args, device)
    if args.task in ('loc', 'all'): train_localization(args, device)
    if args.task in ('seg', 'all'): train_segmentation(args, device)

    print('Done.')


if __name__ == '__main__':
    main()