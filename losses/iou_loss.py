"""Custom IoU loss — range strictly [0, 1].

Autograder import:
    from losses.iou_loss import IoULoss

Requirements per additional instructions:
  - Output range: [0, 1]
  - Reduction types: 'mean' (default), 'sum'
  - Fully differentiable (gradient viable)
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """Intersection-over-Union loss for [x_center, y_center, w, h] boxes.

    Loss = 1 - IoU  ->  range [0, 1].

    Args:
        eps      : Small value to avoid division by zero.
        reduction: 'mean' (default) | 'sum' | 'none'.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        if reduction not in {'none', 'mean', 'sum'}:
            raise ValueError(
                f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'"
            )
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IoU loss.

        Args:
            pred_boxes  : [B, 4] predicted   [x_c, y_c, w, h]
            target_boxes: [B, 4] groundtruth [x_c, y_c, w, h]

        Returns:
            Scalar loss if reduction != 'none', else [B] tensor.
            Values are in [0, 1].
        """
        # Convert xywh -> xyxy, clamp w/h >= 0
        pw  = pred_boxes[:, 2].clamp(min=0)
        ph  = pred_boxes[:, 3].clamp(min=0)
        px1 = pred_boxes[:, 0] - pw / 2
        py1 = pred_boxes[:, 1] - ph / 2
        px2 = pred_boxes[:, 0] + pw / 2
        py2 = pred_boxes[:, 1] + ph / 2

        tw  = target_boxes[:, 2].clamp(min=0)
        th  = target_boxes[:, 3].clamp(min=0)
        tx1 = target_boxes[:, 0] - tw / 2
        ty1 = target_boxes[:, 1] - th / 2
        tx2 = target_boxes[:, 0] + tw / 2
        ty2 = target_boxes[:, 1] + th / 2

        # Intersection
        iw = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
        ih = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
        inter = iw * ih

        # Union
        union = pw * ph + tw * th - inter

        # IoU in [0, 1],  Loss = 1 - IoU in [0, 1]
        iou  = inter / (union + self.eps)
        loss = 1.0 - iou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return f"eps={self.eps}, reduction='{self.reduction}'"