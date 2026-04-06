"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'"
            )
        
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format.
            
        Returns:
            Scalar loss (or [B] tensor if reduction='none')."""
            
        # ------------------------------------------------------------------- #
        # Step 1: Convert [x_center, y_center, w, h] → [x1, y1, x2, y2]
        #         Clamp w/h to ≥ 0 to handle degenerate predictions.
        # ------------------------------------------------------------------- #
        pred_w  = pred_boxes[:, 2].clamp(min=0)
        pred_h  = pred_boxes[:, 3].clamp(min=0)
        pred_x1 = pred_boxes[:, 0] - pred_w / 2
        pred_y1 = pred_boxes[:, 1] - pred_h / 2
        pred_x2 = pred_boxes[:, 0] + pred_w / 2
        pred_y2 = pred_boxes[:, 1] + pred_h / 2
 
        tgt_w  = target_boxes[:, 2].clamp(min=0)
        tgt_h  = target_boxes[:, 3].clamp(min=0)
        tgt_x1 = target_boxes[:, 0] - tgt_w / 2
        tgt_y1 = target_boxes[:, 1] - tgt_h / 2
        tgt_x2 = target_boxes[:, 0] + tgt_w / 2
        tgt_y2 = target_boxes[:, 1] + tgt_h / 2
 
        # ------------------------------------------------------------------- #
        # Step 2: Intersection rectangle
        # ------------------------------------------------------------------- #
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)
 
        # clamp to 0 so non-overlapping boxes give 0 intersection
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h                     # [B]
 
        # ------------------------------------------------------------------- #
        # Step 3: Union area
        # ------------------------------------------------------------------- #
        pred_area = pred_w * pred_h                         # [B]
        tgt_area  = tgt_w  * tgt_h                         # [B]
        union_area = pred_area + tgt_area - inter_area      # [B]
 
        # ------------------------------------------------------------------- #
        # Step 4: IoU and loss
        # ------------------------------------------------------------------- #
        iou  = inter_area / (union_area + self.eps)        # [B]  in [0, 1]
        loss = 1.0 - iou                                   # [B]  in [0, 1]
 
        # ------------------------------------------------------------------- #
        # Step 5: Reduction
        # ------------------------------------------------------------------- #
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:   # 'none'
            return loss
 
    # ----------------------------------------------------------------------- #
    def extra_repr(self) -> str:
        return f"eps={self.eps}, reduction='{self.reduction}'"
    
    

# --------------------------------------------------------------------------- #
# Quick unit tests
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    loss_fn = IoULoss()
 
    # Perfect overlap → IoU=1 → loss=0
    boxes = torch.tensor([[50., 50., 100., 100.]])
    print("perfect:", loss_fn(boxes, boxes).item())          # ~0.0
 
    # No overlap → IoU=0 → loss=1
    pred   = torch.tensor([[0.,  0., 10., 10.]])
    target = torch.tensor([[200., 200., 10., 10.]])
    print("no overlap:", loss_fn(pred, target).item())       # ~1.0
 
    # 50% overlap test
    pred   = torch.tensor([[75., 50., 100., 100.]])          # x1=25 x2=125
    target = torch.tensor([[50., 50., 100., 100.]])           # x1=0  x2=100
    # inter=[25,0]→[100,100] → 75×100=7500; union=10000+10000-7500=12500; iou=0.6
    print("partial:", loss_fn(pred, target).item())          # ~0.4
 
    # Gradient check
    pred = torch.tensor([[50., 50., 100., 100.]], requires_grad=True)
    l = loss_fn(pred, boxes)
    l.backward()
    print("grad:", pred.grad)
 