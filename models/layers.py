"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability.
        """
        super().__init__()
 
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
 
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor for shape [B, C, H, W].

        Returns:
            Output tensor.
        """
        
        if not self.training:
            return x
 
        # p=0 → keep everything (skip mask creation for efficiency)
        if self.p == 0.0:
            return x
        
        keep_prob = 1.0 - self.p
 
        # torch.bernoulli samples from Bernoulli(keep_prob) element-wise.
        # We create a tensor filled with keep_prob and sample from it.
        mask = torch.bernoulli(torch.full_like(x, keep_prob))
 
        # Apply mask and scale
        return x * mask / keep_prob


    # ----------------------------------------------------------------------- #
    def extra_repr(self) -> str:
        return f"p={self.p}"