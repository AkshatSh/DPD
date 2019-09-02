from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

"""
This implementation is adopted from https://github.com/HazyResearch/metal/blob/master/metal/end_model/loss.py
the udpates here allow for a masekd soft cross entropy loss, but the majority of the implementation is
the same
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def flatten(inp: Optional[torch.Tensor]) -> torch.Tensor:
    if inp is None:
        return None
    if len(inp.shape) == 2:
        return inp.view(-1)
    return inp.view(-1, inp.shape[-1])

class SoftCrossEntropyLoss(nn.Module):
    """Computes the CrossEntropyLoss while accepting probabilistic (float) targets
    Args:
        weight: a tensor of relative weights to assign to each class.
            the kwarg name 'weight' is used to match CrossEntropyLoss
        reduction: how to combine the elementwise losses
            'none': return an unreduced list of elementwise losses
            'mean': return the mean loss per elements
            'sum': return the sum of the elementwise losses
    Accepts:
        input: An [n, k] float tensor of prediction logits (not probabilities)
        target: An [n, k] float tensor of target probabilities
    """

    def __init__(self, weight=None, reduction="mean"):
        super().__init__()
        # Register as buffer is standard way to make sure gets moved /
        # converted with the Module, without making it a Parameter
        if weight is None:
            self.weight = None
        else:
            # Note: Sets the attribute self.weight as well
            self.register_buffer("weight", torch.FloatTensor(weight))
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # input: (batch, seq_len, num_classes)
        # target: (batch, seq_len, num_classes)
        # weight: (batch, seq_len)
        flat_in, flat_targ, flat_mask = flatten(input), flatten(target), flatten(mask)
        n, k = flat_in.shape
        # Note that t.new_zeros, t.new_full put tensor on same device as t
        cum_losses = input.new_zeros(n)
        for y in range(k):
            cls_idx = input.new_full((n,), y, dtype=torch.long)
            y_loss = F.cross_entropy(flat_in, cls_idx, reduction="none")
            if self.weight is not None:
                y_loss = y_loss * self.weight[y]
            # add the masked loss
            cum_losses += ((flat_targ[:, y].float() * y_loss) * flat_mask)
        if self.reduction == "none":
            return cum_losses
        elif self.reduction == "mean":
            return cum_losses.mean()
        elif self.reduction == "sum":
            return cum_losses.sum()
        else:
            raise ValueError(f"Unrecognized reduction: {self.reduction}")