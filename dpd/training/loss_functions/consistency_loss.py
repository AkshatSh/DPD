'''
From Unsupervised Data Augmentation by 
Qizhe Xie1, Zihang Dai, Eduard Hovy, Minh-Thang Luong, Quoc V. Le

https://arxiv.org/pdf/1904.12848.pdf

Implements the consistency loss between a pair of instances
'''
from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import os
import sys
from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import allennlp

class ConsistencyLoss(nn.Module):
    def __init__(
        self,
    ):
        super(ConsistencyLoss, self).__init__()
    
    def forward(
        self,
        original_prediction: torch.Tensor,
        augmented_prediciton: torch.Tensor,
    ) -> torch.Tensor:
        return F.kl_div(original_prediction, augmented_prediciton)