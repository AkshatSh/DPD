'''
From Unsupervised Data Augmentation by 
Qizhe Xie1, Zihang Dai, Eduard Hovy, Minh-Thang Luong, Quoc V. Le

https://arxiv.org/pdf/1904.12848.pdf

Implements the Training Signal Annealing method described in the paper above
as a loss function
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

class TrainingSchedulerType(Enum):
    log = 1
    linear = 2
    exponential = 3

class TrainingScheduler(object):
    def __init__(
        self,
        cardinality: int,
        training_scheduler_type: TrainingSchedulerType = TrainingSchedulerType.exponential,
    ):
        self.cardinality = cardinality
        self.training_scheduler_type = training_scheduler_type

    def _compute_lambda(self, step: int, total: int) -> float:
        if self.training_scheduler_type == TrainingSchedulerType.log:
            return np.log(step / total)
        elif self.training_scheduler_type == TrainingSchedulerType.linear:
            return step / total
        elif self.training_scheduler_type == TrainingSchedulerType.exponential:
            return np.exp(step / total)
        else:
            raise Exception(f'Unknown training scheduler: {self.training_scheduler_type}')
    
    def compute_threshold(self, step: int, total: int) -> float:
        expected_pred = 1. / self.cardinality
        return expected_pred + self._compute_lambda(step, total) * (1 - expected_pred)

class TSALoss(nn.Module):
    def __init__(
        self,
        cardinality: int,
        training_scheduler_type: TrainingSchedulerType,
    ):
        super(TSALoss, self).__init__()
        self.training_scheduler = TrainingScheduler(
            cardinality=cardinality,
            training_scheduler_type=training_scheduler_type,
        )
    
    def forward(
        self,
        loss: torch.Tensor,
        negative_log_loss: torch.Tensor,
        step: int,
        total: int,
    ) -> torch.Tensor:
        '''
        compute the TSA loss described in the UDA paper, to help with overfitting
        when there are a small number of instances

        args:
            loss: the 0 to 1 probability loss of the sequence (batch_size, 1)
            negative_log_loss: is the negative transform of the loss above (batch_size, 1)
        returns:
            the TSA loss, only computing a subset of the loss that is below the threshold
            computed with the training schedule (1,)
        '''
        threshold: float = self.training_scheduler.compute_threshold(
            step=step,
            total=total,
        )

        indicator_mask = (loss < threshold).float()
        valid_count = indicator_mask.sum()
        summation = (negative_log_loss * indicator_mask).sum()
        return summation / valid_count

