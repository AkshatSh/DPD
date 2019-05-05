from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import os
import sys
import unittest

import torch
import numpy as np

from dpd.training.loss_functions import TSALoss, TrainingSchedulerType

class TSALossTest(unittest.TestCase):
    def test_loss_linear(self):
        torch.manual_seed(1)
        loss_func = TSALoss(cardinality=2, training_scheduler_type=TrainingSchedulerType.linear)
        input_tensor = torch.Tensor([0.1, 0.5, 1.0])
        neg_log = -input_tensor.log()

        tsa_loss = loss_func(
            loss=input_tensor,
            negative_log_loss=neg_log,
            step=5,
            total=10,
        )

        np.testing.assert_almost_equal(tsa_loss.item(), 1.4979, 2)
    
    def test_loss_log(self):
        torch.manual_seed(1)
        loss_func = TSALoss(cardinality=2, training_scheduler_type=TrainingSchedulerType.log)
        input_tensor = torch.Tensor([0.1, 0.5, 1.0])
        neg_log = -input_tensor.log()

        tsa_loss = loss_func(
            loss=input_tensor,
            negative_log_loss=neg_log,
            step=5,
            total=10,
        )

        np.testing.assert_almost_equal(tsa_loss.item(), 2.30, 2)
    
    def test_loss_expc(self):
        torch.manual_seed(1)
        loss_func = TSALoss(cardinality=2, training_scheduler_type=TrainingSchedulerType.exponential)
        input_tensor = torch.Tensor([0.1, 0.5, 1.0])
        neg_log = -input_tensor.log()

        tsa_loss = loss_func(
            loss=input_tensor,
            negative_log_loss=neg_log,
            step=5,
            total=10,
        )

        np.testing.assert_almost_equal(tsa_loss.item(), 0.99, 2)