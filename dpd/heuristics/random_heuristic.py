from typing import (
    List,
    Tuple,
    Dict,
)

import os

import torch
from torch import nn
from torch.nn import functional as F

from dpd.dataset import UnlabeledBIODataset

class RandomHeuristic(object):
    def __init__(
        self,
    ):
        pass
    
    def evaluate(
        self,
        unlabeled_corpus: UnlabeledBIODataset,
    ) -> torch.Tensor:
        '''
        evaluate the random heuristic on every item and return the
        weights associated with the unlabeled corpus

        input:
            ``unlabeled_corpus`` UnlabeledBIODataset
                the unlabeled corpus to evaluate this heuristic on
        output:
            ``torch.Tensor``
                get the weighted unlabeled corpus
        '''
        distr = torch.zeros((len(unlabeled_corpus),))
        return F.softmax(distr, dim=0)
