from typing import (
    List,
    Tuple,
    Dict,
)

import os
import torch
import allennnlp

from dpd.dataset import UnlabeledBIODataset
from dpd.weak_supervision import WeakFunction, AnnotatedDataType

class GloveLinearFunction(object):
    def __init__(self):
        pass

    def train(self, train_data: AnnotatedDataType):
        raise NotImplementedError()
    
    def evaluate(self, unlabeled_corpus: UnlabeledBIODataset) -> AnnotatedData:
        raise NotImplementedError()
