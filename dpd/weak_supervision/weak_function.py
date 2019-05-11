from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import os

import torch
import allennlp
from dpd.dataset import UnlabeledBIODataset
from dpd.weak_supervision.utils import ABSTAIN_LABEL, NEGATIVE_LABEL

from .types import (
    AnnotatedDataType,
    AnnotationType,
)

class WeakFunction(object):
    '''
    An abstract class for a weak function
    this function can be used to provide weak supervision to a dataset
    it is intended to be called via `WeakFunction.train(train_data)`
    and applied via `Weakfunction.evaluate(unlabeld_corpus)`
    '''
    def __init__(self, positive_label: str, threshold: Optional[float] = None, **kwargs):
        self.positive_label = positive_label
        self.threshold = threshold

    def train(self, train_data: AnnotatedDataType):
        raise NotImplementedError()
    
    def evaluate(self, unlabeled_corpus: UnlabeledBIODataset) -> AnnotatedDataType:
        raise NotImplementedError()

    def get_label(self, index: int) -> str:
        if index == 0:
            return NEGATIVE_LABEL
        return self.positive_label

    def get_probability_label(self, confidence: float) -> str:
        '''
        Assigns positive or negative label if confidence is above a threshold
        otherwise provides an ABSTAIN label
        '''
        assert self.threshold is not None

        if confidence > self.threshold:
            return self.positive_label
        elif confidence < -self.threshold:
            return NEGATIVE_LABEL
        else:
            return ABSTAIN_LABEL
    
    def get_snorkel_index(self, index: int) -> str:
        if index == 2:
            return self.positive_label
        elif index == 1:
            return NEGATIVE_LABEL
        else:
            return ABSTAIN_LABEL