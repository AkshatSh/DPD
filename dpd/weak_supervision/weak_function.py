from typing import (
    List,
    Tuple,
    Dict,
)

import os

import torch
import allennlp
from dpd.dataset import UnlabeledBIODataset


AnnotationType = Dict[str, object]
AnnotatedDataType = List[AnnotationType]

class WeakFunction(object):
    '''
    An abstract class for a weak function
    this function can be used to provide weak supervision to a dataset
    it is intended to be called via `WeakFunction.train(train_data)`
    and applied via `Weakfunction.evaluate(unlabeld_corpus)`
    '''
    def __init__(self):
        pass

    def train(self, train_data: AnnotatedDataType):
        raise NotImplementedError()
    
    def evaluate(self, unlabeled_corpus: UnlabeledBIODataset) -> AnnotatedData:
        raise NotImplementedError()