from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Any,
    Callable
)

import os
import sys

import torch
import allennlp
import numpy as np
from overrides import overrides

from dpd.dataset import UnlabeledBIODataset
from dpd.weak_supervision import WeakFunction, AnnotatedDataType, AnnotationType
from dpd.models.embedder import CachedTextFieldEmbedder
from dpd.utils import TensorList
from dpd.weak_supervision.feature_extractor import FeatureExtractor, FeatureCollator

from ..utils import get_context_window, get_context_range, label_index, NEGATIVE_LABEL
from .window_function import WindowFunction

class BagWindowFunction(WindowFunction):
    def __init__(
        self,
        positive_label: str,
        context_window: int,
        feature_extractor: FeatureExtractor,
        feature_summarizer: Callable[[List[Any]], torch.Tensor] = FeatureCollator.sum,
        use_batch: bool = True,
        **kwargs,
    ):
        self.positive_label = positive_label
        self.feature_extractor = feature_extractor
        self.context_window = context_window
        super(BagWindowFunction, self).__init__(
            positive_label,
            feature_extractor,
            context_window,
            use_batch=use_batch,
        )

        self.dictionary = TensorList()
        self.labels = TensorList()
        self.feature_summarizer = feature_summarizer
    
    def _train_model(self, training_data: List[Tuple[List[str], List[Any], str]]):
        for i, (sentence_window, feature_window, label) in enumerate(training_data):
            window_summary = self.feature_summarizer(feature_window)
            self.dictionary.append(window_summary.float())
            self.labels.append(torch.Tensor([label_index(label)]))
    
    def _predict(self, features: List[torch.Tensor]) -> int:
        feature_summary = self.feature_summarizer(features).long()
        dictionary, labels = self.dictionary.tensor().long(), self.labels.tensor().long()
        for i, (tensor, label) in enumerate(zip(dictionary, labels)):
            if (tensor == feature_summary).all():
                return label.item()
        return 0
    
    def _batch_predict(self, features: List[List[torch.Tensor]]) -> List[int]:
        return list(map(lambda f: self._predict(f), features))
    
    @overrides
    def __str__(self):
        return f'BagWindowFunction({self.context_window})({self.feature_extractor})'