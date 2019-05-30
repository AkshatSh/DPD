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
from tqdm import tqdm

import torch
import allennlp
import numpy as np
from overrides import overrides
from torch import multiprocessing

from dpd.dataset import UnlabeledBIODataset
from dpd.weak_supervision import WeakFunction, AnnotatedDataType, AnnotationType
from dpd.models.embedder import CachedTextFieldEmbedder
from dpd.common import TensorList, SparseTensorList
from dpd.weak_supervision.feature_extractor import FeatureExtractor, FeatureCollator
from dpd.utils import log_time

from ..utils import get_context_window, get_context_range, label_index, NEGATIVE_LABEL, ABSTAIN_LABEL, is_negative
from .window_function import WindowFunction

class BagWindowFunction(WindowFunction):
    def __init__(
        self,
        positive_label: str,
        context_window: int,
        feature_extractor: FeatureExtractor,
        feature_summarizer: Callable[[List[Any]], torch.Tensor] = FeatureCollator.sum,
        use_batch: bool = True,
        threshold: Optional[float] = 0.7,
        parallelize: bool = False, # shared memory issue locally
        use_sparse: bool = False, # store dictionary as sparse matrix
        **kwargs,
    ):
        self.positive_label = positive_label
        self.feature_extractor = feature_extractor
        self.context_window = context_window
        self.parallelize = parallelize
        super(BagWindowFunction, self).__init__(
            positive_label,
            feature_extractor,
            context_window,
            use_batch=use_batch,
            threshold=threshold,
            **kwargs,
        )

        self.dictionary = SparseTensorList() if use_sparse else TensorList()
        self.labels = TensorList()
        self.feature_summarizer = feature_summarizer

    @log_time(function_prefix='bag_window:train')
    def _train_model(self, training_data: List[Tuple[List[str], List[Any], str]]):
        output_dim = training_data[0][1][0].shape[-1]
        if self.feature_summarizer != FeatureCollator.sum:
            output_dim *= len(training_data[0][1])
        
        self.dictionary.preallocate((len(training_data), output_dim))
        self.labels.preallocate((len(training_data),1))
        for i, (sentence_window, feature_window, label) in enumerate(training_data):
            if is_negative(label):
                continue
            window_summary = self.feature_summarizer(feature_window)
            self.dictionary.append(window_summary.float())
            self.labels.append(torch.Tensor([label_index(label)]))
    
    def _predict(self, features: List[torch.Tensor]) -> int:
        feature_summary = self.feature_summarizer(features)
        labels = self.labels.tensor().long()
        found_index = self.dictionary.contains(feature_summary)
        if found_index == -1:
            return 0 # no confidence (should be ABSTAIN)
        label = labels[found_index]
        return label.item()
    
    def _predict_probabilities(self, features: List[torch.Tensor]) -> float:
        feature_summary = self.feature_summarizer(features)
        labels = self.labels.tensor().long()
        found_index = self.dictionary.contains(feature_summary)
        if found_index == -1:
            return 0. # no confidence (should be ABSTAIN)
        label = labels[found_index]
        return 2 * label.item() - 1 # (0 -> -1 ,1 -> 1)
    
    @log_time(function_prefix='bag_window:predict')
    def _batch_predict(self, features: List[List[torch.Tensor]]) -> List[int]:
        return list(map(lambda f: self._predict(f), features))
    
    def _batch_probabilities(self, features: List[List[torch.Tensor]]) -> List[float]:
        if self.parallelize:
            pool = multiprocessing.Pool()
            self.dictionary.share_memory()
            self.labels.share_memory()
            parallel_res = pool.map(self._predict_probabilities, features)
            return list(parallel_res)
        else:
            return list(map(self._predict_probabilities, features))
    
    @overrides
    def __str__(self):
        return f'BagWindowFunction({self.context_window})({self.feature_extractor})'