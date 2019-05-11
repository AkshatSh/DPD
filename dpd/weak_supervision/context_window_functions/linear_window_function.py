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
import time

import torch
import allennlp
import numpy as np
from overrides import overrides

from dpd.dataset import UnlabeledBIODataset
from dpd.weak_supervision import WeakFunction, AnnotatedDataType, AnnotationType
from dpd.models.embedder import CachedTextFieldEmbedder
from dpd.models import LinearType, construct_linear_classifier
from dpd.utils import TensorList, log_time
from dpd.weak_supervision.feature_extractor import FeatureExtractor, FeatureCollator

from ..utils import get_context_window, get_context_range, label_index, NEGATIVE_LABEL
from .window_function import WindowFunction

class LinearWindowFunction(WindowFunction):
    def __init__(
        self,
        positive_label: str,
        context_window: int,
        feature_extractor: FeatureExtractor,
        feature_summarizer: Callable[[List[Any]], torch.Tensor] = FeatureCollator.sum,
        linear_type: LinearType = LinearType.SVM_LINEAR,
        use_batch: bool = True,
        threshold: Optional[float] = 0.7,
        **kwargs,
    ):
        self.positive_label = positive_label
        self.feature_extractor = feature_extractor
        self.context_window = context_window
        super(LinearWindowFunction, self).__init__(
            positive_label,
            feature_extractor,
            context_window,
            use_batch=use_batch,
            threshold=threshold,
            **kwargs,
        )

        self.dictionary = TensorList()
        self.labels = TensorList()
        self.feature_summarizer = feature_summarizer
        self.linear_model = construct_linear_classifier(linear_type=linear_type)
    
    @log_time(function_prefix='linear_window_train')
    def _train_model(self, training_data: List[Tuple[List[str], List[Any], str]]):
        for i, (sentence_window, feature_window, label) in enumerate(training_data):
            window_summary = self.feature_summarizer(feature_window)
            self.dictionary.append(window_summary)
            self.labels.append(torch.Tensor([label_index(label)]))
        x_train = self.dictionary.numpy()
        y_train = self.labels.numpy()
        self.linear_model.fit(x_train, y_train)

    def _predict(self, features: List[torch.Tensor]) -> int:
        feature_summary = self.feature_summarizer(features).numpy()
        label: np.ndarray = self.linear_model.predict(feature_summary)
        return label.item()
    
    def _predict_probabilities(self, features: List[torch.Tensor]) -> float:
        feature_summary = self.feature_summarizer(features).numpy()
        confidence: np.ndarray = self.linear_model.decision_function(feature_summary)
        return confidence.item()

    @log_time(function_prefix='linear_window_snorkel_predict')
    def _batch_probabilities(self, features: List[List[torch.Tensor]]) -> List[float]:
        feature_summaries: List[np.ndarray] = list(map(lambda f: self.feature_summarizer(f).numpy(), features))
        batch_np: np.ndarray = TensorList(feature_summaries).numpy()
        confidence_batch: np.ndarray = self.linear_model.decision_function(batch_np)
        return list(map(lambda conf: conf.item(), TensorList([confidence_batch]).to_list()))

    @log_time(function_prefix='linear_window_predict')
    def _batch_predict(self, features: List[List[torch.Tensor]]) -> List[int]:
        feature_summaries: List[np.ndarray] = list(map(lambda f: self.feature_summarizer(f).numpy(), features))
        batch_np: np.ndarray = TensorList(feature_summaries).numpy()
        label_batch: np.ndarray = self.linear_model.predict(batch_np)
        return list(map(lambda label: label.item(), TensorList([label_batch]).to_list()))

    @overrides
    def __str__(self):
        return f'LinearWindowFunction({self.context_window})({self.feature_extractor})'