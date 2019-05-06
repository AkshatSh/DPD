from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Any,
)

import os
import sys

import torch
import allennlp
import numpy as np

from dpd.dataset import UnlabeledBIODataset
from dpd.weak_supervision import WeakFunction, AnnotatedDataType, AnnotationType
from dpd.models.embedder import CachedTextFieldEmbedder
from dpd.utils import TensorList
from dpd.weak_supervision.feature_extractor import FeatureExtractor, FeaturePadder

from .utils import get_context_window, get_context_range

class WindowFunction(WeakFunction):
    def __init__(
        self,
        positive_label: str,
        feature_extractor: FeatureExtractor,
        context_window: int,
        padder: FeaturePadder = FeaturePadder.zero_tensor,
    ):
        self.positive_label = positive_label
        self.feature_extractor = feature_extractor
        self.context_window = context_window
        self.padder = padder

    def get_label(self, index: int) -> str:
        if index == 0:
            return 'O'
        return self.positive_label

    def _train_model(self, training_data: List[Tuple[List[str], List[Any], str]]):
        raise NotImplementedError()

    def train(self, train_data: AnnotatedDataType, dataset_id: int = 0):
        '''
        Train the keyword matching function on the training data

        input:
            - train_data ``AnnotatedDataType``
                the annotation data to be used for training
        train the function on the specified training data
        '''
        annotated_data = []
        training_data: List[Tuple[List[str], List[Any], str]] = []
        for entry in train_data:
            s_id, sentence, tags = entry['id'], entry['input'], entry['output']
            features = self.feature_extractor.get_features(dataset_id=dataset_id, sentence_id=s_id, sentence=sentence)
            for i, (word, feature, tag) in enumerate(zip(sentence, features, tags)):
                word_window = get_context_window(sentence, index=i, window=self.context_window)
                feature_window = get_context_window(features, index=i, window=self.context_window)
                feature_window = self.padder(feature_window)
                training_data.append((word_window, feature_window, tag))
        self._train_model(training_data)

    def _predict(self, features: List[torch.Tensor]) -> int:
        raise NotImplementedError()

    def predict(self, features: List[Any], index: int) -> str:
        feature_window = get_context_window(features, index=index, window=self.context_window)
        feature_window = self.padder(feature_window)
        predict_index = self._predict(feature_window)
        return self.get_label(predict_index)
    
    def evaluate(self, unlabeled_corpus: UnlabeledBIODataset) -> AnnotatedDataType:
        '''
        evalaute the keyword function on the unlabeled corpus

        input:
            - unlabeled_corpus ``UnlabeledBIODataset``
                the unlabeled corpus to evaluate on
        output:
            - annotations from applying this labeling function
        '''
        annotated_data = []
        for entry in unlabeled_corpus:
            s_id, sentence = entry['id'], entry['input']
            features = self.feature_extractor.get_features(dataset_id=unlabeled_corpus.dataset_id, sentence_id=s_id, sentence=sentence)
            predicted_labels: List[str] = [self.predict(features, i) for i, f in enumerate(features)]
            annotated_data.append({
                'id': s_id,
                'input': sentence,
                'output': predicted_labels,
            })

        return annotated_data

    def __str__(self):
        return f'WindowFunction({self.feature_extractor})'
    
    def __repr__(self):
        return self.__str__()