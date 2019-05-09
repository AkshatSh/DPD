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
from tqdm import tqdm
import logging

from dpd.dataset import UnlabeledBIODataset
from dpd.weak_supervision import WeakFunction, AnnotatedDataType, AnnotationType
from dpd.models.embedder import CachedTextFieldEmbedder
from dpd.utils import TensorList
from dpd.weak_supervision.feature_extractor import FeatureExtractor, FeaturePadder

from ..utils import get_context_window, get_context_range
logger = logging.getLogger(name=__name__)

class WindowFunction(WeakFunction):
    def __init__(
        self,
        positive_label: str,
        feature_extractor: FeatureExtractor,
        context_window: int,
        padder: FeaturePadder = FeaturePadder.zero_tensor,
        use_batch: bool = True,
    ):
        self.positive_label = positive_label
        self.feature_extractor = feature_extractor
        self.context_window = context_window
        self.padder = padder
        self.use_batch = use_batch

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
    
    def _batch_predict(self, features: List[List[torch.Tensor]]) -> List[int]:
        raise NotImplementedError()
    
    def batch_predict(self, features: List[List[Any]], indexes: List[int]) -> List[str]:
        batch_feature_window: List[List[torch.Tensor]] = list(map(
            lambda pair: self.padder(get_context_window(
                pair[0],
                index=pair[1],
                window=self.context_window,
            )),
            zip(features, indexes),
        ))

        batched_predict: List[int] = self._batch_predict(batch_feature_window)
        return list(map(lambda pred: self.get_label(pred), batched_predict))

    def predict(self, features: List[Any], index: int) -> str:
        feature_window = get_context_window(features, index=index, window=self.context_window)
        feature_window = self.padder(feature_window)
        predict_index = self._predict(feature_window)
        return self.get_label(predict_index)
    
    def _single_evalaute(self, unlabeled_corpus: UnlabeledBIODataset) -> AnnotatedDataType:
        # if len(sentence) != len(features) or s_id == 202:
        #     logging.debug(f's_id {s_id}, sentence: {len(sentence)} num features: {len(features)}')
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
    
    def _batch_evalaute(self, unlabeled_corpus: UnlabeledBIODataset) -> AnnotatedDataType:
        annotated_data = []
        batch_features = []
        batch_indexes = []
        offsets: List[Tuple[int, int, int]] = []
        for entry in unlabeled_corpus:
            s_id, sentence = entry['id'], entry['input']
            features = self.feature_extractor.get_features(dataset_id=unlabeled_corpus.dataset_id, sentence_id=s_id, sentence=sentence)
            start = len(batch_features)
            batch_features.extend([features for i in range(len(features))])
            batch_indexes.extend([i for i in range(len(features))])
            end = len(batch_features)
            if len(sentence) != end - start:
                logger.debug(f's_id {s_id}, labels: {end - start} num features: {len(features)}')
            offsets.append((s_id, sentence, start, end))

        batch_labels: List[str] = self.batch_predict(batch_features, batch_indexes)
        for (s_id, sentence, start , end) in offsets:
            predicted_labels = batch_labels[start : end]
            # BERT may mean len(predicted_labels) != len(sentence)
            annotated_data.append({
                'id': s_id,
                'input': sentence,
                'output': predicted_labels,
            })

        return annotated_data
    
    def evaluate(self, unlabeled_corpus: UnlabeledBIODataset) -> AnnotatedDataType:
        '''
        evalaute the keyword function on the unlabeled corpus

        input:
            - unlabeled_corpus ``UnlabeledBIODataset``
                the unlabeled corpus to evaluate on
        output:
            - annotations from applying this labeling function
        '''
        if self.use_batch:
            return self._batch_evalaute(unlabeled_corpus)
        else:
            return self._single_evalaute(unlabeled_corpus)

    def __str__(self):
        return f'WindowFunction({self.feature_extractor})'
    
    def __repr__(self):
        return self.__str__()