from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import os
import sys

import torch
import allennlp
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from dpd.dataset import UnlabeledBIODataset
from dpd.weak_supervision import WeakFunction, AnnotatedDataType, AnnotationType
from dpd.models.embedder import CachedTextFieldEmbedder
from dpd.models import construct_linear_classifier, LinearType
from dpd.common import TensorList

from ..utils import get_label_index, construct_train_data, extract_features, NEGATIVE_LABEL, ABSTAIN_LABEL

class CWRLinear(WeakFunction):
    def __init__(
        self,
        positive_label: str,
        embedder: CachedTextFieldEmbedder,
        linear_type: LinearType = LinearType.SVM_LINEAR,
        threshold: float = 0.7,
        **kwargs,
    ):
        super(CWRLinear, self).__init__(positive_label, threshold, **kwargs)
        self.positive_label = positive_label
        self.embedder = embedder
        self.linear_model = construct_linear_classifier(linear_type=linear_type)
    
    def _prepare_train_data(
        self,
        train_data: AnnotatedDataType,
        dataset_id: int,
        shuffle: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:

        def _feature_extractor(entry: AnnotationType) -> torch.Tensor:
            s_id, sentence = entry['id'], entry['input']
            cwr_embeddings: torch.Tensor = self.embedder.get_embedding(
                sentence_id=s_id,
                dataset_id=dataset_id,
            )
            # assert shape is expected
            assert cwr_embeddings.shape == (len(sentence), self.embedder.get_output_dim())
            return cwr_embeddings

        return extract_features(
            data=train_data,
            dataset_id=dataset_id,
            shuffle=shuffle,
            feature_extractor=_feature_extractor,
        )

    def train(self, train_data: AnnotatedDataType, dataset_id: int = 0):
        '''
        Train the keyword matching function on the training data

        input:
            - train_data ``AnnotatedDataType``
                the annotation data to be used for training
        train the function on the specified training data
        '''
        x_train, y_train = self._prepare_train_data(train_data=train_data, dataset_id=dataset_id, shuffle=True)
        self.linear_model.fit(x_train, y_train)

    
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
            cwr_embeddings: torch.Tensor = self.embedder.get_embedding(
                sentence_id=s_id,
                dataset_id=unlabeled_corpus.dataset_id,
            )

            # (sentence_len, 1)
            if self.threshold is not None:
                confidence: np.ndarray = self.linear_model.decision_function(cwr_embeddings)
                predicted_labels: List[str] = [self.get_probability_label(li) for li in confidence]
            else:
                labels: np.ndarray = self.linear_model.predict(cwr_embeddings)
                predicted_labels: List[str] = [self.get_label(li) for li in labels]
            annotated_data.append({
                'id': s_id,
                'input': sentence,
                'output': predicted_labels,
            })

        return annotated_data

    def __str__(self):
        return f'CWRLinear({self.embedder})'
    
    def __repr__(self):
        return self.__str__()