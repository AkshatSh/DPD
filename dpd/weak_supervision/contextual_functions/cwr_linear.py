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
from dpd.weak_supervision import WeakFunction, AnnotatedDataType
from dpd.models.embedder import CachedTextFieldEmbedder
from dpd.utils import TensorList

from .utils import get_label_index, construct_train_data

class CWRLinear(WeakFunction):
    @classmethod
    def construct_linear_classifier(cls, linear_type: str) -> None:
        if linear_type == 'lr':
            return LogisticRegression()
        elif linear_type == 'svm_linear':
            return SVC(kernel='linear', probability=True) 
        elif linear_type == 'svm_quadratic':
            return SVC(kernel='poly', degree=2, probability=True)
        elif linear_type == 'svm_rbf':
            return SVC(kernel='rbf', probability=True) 
        else:
            raise Exception(f"Unknown Linear type: {linear_type}")

    def __init__(
        self,
        positive_label: str,
        embedder: CachedTextFieldEmbedder,
        linear_type: str,
    ):
        self.positive_label = positive_label
        self.embedder = embedder
        self.linear_model = CWRLinear.construct_linear_classifier(linear_type=linear_type)
    
    def _prepare_train_data(
        self,
        train_data: AnnotatedDataType,
        dataset_id: int,
        shuffle: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        positive_set: TensorList = TensorList()
        negative_set: TensorList = TensorList()
        for entry in train_data:
            s_id, sentence, tags = entry['id'], entry['input'], entry['output']

            cwr_embeddings: torch.Tensor = self.embedder.get_embedding(
                sentence_id=s_id,
                dataset_id=dataset_id,
            )
            
            # assert shape is expected
            assert cwr_embeddings.shape == (len(sentence), self.embedder.get_output_dim())

            pos_idx, neg_idx = get_label_index(tags)

            positive_set.append(cwr_embeddings[pos_idx])
            negative_set.append(cwr_embeddings[neg_idx])
 
        positive_set: np.ndarray = positive_set.numpy()
        negative_set: np.ndarray = negative_set.numpy()
        positive_labels: np.ndarray = np.zeros((len(positive_set),))
        positive_labels.fill(1)
        negative_labels: np.ndarray = np.zeros((len(negative_set)))

        x_train, y_train = construct_train_data(
            pos_data=positive_set,
            neg_data=negative_set,
            pos_labels=positive_labels,
            neg_labels=negative_labels,
            shuffle=shuffle,
        )

        return x_train, y_train

    def train(self, train_data: AnnotatedDataType, dataset_id: int = 0):
        '''
        Train the keyword matching function on the training data

        input:
            - train_data ``AnnotatedDataType``
                the annotation data to be used for training
        train the function on the specified training data
        '''
        x_train, y_train = self._prepare_train_data(train_data=train_data, dataset_id=dataset_id, shuffle=True)
        print(f'x_train: {x_train.shape}, y_train: {y_train.shape}')
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
        return annotated_data

    def __str__(self):
        return f'CWRLinear({self.embedder})'
    
    def __repr__(self):
        return self.__str__()