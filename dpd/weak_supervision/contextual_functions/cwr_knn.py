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
from scipy import stats
import faiss

from dpd.dataset import UnlabeledBIODataset
from dpd.weak_supervision import WeakFunction, AnnotatedDataType, AnnotationType
from dpd.models.embedder import CachedTextFieldEmbedder
from dpd.utils import TensorList

from ..utils import get_label_index, construct_train_data, extract_features, NEGATIVE_LABEL

class CWRkNN(WeakFunction):
    '''
    kNN implementation in the contextual word embedding space
    '''
    def __init__(
        self,
        positive_label: str,
        embedder: CachedTextFieldEmbedder,
        resolve_mode: str = 'weighted',
        k: int = 10,
        threshold: Optional[float] = 0.7,
        **kwargs,
    ):
        super(CWRkNN, self).__init__(positive_label, threshold, **kwargs)
        self.positive_label = positive_label
        self.embedder = embedder
        self.index_np: np.ndarray = None
        self.faiss_index: faiss.IndexFlatIP = None
        self.k = k
        self.resolve_mode = resolve_mode
    
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
            # TODO max position of BERT embedder may make this not true
            # assert cwr_embeddings.shape == (len(sentence), self.embedder.get_output_dim())
            return cwr_embeddings

        return extract_features(
            data=train_data,
            dataset_id=dataset_id,
            shuffle=shuffle,
            feature_extractor=_feature_extractor,
        )
    
    def resolve_label_weighted(self, index: np.ndarray, distances: np.ndarray) -> np.ndarray:
        '''
        given the resutls from the kNN search, determine the label of the index

        args:
            index: (num_words, CWRkNN.k)
            distances: (num_words, CWRkNN.k)
        returns:
            (num_words, 1)
                each index means 1 if the word at that index resolves to positive
                and 0 if negative
        '''
        labels: np.ndarray = self.labels

        # get all the labels and convert them to the same shape as indexs
        neighbor_labels = self.labels[index.flatten()].reshape(index.shape)

        # positive and negative masks
        pos_neighbors = neighbor_labels == 1
        neg_neighbors = neighbor_labels == 0


        # (num_words, 1)
        num_pos_neighbors = pos_neighbors.sum(axis=1)

        # (num_words, 1)
        pos_distances = (distances * pos_neighbors).sum(axis=1) / num_pos_neighbors
        np.nan_to_num(pos_distances, copy=False)

        # (num_words, 1)
        num_neg_neighbors = neg_neighbors.sum(axis=1)

        # (num_words, 1)
        neg_distances = (distances * neg_neighbors).sum(axis=1) / num_neg_neighbors
        np.nan_to_num(neg_distances, copy=False)
        if self.threshold is not None:
            # (num_words, 1)
            total_valid_neighbors = pos_distances + neg_distances

            # ratio pos
            rp = pos_distances / total_valid_neighbors
            
            # raio neg
            rn = neg_distances / total_valid_neighbors

            res = np.zeros(rn.shape)
            res[rp > self.threshold] = 2
            res[rn > self.threshold] = 1
            return res.astype(int)

        # return the modes as (num_words, 1)
        return (pos_distances > neg_distances).astype(int)

    def resolve_label_mode(self, index: np.ndarray, distances: np.ndarray) -> np.ndarray:
        '''
        given the resutls from the kNN search, determine the label of the index

        args:
            index: (num_words, CWRkNN.k)
            distances: (num_words, CWRkNN.k)
        returns:
            (num_words, 1)
                each index means 1 if the word at that index resolves to positive
                and 0 if negative
        '''
        labels: np.ndarray = self.labels

        # get all the labels and convert them to the same shape as indexs
        neighbor_labels = self.labels[index.flatten()].reshape(index.shape)

        # select the mode of each column
        modes, counts = stats.mode(neighbor_labels, axis=1)

        # return the modes as (num_words, 1)
        return np.array(modes)
    
    def resolve_label(self, index: np.ndarray, distances: np.ndarray, mode='mode'):
        if mode == 'mode':
            return self.resolve_label_mode(index, distances)
        elif mode == 'weighted':
            return self.resolve_label_weighted(index, distances)
        else:
            raise Exception(f'Unknown mode: {mode}')

    def train(self, train_data: AnnotatedDataType, dataset_id: int = 0):
        '''
        Train the keyword matching function on the training data

        input:
            - train_data ``AnnotatedDataType``
                the annotation data to be used for training
        train the function on the specified training data
        '''
        x_train, y_train = self._prepare_train_data(train_data=train_data, dataset_id=dataset_id, shuffle=False)
        self.index_np = x_train.astype('float32')
        self.labels = y_train
        self.faiss_index = faiss.IndexFlatIP(self.index_np.shape[1])
        faiss.normalize_L2(self.index_np)
        self.faiss_index.add(self.index_np)

    
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

            distances, indexes = self.faiss_index.search(cwr_embeddings.detach().cpu().numpy(), self.k)
            resolved_labels = self.resolve_label(indexes, distances, mode=self.resolve_mode)
            if self.threshold is not None:
                predicted_labels: List[str] = [self.get_snorkel_index(pi) for pi in resolved_labels]
            else:
                predicted_labels: List[str] = [self.get_label(li) for li in resolved_labels]
            annotated_data.append({
                'id': s_id,
                'input': sentence,
                'output': predicted_labels,
            })

        return annotated_data

    def __str__(self):
        return f'CWRkNN({self.embedder})'
    
    def __repr__(self):
        return self.__str__()