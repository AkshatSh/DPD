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
from dpd.common import TensorList

class CWRFeatureExtractor(WeakFunction):
    '''
    kNN implementation in the contextual word embedding space
    '''
    def __init__(
        self,
        embedder: CachedTextFieldEmbedder,
        *args,
        **kwargs,
    ):
        self.embedder = embedder
    
    def get_features(
        self,
        dataset_id: int,
        sentence_id: int,
        sentence: List[str],
    ) -> List[torch.Tensor]:
        features = torch.zeros(len(sentence), self.embedder.get_output_dim())
        cwr_embeddings: torch.Tensor = self.embedder.get_embedding(
            sentence_id=sentence_id,
            dataset_id=dataset_id,
        )

        features[:len(cwr_embeddings)] = cwr_embeddings

        return list(map(lambda t: t.unsqueeze(0), features))

    def __str__(self):
        return f'CWR({self.embedder})'
    
    def __repr__(self):
        return self.__str__()