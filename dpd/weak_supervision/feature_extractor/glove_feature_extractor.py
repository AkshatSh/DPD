from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Any,
    Iterator,
)

import os
import sys
import pickle
from tqdm import tqdm

import spacy
import nltk
import torch
import allennlp
from allennlp.data import Instance, Vocabulary
from dpd.utils import TensorList
from dpd.models.embedder import GloVeWordEmbeddingIndex

from .feature_extractor import FeatureExtractor

class GloVeFeatureExtractor(FeatureExtractor):
    def __init__(self, *args, **kwargs):
        self.glove = GloVeWordEmbeddingIndex.instance()

    def get_glove(
        self,
        word: str,
    ) -> torch.Tensor:
        embedding_vec = self.glove.get_embedding(word)
        return torch.Tensor(embedding_vec).view(1, -1)
    
    def get_features(
        self,
        dataset_id: int,
        sentence_id: int,
        sentence: List[str],
    ) -> List[torch.Tensor]:
        return [
            self.get_glove(w) for w in sentence
        ]
    
    def __str__(self):
        return f'GloVeFeatureExtractor'
    
    def __repr__(self):
        return self.__str__()