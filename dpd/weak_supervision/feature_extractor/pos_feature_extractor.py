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
from dpd.constants import SPACY_POS_INDEX

from .feature_extractor import FeatureExtractor
from .spacy_feature_extractor import SpaCyFeatureExtractor

class POSFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        spacy_module: SpaCyFeatureExtractor,
        *args,
        **kwargs,
    ):
        self.spacy_module = spacy_module
    
    def encode(
        self,
        pos: str
    ) -> torch.Tensor:
        tensor = torch.zeros((len(SPACY_POS_INDEX)))
        word_i = SPACY_POS_INDEX[pos]
        tensor[word_i] = 1
        return tensor.view(1, -1)
    
    def get_features(
        self,
        dataset_id: int,
        sentence_id: int,
        sentence: List[str],
    ) -> List[torch.Tensor]:
        spacy_features = self.spacy_module.get_features(
            dataset_id=dataset_id,
            sentence_id=sentence_id,
            sentence=sentence,
        )
        pos_features = list(map(lambda f: f.pos_, spacy_features))
        return [
            self.encode(p) for p in pos_features
        ]
    
    def __str__(self):
        return f'POSFeatureExtractor'
    
    def __repr__(self):
        return self.__str__()