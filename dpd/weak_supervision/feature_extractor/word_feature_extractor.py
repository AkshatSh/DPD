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

from .feature_extractor import FeatureExtractor

class WordFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        vocab: Vocabulary,
        *args,
        **kwargs,
    ):
        self.vocab = vocab
    
    def get_one_hot_encoding(
        self,
        word: str
    ) -> torch.Tensor:
        tensor = torch.zeros((self.vocab.get_vocab_size()))
        word_i = self.vocab.get_token_index(word)
        tensor[word_i] = 1
        return tensor.view(1, -1)
    
    def get_word_tensor(
        self,
        word: str
    ) -> torch.Tensor:
        # significantly faster for concat of word onehot
        word_i = self.vocab.get_token_index(word)
        tensor = torch.Tensor([word_i])
        return tensor.view(1, -1)

    def get_features(
        self,
        dataset_id: int,
        sentence_id: int,
        sentence: List[str],
    ) -> List[torch.Tensor]:
        return [
            self.get_one_hot_encoding(w) for w in sentence
        ]
    
    def __str__(self):
        return f'WordFeatureExtractor'
    
    def __repr__(self):
        return self.__str__()