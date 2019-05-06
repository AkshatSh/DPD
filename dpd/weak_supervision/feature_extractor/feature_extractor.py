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

class FeatureExtractor(object):
    def __init__(
        self,
    ):
        pass
    
    def get_features(
        self,
        dataset_id: int,
        sentence_id: int,
        sentence: List[str],
    ) -> Any:
        raise NotImplementedError()