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

from dpd.utils import SaveFile, PickleSaveFile
from dpd.constants import SPACY_NLP

class CachedSpaCyFeatures(object):
    def __init__(
        self,
        dataset_id: int,
    ):
        self.id_to_doc: Dict[int, Any] = {}
    
    def cache_features(
        self,
        sentence_id: int,
        sentence: str,
    ):
        self.id_to_doc[sentence_id] = SPACY_NLP(sentence)
    
    def get_features(
        self,
        sentence_id: int,
        sentence: Optional[str],
    ) -> Any:
        if sentence_id not in self.id_to_doc:
            if sentence is not None:
                return SPACY_NLP(sentence)
            else:
                raise Exception(f'Sentence id: {sentence_id} is not cached')
        else:
            return self.id_to_doc[sentence_id]
    
    def save(
        self,
        key: str,
        save_file: SaveFile,
    ):
        save_file.save_dict(item=self.id_to_doc, key=f'{key}/id_to_doc')
    
    def load(
        self,
        key: str,
        save_file: SaveFile,
    ):
        self.id_to_doc = save_file.load_dict(key=f'{key}/id_to_doc')
    
    @classmethod
    def cache_dataset(
        cls,
        dataset_id: int,
        dataset: Iterator[Instance],
    ):
        cached_fetures = cls(dataset_id)
        for instance in dataset:
            sentence = [t.text for t in instance.fields['sentence']]
            sentence_id = instance.fields['entry_id'].as_tensor(None).item()
            sentence_str = ' '.join(sentence)
            cached_fetures.cache_features(
                sentence_id=sentence_id,
                sentence=sentence_str,
            )
        return cached_fetures

class SpaCyFeatureExtractor(object):
    '''
    spacy lingustic feature extractor
    '''
    def __init__(self):
        self.datasets: Dict[int, CachedSpaCyFeatures] = {}
    
    def cache(
        self,
        dataset_id: int,
        dataset: Iterator[Instance],
        vocab: Vocabulary,
    ):
        self.datasets[dataset_id] = CachedSpaCyFeatures.cache_dataset(
            dataset_id=dataset_id,
            dataset=dataset,
        )
    
    def get_features(
        self,
        dataset_id: int,
        sentence_id: int,
        sentence_str: Optional[str] = None,
    ) -> Any:
        if dataset_id not in self.datasets:
            if sentence_str is not None:
                return SPACY_NLP(sentence_str)
            else:
                raise Exception(f'Unknown dataset: {dataset_id}')
        else:
            return self.datasets[dataset_id].get_features(
                sentence_id=sentence_id,
                sentence=sentence_str,
            )
    
    def enable_dataset(self, dataset_id: int):
        self.datasets[dataset_id] = CachedSpaCyFeatures(dataset_id=dataset_id)
    
    def save(self, save_file: SaveFile):
        for d_id, features in self.datasets.items():
            features.save(key=f'dataset_id_{d_id}', save_file=save_file)
    
    def load(self, save_file: SaveFile):
        for d_id, features in self.datasets.items():
            features.load(key=f'dataset_id_{d_id}', save_file=save_file)
    
    @classmethod
    def setup(cls, dataset_ids: List[int] = [0, 1]):
        extractor = cls()
        for dataset_id in dataset_ids:
            extractor.enable_dataset(dataset_id)
        return extractor
