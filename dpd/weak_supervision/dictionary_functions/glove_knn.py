from typing import (
    List,
    Tuple,
    Dict,
)

import os
import torch
import allennlp

from dpd.dataset import UnlabeledBIODataset
from dpd.weak_supervision import WeakFunction, AnnotatedDataType
from dpd.models.embedder import GloVeWordEmbeddingIndex
from dpd.weak_supervision.dictionary_functions import KeywordMatchFunction

from ..utils import build_gold_dictionary

class GlovekNNFunction(object):
    def __init__(
        self,
        binary_class: str,
        k: int = 5,
        **kwargs,
    ):
        self.word_embedding_index = GloVeWordEmbeddingIndex.instance()
        self.similar_words: Dict[str, Counter] = {}
        self.similar_phrases: Dict[str, Counter] = {}
        self.k = k
        self.keywords_func: KeywordMatchFunction = None 
        self.binary_class = binary_class

    def train(self, train_data: AnnotatedDataType):
        self.word_counter, self.phrase_counter = build_gold_dictionary(train_data, self.binary_class)
        self.similar_words = self.word_embedding_index.find_similar_words(
            list(self.word_counter['pos'].keys()),
            k=self.k,
        )

        # self.similar_phrases = self.word_embedding_index.find_similar_phrases(
        #     list(self.phrase_counter.keys()),
        #     k=self.k,
        # )

        self.keywords = {
            'pos': self.similar_words + self.word_counter['pos'],
            # 'neg': self.similar_words['neg'] + self.word_counter['neg'],
        }

        self.keywords_func = KeywordMatchFunction(self.binary_class)
        self.keywords_func.set_keywords(self.keywords)

    
    def evaluate(self, unlabeled_corpus: UnlabeledBIODataset) -> AnnotatedDataType:
        return self.keywords_func.evaluate(unlabeled_corpus)
    
    def __str__(self):
        return f'GloVekNNFunction'

    def __repr__(self):
        return self.__str__()
