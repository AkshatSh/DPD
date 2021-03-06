from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

from collections import Counter

import os
import torch
import allennlp
import logging

from dpd.dataset import UnlabeledBIODataset
from dpd.weak_supervision import WeakFunction, AnnotatedDataType
from dpd.utils import (
    remove_bio,
    get_words,
)
from dpd.weak_supervision.utils import ABSTAIN_LABEL

logger =logging.getLogger(name=__name__)

class KeywordMatchFunction(WeakFunction):
    '''
    A keyword matching function, trains by keeping track of all the
    keywords in the train set

    During test time predicts any seen words as positive
    '''
    def __init__(
        self,
        binary_class: str,
        threshold: Optional[float] = 0.7,
        **kwargs,
    ):
        super(KeywordMatchFunction, self).__init__(binary_class, threshold=threshold, **kwargs)
        '''
        Initializes a keyword tracker that checks for the binary class

        input:
            - binary_class: ``str``
                the binary class to keep track of
        output:
            initializes the KeywordMatchingFunction
        '''
        self.keywords = {
            'pos': Counter(),
            'neg': Counter(),
        }

        self.binary_class = binary_class
        if self.threshold is None:
            raise Exception('some error')
    
    def set_keywords(self, keywords: Dict[str, Counter]):
        self.keywords = keywords

    def train(self, train_data: AnnotatedDataType):
        '''
        Train the keyword matching function on the training data

        input:
            - train_data ``AnnotatedDataType``
                the annotation data to be used for training
        train the function on the specified training data
        '''
        for data_entry in train_data:
            sentence, tags = data_entry['input'], data_entry['output']
            keywords = get_words(sentence, tags, self.binary_class)
            negative_words = get_words(sentence, tags, 'O')
            for w in keywords:
                self.keywords['pos'][w] += 1
            
            for w in negative_words:
                self.keywords['neg'][w] += 1
    
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
        for data_entry in unlabeled_corpus:
            data_entry = data_entry.copy()
            sentence = data_entry['input']
            annotations = []
            for s_word in sentence:
                if s_word in self.keywords['pos']:
                    annotations.append(self.binary_class)
                elif self.threshold is None or ('neg' in self.keywords and s_word in self.keywords['neg']):
                    annotations.append('O')
                else:
                    annotations.append(ABSTAIN_LABEL)
            data_entry['output'] = annotations
            annotated_data.append(data_entry)
        return annotated_data

    def __str__(self):
        return f'KeywordMatchingFunction'
    
    def __repr__(self):
        return self.__str__()
