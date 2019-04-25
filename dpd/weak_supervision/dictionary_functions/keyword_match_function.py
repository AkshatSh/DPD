from typing import (
    List,
    Tuple,
    Dict,
)

from collections import Counter

import os
import torch
import allennlp

from dpd.dataset import UnlabeledBIODataset
from dpd.weak_supervision import WeakFunction, AnnotatedDataType
from dpd.utils import (
    remove_bio,
    get_words
)

class KeywordMatchFunction(object):
    '''
    A keyword matching function, trains by keeping track of all the
    keywords in the train set

    During test time predicts any seen words as positive
    '''
    def __init__(
        self,
        binary_class: str,
    ):
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
    
    def convert_to_bio(self, sentence: List[str], predictions: List[str], class_tag: str) -> List[str]:
        proc_pred = list(predictions) # create copy
        for i, (w, p_i) in enumerate(zip(sentence, predictions)):
            if i == 0:
                if p_i == class_tag:
                    proc_pred[i] = f'B-{class_tag}'
                continue
            
            if p_i == class_tag:
                prev_tag = predictions[i - 1]
                if prev_tag == 'O':
                    proc_pred[i] = f'B-{class_tag}'
                else:
                    proc_pred[i] = f'I-{class_tag}'
        return proc_pred
    
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
                else:
                    annotations.append('O')
            data_entry['output'] = self.convert_to_bio(sentence, annotations, self.binary_class)
            annotated_data.append(data_entry)
        return annotated_data

    def __str__(self):
        return f'KeywordMatchingFunction'
