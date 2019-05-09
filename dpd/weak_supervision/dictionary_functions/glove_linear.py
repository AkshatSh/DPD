from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import os
from collections import Counter
import string
import torch
import allennlp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from dpd.dataset import UnlabeledBIODataset
from dpd.weak_supervision import WeakFunction, AnnotatedDataType
from dpd.weak_supervision.dictionary_functions.utils import build_gold_dictionary, build_sklearn_train_data
from dpd.weak_supervision.dictionary_functions import KeywordMatchFunction
from dpd.models import construct_linear_classifier, LinearType
from dpd.models.embedder import GloVeWordEmbeddingIndex
from dpd.constants import (
    STOP_WORDS,
)

class GloveLinearFunction(object):
    def __init__(
        self,
        binary_class: str,
        linear_function: LinearType = LinearType.SVM_LINEAR,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        self.word_embedding_index = GloVeWordEmbeddingIndex.instance()
        self.similar_words: Dict[str, Counter] = {}
        self.similar_phrases: Dict[str, Counter] = {}
        self.linear_function = linear_function
        self.keywords_func: KeywordMatchFunction = None 
        self.binary_class = binary_class
        self.linear_classifier = construct_linear_classifier(
            linear_type=linear_function,
        )
        self.threshold = threshold

    def train(self, train_data: AnnotatedDataType):
        self.word_counter, self.phrase_counter = build_gold_dictionary(train_data, self.binary_class)
        x_train, y_train = build_sklearn_train_data(self.word_counter, self.word_embedding_index)

        # fit our linear model
        self.linear_classifier.fit(x_train, y_train)

        # predict hard labels
        # and predict probabilities for thresholding
        labels = self.linear_classifier.predict(self.word_embedding_index.index_np)
        probs = self.linear_classifier.predict_proba(self.word_embedding_index.index_np)

        self.similar_words = {'pos': Counter(), 'neg': Counter()}
        for i, (label, prob) in enumerate(zip(labels, probs)):
            if self.threshold is not None:
                if prob[1] > threshold:
                    label = 1
                else:
                    label = 0
            elif prob[1] > prob[0]:
                label = 1
            else:
                label = 0

            key = 'pos' if label == 1 else 'neg'
            counter = self.similar_words[key]
            word = self.word_embedding_index.index_to_word[i]
            if word in STOP_WORDS or word in string.punctuation or word in self.word_counter['pos']:
                # ignore random stop words
                # ignore words already in the dictionary
                continue
            counter[word] += prob[label]

        self.keywords = {
            'pos': self.similar_words['pos'] + self.word_counter['pos'],
            'neg': self.similar_words['neg'] + self.word_counter['neg'],
        }

        self.keywords_func = KeywordMatchFunction(self.binary_class)
        self.keywords_func.set_keywords(self.keywords)
    
    def evaluate(self, unlabeled_corpus: UnlabeledBIODataset) -> AnnotatedDataType:
        return self.keywords_func.evaluate(unlabeled_corpus)
    
    def __str__(self):
        linear_type = "SVM_LINEAR"
        if self.linear_function == LinearType.LOGISTIC_REGRESSION:
            linear_type = "LOGISTIC_REGRESSION"
        elif self.linear_function == LinearType.SVM_QUADRATIC:
            linear_type = "SVM_QUADRATIC"
        elif self.linear_function == LinearType.SVM_RBF:
            linear_type = "SVM_RBF"

        return f'GloVeLinearFunction({linear_type})'
    
    def __repr__(self):
        return self.__str__()
