from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import os
from collections import Counter
import string
import numpy as np

from dpd.models.embedder import GloVeWordEmbeddingIndex
from dpd.utils import get_words, explain_labels
from dpd.constants import STOP_WORDS

def build_gold_dictionary(gold_set, binary_class: str) -> Tuple[Counter, Counter]:
    '''
    Parse a gold set to get the positive words and the negative words
    '''
    word_counter: Dict[str, Counter] = {'pos': Counter(), 'neg': Counter()}
    phrase_counter: Counter = Counter()
    for entry in gold_set:
        sentence, tags = entry['input'], entry['output']
        pos_words = get_words(sentence, tags, binary_class)

        # get the negative words
        neg_words = get_words(sentence, tags, 'O')

        pos_ranges, pos_phrases = explain_labels(sentence, tags)
        for w in pos_words:
            word_counter['pos'][w] += 1

        for w in neg_words:
            word_counter['neg'][w] += 1

        for phrase in pos_phrases:
            phrase_counter[tuple(phrase)] += 1
    
    return word_counter, phrase_counter

def build_sklearn_train_data(
    word_counter: Dict[str, Counter],
    word_embedding_index: GloVeWordEmbeddingIndex
) -> Tuple[np.ndarray, np.ndarray]:
    train_set_embeddings = []
    train_set_labels = []

    num_pos = 0
    for w in word_counter['pos'].keys():
        if w in STOP_WORDS or w in string.punctuation:
            continue
        embedding_vec = word_embedding_index.get_embedding(w)
        train_set_labels.append(1)
        train_set_embeddings.append(embedding_vec)
        num_pos += 1

    num_neg = 0
    for w in word_counter['neg'].keys():
        if w in STOP_WORDS or w in string.punctuation:
            continue
        embedding_vec = word_embedding_index.get_embedding(w)
        train_set_labels.append(0)
        train_set_embeddings.append(embedding_vec)
        num_neg += 1
    x_train = np.array(train_set_embeddings)
    y_train = np.array(train_set_labels)

    return x_train, y_train