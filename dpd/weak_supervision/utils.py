from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Any,
    Callable,
)

import os
import sys
import string
from collections import Counter

import torch
import allennlp
import numpy as np

from dpd.models.embedder import GloVeWordEmbeddingIndex
from dpd.common import TensorList
from dpd.utils import get_words, explain_labels
from dpd.constants import STOP_WORDS
from dpd.weak_supervision import AnnotatedDataType, AnnotationType

NEGATIVE_LABEL = 'O'
ABSTAIN_LABEL = '<ABS>'

def is_negative(label: str) -> bool:
    return label == NEGATIVE_LABEL

def is_abstain(label: str) -> bool:
    return label == ABSTAIN_LABEL

def is_positive(label: str) -> bool:
    return not is_negative(label) and not is_abstain(label)

def label_index(label: str) -> int:
    if is_positive(label):
        return 1 
    return 0

def get_context_range(features: List[Any], index: int, window: int) -> Tuple[int, int, int, int]:
    left_half: int = max(index - window, 0)
    left_pad: int = 0
    if index - window < 0:
        left_pad = window - index
    # TODO: change this to `window + 1` (Off by one error)
    right_half: int = min(index + window, len(features))
    right_pad: int = 0
    if index + window > len(features):
        right_pad = (index + window) - len(features)
    return left_half, right_half, left_pad, right_pad

def get_context_window(features: List[Any], index: int, window: int, pad_val: Optional[Any] = None) -> List[Any]:
    left_half, right_half, left_pad, right_pad = get_context_range(features, index, window)
    left_padding = [pad_val for i in range(left_pad)]
    right_padding = [pad_val for i in range(right_pad)]
    return left_padding + features[left_half:right_half] + right_padding

def get_label_index(labels: List[str]) -> Tuple[List[int], List[int]]:
    '''
    gets the indexes for the positive and negative labels
    '''
    pos_idx: List[int] = []
    neg_idx: List[int] = []
    for i, label in enumerate(labels):
        if is_negative(label):
            neg_idx.append(i)
        else:
            pos_idx.append(i)
    return pos_idx, neg_idx

def construct_train_data(
    pos_data: np.ndarray,
    neg_data: np.ndarray,
    pos_labels: np.ndarray,
    neg_labels: np.ndarray,
    shuffle: Optional[bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:
    train_data: TensorList = TensorList()
    train_labels: TensorList = TensorList()

    train_data.append(pos_data)
    train_data.append(neg_data)

    train_labels.append(pos_labels)
    train_labels.append(neg_labels)

    if shuffle:
        x = train_data.tensor()
        y = train_labels.tensor()
        idx = torch.randperm(len(x))
        return x[idx].numpy(), y[idx].numpy()
    else:
        return train_data.numpy(), train_labels.numpy()

def extract_features(
    data: AnnotatedDataType,
    dataset_id: int,
    shuffle: bool,
    feature_extractor: Callable[[AnnotationType], torch.Tensor],
):
    positive_set: TensorList = TensorList()
    negative_set: TensorList = TensorList()
    for entry in data:
        tags: List[str] = entry['output']
        features: torch.Tensor = feature_extractor(entry)
        pos_idx, neg_idx = get_label_index(tags)

        positive_set.append(features[pos_idx])
        negative_set.append(features[neg_idx])

    positive_set: np.ndarray = positive_set.numpy()
    negative_set: np.ndarray = negative_set.numpy()
    positive_labels: np.ndarray = np.zeros((len(positive_set),))
    positive_labels.fill(1)
    negative_labels: np.ndarray = np.zeros((len(negative_set)))
    x_train, y_train = construct_train_data(
        pos_data=positive_set,
        neg_data=negative_set,
        pos_labels=positive_labels,
        neg_labels=negative_labels,
        shuffle=shuffle,
    )

    return x_train, y_train

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