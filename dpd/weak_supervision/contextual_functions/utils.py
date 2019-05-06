from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Callable,
    Any,
)

import torch
import numpy as np
from allennlp.data import Instance
from dpd.weak_supervision import AnnotatedDataType, AnnotationType
from dpd.utils import TensorList


NEGATIVE_LABEL = 'O'

def is_negative(label: str) -> bool:
    return label == NEGATIVE_LABEL

def is_positive(label: str) -> bool:
    return not is_negative(label)

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
    

