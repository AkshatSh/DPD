from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import torch
import numpy as np
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
    

