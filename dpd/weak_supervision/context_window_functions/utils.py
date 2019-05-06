from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Any,
)

import os
import sys

import torch
import allennlp
import numpy as np

NEGATIVE_LABEL = 'O'

def is_negative(label: str) -> bool:
    return label == NEGATIVE_LABEL

def is_positive(label: str) -> bool:
    return not is_negative(label)

def label_index(label: str) -> int:
    if is_positive(label):
        return 1 
    return 0

def get_context_range(features: List[Any], index: int, window: int) -> Tuple[int, int, int, int]:
    left_half: int = max(index - window, 0)
    left_pad: int = 0
    if index - window < 0:
        left_pad = window - index
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