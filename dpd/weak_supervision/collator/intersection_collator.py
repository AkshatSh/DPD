from typing import (
    List,
    Tuple,
    Dict,
    Iterator,
    Optional,
)

import os
import sys

import torch
import numpy as np
import allennlp

from dpd.weak_supervision import AnnotatedDataType
from .collator import Collator
from ..utils import is_negative, is_positive, NEGATIVE_LABEL

class IntersectionCollator(Collator):
    def __init__(
        self,
        positive_label: str,
    ):
        self.positive_label = positive_label

    def _intersect(self, potential_tags: List[str]) -> str:
        for tag in potential_tags:
            if is_negative(tag):
                return NEGATIVE_LABEL
        return self.positive_label

    def _collate_fn(self, outputs: List[List[str]]) -> List[str]:
        output: List[str] = []
        for i, out_i in enumerate(zip(*outputs)):
            tag = self._intersect(out_i)
            output.append(tag)
        return output