from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import os
import sys

import torch
import allennlp
import numpy as np
import scipy

from ..types import (
    AnnotatedDataType,
    AnnotationType,
)

from .collator import Collator

class SnorkeMeTalCollator(Collator):
    def __init__(
        self,
        positive_label: str,
    ):
        self.positive_label = positive_label