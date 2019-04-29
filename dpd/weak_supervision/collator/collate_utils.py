from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import os
import sys

NEGATIVE_LABEL = 'O'

def bio_negative(label: str) -> bool:
    return label == NEGATIVE_LABEL

def bio_positive(label: str) -> bool:
    return not bio_negative(label)