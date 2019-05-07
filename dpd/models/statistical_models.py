from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import torch
import numpy as np
import allennlp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from enum import Enum

class LinearType(Enum):
    LOGISTIC_REGRESSION = 'lr'
    SVM_LINEAR = 'svm_linear'
    SVM_QUADRATIC = 'svm_quadratic'
    SVM_RBF = 'svm_rbf'

def construct_linear_classifier(linear_type: str) -> None:
    if linear_type == LinearType.LOGISTIC_REGRESSION:
        return LogisticRegression()
    elif linear_type == LinearType.SVM_LINEAR:
        return SVC(kernel='linear', probability=True) 
    elif linear_type == LinearType.SVM_QUADRATIC:
        return SVC(kernel='poly', degree=2, probability=True)
    elif linear_type == LinearType.SVM_RBF:
        return SVC(kernel='rbf', probability=True) 
    else:
        raise Exception(f"Unknown Linear type: {linear_type}")