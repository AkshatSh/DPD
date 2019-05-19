from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import logging

import torch
import numpy as np
import allennlp
from sklearn.linear_model import LogisticRegression
from enum import Enum

logger = logging.getLogger(name=__name__)

try:
    from thundersvm import SVC
except ModuleNotFoundError as e:
    logger.warn('Unable to import ThunderSVM, using scikit learn instead')
    from sklearn.svm import SVC
except FileNotFoundError as e:
    logger.warn('Broken ThunderSVM, using scikit learn instead')
    logger.exception(e)
    from sklearn.svm import SVC

class LinearType(Enum):
    LOGISTIC_REGRESSION = 'lr'
    SVM_LINEAR = 'svm_linear'
    SVM_QUADRATIC = 'svm_quadratic'
    SVM_RBF = 'svm_rbf'

def construct_linear_classifier(linear_type: LinearType) -> None:
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