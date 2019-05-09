from typing import (
    List,
    Tuple,
    Callable,
    Optional,
)

import os
import sys

import torch
import numpy as np
from dpd.utils import TensorList

import unittest

TENSOR_EMBEDDING_DIM = 4

class TensorListTest(unittest.TestCase):
    def test_empty_construct(self):
        tl = TensorList()
        assert len(tl) == 0
        assert tl.shape == (0,)
    
    def test_constructor_tensor(self):
        tl = TensorList(tensor_list=[
            torch.zeros(1,TENSOR_EMBEDDING_DIM),
            torch.zeros(1,TENSOR_EMBEDDING_DIM),
            torch.zeros(1,TENSOR_EMBEDDING_DIM),
        ])
        assert len(tl) == 3
        assert tl.shape == (3, TENSOR_EMBEDDING_DIM)
    
    def test_constructor_numpy(self):
        tl = TensorList(tensor_list=[
            np.zeros((1,TENSOR_EMBEDDING_DIM)),
            np.zeros((1,TENSOR_EMBEDDING_DIM)),
            np.zeros((1,TENSOR_EMBEDDING_DIM)),
        ])
        assert len(tl) == 3
        assert tl.shape == (3, TENSOR_EMBEDDING_DIM)
    
    def test_append(self):
        tl = TensorList(tensor_list=[
            torch.zeros(1,TENSOR_EMBEDDING_DIM),
            torch.zeros(1,TENSOR_EMBEDDING_DIM),
            torch.zeros(1,TENSOR_EMBEDDING_DIM),
        ])
        assert len(tl) == 3
        assert tl.shape == (3, TENSOR_EMBEDDING_DIM)
        tl.append(np.zeros((1, TENSOR_EMBEDDING_DIM)))
        assert len(tl) == 4
        assert tl.shape == (4, TENSOR_EMBEDDING_DIM)
    
    def test_extend(self):
        def _create_list():
            return [
                torch.zeros(1,TENSOR_EMBEDDING_DIM),
                torch.zeros(1,TENSOR_EMBEDDING_DIM),
                torch.zeros(1,TENSOR_EMBEDDING_DIM),
            ]
        tl = TensorList(tensor_list=_create_list())
        assert len(tl) == 3
        assert tl.shape == (3, TENSOR_EMBEDDING_DIM)
        tl.extend(_create_list())
        assert len(tl) == 6
        assert tl.shape == (6, TENSOR_EMBEDDING_DIM)
    
    def test_numpy(self):
        def _create_list():
            return [
                torch.zeros(1,TENSOR_EMBEDDING_DIM),
                torch.zeros(1,TENSOR_EMBEDDING_DIM),
                torch.zeros(1,TENSOR_EMBEDDING_DIM),
            ]
        tl = TensorList(tensor_list=_create_list())
        assert type(tl.numpy()) == np.ndarray
    
    def test_tensor(self):
        def _create_list():
            return [
                torch.zeros(1,TENSOR_EMBEDDING_DIM),
                torch.zeros(1,TENSOR_EMBEDDING_DIM),
                torch.zeros(1,TENSOR_EMBEDDING_DIM),
            ]
        tl = TensorList(tensor_list=_create_list())
        assert type(tl.tensor()) == torch.Tensor

    def test_tensor_list(self):
        def _create_list():
            return [
                torch.zeros(1,TENSOR_EMBEDDING_DIM),
                torch.zeros(1,TENSOR_EMBEDDING_DIM),
                torch.zeros(1,TENSOR_EMBEDDING_DIM),
            ]
        list_tensor = _create_list()
        tl = TensorList(tensor_list=list_tensor)
        created_list = tl.to_list()
        assert type(created_list) == type(list_tensor)
        assert all((t1 == t2).all() for t1, t2 in zip(list_tensor, created_list))
