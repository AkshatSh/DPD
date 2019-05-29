from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Union
)

import os
import sys
from overrides import overrides
import logging
from enum import Enum

import torch
import numpy as np
import scipy
from scipy import sparse

logger = logging.getLogger(name=__name__)

from .tensor_list import (
    TensorList,
    TensorType,
    get_tensor,
    tensor_to_sparse,
    numpy,
)

class SparseTensorList(TensorList):
    @classmethod
    def create_sparse_tensor_list(
        cls,
        tensor: Optional[sparse.csr_matrix],
        incoming_tensor: TensorType,
    ):
        incoming_tensor = get_tensor(incoming_tensor)
        if tensor is None:
            return tensor_to_sparse(incoming_tensor)

        # format = csr to enforce result being csr matrix not coo matrix
        return sparse.vstack((tensor, incoming_tensor), format='csr')

    def __init__(
        self,
        tensor_list: Optional[TensorType] = None,
        device: str = 'cpu'
    ):
        if device != 'cpu':
            raise ValueError(f'Sparse Tensors are only on CPU, unsupported device: {device}')
        logger.info('using sparse matrix')
        self.tensor_list: sparse.csr_matrix = SparseTensorList.create_sparse_tensor_list(
            tensor=None, 
            incoming_tensor=TensorList.create_tensor_from_list(tensor_list)
        )
        self.size = self.tensor_list.shape[0]
        self.device = device
    
    @overrides
    def contains(self, item: TensorType) -> torch.Tensor:
        item_np = numpy(item)
        comp_sparse = self.tensor_list - item_np
        search = np.where(~comp_sparse.any(axis=1))[0]
        if len(search) == 0:
            return -1
        return search[0]

    @overrides
    def append(
        self,
        tensor: TensorType,
    ):
        tensor = get_tensor(tensor).to(self.device)
        new_size = tensor.shape[0]
        self.tensor_list = SparseTensorList.create_sparse_tensor_list(
            tensor=self.tensor_list if len(self) != 0 else None,
            incoming_tensor=tensor,
        )

        self.size += new_size
    
    def tensor_list(self) -> TensorList:
        tensor = self.tensor()
        return TensorList(tensor)

    @overrides
    def to(self, device: str):
        raise ValueError('SparseTensorList does not support CUDA operations, export to tensor')
    
    @overrides
    def to_list(self) -> List[torch.Tensor]:
        raise ValueError('SparseTensorList does not support list conversion, export to tensor')

    @overrides
    def numpy(self) -> np.ndarray:
        return np.array(self.tensor_list[:self.size].todense())

    @overrides
    def tensor(self) -> torch.Tensor:
        return torch.Tensor(self.numpy())
    
    @overrides
    def __len__(self) -> int:
        return self.tensor_list.shape[0] if self.tensor_list.shape[1] != 0 else 0
    
    @overrides
    def __str__(self) -> str:
        return f'SparseTensorList({self.shape})'
    @overrides
    def share_memory(self):
        # this is a no-op for sparse matricies
        pass