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

TensorType = Union[
    np.ndarray,
    torch.Tensor
]

class OperationMode(Enum):
    FAST = 'fast'
    MEMORY_EFFICENT = 'memory'

def numpy(in_t: TensorType) -> np.ndarray:
    if type(in_t) == torch.Tensor:
        return in_t.detach().cpu().numpy()
    elif type(in_t) == sparse.csr_matrix:
        return np.array(in_t.todense())
    elif type(in_t) == np.ndarray:
        return in_t
    else:
        raise ValueError(f'Unsupported tensor type: {type(in_t)}')

def get_tensor(in_t: TensorType) -> torch.Tensor:
    if type(in_t) == torch.Tensor:
        return in_t
    return torch.Tensor(in_t)

def tensor_to_sparse(in_t: TensorType) -> sparse.csr_matrix:
    return sparse.csr_matrix(in_t)

def sparse_to_tensor(in_sparse: sparse.csr_matrix) -> torch.Tensor:
    return torch.Tensor(in_sparse.todense())

def sparse_equal(sparse_a: sparse.csr_matrix, sparse_b: sparse.csr_matrix) -> bool:
    return (sparse_a != sparse_b).nnz==0 

class TensorList(object):
    @classmethod
    def create_tensor_list(
        cls,
        tensor: Optional[torch.Tensor],
        incoming_tensor: TensorType,
    ):
        incoming_tensor = get_tensor(incoming_tensor)
        if tensor is None:
            return incoming_tensor

        return torch.cat((tensor, incoming_tensor), dim=0)

    @classmethod
    def create_tensor_from_list(
        cls,
        tensor_list: Optional[List[TensorType]],
    ):
        if tensor_list is None:
            return torch.Tensor()
        tensor_list = [get_tensor(t) for t in tensor_list]
        return torch.cat(tensor_list, dim=0)

    def __init__(
        self,
        tensor_list: Optional[TensorType] = None,
        device: str = 'cpu',
        operation_mode: OperationMode = OperationMode.MEMORY_EFFICENT,
    ):
        self.tensor_list = TensorList.create_tensor_list(
            tensor=None, 
            incoming_tensor=TensorList.create_tensor_from_list(tensor_list)
        )
        self.device = device
        self.tensor_list.to(device)
        self.operation_mode = operation_mode

    def append(
        self,
        tensor: TensorType,
    ):
        tensor = get_tensor(tensor).to(self.device)
        self.tensor_list = TensorList.create_tensor_list(tensor=self.tensor_list, incoming_tensor=tensor)
    
    def extend(
        self,
        tensor_list: List[TensorType],
    ):
        tensor_list = [get_tensor(t).to(self.device) for t in tensor_list]
        tensor_list = TensorList.create_tensor_from_list(tensor_list)
        self.append(tensor=tensor_list)

    def to(self, device: str):
        self.tensor_list.to(device)
        self.device = device
    
    def to_list(self) -> List[torch.Tensor]:
        return list(map(lambda t: t.unsqueeze(0), self.tensor_list))
    
    def numpy(self) -> np.ndarray:
        return self.tensor_list.cpu().detach().numpy()

    def tensor(self) -> torch.Tensor:
        return self.tensor_list
    
    def contains(self, item: TensorType) -> int:
        if self.operation_mode == OperationMode.FAST:
            return self._fast_contains(item)
        else:
            return self._memory_efficient_contains(item)
    
    def _fast_contains(self, item: TensorType) -> int:
        search = ((self.tensor_list - item) == 0).all(dim=1).nonzero()
        if len(search) == 0:
            return -1
        res = search[0].item()
        return res
    
    def _memory_efficient_contains(self, item: TensorType) -> int:
        for i, (tensor) in enumerate(self.tensor_list):
            if (tensor == item).all():
                return i
        return -1

    def __getattribute__(self, name):
        '''
        TODO: hacky, but overrides shape field
        '''
        if name == 'shape':
            return self.tensor_list.shape
        return super().__getattribute__(name)
    
    def __len__(self) -> int:
        return len(self.tensor_list)
    
    def __get_item__(self, idx: int) -> torch.Tensor:
        return self.tensor_list[idx]
    
    def __str__(self) -> str:
        return f'TensorList({self.shape})'
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def share_memory(self):
        self.tensor_list.share_memory_()

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
        self.tensor_list = SparseTensorList.create_sparse_tensor_list(
            tensor=self.tensor_list if len(self) != 0 else None,
            incoming_tensor=tensor,
        )
    
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
        return np.array(self.tensor_list.todense())

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