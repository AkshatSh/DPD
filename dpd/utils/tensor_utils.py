from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Union
)

import os
import sys

import torch
import numpy as np

TensorType = Union[
    np.ndarray,
    torch.Tensor
]

def get_tensor(in_t: TensorType) -> torch.Tensor:
    if type(in_t) == torch.Tensor:
        return in_t
    return torch.Tensor(in_t)

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
        device: str = 'cpu'
    ):
        self.tensor_list = TensorList.create_tensor_list(
            tensor=None, 
            incoming_tensor=TensorList.create_tensor_from_list(tensor_list)
        )
        self.device = device
        self.tensor_list.to(device)

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
    
    def numpy(self) -> np.ndarray:
        return self.tensor_list.cpu().detach().numpy()

    def tensor(self) -> torch.Tensor:
        return self.tensor_list
    
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