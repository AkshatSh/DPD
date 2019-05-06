from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Any,
    Callable
)

import os
import sys

import torch
import numpy as np
import allennlp

from dpd.utils import TensorList

class FeaturePadder:
    @classmethod
    def pad_tensor(
        cls,
        tensors: List[Optional[torch.Tensor]],
        padding_constructor: Callable[[Tuple[int, int]], torch.Tensor],
    ) -> List[torch.Tensor]:
        non_null = list(filter(lambda t: t is not None, tensors))
        if len(non_null) == 0:
            raise Exception(f'All nulls passed into padder')

        non_null_shapes = set(map(lambda t: t.shape, non_null))
        if len(non_null_shapes) != 1:
            raise Exception(f'All shapes must be the same got: {non_null_shapes}')
        t_shape = non_null[0].shape

        ret = []
        for tensor in tensors:
            if tensor is None:
                ret.append(padding_constructor(size=t_shape))
            else:
                ret.append(tensor)
        return ret

    @classmethod
    def zero_tensor(cls, tensors: List[Optional[torch.Tensor]]) -> List[torch.Tensor]:
        return cls.pad_tensor(tensors, padding_constructor=torch.zeros)

    @classmethod
    def randn_tensor(cls, tensors: List[Optional[torch.Tensor]]) -> List[torch.Tensor]:
        return cls.pad_tensor(tensors, padding_constructor=torch.randn)


class FeatureCollator:
    @classmethod
    def sum(cls, features: List[torch.Tensor]) -> torch.Tensor:
        return sum(map(lambda t: t.float(), features)).long()

    @classmethod
    def concat(cls, features: List[torch.Tensor]) -> torch.Tensor:
        tl = TensorList(features)
        return tl.tensor().reshape(1, -1)

    @classmethod
    def average(cls, features: List[torch.Tensor]) -> torch.Tensor:
        return cls.sum(features) / len(features)