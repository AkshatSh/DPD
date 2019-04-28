import warnings
from typing import (
    Dict,
    List,
    Union,
    Any,
    Optional,
    Iterator,
    Callable,
    Tuple,
)

import torch
from torch import nn
import numpy as np
from overrides import overrides
import logging

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary, DatasetReader, Instance
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder

class CachedDataset(object):
    def __init__(
        self,
        dataset_id: int,
    ):
        self.dataset_id = dataset_id
        self.embedding_dataset_list: Optional[List[torch.Tensor]] = []
        self.index_to_sid: Dict[int, int] = {}
        self.sid_to_start: Dict[int, int] = {}
        self.sid_to_end: Dict[int, int] = {}
        self.embedded_dataset: Optional[np.ndarray] = None
        self.is_continous: bool = False

    def cache_entry(
        self,
        s_id: int,
        embedding_tensor: torch.Tensor,
    ):
        index = len(self.embedding_dataset)
        self.embedding_dataset_list.append(embedding_tensor.cpu().detach())
        self.sid_to_index[s_id] = index
        self.index_to_sid[index] = s_id

    def continous(self):
        self.embedded_dataset = torch.cat(self.embedded_dataset, dim=1).cpu().deatch().numpy()
        self.embedding_dataset_list = None
        self.is_continous = True

    def _ready(self):
        if not self.is_continous:
            raise Exception(f'Cached dataset needs to be continous for retrieval')

    
    def get_embedding(self, s_id: int) -> Optional[torch.Tensor]:
        self._ready()
        
        if s_id not in self.sid_to_start:
            return None
        
        start = self.sid_to_start[s_id]
        end = self.sid_to_end[s_id]

        tensor = torch.Tensor(self.embedded_dataset[start:end])
        return tensor

    def numpy(self):
        self._ready()

        if type(self.embedded_dataset) == torch.Tensor:
            self.embedded_dataset = self.embedded_dataset.cpu().deatch().numpy()
        elif type(self.embedded_dataset) == np.ndarray:
            # already numpy
            pass
        else:
            raise Exception(f'Unknown type for embedded dataset {type(self.embedded_dataset)}')

    def tensor(self, device: str):
        self._ready()

        if type(self.embedded_dataset) == torch.Tensor:
            # already tensor
            pass
        elif type(self.embedded_dataset) == np.ndarray:
            self.embedded_dataset = torch.Tensor(self.embedded_dataset).to(device)
        else:
            raise Exception(f'Unknown type for embedded dataset {type(self.embedded_dataset)}')

    @classmethod
    def cache_dataset(
        cls,
        dataset_id: int,
        dataset: Iterator[Instance],
        embedding_function: Callable[Instance, torch.Tensor],
    ):
        cached_dataset = cls(dataset_id=dataset_id)
        for instance in dataset:
            # cache the instance
            # with the embedding function
            pass
        cached_dataset.contious()
        return cached_dataset


class CachedTextFieldEmbedder(nn.Module):
    '''
    Given an AllenNLP text field embedder, assumes the embedder to be frozen
    and caches the result so it never needs to be recomputed based on dataset id
    and entry id

    Can also be used for futher analysis of an embedding space through TSNE, PCA, and UMAP Projections
    '''
    def __init__(
        self,
        text_field_embedder: TextFieldEmbedder,
        silent_error: bool = False,
    ):
        super(CachedTextFieldEmbedder, self).__init__()
        self.text_field_embedder = text_field_embedder

        # maps dataset id to a cached instance of the dataset
        self.cached_datasets: Dict[int, CachedDataset] = {}
        self.silent_error = silent_error
    
    def get_output_dim(self) -> int:
        return self.text_field_embedder.get_output_dim()
    
    def forward(
        self,
        input_tensor: torch.Tensor, # (batch_size, input_dim)
        sentence_ids: Optional[torch.Tensor] = None, # (batch_size, 1)
        dataset_ids: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        '''
        forward cached, forwards the text field embedder, but tries to retrieve from the
        cached datasets if possible

        relies retrieval from ``sentence_ids`` and ``dataset_ids``

        to disable cache set ``use_cache`` to false
        '''
        pass
    
    def cache(
        self,
        dataset_id: int,
        dataset: Iterator[Instance],
    ) -> bool:
        '''
        takes the input ``dataset`` and caches the entire thing to stop retrieval

        returns success
        '''
        try:
            self.cached_datasets[dataset_id] = CachedDataset.cache_dataset(
                dataset_id=dataset_id,
                dataset=dataset,
            )
        except Exception as e:
            if self.silent_error:
                return False
            else:
                raise e

        return True
    