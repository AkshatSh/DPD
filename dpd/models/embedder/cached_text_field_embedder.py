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
from overrides import overrides
import h5py
from tqdm import tqdm
import copy

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary, DatasetReader, Instance
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn.util import get_text_field_mask, move_to_device

from dpd.utils import SaveFile

class CachedDataset(object):
    def __init__(
        self,
        dataset_id: int,
    ):
        self.dataset_id = dataset_id
        self.index_to_sid: Dict[int, int] = {}
        self.sid_to_start: Dict[int, int] = {}
        self.sid_to_end: Dict[int, int] = {}
        self.embedded_dataset: Optional[np.ndarray] = None

    def cache_entry(
        self,
        s_id: int,
        embedding_tensor: torch.Tensor,
    ):
        index = len(self.embedded_dataset) if self.embedded_dataset is not None else 0
        embedding_tensor = embedding_tensor.detach().cpu()

        if len(embedding_tensor.shape) == 3:
            # make sure batch size is 1
            assert len(embedding_tensor) == 1
            embedding_tensor = embedding_tensor[0]

        if self.embedded_dataset is None:
            self.embedded_dataset = embedding_tensor
        else:
            self.embedded_dataset = torch.cat((self.embedded_dataset, embedding_tensor), dim=0)
        self.sid_to_start[s_id] = index
        self.index_to_sid[index] = s_id
        self.sid_to_end[s_id] = len(self.embedded_dataset)
    
    def get_embedding(self, s_id: int) -> Optional[torch.Tensor]:
        if s_id not in self.sid_to_start:
            return None
        
        start = self.sid_to_start[s_id]
        end = self.sid_to_end[s_id]

        tensor = torch.Tensor(self.embedded_dataset[start:end])
        return tensor

    def numpy(self):
        if type(self.embedded_dataset) == torch.Tensor:
            self.embedded_dataset = self.embedded_dataset.cpu().deatch().numpy()
        elif type(self.embedded_dataset) == np.ndarray:
            # already numpy
            pass
        else:
            raise Exception(f'Unknown type for embedded dataset {type(self.embedded_dataset)}')

    def tensor(self, device: str):
        if type(self.embedded_dataset) == torch.Tensor:
            # already tensor
            pass
        elif type(self.embedded_dataset) == np.ndarray:
            self.embedded_dataset = torch.Tensor(self.embedded_dataset).to(device)
        else:
            raise Exception(f'Unknown type for embedded dataset {type(self.embedded_dataset)}')
    
    def get_embeddings(self):
        return self.embedded_dataset
    
    def save(self, key: str, save_file: SaveFile):
        save_file.save_dict(item=self.index_to_sid, key=f'{key}/index_to_sid')
        save_file.save_dict(item=self.sid_to_start, key=f'{key}/sid_to_start')
        save_file.save_dict(item=self.sid_to_end, key=f'{key}/sid_to_end')
        save_file.save_np(item=self.embedded_dataset, key=f'{key}/embedded_dataset')
    
    def load(self, key: str, save_file: SaveFile):
        self.index_to_sid = save_file.load_dict(key=f'{key}/index_to_sid')
        self.sid_to_start = save_file.load_dict(key=f'{key}/sid_to_start')
        self.sid_to_end = save_file.load_dict(key=f'{key}/sid_to_end')
        self.embedded_dataset = save_file.load_np(key=f'{key}/embedded_dataset')
    
    @overrides
    def __str__(self) -> str:
        return f'CachedDataset({self.embedded_dataset.shape})'

    @classmethod
    def cache_dataset(
        cls,
        dataset_id: int,
        dataset: Iterator[Instance],
        embedding_function: Callable[[Instance], torch.Tensor],
    ):
        cached_dataset = cls(dataset_id=dataset_id)
        for instance in tqdm(dataset):
            # cache the instance
            # with the embedding function
            s_id = instance.fields['entry_id'].as_tensor(padding_lengths=None).item()
            e_t = embedding_function(instance)
            cached_dataset.cache_entry(s_id=s_id, embedding_tensor=e_t)
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
        sentence: Dict[str, torch.Tensor], # (batch_size, seq_len)
        sentence_ids: Optional[torch.Tensor] = None, # (batch_size, 1)
        dataset_ids: Optional[torch.Tensor] = None, # (batch_size, 1)
        use_cache: bool = True,
        padding_val: float = 0.,
    ) -> torch.Tensor:
        '''
        forward cached, forwards the text field embedder, but tries to retrieve from the
        cached datasets if possible

        relies retrieval from ``sentence_ids`` and ``dataset_ids``

        to disable cache set ``use_cache`` to false
        '''
        mask = get_text_field_mask(sentence)
        embedding_tensor: torch.Tensor = None
        if not use_cache or sentence_ids is None or dataset_ids[0].item() not in self.cached_datasets:
            return self.text_field_embedder(sentence)
        else:
            input_tensor = sentence['tokens']
            output_tensor: torch.Tensor = torch.zeros(input_tensor.shape[:2] + (self.get_output_dim(),)).to(input_tensor.device)
            for i, (s_id, d_id, input_tensor) in enumerate(zip(sentence_ids, dataset_ids, input_tensor)):
                d_id: int = d_id.item()
                s_id: int = s_id.item()
                et: torch.Tensor = self.cached_datasets[d_id].get_embedding(s_id)
                output_tensor[i, :len(et)] = et
            return output_tensor

    def cache(
        self,
        dataset_id: int,
        dataset: Iterator[Instance],
        vocab: Vocabulary,
        cuda_device: int = -1,
    ) -> bool:
        '''
        takes the input ``dataset`` and caches the entire thing to stop retrieval

        returns success
        '''
        def _ef(inst: Instance, vocab: Vocabulary):
            inst.fields['sentence'].index(vocab)
            pl = inst.fields['sentence'].get_padding_lengths()
            # adds batch dimension
            input_tensor = copy.deepcopy(inst.fields['sentence'].as_tensor(padding_lengths=pl))
            input_tensor['tokens'] = input_tensor['tokens'].unsqueeze(0)
            input_tensor = move_to_device(input_tensor, cuda_device)

            return self.text_field_embedder(
                input_tensor,
            )
        try:
            self.cached_datasets[dataset_id] = CachedDataset.cache_dataset(
                dataset_id=dataset_id,
                dataset=dataset,
                embedding_function=lambda inst: _ef(inst, vocab),
            )
        except Exception as e:
            if self.silent_error:
                return False
            else:
                raise e

        return True
    
    def setup_cache(
        self,
        dataset_id: int,
    ):
        if dataset_id not in self.cached_datasets:
            self.cached_datasets[dataset_id] = CachedDataset(dataset_id=dataset_id)
        else:
            logging.info(f'Already cached dataset with id: {dataset_id}')

    def save(
        self,
        save_file: SaveFile,
    ):
        for i, (d_id, cached_dataset) in enumerate(self.cached_datasets.items()):
            key = f'cached_dataset_{d_id}'
            cached_dataset.save(key=key, save_file=save_file)
    
    def load(
        self,
        save_file: SaveFile,
    ):
        for i, (d_id, cached_dataset) in enumerate(self.cached_datasets.items()):
            key = f'cached_dataset_{d_id}'
            cached_dataset.load(key=key, save_file=save_file)

    def __str__(self) -> str:
        return str(self.cached_datasets)
