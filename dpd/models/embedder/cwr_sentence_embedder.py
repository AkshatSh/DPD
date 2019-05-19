from typing import (
    List,
)

import torch
from torch import nn

from dpd.common import TensorList
from dpd.dataset import BIODataset, UnlabeledBIODataset

from .cached_text_field_embedder import CachedTextFieldEmbedder

class SentenceEmbedder(nn.Module):
    @classmethod
    def build_index(cls, sentence_embedder, dataset: UnlabeledBIODataset) -> TensorList:
        index = TensorList()
        for inst in dataset:
            sentence_embedding: torch.Tensor = sentence_embedder(
                sentence_ids=torch.Tensor([inst['id']]),
                dataset_ids=torch.Tensor([dataset.dataset_id]),
            )
            index.append(sentence_embedding)
        return index

    def __init__(
        self,
        cwr: CachedTextFieldEmbedder,
        mode: str = 'avg',
    ):
        super(SentenceEmbedder, self).__init__()
        self.cwr = cwr
        self.mode = mode
    
    def _average_embedding(self, sentence_id: int, dataset_id: int) -> torch.Tensor:
        cwr_embedding: torch.Tensor = self.cwr.get_embedding(
            sentence_id=sentence_id,
            dataset_id=dataset_id,
        )
        
        return cwr_embedding.sum(dim=0) / len(cwr_embedding)
    
    def forward(
        self,
        sentence_ids: torch.Tensor, # (batch_size, )
        dataset_ids: torch.Tensor, # (batch_size, )
    ) -> torch.Tensor:
        embeddings: List[torch.Tensor] = [
            self._average_embedding(
                sentence_id=sentence_id.item(),
                dataset_id=dataset_id.item(),
            ).unsqueeze(0)
            for sentence_id, dataset_id in zip(sentence_ids, dataset_ids)
        ]
        return torch.cat(embeddings, dim=0)