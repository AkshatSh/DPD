from typing import (
    List,
    Tuple,
    Dict,
    Callable,
    Optional,
    Any,
)

from overrides import overrides

# PyTorch imports
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.linear import Linear

# AllenNLP imports
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.modules import TimeDistributed
import allennlp.nn.util as util

# local imports
from dpd.utils import H5SaveFile
from ..embedder import NERElmoTokenEmbedder, CachedTextFieldEmbedder


class SoftmaxHead(nn.Module):
    def __init__(
        self,
        vocab: Vocabulary,
        hidden_dim: int,
        label_namespace: str = 'labels',
        dropout: Optional[float] = None,
    ):
        super().__init__(vocab)
        self.tag_projection_layer = TimeDistributed(
            Linear(hidden_dim, self.num_tags)
        )
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    @overrides
    def forward(
        self,
        encoded_text: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.dropout:
            encoded_text = self.dropout(encoded_text)
        logits = self.text_to_logits(encoded_text)
        logits = F.softmax(logits, dim=0)
        return {'logits': logits}

        