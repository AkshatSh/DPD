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
from allennlp.modules import TimeDistributed, TokenEmbedder
import allennlp.nn.util as util


# local imports
from dpd.constants import CADEC_NER_ELMo, CADEC_BERT
from dpd.utils import H5SaveFile
from ..embedder import NERElmoTokenEmbedder, CachedTextFieldEmbedder


class TaskBase(nn.Module):
    def __init__(
        self,
        vocab: Vocabulary,
        hidden_dim: int,
        class_labels: List[str],
        cached: bool,
        word_embedder: TokenEmbedder,
        label_namespace: str = 'labels',
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.vocab = vocab
        self.word_embeddings = BasicTextFieldEmbedder(
            {"tokens": word_embedder},
        )
        self.num_tags = self.vocab.get_vocab_size(label_namespace)
        self.cached_embeddings = cached
        self.hidden_dim = hidden_dim
        if cached:
            self.word_embeddings = CachedTextFieldEmbedder(
                text_field_embedder=self.word_embeddings,
            )

            self.word_embeddings.setup_cache(dataset_id=0)
            self.word_embeddings.setup_cache(dataset_id=1)

            self.word_embeddings.load(save_file=H5SaveFile(CADEC_NER_ELMo))

        self.seq2seq_model = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(
                word_embedder.get_output_dim(),
                hidden_dim,
                bidirectional=True,
                batch_first=True,
            ),
        )

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    
    @overrides
    def forward(
        self,
        tokens: Dict[str, torch.LongTensor],
        entry_id: Optional[torch.LongTensor] = None,
        dataset_id: Optional[torch.LongTensor] = None,
        tags: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        weight: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if not self.cached_embeddings:
            embedded_text_input = self.word_embeddings(tokens)
        else:
            embedded_text_input = self.word_embeddings(
                tokens,
                sentence_ids=entry_id,
                dataset_ids=dataset_id,
            )
        mask = util.get_text_field_mask(tokens)

        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        encoded_text = self.seq2seq_model(embedded_text_input, mask)
        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        return {'encoded_text': encoded_text, 'mask': mask}

    def get_output_dim(self) -> int:
        return self.seq2seq_model.get_output_dim()

        