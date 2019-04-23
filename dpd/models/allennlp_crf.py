from typing import (
    List,
    Tuple,
    Dict,
    Callable,
)

# PyTorch imports
import torch
from torch import nn

# AllenNLP imports
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask


# local imports
from .allennlp_crf_tagger import CrfTagger
from .embedder.ner_elmo import NERELMoTokenEmbedder


'''
This is a wrapper for the CRF models to be used in the AllenNLP pipeline
'''
class ELMoCrfTagger(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        hidden_dim: int,
        class_labels: List[str],
    ) -> None:
        super().__init__(vocab)
        elmo_embedder = NERElmoTokenEmbedder()
        self.vocab = vocab
        self.word_embeddings = BasicTextFieldEmbedder(
            {"tokens": elmo_embedder},
        )

        self.seq2seq_model = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(
                elmo_embedder.get_output_dim(),
                hidden_dim,
                bidirectional=True,
                batch_first=True,
            ),
        )

        self.model = CrfTagger(
            vocab,
            self.word_embeddings,
            self.seq2seq_model,
            label_encoding='BIO',
            calculate_span_f1=True,
            # constrain_crf_decoding=True,
            verbose_metrics=False,
            class_labels=class_labels,
        )
    
    def forward(
        self,
        sentence: Dict[str, torch.Tensor],
        dataset_id: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor = None,
        entry_id: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        model_out = self.model(
            tokens=sentence,
            tags=labels,
        )

        return model_out
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.model.get_metrics(reset)
        return metrics