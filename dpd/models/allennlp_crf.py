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
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask


# local imports
from .crf import CRF


'''
This is a wrapper for the CRF models to be used in the AllenNLP pipeline
'''

class AllenNLPCRF(Model):
    def __init__(
        self,
        word_embeddings: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        vocab: Vocabulary,
    ):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.crf = CRF(
            vocab=vocab,
            tagset=vocab,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
        )
    
    def forward(
        self,
        sentence: Dict[str, torch.Tensor],
        labels: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        features = self.encoder(embeddings, mask)
        if labels is not None:
            # compute MLE
            partition = self.crf.compute_partion(features, mask)
            gold_score = self.crf.score(features, labels, mask)

            # to make this a gradient descent problem we want to minimze, hence
            # the order is switched from the MLE equation
            output['loss'] = partition - gold_score
        else:
            # compute viterbi
            tag_seq = self.crf.viterbi_decode(features, mask)
            output['tag_seq'] = tag_seq

        return output