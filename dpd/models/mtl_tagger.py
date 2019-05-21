from typing import (
    List,
    Tuple,
    Dict,
    Callable,
    Any,
    Optional,
)

from overrides import overrides
import logging
import random

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
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.token_indexers import PretrainedBertIndexer
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from allennlp.modules import TimeDistributed, TokenEmbedder
from allennlp.models import SimpleTagger
from allennlp.common.checks import check_dimensions_match, ConfigurationError

# Probabilistic Loss Function
# Soft Cross Entropy Loss: A cross entropy loss for probabilisitc
# distributions. Defined here:
# https://github.com/HazyResearch/metal/blob/master/metal/end_model/loss.py
from metal.end_model.loss import SoftCrossEntropyLoss

# local imports
from dpd.constants import CADEC_NER_ELMo, CADEC_BERT
from dpd.utils import H5SaveFile
from dpd.training.metrics import TagF1, AverageTagF1
from .linear_tagger import LinearTagger
from .crf_tagger import CrfTagger

logger = logging.getLogger(name=__name__)

class MTLTagger(Model):
    def __init__(
        self, 
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        calculate_span_f1: bool = True,
        label_encoding: Optional[str] = 'BIO',
        label_namespace: str = "labels",
        verbose_metrics: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        class_labels: Optional[List[str]] = None,
        constrain_crf_decoding: bool = None,
        noisy_threshold: float = 0.8,
        **kwargs,
    ) -> None:
        super(MTLTagger, self).__init__(vocab, regularizer)
        self.noisy_threshold = noisy_threshold

        self.noisy_tagger = LinearTagger(
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            encoder=encoder,
            calculate_span_f1=calculate_span_f1,
            label_encoding=label_encoding,
            label_namespace=label_namespace,
            verbose_metrics=verbose_metrics,
            initializer=initializer,
            regularizer=regularizer,
            class_labels=class_labels,
            use_probabillity_labels=True,
            **kwargs,
        )

        self.gold_tagger = CrfTagger(
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            encoder=encoder,
            calculate_span_f1=calculate_span_f1,
            label_encoding=label_encoding,
            label_namespace=label_namespace,
            verbose_metrics=verbose_metrics,
            initializer=initializer,
            regularizer=regularizer,
            class_labels=class_labels,
            constrain_crf_decoding=constrain_crf_decoding,
            **kwargs,
        )

        self.is_noisy = False

    @overrides
    def forward(
        self,
        sentence: Dict[str, torch.LongTensor],
        entry_id: Optional[torch.LongTensor] = None,
        dataset_id: Optional[torch.LongTensor] = None,
        tags: torch.LongTensor = None,
        weight: torch.Tensor = None,
        metadata: List[Dict[str, Any]] = None,
        prob_labels: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        prev: bool = self.is_noisy
        self.is_noisy = prob_labels is not None
        if prev != self.is_noisy and tags is not None:
            logger.warn(f'Switching MTL train mode noisy: {self.is_noisy}')

        # allow random switching with low probability from noisy to gold
        # task head
        if self.is_noisy and random.random() < self.noisy_threshold:
            return self.noisy_tagger.forward(
                sentence=sentence,
                entry_id=entry_id,
                dataset_id=dataset_id,
                tags=tags,
                weight=weight,
                metadata=metadata,
                prob_labels=prob_labels,
                **kwargs,
            )
        else:
            return self.gold_tagger.forward(
                tokens=sentence,
                entry_id=entry_id,
                dataset_id=dataset_id,
                tags=tags,
                weight=weight,
                metadata=metadata,
                prob_labels=prob_labels,
                freeze_encoder=True,
                **kwargs,
            )

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        noisy_metrics = self.noisy_tagger.get_metrics(reset=reset)
        gold_metrics = self.gold_tagger.get_metrics(reset=reset)
        if self.is_noisy:
            return noisy_metrics
        else:
            return gold_metrics