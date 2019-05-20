from typing import (
    List,
    Tuple,
    Dict,
    Callable,
    Any,
    Optional,
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
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.token_indexers import PretrainedBertIndexer
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from allennlp.modules import TimeDistributed, TokenEmbedder
from allennlp.models import SimpleTagger
from allennlp.common.checks import check_dimensions_match, ConfigurationError

# local imports
from dpd.constants import CADEC_NER_ELMo, CADEC_BERT
from dpd.utils import H5SaveFile
from dpd.training.metrics import TagF1, AverageTagF1
from dpd.training.loss_functions import SoftCrossEntropyLoss

class LinearTagger(SimpleTagger):
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
        use_probabillity_labels: bool = True,
    ) -> None:
        super(SimpleTagger, self).__init__(vocab, regularizer)

        self.use_probabillity_labels = use_probabillity_labels
        self.prob_loss = SoftCrossEntropyLoss()
        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size(label_namespace)
        self.encoder = encoder
        self._verbose_metrics = verbose_metrics
        self.tag_projection_layer = TimeDistributed(
            Linear(self.encoder.get_output_dim(),self.num_classes),
        )

        check_dimensions_match(
            text_field_embedder.get_output_dim(),
            encoder.get_input_dim(),
            "text field embedding dim",
            "encoder input dim",
        )

        self.metrics = {
                "accuracy": CategoricalAccuracy(),
        }

        self._f1_metric = SpanBasedF1Measure(
            vocab,
            tag_namespace=label_namespace,
            label_encoding=label_encoding,
        )
        self._tag_f1_metric = TagF1(vocab, class_labels=class_labels)
        self._average_f1_metric = AverageTagF1(vocab, class_labels=class_labels)

        initializer(self)

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
        embedded_text_input = self.text_field_embedder(sentence)
        batch_size, sequence_length, _ = embedded_text_input.size()
        mask = get_text_field_mask(sentence)
        encoded_text = self.encoder(embedded_text_input, mask)

        logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
            [
                batch_size,
                sequence_length,
                self.num_classes,
            ]
        )

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if tags is not None:
            if prob_labels is not None and self.use_probabillity_labels:
                loss = self.prob_loss(logits, prob_labels, weight=mask.float())
            else:
                loss = sequence_cross_entropy_with_logits(logits, tags, mask)
            for metric in self.metrics.values():
                metric(logits, tags, mask.float())
            if self._f1_metric is not None:
                self._f1_metric(logits, tags, mask.float())
                self._tag_f1_metric(class_probabilities, tags, mask.float())
                self._average_f1_metric(class_probabilities, tags, mask.float())
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {
            metric_name: metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }
    
        tag_f1_metrics = self._tag_f1_metric.get_metric(reset=reset)
        average_f1_metrics = self._average_f1_metric.get_metric(reset=reset)
        metrics_to_return.update({'tag_f1': tag_f1_metrics['f1']})
        metrics_to_return.update({'bio_tag_f1': average_f1_metrics['avg_f1']})

        other_metrics = {}
        other_metrics.update({f'tag_{m}': v for m, v in tag_f1_metrics.items() if m != 'f1'})
        other_metrics.update({f'bio_tag_{m}': v for m, v in average_f1_metrics.items() if m != 'avg_f1'})

        # ignore all other metrics with _ so they do not appear in tqdm
        for metric, val in other_metrics.items():
            metrics_to_return[f'_{metric}'] = val

        f1_dict = self._f1_metric.get_metric(reset=reset)
        if self._verbose_metrics:
            metrics_to_return.update(f1_dict)
        else:
            metrics_to_return.update({
                    x: y for x, y in f1_dict.items() if
                    "f1-measure-overall" in x})
        return metrics_to_return