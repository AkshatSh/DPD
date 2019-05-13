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
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.token_indexers import PretrainedBertIndexer
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from allennlp.modules import TimeDistributed, TokenEmbedder

# local imports
from dpd.constants import CADEC_NER_ELMo, CADEC_BERT
from dpd.utils import H5SaveFile
from dpd.training.metrics import TagF1, AverageTagF1
from .crf_tagger import CrfTagger
from .embedder import NERElmoTokenEmbedder, CachedTextFieldEmbedder
from .modules import TaskBase, WeightedCRF

class MultiTaskTagger(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        hidden_dim: int,
        class_labels: List[str],
        cached: bool,
        dropout: Optional[float] = None,
        word_embedder: TokenEmbedder = NERElmoTokenEmbedder(),
        noisy_weight: Optional[float] = 0.1,
    ) -> None:
        super().__init__(vocab)
        self.label_namespace = 'labels'
        self.num_tags = self.vocab.get_vocab_size(self.label_namespace)
        self._verbose_metrics = False

        self.label_encoding = 'BIO'
        constraints = None
        self.include_start_end_transitions = True
        self.metrics = {
                "accuracy": CategoricalAccuracy(),
        }
        self.calculate_span_f1 = True
        self._f1_metric = SpanBasedF1Measure(
            vocab,
            tag_namespace=self.label_namespace,
            label_encoding=self.label_encoding,
        )
        self._tag_f1_metric = TagF1(vocab, class_labels=class_labels)
        self._average_f1_metric = AverageTagF1(vocab, class_labels=class_labels)
        
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.word_embedder = word_embedder
        
        self.task_base = TaskBase(
            vocab,
            hidden_dim,
            class_labels,
            cached,
            word_embedder=self.word_embedder,
        )

        self.tag_projection_layer = TimeDistributed(
            Linear(self.task_base.get_output_dim(), self.num_tags)
        )

        self.gold_head = WeightedCRF(
            self.num_tags,
            constraints,
            include_start_end_transitions=self.include_start_end_transitions
        )

        self.noisy_head = WeightedCRF(
            self.num_tags,
            constraints,
            include_start_end_transitions=self.include_start_end_transitions
        )

        self.noisy_weight = noisy_weight
    
    def _forward_noisy(
        self,
        encoding: torch.Tensor,
        mask: torch.Tensor,
        tags: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = self.tag_projection_layer(encoding)
        best_paths = self.noisy_head.viterbi_tags(logits, mask)

        # Just get the tags and ignore the score.
        predicted_tags = [x for x, y in best_paths]

        output = {"logits": logits, "mask": mask, "tags": predicted_tags}

        if tags is not None:
            # Add negative log-likelihood as loss
            log_likelihood = self.gold_head(logits, tags, mask, weight=weight)
            output["loss"] = -log_likelihood

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, tags, mask.float())
            if self.calculate_span_f1:
                self._f1_metric(class_probabilities, tags, mask.float())
                self._tag_f1_metric(class_probabilities, tags, mask.float())
                self._average_f1_metric(class_probabilities, tags, mask.float())
        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]
        return output
    
    def _forward_gold(
        self,
        encoding: torch.Tensor,
        mask: torch.Tensor,
        tags: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = self.tag_projection_layer(encoding)
        best_paths = self.gold_head.viterbi_tags(logits, mask)

        # Just get the tags and ignore the score.
        predicted_tags = [x for x, y in best_paths]

        output = {"logits": logits, "mask": mask, "tags": predicted_tags}

        if tags is not None:
            # Add negative log-likelihood as loss
            log_likelihood = self.gold_head(logits, tags, mask, weight=weight)
            output["loss"] = -log_likelihood

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, tags, mask.float())
            if self.calculate_span_f1:
                self._f1_metric(class_probabilities, tags, mask.float())
                self._tag_f1_metric(class_probabilities, tags, mask.float())
                self._average_f1_metric(class_probabilities, tags, mask.float())
        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]
        return output

    @overrides
    def forward(
        self,
        sentence: Dict[str, torch.LongTensor],
        entry_id: Optional[torch.LongTensor] = None,
        dataset_id: Optional[torch.LongTensor] = None,
        tags: torch.LongTensor = None,
        weight: torch.Tensor = None,
        metadata: List[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        encoding: Dict[str, Any]= self.task_base(
            sentence,
            entry_id,
            dataset_id,
            tags,
            metadata,
            weight,
        )

        encoded_text: torch.Tensor = encoding['encoded_text']
        mask: torch.Tensor = encoding['mask']
        if weight is not None:
            gold_idx = (weight.squeeze(1) > self.noisy_weight).nonzero().squeeze(1)
            noise_idx = (weight.squeeze(1) <= self.noisy_weight).nonzero().squeeze(1)
            def _safe_partition(array: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
                if array is None:
                    return None
                ret_val = array[idx]
                return ret_val
            gold_dict = self._forward_gold(
                _safe_partition(encoded_text, gold_idx),
                _safe_partition(mask, gold_idx),
                _safe_partition(tags, gold_idx),
                weight=_safe_partition(weight, gold_idx),
            )
            noisy_dict = self._forward_noisy(
                _safe_partition(encoded_text, noise_idx),
                _safe_partition(mask, noise_idx),
                _safe_partition(tags,noise_idx),
                weight=_safe_partition(weight, noise_idx),
            )

            return None
        return self._forward_gold(encoded_text, mask, tags, weight=weight, metadata=metadata)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["tags"] = [
                [self.vocab.get_token_from_index(tag, namespace=self.label_namespace)
                 for tag in instance_tags]
                for instance_tags in output_dict["tags"]
        ]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items()}
    
        tag_f1_metrics = self._tag_f1_metric.get_metric(reset)
        average_f1_metrics = self._average_f1_metric.get_metric(reset)
        metrics_to_return.update({'tag_f1': tag_f1_metrics['f1']})
        metrics_to_return.update({'bio_tag_f1': average_f1_metrics['avg_f1']})

        other_metrics = {}
        other_metrics.update({f'tag_{m}': v for m, v in tag_f1_metrics.items() if m != 'f1'})
        other_metrics.update({f'bio_tag_{m}': v for m, v in average_f1_metrics.items() if m != 'avg_f1'})

        # ignore all other metrics with _ so they do not appear in tqdm
        for metric, val in other_metrics.items():
            metrics_to_return[f'_{metric}'] = val

        if self.calculate_span_f1:
            f1_dict = self._f1_metric.get_metric(reset=reset)
            if self._verbose_metrics:
                metrics_to_return.update(f1_dict)
            else:
                metrics_to_return.update({
                        x: y for x, y in f1_dict.items() if
                        "f1-measure-overall" in x})
        return metrics_to_return