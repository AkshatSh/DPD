from typing import (
    Optional,
    List,
    Tuple,
    Dict,
)

import torch

from allennlp.training.metrics.metric import Metric
from allennlp.training.metrics import F1Measure
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary


@Metric.register("average_tag_f1")
class AverageTagF1(Metric):
    """
    Computes Precision, Recall and F1 with respect to a given ``positive_label``.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.
    """
    def __init__(self, vocab: Vocabulary, class_labels: List[str]) -> None:
        self.class_labels = class_labels
        positive_labels = {}
        for class_label in class_labels:
            cls_index = vocab.get_token_index(class_label, namespace='labels')
            positive_labels[class_label] = (cls_index, F1Measure(positive_label=cls_index))
        
        self._postiive_labels = positive_labels

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        for cls_label, (cls_index, metric) in self._postiive_labels.items():
            metric(predictions, gold_labels, mask)

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        Aprecision : float
        Arecall : float
        Af1-measure : float
        """
        metric_values = []
        for cls_label, (cls_index, metric) in self._postiive_labels.items():
            precision, recall, f1_measure = metric.get_metric()

            metric_values.append({
                'class': cls_label,
                'recall': recall,
                'precision': precision,
                'f1': f1_measure,
            })

        if reset:
            self.reset()
        
        def _get_average(values: List[Dict[str, float]], key: str) -> float:
            metric_values = [v[key] for v in values]
            if len(metric_values) == 0:
                return 0.
            return sum(metric_values) / len(metric_values)

        return {
            'avg_precision' : _get_average(metric_values, 'precision'),
            'avg_recall' : _get_average(metric_values, 'recall'),
            'avg_f1' : _get_average(metric_values, 'f1'),
            'class_metric_values': metric_values,
        }

    def reset(self):
        for cls_label, (cls_index, metric) in self._postiive_labels.items():
            metric.reset()