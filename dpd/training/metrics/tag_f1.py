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


@Metric.register("tag_f1")
class TagF1(F1Measure):
    """
    Computes Precision, Recall and F1 with respect to a given ``positive_label``.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.
    """
    def __init__(self, vocab: Vocabulary, class_labels: List[str]) -> None:
        self.class_labels = class_labels
        positive_labels = []
        for class_label in class_labels:
            positive_labels.append(vocab.get_token_index(class_label, namespace='labels'))
        
        self._pos_labels = set(positive_labels)
        super(TagF1, self).__init__(positive_label=1)
    
    def project_class_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        output = tensor.clone()
        # vocab values should never be negative
        # HACK(akshatsh): there should be a better way
        # but algorithm is:
        # multiply everything by -1
        output *= -1
        # replace all numbers in positive class with 1
        for class_id in self._pos_labels:
            output[output == -class_id] = 1
        # convert all other numbers to 0
        output[output < 0] = 0

        return output


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
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to F1Measure contains an id >= {}, "
                                     "the number of classes.".format(num_classes))
        if mask is None:
            mask = torch.ones_like(gold_labels)
        mask = mask.float()
        gold_labels = self.project_class_tensor(gold_labels)
        gold_labels = gold_labels.float()
        positive_label_mask = gold_labels.eq(self._positive_label).float()
        negative_label_mask = 1.0 - positive_label_mask

        argmax_predictions = predictions.max(-1)[1].float().squeeze(-1)
        argmax_predictions = self.project_class_tensor(argmax_predictions)

        # True Negatives: correct non-positive predictions.
        correct_null_predictions = (argmax_predictions !=
                                    self._positive_label).float() * negative_label_mask
        self._true_negatives += (correct_null_predictions.float() * mask).sum()

        # True Positives: correct positively labeled predictions.
        correct_non_null_predictions = (argmax_predictions ==
                                        self._positive_label).float() * positive_label_mask
        self._true_positives += (correct_non_null_predictions * mask).sum()

        # False Negatives: incorrect negatively labeled predictions.
        incorrect_null_predictions = (argmax_predictions !=
                                      self._positive_label).float() * positive_label_mask
        self._false_negatives += (incorrect_null_predictions * mask).sum()

        # False Positives: incorrect positively labeled predictions
        incorrect_non_null_predictions = (argmax_predictions ==
                                          self._positive_label).float() * negative_label_mask
        self._false_positives += (incorrect_non_null_predictions * mask).sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        precision = float(self._true_positives) / float(self._true_positives + self._false_positives + 1e-13)
        recall = float(self._true_positives) / float(self._true_positives + self._false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        if reset:
            self.reset()

        return {
            'precision' : precision,
            'recall' : recall,
            'f1' : f1_measure,
        }

    def reset(self):
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0