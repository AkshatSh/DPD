from typing import Dict, List, Union, Set, Iterator
import logging
import textwrap

from overrides import overrides
import torch
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.fields.field import Field
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.vocabulary import Vocabulary

from dpd.common import TensorType, TensorList

logger = logging.getLogger(name=__name__)


class ProbabilisticLabelField(Field[torch.Tensor]):
    def __init__(
            self,
            labels: Union[List[TensorType], TensorList],
            sequence_field: SequenceField,
            label_namespace: str = 'labels'
        ) -> None:
        self.labels = labels
        self.sequence_field = sequence_field
        self._indexed_labels: List[torch.Tensor] = None
        if type(labels) == list:
            self._indexed_labels = TensorList(labels).to_list()
        elif type(labels) == TensorList:
            self._indexed_labels = labels.to_list()
        else:
            raise Exception(f'Unknown type for labels {type(labels)}')

        self._label_namespace = label_namespace
        if len(labels) != sequence_field.sequence_length():
            raise ConfigurationError("Label length and sequence length "
                                     "don't match: %d and %d" % (len(labels), sequence_field.sequence_length()))

    # Sequence methods
    def __iter__(self) -> Iterator[TensorType]:
        return iter(self.labels)

    def __getitem__(self, idx: int) -> Union[str, int]:
        return self.labels[idx]

    def __len__(self) -> int:
        return len(self.labels)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {'num_tokens': self.sequence_field.sequence_length()}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        desired_num_tokens = padding_lengths['num_tokens']
        padded_tags = pad_sequence_to_length(self._indexed_labels, desired_num_tokens, default_value=lambda: torch.zeros(self._indexed_labels[0].shape))
        tensor = TensorList(padded_tags).tensor()
        return tensor

    @overrides
    def empty_field(self) -> 'SequenceLabelField': 
        # pylint: disable=protected-access
        # The empty_list here is needed for mypy
        empty_list: List[TensorType] = []
        sequence_label_field = ProbabilisticLabelField(empty_list, self.sequence_field.empty_field())
        sequence_label_field._indexed_labels = empty_list
        return sequence_label_field

    def __str__(self) -> str:
        length = self.sequence_field.sequence_length()
        return f"ProbabilisticLabelField of length {length}"