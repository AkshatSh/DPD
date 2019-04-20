# pylint: disable=access-member-before-definition
from typing import Dict

from overrides import overrides
import torch

from allennlp.data.fields.field import Field


class IntField(Field[torch.Tensor]):
    """
    A ``IntField`` is a field that represents a number
    ----------
    number : ``int``, required.
        the number to be stored
    """
    def __init__(self, number: int) -> None:
        self.number = number

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        # pylint: disable=no-self-use
        return {}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        # pylint: disable=unused-argument
        tensor = torch.LongTensor([self.number])
        return tensor

    @overrides
    def empty_field(self):
        return IntField(-1)

    def __str__(self) -> str:
        return f"IntField with value: ({self.number})."

    def __eq__(self, other) -> bool:
        if isinstance(other, int):
            return other == self.number
        else:
            return id(self) == id(other)