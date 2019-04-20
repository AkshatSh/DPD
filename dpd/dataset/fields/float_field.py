# pylint: disable=access-member-before-definition
from typing import Dict

from overrides import overrides
import torch

from allennlp.data.fields.field import Field
from allennlp.common.checks import ConfigurationError


class FloatField(Field[torch.Tensor]):
    """
    A ``FloatField`` is a field that represents a number
    ----------
    number : ``int``, required.
        the number to be stored
    """
    def __init__(self, number: float) -> None:
        if type(number) != float:
            raise ConfigurationError(
                f'Not supported type for FloatField: {number}'
            )
        self.number = number

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        # pylint: disable=no-self-use
        return {}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        # pylint: disable=unused-argument
        tensor = torch.Tensor([self.number])
        return tensor

    @overrides
    def empty_field(self):
        return FloatField(-1.0)

    def __str__(self) -> str:
        return f"FloatField with value: ({self.number})."

    def __eq__(self, other) -> bool:
        if isinstance(other, float):
            return other == self.number
        else:
            return id(self) == id(other)