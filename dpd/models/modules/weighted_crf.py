from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import torch
from overrides import overrides
import allennlp
from allennlp.modules import ConditionalRandomField

class WeightedCRF(ConditionalRandomField):
    '''
    A weighted Conditional Random Field implementation

    It allows weighting at the input level by specifying a weight tensor of size (batch_size,)

    The equivalent would be multiplying the forward backward loss for each instance by the appropriate
    weight in the weight tensor
    '''
    @overrides
    def forward(
        self,
        inputs: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.ByteTensor = None,
        # (batch,)
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Computes the log likelihood.
        """
        # pylint: disable=arguments-differ
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.long)

        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)

        batch_loss = log_numerator - log_denominator
        weighted_batch_loss = weight * batch_loss
        return torch.sum(weighted_batch_loss)