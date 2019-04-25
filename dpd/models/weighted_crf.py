from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import torch
import allennlp
from allennlp.modules import ConditionalRandomField

class WeightedCRF(ConditionalRandomField):
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