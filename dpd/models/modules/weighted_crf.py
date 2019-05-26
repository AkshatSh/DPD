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
from allennlp.nn import util

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

        if weight is not None:
            batch_loss = weight * batch_loss
        return torch.sum(batch_loss)

    def soft_likelihood(
        logits: torch.Tensor,
        probability_tags: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        """
        batch_size, sequence_length, num_tags = logits.size()

        # Transpose batch size and sequence dimensions
        mask = mask.float().transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]

        # For each i we compute logits for the transitions from timestep i-1 to timestep i.
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are
        # (instance, current_tag, next_tag)
        for i in range(1, sequence_length):
            # The emit scores are for time i ("next_tag") so we broadcast along the current_tag axis.
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            # Transition scores are (current_tag, next_tag) so we broadcast along the instance axis.
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            # Alpha is for the current_tag, so we broadcast along the next_tag axis.
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            # TODO describe
            prob_scores = probability_tags[i].view(batch_size, 1, num_tags)

            # Add all the scores together and logexp over the current_tag axis
            inner = broadcast_alpha + emit_scores + prob_scores + transition_scores

            # In valid positions (mask == 1) we want to take the logsumexp over the current_tag dimension
            # of ``inner``. Otherwise (mask == 0) we want to retain the previous alpha.
            alpha = (util.logsumexp(inner, 1) * mask[i].view(batch_size, 1) +
                     alpha * (1 - mask[i]).view(batch_size, 1))

        # Every sequence needs to end with a transition to the stop_tag.
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return util.logsumexp(stops)