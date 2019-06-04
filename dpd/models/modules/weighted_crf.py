from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import torch
from torch.nn import functional as F

from overrides import overrides
import allennlp
from allennlp.modules import ConditionalRandomField
from allennlp.nn import util

def _mask_mult(inc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    '''
    inc: (batch, num_tags)
    mask: (batch)

    return (batch, num_tags)
    '''
    batch, num_tags = inc.shape
    for i in range(batch):
        val: int = mask[i]
        inc[i] = inc[i] * val
    
    return inc


class WeightedCRF(ConditionalRandomField):
    '''
    A weighted Conditional Random Field implementation

    It allows weighting at the input level by specifying a weight tensor of size (batch_size,)

    The equivalent would be multiplying the forward backward loss for each instance by the appropriate
    weight in the weight tensor
    '''

    def __init__(
        self,
        *args,
        use_soft_label_training: bool = False,
        **kwargs,
    ):
        self.use_soft_label_training = use_soft_label_training
        super(WeightedCRF, self).__init__(
            *args,
            **kwargs,
        )

    @overrides
    def forward(
        self,
        inputs: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.ByteTensor = None,
        # (batch,)
        weight: torch.Tensor = None,
        # (batch, seq_len, num_tags)
        prob_labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Computes the log likelihood.
        """
        # pylint: disable=arguments-differ
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.long)

        log_denominator = self._input_likelihood(inputs, mask)

        if prob_labels is None or not self.use_soft_label_training:
            log_numerator = self._joint_likelihood(inputs, tags, mask)
        else:
            log_numerator = self._soft_joint_likelihood(inputs, prob_labels, mask)

        batch_loss = log_numerator - log_denominator

        if weight is not None:
            batch_loss = weight * batch_loss
        return torch.sum(batch_loss)

    def _soft_likelihood(
        self,
        logits: torch.Tensor,
        probability_tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        """
        batch_size, sequence_length, num_tags = logits.size()

        # Transpose batch size and sequence dimensions
        mask = mask.float().transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()
        probability_tags = probability_tags.transpose(0, 1).contiguous()
        # probability_tags = torch.log(F.softmax(probability_tags.transpose(0, 1).contiguous(), dim=1))

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
            # if i < sequence_length - 1:
            #     curr_prob = probability_tags[i].view(batch_size, 1, num_tags)
            #     next_prob = probability_tags[i + 1].view(batch_size, 1, num_tags)

            #     # shape: (batch_size, num_tags, num_tags)
            #     prob_scores = curr_prob.transpose(1,2) * next_prob

            #     # Add all the scores together and logexp over the current_tag axis
            #     inner = broadcast_alpha + emit_scores * curr_prob + transition_scores * prob_scores
            # else:
            #     curr_prob = probability_tags[i].view(batch_size, 1, num_tags)
            #     if self.include_start_end_transitions:
            #         prob_scores = curr_prob.transpose(1,2) * self.end_transitions
            #         prob_scores = torch.Tensor((1, num_tags, num_tags)).fill_(1.)
            #     else:
            #         prob_scores = torch.Tensor((1, num_tags, num_tags)).fill_(1.)
                
            #     print('here', prob_scores.shape, transition_scores.shape)
            #     print('next', (transition_scores * prob_scores).shape)

            #     # Add all the scores together and logexp over the current_tag axis
            #     inner = broadcast_alpha + emit_scores * curr_prob + transition_scores * prob_scores
        
            curr_prob = probability_tags[i].view(batch_size, 1, num_tags)
            prev_prob = probability_tags[i - 1].view(batch_size, 1, num_tags)

            # shape: (batch_size, num_tags, num_tags)
            prob_scores = prev_prob.transpose(1,2) * curr_prob

            # Add all the scores together and logexp over the current_tag axis
            inner = broadcast_alpha + emit_scores * curr_prob + transition_scores * prob_scores

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

    def _soft_joint_likelihood(
        self,
        logits: torch.Tensor,
        probability_tags: torch.Tensor,
        mask: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
        """
        batch_size, sequence_length, num_tags = logits.data.shape

        # Transpose batch size and sequence dimensions:
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        probability_tags = probability_tags.transpose(0, 1).contiguous()

        # Start with the transition scores from start_tag to the first tag in each input
        if self.include_start_end_transitions:
            score = self.start_transitions * probability_tags[0]
        else:
            score = torch.zeros(self.start_transitions.shape)


        # score is distribution over current states

        # Add up the scores for the observed transitions and all the inputs but the last
        for i in range(sequence_length - 1):
            # Each is shape (batch_size,)
            current_tag, next_tag = probability_tags[i].unsqueeze(1), probability_tags[i+1].unsqueeze(1)

            # The scores for transitioning from current_tag to next_tag
            # print(current_tag.shape, self.transitions.shape, next_tag.shape, (current_tag * self.transitions).shape, (current_tag @ self.transitions).shape)
            transition_score = ((current_tag @ self.transitions) * next_tag).squeeze(1)

            # The score for using current_tag
            emit_score = (logits[i].unsqueeze(1) * current_tag).squeeze(1)
            # print('emit', emit_score.shape, (logits[i].unsqueeze(1) * current_tag).shape, logits[i].unsqueeze(1).shape, current_tag.shape )

            # Include transition score if next element is unmasked,
            # input_score if this element is unmasked.
            score = score + _mask_mult(emit_score, mask[i]) + _mask_mult(transition_score, mask[i + 1])
        # return score.sum(dim=1)
        # Transition from last state to "stop" state. To start with, we need to find the last tag
        # for each instance.
        last_tag_index = mask.sum(0).long() - 1
        last_tags = probability_tags.gather(0, last_tag_index.view(1, batch_size, 1).repeat(1, 1, 5)).squeeze(0)
        # (2, 5)

        # Compute score of transitioning to `stop_tag` from each "last tag".
        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions * last_tags
        else:
            last_transition_score = torch.zeros(num_tags)

        # Add the last input if it's not masked.
        last_inputs = logits[-1]                                         # (batch_size, num_tags)
        last_input_score = last_inputs * last_tags  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()                    # (batch_size,)

        # print(last_inputs.shape, last_input_score.shape, last_tags.shape, last_transition_score.shape, mask[-1].shape)
        score = score + last_transition_score + _mask_mult(last_input_score, mask[-1])

        return score.sum(dim=1)