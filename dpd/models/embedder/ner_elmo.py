from typing import List
import torch

from allennlp.common import Params
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.elmo import Elmo
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.data import Vocabulary

from .elmo import load_ner_elmo


class NERElmoTokenEmbedder(TokenEmbedder):
    """
    Compute a single layer of ELMo representations.
    This class serves as a convenience when you only want to use one layer of
    ELMo representations at the input of your network.  It's essentially a wrapper
    around Elmo(num_output_representations=1, ...) with the NER scalar mix parameters
    """
    def __init__(self) -> None:
        super(NERElmoTokenEmbedder, self).__init__()

        self._elmo = load_ner_elmo()

    def get_output_dim(self) -> int:
        return 1024

    def forward(self, # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                word_inputs: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``
            Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
        Returns
        -------
        The ELMo representations for the input sequence, shape
        ``(batch_size, timesteps, embedding_dim)``
        """
        with torch.no_grad():
            elmo_output = self._elmo(inputs)
            elmo_representations = elmo_output['elmo_representations'][0]
        return elmo_representations