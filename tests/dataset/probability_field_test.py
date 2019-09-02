import unittest

import torch
import numpy as np
from allennlp.common.checks import ConfigurationError
from allennlp.data import Token, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer

from dpd.dataset.fields.probability_field import ProbabilisticLabelField
from dpd.common import TensorType, TensorList


class ProbabilisticLabelFieldTest(unittest.TestCase):
    def setUp(self):
        super(ProbabilisticLabelFieldTest, self).setUp()
        self.text = TextField([Token(t) for t in ["here", "are", "some", "words", "."]],
                              {"words": SingleIdTokenIndexer("words")})

    def test_simple_tensor_list_constructor(self):
        vocab = Vocabulary()
        tags = np.random.randn(len(self.text),3)
        prob_tags = TensorList([tags])
        sequence_label_field = ProbabilisticLabelField(prob_tags, self.text, label_namespace="*labels")
        padding_lengths = sequence_label_field.get_padding_lengths()
        tensor = sequence_label_field.as_tensor(padding_lengths).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(tensor, prob_tags.numpy())
        assert tensor.shape == prob_tags.numpy().shape