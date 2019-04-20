import unittest

import torch
import numpy as np

from allennlp.data import Vocabulary
from dpd.training.metrics import TagF1

class TestTagF1(unittest.TestCase):

    @classmethod
    def _create_vocab(cls) -> Vocabulary:
        vocab = Vocabulary()
        vocab.add_token_to_namespace("O", "labels")
        vocab.add_token_to_namespace("B-Tag", "labels")
        vocab.add_token_to_namespace("I-Tag", "labels")
        return vocab
    
    def test_project_class_tensor(self):
        vocab = TestTagF1._create_vocab()
        class_labels = ['B-Tag', 'I-Tag']

        metric = TagF1(
            vocab=vocab,
            class_labels=class_labels,
        )

        gold_indicies = [
            # O B-ADR I-ADR O
            [0, 1, 2, 0],
            # O O O B-ADR
            [0, 0, 0, 1],
        ]

        gold_tensor = torch.Tensor(gold_indicies)

        expected_result = [
            # O ADR ADR O
            [0, 1, 1, 0],
            # O O O ADR
            [0, 0, 0, 1],
        ]

        expected_tensor = torch.Tensor(expected_result)

        projected = metric.project_class_tensor(gold_tensor)

        assert (projected == expected_tensor).all()

    def test_tag_f1(self):
        '''
        A simple test to ensure tag f1 is operating as we expect it to
        '''
        vocab = TestTagF1._create_vocab()
        class_labels = ['B-Tag', 'I-Tag']

        metric = TagF1(
            vocab=vocab,
            class_labels=class_labels,
        )

        gold_indicies = [
            # O B-ADR I-ADR O
            [0, 1, 2, 0],
            # O O O B-ADR
            [0, 0, 0, 1],
        ]

        gold_tensor = torch.Tensor(gold_indicies)

        predictions_tensor = torch.rand([gold_tensor.shape[0], gold_tensor.shape[1], vocab.get_vocab_size('labels')])

        '''
        Predictions should follow:
            O B-ADR B-ADR O
            O   O   O   O
        '''
        expected_predictions = [
            # True Negative # True Positive # True Positive # False Positive
            ['O', 'B-Tag', 'B-Tag', 'I-Tag'],
            # True Negative # True Negative # True Negative # False Negative
            ['O', 'O', 'O', 'O'],

        ]

        # since prediction tensor sets all the values to be between 0 and 1
        # this loop sets what we want to have the highest values (1.0)
        for i, pred_seq in enumerate(expected_predictions):
            for j, pred_label in enumerate(pred_seq):
                predictions_tensor[i, j, vocab.get_token_index(pred_label, namespace='labels')] = 1.
        
        metric(predictions_tensor, gold_tensor)
        precision, recall, f1 = metric.get_metric()

        assert metric._true_positives == 2.0
        assert metric._true_negatives == 4.0
        assert metric._false_positives == 1.0
        assert metric._false_negatives == 1.0

        np.testing.assert_almost_equal(precision, 0.6666666666666666)
        np.testing.assert_almost_equal(recall, 0.666666666)
        np.testing.assert_almost_equal(f1, 0.6666666666666167)
        







