import unittest

import torch
import numpy as np

from allennlp.data import Vocabulary
from allennlp.training.metrics import SpanBasedF1Measure
from dpd.training.metrics import AverageTagF1

class TestTagF1(unittest.TestCase):

    @classmethod
    def _create_vocab(cls) -> Vocabulary:
        vocab = Vocabulary()
        vocab.add_token_to_namespace("O", "labels")
        vocab.add_token_to_namespace("B-Tag", "labels")
        vocab.add_token_to_namespace("I-Tag", "labels")
        return vocab

    def test_tag_f1(self):
        '''
        A simple test to ensure tag f1 is operating as we expect it to
        '''
        vocab = TestTagF1._create_vocab()
        class_labels = ['B-Tag', 'I-Tag']

        metric = AverageTagF1(
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
            ['O', 'B-Tag', 'I-Tag', 'I-Tag'],
            # True Negative # True Negative # True Negative # False Negative
            ['O', 'O', 'O', 'O'],
        ]

        '''
        expected result

        B-ADR
        
        TN TP TN TN
        TN TN TN FN

        p: 1/(1 + 0) = 1 r: 1/(1 + 1) = 0.5 f1: 0.25

        I-ADR

        TN TN TP FP
        TN TN TN TN

        p: 1/(1 + 1) = 0.5 r: 1/(1 + 0) = 1 f1: 0

        '''

        # since prediction tensor sets all the values to be between 0 and 1
        # this loop sets what we want to have the highest values (1.0)
        for i, pred_seq in enumerate(expected_predictions):
            for j, pred_label in enumerate(pred_seq):
                predictions_tensor[i, j, vocab.get_token_index(pred_label, namespace='labels')] = 1.
        
        metric(predictions_tensor, gold_tensor)
        metric_dict = metric.get_metric()
        precision, recall, f1 = metric_dict['Aprecision'], metric_dict['Arecall'], metric_dict['Af1']

        print(precision, recall, f1)

        np.testing.assert_almost_equal(precision, 0.75)
        np.testing.assert_almost_equal(recall, 0.75)
        np.testing.assert_almost_equal(f1, 0.6666666666666167)
