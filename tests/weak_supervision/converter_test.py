from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Any,
)

import unittest
import numpy as np
from numpy import array

from allennlp.data import Vocabulary

from dpd.dataset import BIODataset, BIODatasetReader
from dpd.constants import STOP_WORDS
from dpd.weak_supervision import BIOConverter

def create_entry(input: List[str], output: List[str], in_id: int) -> Dict[str, List[str]]:
    assert len(input) == len(output)
    return {
        'input': input,
        'output': output,
        'id': in_id,
        'weight': 1.0,
    }

def compare_lists(
    expected: List[Any],
    actual: List[Any],
) -> bool:
    if len(expected) != len(actual):
        return False
    for e_item, a_item in zip(expected, actual):
        if not compare_items(e_item, a_item):
            return False
    return True

def compare_dictionaries(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
) -> bool:
    if expected.keys() != actual.keys():
        return False
    for key in expected.keys():
        e_val = expected[key]
        a_val = actual[key]
        if not compare_items(e_val, a_val):
            return False
    return True

def compare_items(
    expected: Any,
    actual: Any,
) -> bool:
    if type(expected) != type(actual):
        return False

    if type(expected) == list:
        return compare_lists(expected, actual)
    elif type(expected) == dict:
        return compare_dictionaries(expected, actual)
    elif type(expected) == np.ndarray:
        return (expected==actual).all()
    else:
        return expected == actual

class ConverterTest(unittest.TestCase):
    @classmethod
    def create_label_vocab(cls) -> Vocabulary:
        data = [
            create_entry(['A', 'A', 'A', 'A'], ['O', 'B-ADR', 'I-ADR', 'O'], 0),
        ]

        dataset = BIODataset(0, 'fake_file.txt', None)

        # hack around reading a file
        dataset.data = data

        train_reader = BIODatasetReader(
            bio_dataset=dataset,
        )

        train_data = train_reader.read('temp.txt')
        vocab = Vocabulary.from_instances(train_data)

        return vocab
    @classmethod
    def create_fake_data(cls, binary_class: Optional[str] = None) -> BIODataset:
        data = [
            create_entry(['this', 'is', 'an', 'reaction'], ['O', 'O', 'O', 'ADR'], 0),
            create_entry(['this', 'is', 'an', 'reaction', 'an', 'reaction'], ['O', 'O', 'O', 'O', 'O', 'ADR'], 1)
        ]

        dataset = BIODataset(0, 'fake_file.txt', binary_class)

        # hack around reading a file
        dataset.data = data

        train_reader = BIODatasetReader(
            bio_dataset=dataset,
        )

        train_data = train_reader.read('temp.txt')
        vocab = Vocabulary.from_instances(train_data)

        return dataset, vocab

    def test_stop_words(self):
        _, vocab = ConverterTest.create_fake_data()
        sentence = ['this', 'is', 'an', 'reaction']
        tags = ['O', 'O', 'O', 'ADR']
        conv = BIOConverter('ADR', vocab=vocab)
        new_tags = conv.stop_word_heuristic(sentence, tags, 'ADR')
        assert new_tags == ['ADR', 'ADR', 'ADR', 'ADR']

    def test_stop_words_end(self):
        _, vocab = ConverterTest.create_fake_data()
        sentence = ['this', 'is', 'an', 'reaction', 'an', 'reaction']
        tags = ['O', 'O', 'O', 'O', 'O', 'ADR']
        conv = BIOConverter('ADR', vocab=vocab)
        new_tags = conv.stop_word_heuristic(sentence, tags, 'ADR')
        assert new_tags == ['O', 'O', 'O', 'O', 'ADR', 'ADR']

    def test_no_stop_words(self):
        _, vocab = ConverterTest.create_fake_data()
        sentence = ['this', 'is', 'an', 'reaction', 'an', 'reaction']
        tags = ['O', 'O', 'O', 'O', 'O', 'O']
        conv = BIOConverter('ADR', vocab=vocab)
        new_tags = conv.stop_word_heuristic(sentence, tags, 'ADR')
        assert new_tags == ['O', 'O', 'O', 'O', 'O', 'O']

    def test_post_stop_word(self):
        _, vocab = ConverterTest.create_fake_data()
        sentence = ['this', 'is', 'temp', 'reaction', 'an', 'reaction']
        tags = ['O', 'O', 'O', 'ADR', 'O', 'O']
        conv = BIOConverter('ADR', vocab=vocab)
        new_tags = conv.stop_word_heuristic(sentence, tags, 'ADR')
        assert new_tags == ['O', 'O', 'O', 'ADR', 'ADR', 'O']
    
    def test_conversion(self):
        dataset, _ = ConverterTest.create_fake_data()
        vocab = ConverterTest.create_label_vocab()
        converter = BIOConverter('ADR', vocab=vocab)
        bio = converter.convert(dataset)
        res = all([compare_items(e, a) for (e, a) in zip(
            bio, 
            [
                {
                    'input': ['this', 'is', 'an', 'reaction'],
                    'output': ['O', 'O', 'O', 'B-ADR'],
                    'id': 0,
                    'weight': 1.0,
                    'prob_labels': [array([[1., 0., 0.]]), array([[1., 0., 0.]]), array([[1., 0., 0.]]), array([[0., 1., 0.]])],
                },
                {
                    'input': ['this', 'is', 'an', 'reaction', 'an', 'reaction'],
                    'output': ['O', 'O', 'O', 'O', 'O', 'B-ADR'],
                    'id': 1,
                    'weight': 1.0,
                    'prob_labels': [array([[1., 0., 0.]]), array([[1., 0., 0.]]), array([[1., 0., 0.]]), array([[1., 0., 0.]]), array([[1., 0., 0.]]), array([[0., 1., 0.]])],
                }
            ]
        )])

        assert res