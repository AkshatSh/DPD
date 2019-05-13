from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import unittest
import numpy as np

from dpd.dataset import BIODataset
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

class ConverterTest(unittest.TestCase):
    @classmethod
    def create_fake_data(cls, binary_class: Optional[str] = None) -> BIODataset:
        data = [
            create_entry(['this', 'is', 'an', 'reaction'], ['O', 'O', 'O', 'ADR'], 0),
            create_entry(['this', 'is', 'an', 'reaction', 'an', 'reaction'], ['O', 'O', 'O', 'O', 'O', 'ADR'], 1)
        ]

        dataset = BIODataset(0, 'fake_file.txt', binary_class)

        # hack around reading a file
        dataset.data = data

        return dataset

    def test_stop_words(self):
        sentence = ['this', 'is', 'an', 'reaction']
        tags = ['O', 'O', 'O', 'ADR']
        conv = BIOConverter('ADR')
        new_tags = conv.stop_word_heuristic(sentence, tags, 'ADR')
        assert new_tags == ['ADR', 'ADR', 'ADR', 'ADR']

    def test_stop_words_end(self):
        sentence = ['this', 'is', 'an', 'reaction', 'an', 'reaction']
        tags = ['O', 'O', 'O', 'O', 'O', 'ADR']
        conv = BIOConverter('ADR')
        new_tags = conv.stop_word_heuristic(sentence, tags, 'ADR')
        assert new_tags == ['O', 'O', 'O', 'O', 'ADR', 'ADR']

    def test_no_stop_words(self):
        sentence = ['this', 'is', 'an', 'reaction', 'an', 'reaction']
        tags = ['O', 'O', 'O', 'O', 'O', 'O']
        conv = BIOConverter('ADR')
        new_tags = conv.stop_word_heuristic(sentence, tags, 'ADR')
        assert new_tags == ['O', 'O', 'O', 'O', 'O', 'O']

    def test_post_stop_word(self):
        sentence = ['this', 'is', 'temp', 'reaction', 'an', 'reaction']
        tags = ['O', 'O', 'O', 'ADR', 'O', 'O']
        conv = BIOConverter('ADR')
        new_tags = conv.stop_word_heuristic(sentence, tags, 'ADR')
        assert new_tags == ['O', 'O', 'O', 'ADR', 'ADR', 'O']
    
    def test_conversion(self):
        dataset = ConverterTest.create_fake_data()
        converter = BIOConverter('ADR')
        bio = converter.convert(dataset)
        assert bio == [
            {
                'input': ['this', 'is', 'an', 'reaction'],
                'output': ['B-ADR', 'I-ADR', 'I-ADR', 'I-ADR'],
                'id': 0,
                'weight': 1.0,
            },
            {
                'input': ['this', 'is', 'an', 'reaction', 'an', 'reaction'],
                'output': ['O', 'O', 'O', 'O', 'B-ADR', 'I-ADR'],
                'id': 1,
                'weight': 1.0,
            },
        ]