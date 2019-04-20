from typing import (
    List,
    Dict,
    Tuple,
    Optional,
)
import unittest

import torch
import allennlp

from dpd.dataset import BIODataset, BIODatasetReader


class BIODatasetTest(unittest.TestCase):

    @classmethod
    def create_entry(cls, sentence: List[str], labels: List[str], entry_id: int, weight: float) -> Dict[str, object]:
        assert len(sentence) == len(labels)
        return {
            'id': entry_id,
            'input': sentence,
            'output': labels,
            'weight': weight,
        }
    
    @classmethod
    def create_fake_data(cls, binary_class: Optional[str] = None) -> BIODataset:
        data = [
            BIODatasetTest.create_entry(['single'], ['B-Tag'], 0, 1.0),
            BIODatasetTest.create_entry(['single', 'double'], ['B-Tag', 'I-Tag'], 1, 1.0),
            BIODatasetTest.create_entry(['single', 'double', 'triple'], ['B-TTag', 'I-TTag', 'O'], 2, 1.0),
            BIODatasetTest.create_entry(['no_label'], ['O'], 3, 1.0),
        ]

        dataset = BIODataset(0, 'fake_file.txt', binary_class)

        # hack around reading a file
        dataset.data = data

        return dataset
    
    def test_multiclass(self):
        reader = BIODatasetReader(
            bio_dataset=BIODatasetTest.create_fake_data(),
        )

        instances = reader.read('fake_file.txt')

        assert type(instances) == list

        expected_labels = [['B-Tag'], ['B-Tag', 'I-Tag']]

        fields = instances[0].fields
        tokens = [t.text for t in fields['sentence'].tokens]
        assert tokens == ['single']
        assert fields['labels'].labels == expected_labels[0]
        assert fields['weight'] == 1.0
        assert fields['entry_id'] == 0

        fields = instances[1].fields
        tokens = [t.text for t in fields['sentence'].tokens]
        assert tokens == ['single', 'double']
        assert fields['labels'].labels == expected_labels[1]
        assert fields['weight'] == 1.0
        assert fields['entry_id'] == 1