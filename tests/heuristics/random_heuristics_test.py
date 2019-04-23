from typing import (
    List,
    Dict,
    Tuple,
    Optional,
)
import unittest

import torch
import allennlp

from dpd.dataset import UnlabeledBIODataset, BIODataset
from dpd.heuristics import RandomHeuristic


class RandomHeuristicTests(unittest.TestCase):
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
            RandomHeuristicTests.create_entry(['single'], ['B-Tag'], 0, 1.0),
            RandomHeuristicTests.create_entry(['single', 'double'], ['B-Tag', 'I-Tag'], 1, 1.0),
            RandomHeuristicTests.create_entry(['single', 'double', 'triple'], ['B-TTag', 'I-TTag', 'O'], 2, 1.0),
            RandomHeuristicTests.create_entry(['no_label'], ['O'], 3, 1.0),
        ]

        dataset = BIODataset(0, 'fake_file.txt', binary_class)

        # hack around reading a file
        dataset.data = data

        return dataset
    
    def test_unlabeled(self):
        bio_data = RandomHeuristicTests.create_fake_data()
        unlabeled_corpus = UnlabeledBIODataset(dataset_id=0, bio_data=bio_data)

        h = RandomHeuristic()
        h_res = h.evaluate(unlabeled_corpus)

        assert type(h_res) == torch.Tensor
        assert h_res.shape == (len(unlabeled_corpus),)