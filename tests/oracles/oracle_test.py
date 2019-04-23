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
from dpd.oracles import GoldOracle

class GoldOracleTest(unittest.TestCase):
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
            GoldOracleTest.create_entry(['single'], ['B-Tag'], 0, 1.0),
            GoldOracleTest.create_entry(['single', 'double'], ['B-Tag', 'I-Tag'], 1, 1.0),
            GoldOracleTest.create_entry(['single', 'double', 'triple'], ['B-TTag', 'I-TTag', 'O'], 2, 1.0),
            GoldOracleTest.create_entry(['no_label'], ['O'], 3, 1.0),
        ]

        dataset = BIODataset(0, 'fake_file.txt', binary_class)

        # hack around reading a file
        dataset.data = data

        return dataset

    def test_gold_oracle(self):
        bio_data = GoldOracleTest.create_fake_data()
        unlabeled_corpus = UnlabeledBIODataset(dataset_id=0, bio_data=bio_data)

        oracle = GoldOracle(bio_data)
        
        for item in bio_data:
            s_id, s_input, s_output = item['id'], item['input'], item['output']

            oracle_out = oracle.get_query(query=(s_id, s_input))
            o_input = oracle_out['input']
            o_output = oracle_out['output']
            o_id = oracle_out['id']
            o_weight = oracle_out['weight']

            assert o_id == s_id
            assert s_input == o_input
            assert o_output == s_output
            assert o_weight == 1.0
