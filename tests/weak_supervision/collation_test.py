from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import unittest

from dpd.dataset import BIODataset, UnlabeledBIODataset
from dpd.weak_supervision.collator import Collator, UnionCollator, IntersectionCollator

def create_entry(input: List[str], output: List[str], in_id: int) -> Dict[str, List[str]]:
    assert len(input) == len(output)
    return {
        'input': input,
        'output': output,
        'id': in_id,
        'weight': 1.0,
    }

class CollatorTest(unittest.TestCase):
    @classmethod
    def sample_data(cls):
        return [
            [
                create_entry(
                    input=['This', 'is', 'a', 'sentence'],
                    output=['O', 'O', 'O', 'Tag'],
                    in_id=1,
                ),
                create_entry(
                    input=['This', 'is', 'a', 'word'],
                    output=['O', 'Tag', 'O', 'O'],
                    in_id=2,
                ),
            ],
            [
                create_entry(
                    input=['This', 'is', 'a', 'sentence'],
                    output=['O', 'Tag', 'O', 'Tag'],
                    in_id=1,
                ),
                create_entry(
                    input=['This', 'is', 'a', 'word'],
                    output=['O', 'O', 'O', 'O'],
                    in_id=2,
                ),
            ],
            [
                create_entry(
                    input=['This', 'is', 'a', 'sentence'],
                    output=['O', 'O', 'Tag', 'Tag'],
                    in_id=1,
                ),
                create_entry(
                    input=['This', 'is', 'a', 'word'],
                    output=['Tag', 'O', 'O', 'O'],
                    in_id=2,
                ),
            ],
        ]
    
    @classmethod
    def _test_collator(cls, collator_construct):
        sample_data = cls.sample_data()
        collator = collator_construct(positive_label='Tag')
        return collator.collate(sample_data, should_verify=True)

    def test_union(self):
        union_collation = CollatorTest._test_collator(collator_construct=UnionCollator)
        expected_result = [
            {
                'input': ['This', 'is', 'a', 'sentence'],
                'output': ['O', 'Tag', 'Tag', 'Tag'],
                'id': 1,
            },
            {
                'input': ['This', 'is', 'a', 'word'],
                'output': ['Tag', 'Tag', 'O', 'O'],
                'id': 2,
            },
        ]
        
        assert union_collation == expected_result
    
    def test_intersect(self):
        intersect_collation = CollatorTest._test_collator(collator_construct=IntersectionCollator)
        expected_result = [
            {
                'input': ['This', 'is', 'a', 'sentence'],
                'output': ['O', 'O', 'O', 'Tag'],
                'id': 1,
            },
            {
                'input': ['This', 'is', 'a', 'word'],
                'output': ['O', 'O', 'O', 'O'],
                'id': 2,
            },
        ]
        assert intersect_collation == expected_result
