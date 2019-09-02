from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Any,
)

import unittest
import numpy as np
import logging

from dpd.dataset import BIODataset, UnlabeledBIODataset
from dpd.weak_supervision.collator import Collator, UnionCollator, IntersectionCollator, SnorkeMeTalCollator, SnorkelCollator
from dpd.weak_supervision.utils import ABSTAIN_LABEL

logger = logging.getLogger(name=__name__)

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
    def _verify_snorkel_result(cls, expected_data: Dict[str, Any], result: Dict[str, Any]):
        assert len(expected_data) == len(result)
        for expected_data_item, result_item in zip(expected_data, result):
            for key in expected_data_item:
                assert expected_data_item[key] == result_item[key]
            
            assert type(result_item['prob_labels']) == np.ndarray
            prob_labels_shape = result_item['prob_labels'].shape
            assert len(prob_labels_shape) == 2
            assert prob_labels_shape[1] == 2
    
    @classmethod
    def _test_collator(cls, collator_construct, extra_data: list = [], **kwargs):
        sample_data = cls.sample_data() + extra_data
        collator = collator_construct(positive_label='Tag', **kwargs)
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

    def test_snorkel_metal_collator(self):
        abstain_point = [
            create_entry(
                input=['This', 'is', 'a', 'sentence'],
                output=[ABSTAIN_LABEL, 'O', 'Tag', 'Tag'],
                in_id=1,
            ),
            create_entry(
                input=['This', 'is', 'a', 'word'],
                output=['Tag', ABSTAIN_LABEL, ABSTAIN_LABEL, 'O'],
                in_id=2,
            ),
        ]
        snorkel_collation = CollatorTest._test_collator(collator_construct=SnorkeMeTalCollator, extra_data=[abstain_point], seed=123)
        expected_result = [
            {
                'id': 1,
                'input': ['This', 'is', 'a', 'sentence'],
                'output': ['O', 'O', 'Tag', 'Tag']},
            {
                'id': 2,
                'input': ['This', 'is', 'a', 'word'],
                'output': ['Tag', 'O', 'O', 'O']
            }
        ]
        
        CollatorTest._verify_snorkel_result(expected_data=expected_result, result=snorkel_collation)
    
    def test_snorkel_collator(self):
        try:
            import snorkel
        except Exception:
            # snorkel module not located
            logger.warning(f'Not running Snorkel Collation Test because `snorkel` not installed properly')
            return

        abstain_point = [
            create_entry(
                input=['This', 'is', 'a', 'sentence'],
                output=[ABSTAIN_LABEL, 'O', 'Tag', 'Tag'],
                in_id=1,
            ),
            create_entry(
                input=['This', 'is', 'a', 'word'],
                output=['Tag', ABSTAIN_LABEL, ABSTAIN_LABEL, 'O'],
                in_id=2,
            ),
        ]
        snorkel_collation = CollatorTest._test_collator(collator_construct=SnorkelCollator, extra_data=[abstain_point], seed=123)
        expected_result = [
            {
                'id': 1,
                'input': ['This', 'is', 'a', 'sentence'],
                'output': ['O', 'Tag', 'Tag', 'Tag']},
            {
                'id': 2,
                'input': ['This', 'is', 'a', 'word'],
                'output': ['Tag', 'O', 'O', 'O']
            }
        ]
        assert snorkel_collation == expected_result
