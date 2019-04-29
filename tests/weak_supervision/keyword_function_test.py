from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import unittest
from collections import Counter

from dpd.dataset import BIODataset, UnlabeledBIODataset
from dpd.weak_supervision.dictionary_functions import KeywordMatchFunction
from dpd.weak_supervision import BIOConverter

class KeywordFunctionTest(unittest.TestCase):

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
            KeywordFunctionTest.create_entry(['single'], ['B-Tag'], 0, 1.0),
            KeywordFunctionTest.create_entry(['single', 'double'], ['B-Tag', 'I-Tag'], 1, 1.0),
            KeywordFunctionTest.create_entry(['single', 'double', 'triple'], ['B-TTag', 'I-TTag', 'O'], 2, 1.0),
            KeywordFunctionTest.create_entry(['no_label'], ['O'], 3, 1.0),
        ]

        dataset = BIODataset(0, 'fake_file.txt', binary_class)

        # hack around reading a file
        dataset.data = data

        return dataset
    
    def _verify_bio_scheme(self, predictions: List[str], class_tag: str) -> bool:
        for i, p_i in enumerate(predictions):
            if i == 0:
                if p_i != 'O' and p_i != f'B-{class_tag}':
                    return False
                continue

            if p_i == 'O':
                pass
            elif p_i == f'B-{class_tag}':
                pass
            elif p_i == f'I-{class_tag}':
                if predictions[i-1] != f'B-{class_tag}' and predictions[i - 1] != f'I-{class_tag}':
                    return False
            else:
                return False
        return True

    def test_keyword_func(self):
        dataset = KeywordFunctionTest.create_fake_data()
        unlabeled_corpus = UnlabeledBIODataset(bio_data=dataset, dataset_id=dataset.dataset_id)

        func = KeywordMatchFunction('Tag')
        func.train(train_data=dataset)
        annotated_corpus = func.evaluate(unlabeled_corpus=unlabeled_corpus)

        expected_counter = Counter()
        expected_counter['single'] = 2 
        expected_counter['double'] = 1

        expected_neg_counter = Counter()
        expected_neg_counter['triple'] = 1
        expected_neg_counter['no_label'] = 1 

        assert expected_counter == func.keywords['pos']
        assert expected_neg_counter == func.keywords['neg']

        converter = BIOConverter(binary_class='Tag')
        annotated_corpus = converter.convert(annotated_corpus)
        for ann_entry in annotated_corpus:
            assert self._verify_bio_scheme(ann_entry['output'], 'Tag')
