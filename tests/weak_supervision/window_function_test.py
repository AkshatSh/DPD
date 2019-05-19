from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Any,
)

import os
import sys
import unittest

import torch
import allennlp
import spacy
import nltk

from allennlp.data import Vocabulary

from dpd.utils import TensorList
from dpd.dataset import BIODataset, BIODatasetReader
from dpd.weak_supervision.feature_extractor import WordFeatureExtractor, FeatureCollator, FeatureExtractor, GloVeFeatureExtractor
from dpd.weak_supervision.context_window_functions import BagWindowFunction, WindowFunction, LinearWindowFunction
from dpd.constants import GLOVE_DIR
from dpd.weak_supervision.collator import SnorkeMeTalCollator

GLOVE_ENABLED = os.path.exists(GLOVE_DIR)

class WindowFunctionTest(unittest.TestCase):
    COMPARISON_FEATURES = ['pos_', 'lemma_', 'text', 'tag_', 'dep_']

    @classmethod
    def features_eq(cls, f1, f2) -> bool:
        assert len(f1) == len(f2)
        for attr in cls.COMPARISON_FEATURES:
            f1_a = getattr(f1, attr, None)
            f2_a = getattr(f2, attr, None)
            if f1_a != f2_a:
                return False
        return True

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
            cls.create_entry(['single'], ['B-Tag'], 0, 1.0),
            cls.create_entry(['single', 'double'], ['B-Tag', 'I-Tag'], 1, 1.0),
            cls.create_entry(['single', 'double', 'triple'], ['B-Tag', 'I-Tag', 'O'], 2, 1.0),
            cls.create_entry(['no_label'], ['O'], 3, 1.0),
        ]

        dataset = BIODataset(0, 'fake_file.txt', binary_class)

        # hack around reading a file
        dataset.data = data

        return dataset
    
    def _test_word_feature(
        self,
        feature_summarizer: FeatureCollator,
        window_function: WindowFunction = BagWindowFunction,
        feature_extractor_obj: FeatureExtractor = WordFeatureExtractor,
        context_window: int = 2,
        threshold: Optional[float] =None,
    ):
        dataset = WindowFunctionTest.create_fake_data()
        dataset_reader = BIODatasetReader(dataset)
        instances = dataset_reader.read('fake.txt')
        vocab = Vocabulary.from_instances(instances)
        feature_extractor = feature_extractor_obj(vocab=vocab)
        batch_func = window_function(
            positive_label='Tag',
            context_window=context_window,
            feature_extractor=feature_extractor,
            feature_summarizer=feature_summarizer,
            use_batch=True,
            threshold=threshold,
        )

        sparse_batch_func = None
        if window_function == BagWindowFunction:
            sparse_batch_func = window_function(
                positive_label='Tag',
                context_window=context_window,
                feature_extractor=feature_extractor,
                feature_summarizer=feature_summarizer,
                use_batch=True,
                use_sparse=True,
                threshold=threshold,
            )

        single_func = window_function(
            positive_label='Tag',
            context_window=context_window,
            feature_extractor=feature_extractor,
            feature_summarizer=feature_summarizer,
            use_batch=False,
            threshold=threshold,
        )

        batch_func.train(dataset.data)
        single_func.train(dataset.data)
        if sparse_batch_func:
            sparse_eval = sparse_batch_func.train(dataset.data)

        assert batch_func.dictionary.shape[0] == batch_func.labels.shape[0]
        
        batch_eval = batch_func.evaluate(dataset)
        single_eval = single_func.evaluate(dataset)
        if sparse_batch_func:
            sparse_eval = sparse_batch_func.evaluate(dataset)
            assert batch_eval == sparse_eval

        assert batch_eval == single_eval

        return batch_eval
    
    def test_word_feature_context_window_concat(self):
        expected_result = [
            {'id': 0, 'input': ['single'], 'output': ['Tag']},
            {'id': 1, 'input': ['single', 'double'], 'output': ['Tag', 'Tag']},
            {'id': 2, 'input': ['single', 'double', 'triple'], 'output': ['Tag', 'Tag', 'O']},
            {'id': 3, 'input': ['no_label'], 'output': ['O']},
        ]

        annotations = self._test_word_feature(
            feature_summarizer=FeatureCollator.concat,
        )

        assert annotations == expected_result

    def test_word_feature_context_window_sum(self):
        expected_result = [
            {'id': 0, 'input': ['single'], 'output': ['Tag']},
            {'id': 1, 'input': ['single', 'double'], 'output': ['Tag', 'Tag']},
            {'id': 2, 'input': ['single', 'double', 'triple'], 'output': ['Tag', 'Tag', 'Tag']},
            {'id': 3, 'input': ['no_label'], 'output': ['O']},
        ]

        annotations = self._test_word_feature(
            feature_summarizer=FeatureCollator.sum,
        )
        assert annotations == expected_result
    
    def test_linear_feature_context_window_sum(self):
        if not GLOVE_ENABLED:
            return
        expected_result = [
            {'id': 0, 'input': ['single'], 'output': ['Tag']},
            {'id': 1, 'input': ['single', 'double'], 'output': ['Tag', 'Tag']},
            {'id': 2, 'input': ['single', 'double', 'triple'], 'output': ['Tag', 'Tag', 'Tag']},
            {'id': 3, 'input': ['no_label'], 'output': ['O']},
        ]

        annotations = self._test_word_feature(
            feature_summarizer=FeatureCollator.sum,
            window_function=LinearWindowFunction,
            feature_extractor_obj=GloVeFeatureExtractor,
        )

        assert annotations == expected_result

    def test_linear_feature_context_window_concat(self):
        if not GLOVE_ENABLED:
            return
        expected_result = [
            {'id': 0, 'input': ['single'], 'output': ['Tag']},
            {'id': 1, 'input': ['single', 'double'], 'output': ['Tag', 'Tag']},
            {'id': 2, 'input': ['single', 'double', 'triple'], 'output': ['Tag', 'Tag', 'O']},
            {'id': 3, 'input': ['no_label'], 'output': ['O']},
        ]

        annotations = self._test_word_feature(
            feature_summarizer=FeatureCollator.concat,
            window_function=LinearWindowFunction,
            feature_extractor_obj=GloVeFeatureExtractor,
        )

        assert annotations == expected_result

    def test_linear_feature_context_window_sum_zero_threshold(self):
        if not GLOVE_ENABLED:
            return
        expected_result = [
            {'id': 0, 'input': ['single'], 'output': ['Tag']},
            {'id': 1, 'input': ['single', 'double'], 'output': ['Tag', 'Tag']},
            {'id': 2, 'input': ['single', 'double', 'triple'], 'output': ['Tag', 'Tag', 'Tag']},
            {'id': 3, 'input': ['no_label'], 'output': ['O']},
        ]

        annotations = self._test_word_feature(
            feature_summarizer=FeatureCollator.sum,
            window_function=LinearWindowFunction,
            feature_extractor_obj=GloVeFeatureExtractor,
            threshold=0.,
        )

        assert annotations == expected_result
    
    def test_linear_feature_context_window_sum_high_threshold(self):
        if not GLOVE_ENABLED:
            return
        expected_result = [
            {'id': 0, 'input': ['single'], 'output': ['Tag']},
            {'id': 1, 'input': ['single', 'double'], 'output': ['Tag', 'Tag']},
            {'id': 2, 'input': ['single', 'double', 'triple'], 'output': ['Tag', 'Tag', 'Tag']},
            {'id': 3, 'input': ['no_label'], 'output': ['<ABS>']},
        ]

        annotations = self._test_word_feature(
            feature_summarizer=FeatureCollator.sum,
            window_function=LinearWindowFunction,
            feature_extractor_obj=GloVeFeatureExtractor,
            threshold=0.7,
        )

        assert annotations == expected_result
    
    def test_word_feature_context_window_concat_bag(self):
        expected_result = [
            {'id': 0, 'input': ['single'], 'output': ['Tag']},
            {'id': 1, 'input': ['single', 'double'], 'output': ['Tag', 'Tag']},
            {'id': 2, 'input': ['single', 'double', 'triple'], 'output': ['Tag', 'Tag', 'O']},
            {'id': 3, 'input': ['no_label'], 'output': ['O']},
        ]

        annotations = self._test_word_feature(
            feature_summarizer=FeatureCollator.concat,
            window_function=LinearWindowFunction,
        )

        collator = SnorkeMeTalCollator('Tag')

        res = collator.collate(
            [annotations],
        )

        assert annotations == expected_result 