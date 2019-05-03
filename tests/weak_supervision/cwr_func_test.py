from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Callable,
)

import os
import sys

import unittest

from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules import TextFieldEmbedder, TokenEmbedder
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.data.token_indexers import TokenIndexer

from dpd.dataset import BIODataset, BIODatasetReader
from dpd.models.embedder import NERElmoTokenEmbedder, CachedTextFieldEmbedder
from dpd.weak_supervision.contextual_functions import CWRLinear

SHOULD_RUN = False

class CWRFuncTest(unittest.TestCase):
    @classmethod
    def get_embedder_info(cls) -> Tuple[TokenEmbedder, TokenIndexer]:
        return NERElmoTokenEmbedder(), ELMoTokenCharactersIndexer()

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

    @classmethod
    def setup_embedder(cls, cache: bool = True) -> CachedTextFieldEmbedder:
        token_embedder, token_indexer = CWRFuncTest.get_embedder_info()
        train_bio = CWRFuncTest.create_fake_data('Tag')
        train_reader = BIODatasetReader(
            bio_dataset=train_bio,
            token_indexers={
                'tokens': token_indexer,
            },
        )

        train_data = train_reader.read('temp.txt')
        vocab = Vocabulary.from_instances(train_data)
        text_field_embedder = BasicTextFieldEmbedder({"tokens": token_embedder})
        cached_embedder = CachedTextFieldEmbedder(
            text_field_embedder=text_field_embedder,
        )

        cached_embedder.cache(
            dataset_id=train_bio.dataset_id,
            dataset=train_data,
            vocab=vocab,
        )

        return cached_embedder

    @classmethod
    def _exec_test(cls, test: callable):
        if SHOULD_RUN:
            test()

    def test_cwr(self):
        def _test():
            dataset = CWRFuncTest.create_fake_data('Tag')
            embedder = CWRFuncTest.setup_embedder()
            cwr_linear = CWRLinear(
                positive_label='Tag',
                embedder=embedder,
                linear_type='svm_linear',
            )
            cwr_linear.train(dataset, dataset_id=dataset.dataset_id)
            annotations = cwr_linear.evaluate(dataset)
        CWRFuncTest._exec_test(_test)
