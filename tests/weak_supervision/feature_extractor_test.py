from typing import (
    List,
    Tuple,
    Dict,
    Optional,
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
from dpd.weak_supervision.feature_extractor import SpaCyFeatureExtractor, WordFeatureExtractor, GloVeFeatureExtractor, POSFeatureExtractor
from dpd.constants import SPACY_NLP, GLOVE_DIR, SPACY_POS

GLOVE_ENABLED = os.path.exists(GLOVE_DIR)

class FeatureExtractorTest(unittest.TestCase):
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
    
    def test_spacy_extractor(self):
        dataset = FeatureExtractorTest.create_fake_data()
        dataset_reader = BIODatasetReader(dataset)
        instances = dataset_reader.read('fake.txt')
        vocab = Vocabulary.from_instances(instances)
        feature_extractor = SpaCyFeatureExtractor()
        feature_extractor.cache(
            dataset_id=0,
            dataset=instances,
            vocab=vocab,
        )

        features = feature_extractor.get_features(dataset_id=0, sentence_id=1)
        computed_features = SPACY_NLP(' '.join([w for w in dataset.data[1]['input']]))
        assert len(features) == len(dataset.data[1]['input'])
        assert len(features) == len(computed_features)
        for i, (f, c) in enumerate(zip(features, computed_features)):
            assert FeatureExtractorTest.features_eq(f, c)
    
    def test_word_feature_extractor(self):
        dataset = FeatureExtractorTest.create_fake_data()
        dataset_reader = BIODatasetReader(dataset)
        instances = dataset_reader.read('fake.txt')
        vocab = Vocabulary.from_instances(instances)
        feature_extractor = WordFeatureExtractor(vocab=vocab)
        for entry in dataset.data:
            sentence = entry['input']
            feats = feature_extractor.get_features(sentence_id=None, dataset_id=None, sentence=sentence)
            for word, feat in zip(sentence, feats):
                assert feat.shape == (1, vocab.get_vocab_size())
                word_i = vocab.get_token_index(word)
                assert feat.sum() == 1
                assert feat.argmax() == word_i
    
    def test_glove_feature_extractor(self):
        if not GLOVE_ENABLED:
            return
        dataset = FeatureExtractorTest.create_fake_data()
        dataset_reader = BIODatasetReader(dataset)
        instances = dataset_reader.read('fake.txt')
        vocab = Vocabulary.from_instances(instances)
        feature_extractor = GloVeFeatureExtractor()
        for entry in dataset.data:
            sentence = entry['input']
            feats = feature_extractor.get_features(sentence_id=None, dataset_id=None, sentence=sentence)
            for word, feat in zip(sentence, feats):
                assert feat.shape == (1, 300)
    
    def test_pos_feature_extractor(self):
        dataset = FeatureExtractorTest.create_fake_data()
        dataset_reader = BIODatasetReader(dataset)
        instances = dataset_reader.read('fake.txt')
        vocab = Vocabulary.from_instances(instances)
        spacy_features = SpaCyFeatureExtractor()
        spacy_features.cache(
            dataset_id=0,
            dataset=instances,
            vocab=vocab,
        )
        feature_extractor = POSFeatureExtractor(spacy_module=spacy_features)
        for entry in dataset.data:
            sentence = entry['input']
            feats = feature_extractor.get_features(sentence_id=None, dataset_id=None, sentence=sentence)
            for word, feat in zip(sentence, feats):
                assert feat.shape == (1, len(SPACY_POS))
