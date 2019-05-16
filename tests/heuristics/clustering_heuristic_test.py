from typing import (
    List,
    Dict,
    Tuple,
    Optional,
)
import unittest

import torch
import allennlp
from allennlp.data import Vocabulary
from allennlp.data.token_indexers import TokenIndexer
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules import Embedding
import numpy as np

from dpd.dataset import UnlabeledBIODataset, BIODataset, BIODatasetReader
from dpd.heuristics import ClusteringHeuristic
from dpd.models.embedder import CachedTextFieldEmbedder

class ClusteringHeuristicTest(unittest.TestCase):
    EMBEDDING_DIM = 5
    SAMPLE_SIZE = 2

    @classmethod
    def build_cache_cwr(cls):
        bio_dataset = cls.create_fake_data()
        reader = BIODatasetReader(
            bio_dataset=bio_dataset,
        )

        instances = reader.read('fake_file.txt')
        vocab = Vocabulary.from_instances(instances)
        token_embedding = Embedding(
            num_embeddings=vocab.get_vocab_size('tokens'),
            embedding_dim=cls.EMBEDDING_DIM,
        )
        word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
        cached_embedder = CachedTextFieldEmbedder(
            text_field_embedder=word_embeddings,
        )

        cached_embedder.cache(
            dataset_id=bio_dataset.dataset_id,
            dataset=instances,
            vocab=vocab,
        )

        return cached_embedder

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
            cls.create_entry(['single', 'double', 'triple'], ['B-TTag', 'I-TTag', 'O'], 2, 1.0),
            cls.create_entry(['no_label'], ['O'], 3, 1.0),
        ]

        dataset = BIODataset(0, 'fake_file.txt', binary_class)

        # hack around reading a file
        dataset.data = data

        return dataset
    
    def test_unlabeled(self):
        bio_data = ClusteringHeuristicTest.create_fake_data()
        cwr = ClusteringHeuristicTest.build_cache_cwr()
        unlabeled_corpus = UnlabeledBIODataset(dataset_id=0, bio_data=bio_data)

        h = ClusteringHeuristic(cwr, unlabeled_corpus)
        h_res = h.evaluate(unlabeled_corpus, ClusteringHeuristicTest.SAMPLE_SIZE)

        assert type(h_res) == torch.Tensor
        assert h_res.shape == (len(unlabeled_corpus),)

        new_points = sorted(
            range(len(unlabeled_corpus)), 
            reverse=True,
            key=lambda ind: h_res[ind]
        )

        select_val = h_res[new_points][0]
        unselect_val = h_res[new_points][-1]

        assert (h_res == unselect_val).sum() == (len(unlabeled_corpus) - ClusteringHeuristicTest.SAMPLE_SIZE)
        assert (h_res == select_val).sum() == ClusteringHeuristicTest.SAMPLE_SIZE