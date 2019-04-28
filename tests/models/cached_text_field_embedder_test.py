from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Iterator,
)

import unittest
import os
import sys

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data.vocabulary import Vocabulary
from allennlp.data import Instance
from allennlp.data.iterators import BucketIterator

from dpd.dataset import BIODataset, BIODatasetReader
from dpd.models.embedder import CachedTextFieldEmbedder

class CachecTextFieldEmbedderTest(unittest.TestCase):
    EMBEDDING_DIM = 5
    MINIBATCH_SIZE = 2
    NUM_EPOCHS = 1

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
            CachecTextFieldEmbedderTest.create_entry(['single'], ['B-Tag'], 0, 1.0),
            CachecTextFieldEmbedderTest.create_entry(['single', 'double'], ['B-Tag', 'I-Tag'], 1, 1.0),
            CachecTextFieldEmbedderTest.create_entry(['single', 'double', 'triple'], ['B-TTag', 'I-TTag', 'O'], 2, 1.0),
            CachecTextFieldEmbedderTest.create_entry(['no_label'], ['O'], 3, 1.0),
        ]

        dataset = BIODataset(0, 'fake_file.txt', binary_class)

        # hack around reading a file
        dataset.data = data

        return dataset
    
    def test_cache_structure(self):
        bio_dataset = CachecTextFieldEmbedderTest.create_fake_data()
        reader = BIODatasetReader(
            bio_dataset=bio_dataset,
        )

        instances = reader.read('fake_file.txt')
        vocab = Vocabulary.from_instances(instances)
        token_embedding = Embedding(
            num_embeddings=vocab.get_vocab_size('tokens'),
            embedding_dim=CachecTextFieldEmbedderTest.EMBEDDING_DIM,
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

        def get_num_words(instances: Iterator[Instance]) -> int:
            num_words: int = 0
            for inst in instances:
                num_words += len(inst['sentence'])
            return num_words

        num_words: int = get_num_words(instances)

        # only one dataset is cached
        assert len(cached_embedder.cached_datasets) == 1
        # make sure the dataset id is cached
        assert 0 in cached_embedder.cached_datasets
        # make sure every entry is cached
        cd = cached_embedder.cached_datasets[0]
        assert len(cd.embedded_dataset) == num_words
        assert cd.embedded_dataset.shape == (num_words, CachecTextFieldEmbedderTest.EMBEDDING_DIM)

        for inst in instances:
            s_id = inst['entry_id'].as_tensor(None).item()
            sent = inst['sentence']
            assert s_id in cd.sid_to_start
            et = cd.get_embedding(s_id)
            assert et.shape == (len(sent), CachecTextFieldEmbedderTest.EMBEDDING_DIM)
    
    def test_cache_forward(self):
        bio_dataset = CachecTextFieldEmbedderTest.create_fake_data()
        reader = BIODatasetReader(
            bio_dataset=bio_dataset,
        )

        instances = reader.read('fake_file.txt')
        vocab = Vocabulary.from_instances(instances)
        token_embedding = Embedding(
            num_embeddings=vocab.get_vocab_size('tokens'),
            embedding_dim=CachecTextFieldEmbedderTest.EMBEDDING_DIM,
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

        iterator = BucketIterator(batch_size=1, sorting_keys=[("sentence", "num_tokens")])
        iterator.index_with(vocab)
        train_generator = iterator(instances, num_epochs=CachecTextFieldEmbedderTest.NUM_EPOCHS, shuffle=False)
        for inst in train_generator:
            cached_result = cached_embedder.forward(
                sentence=inst['sentence'],
                sentence_ids=inst['entry_id'],
                dataset_ids=inst['dataset_id'],
                use_cache=True,
            )
            
            non_cached_result = cached_embedder.forward(
                sentence=inst['sentence'],
                sentence_ids=inst['entry_id'],
                dataset_ids=inst['dataset_id'],
                use_cache=False,
            )

            assert cached_result.shape == non_cached_result.shape
    
    def test_cache_forward_minibatch(self):
        bio_dataset = CachecTextFieldEmbedderTest.create_fake_data()
        reader = BIODatasetReader(
            bio_dataset=bio_dataset,
        )

        instances = reader.read('fake_file.txt')
        vocab = Vocabulary.from_instances(instances)
        token_embedding = Embedding(
            num_embeddings=vocab.get_vocab_size('tokens'),
            embedding_dim=CachecTextFieldEmbedderTest.EMBEDDING_DIM,
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

        iterator = BucketIterator(batch_size=CachecTextFieldEmbedderTest.MINIBATCH_SIZE, sorting_keys=[("sentence", "num_tokens")])
        iterator.index_with(vocab)
        train_generator = iterator(instances, num_epochs=CachecTextFieldEmbedderTest.NUM_EPOCHS, shuffle=False)
        for inst in train_generator:
            cached_result = cached_embedder.forward(
                sentence=inst['sentence'],
                sentence_ids=inst['entry_id'],
                dataset_ids=inst['dataset_id'],
                use_cache=True,
            )
            
            non_cached_result = cached_embedder.forward(
                sentence=inst['sentence'],
                sentence_ids=inst['entry_id'],
                dataset_ids=inst['dataset_id'],
                use_cache=False,
            )

            assert cached_result.shape == non_cached_result.shape