from typing import (
    List,
    Optional,
    Dict,
    Any,
)

import unittest

from allennlp.data import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.data import Instance
from allennlp.data.iterators import BucketIterator

from dpd.dataset import BIODataset, BIODatasetReader
from dpd.models import MultiTaskTagger

class MultiTaskTest(unittest.TestCase):
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 2
    MINIBATCH_SIZE = 2
    NUM_EPOCHS = 1

    @classmethod
    def create_entry(cls, sentence: List[str], labels: List[str], entry_id: int, weight: float) -> Dict[str, Any]:
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
            cls.create_entry(['no_label'], ['O'], 3, 0.1),
        ]

        dataset = BIODataset(0, 'fake_file.txt', binary_class)

        # hack around reading a file
        dataset.data = data

        return dataset

    def test_forward(self):
        bio_dataset = MultiTaskTest.create_fake_data()
        reader = BIODatasetReader(
            bio_dataset=bio_dataset,
        )

        instances = reader.read('fake_file.txt')
        vocab = Vocabulary.from_instances(instances)

        token_embedding = Embedding(
            num_embeddings=vocab.get_vocab_size('tokens'),
            embedding_dim=MultiTaskTest.EMBEDDING_DIM,
        )

        tagger = MultiTaskTagger(
            vocab=vocab,
            hidden_dim=MultiTaskTest.HIDDEN_DIM,
            class_labels=['B-Tag', 'I-Tag'],
            cached=False,
            word_embedder=token_embedding,
        )

        iterator = BucketIterator(batch_size=MultiTaskTest.MINIBATCH_SIZE, sorting_keys=[("sentence", "num_tokens")])
        iterator.index_with(vocab)
        train_generator = iterator(instances, num_epochs=MultiTaskTest.NUM_EPOCHS, shuffle=False)
        for inst in train_generator:
            tagged_res = tagger.forward(
                sentence=inst['sentence'],
                sentence_ids=inst['entry_id'],
                dataset_ids=inst['dataset_id'],
                weight=inst['weight'],
                use_cache=True,
            )

            batch_size, token_num = inst['sentence']['tokens'].shape
            expected_shape = (batch_size, token_num, vocab.get_vocab_size('labels'))