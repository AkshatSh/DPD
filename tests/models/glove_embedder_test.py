from typing import (
    List,
    Tuple,
    Dict,
)

import unittest
import os
import logging

from dpd.constants import (
    GLOVE_DIR
)
from dpd.models.embedder import (
    GloVeWordEmbeddingIndex
)

GLOVE_ENABLED = os.path.exists(GLOVE_DIR)

class GloveEmbedderTest(unittest.TestCase):
    def test_glove_singleton(self):
        if not GLOVE_ENABLED:
            logging.warning(
                f'Skipping because glove dir not found: {GLOVE_DIR}'
            )
            return
        word_embedding_index = GloVeWordEmbeddingIndex.instance()
        word_embedding_index_2 = GloVeWordEmbeddingIndex.instance()
        assert word_embedding_index is word_embedding_index_2
    
    def test_glove_singleton(self):
        if not GLOVE_ENABLED:
            logging.warning(
                f'Skipping because glove dir not found: {GLOVE_DIR}'
            )
            return
        word_embedding_index = GloVeWordEmbeddingIndex.instance()
        assert len(word_embedding_index.embedding_space) == 400000
        assert len(word_embedding_index.index_np) == 400001 # adds UNK
        assert len(word_embedding_index.word_to_index) == 400001 # has UNK
        assert len(word_embedding_index.index_to_word) == 400001 # has UNK
