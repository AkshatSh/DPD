from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import os
import sys

import torch
import numpy as np
import allennlp

from dpd.utils import SaveFile, H5SaveFile
from dpd.models.embedder import CachedTextFieldEmbedder
from dpd.constants import (
    CADEC_NER_ELMo,
    CADEC_ELMo,
    CADEC_BERT,
)

def main():
    save_file = H5SaveFile(file_name=CADEC_BERT)
    ce = CachedTextFieldEmbedder(text_field_embedder=None)
    ce.setup_cache(dataset_id=0)
    ce.setup_cache(dataset_id=1)
    ce.load(save_file)

    res = ce.get_embedding(sentence_id=0, dataset_id=0)

if __name__ == "__main__":
    main()
