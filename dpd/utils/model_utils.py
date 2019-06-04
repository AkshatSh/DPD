from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import os
import sys

import numpy as np
import torch

import allennlp
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.modules.token_embedders import ElmoTokenEmbedder

from dpd.models.embedder import CachedTextFieldEmbedder, NERElmoTokenEmbedder
from dpd.utils import SaveFile, H5SaveFile
from dpd.constants import (
    ELMo_file,
    BERT_file,
    NER_ELMo_file,
    ELMO_OPTIONS_FILE,
    ELMO_WEIGHT_FILE,
)

def setup_embedder(text_field_embedder: BasicTextFieldEmbedder, save_file: SaveFile) -> CachedTextFieldEmbedder:

    word_embeddings = CachedTextFieldEmbedder(
        text_field_embedder=text_field_embedder,
    )

    word_embeddings.setup_cache(dataset_id=0)
    word_embeddings.setup_cache(dataset_id=1)

    word_embeddings.load(save_file=save_file)

    return word_embeddings

def get_elmo_embedder(dataset_file: str) -> CachedTextFieldEmbedder:
    elmo_embedder = ElmoTokenEmbedder(ELMO_OPTIONS_FILE, ELMO_WEIGHT_FILE)

def get_ner_elmo_embedder(dataset_file: str) -> CachedTextFieldEmbedder:
    elmo_embedder = NERElmoTokenEmbedder()
    word_embeddings = BasicTextFieldEmbedder(
        {"tokens": elmo_embedder},
    )

    return setup_embedder(
        text_field_embedder=word_embeddings,
        save_file=H5SaveFile(dataset_file),
    )

def get_bert_embedder(dataset_file: str) -> CachedTextFieldEmbedder:
    bert_embedder = PretrainedBertEmbedder(
        pretrained_model="bert-base-uncased",
        top_layer_only=True, # conserve memory
        # max_pieces=512, # max pieces allowed for positional embeddings
    )

    word_embeddings = BasicTextFieldEmbedder(
        {"tokens": bert_embedder},
        allow_unmatched_keys=True,
        embedder_to_indexer_map={"tokens": ["tokens", "tokens-offsets"]},
    )

    return setup_embedder(
        text_field_embedder=word_embeddings,
        save_file=H5SaveFile(dataset_file),
    )

def get_cached_embedder(dataset_name: str, e_type: str) -> CachedTextFieldEmbedder:
    if e_type == 'elmo':
        return get_elmo_embedder(ELMo_file[dataset_name])
    elif e_type == 'bert':
        return get_bert_embedder(BERT_file[dataset_name])
    elif e_type == 'ner_elmo':
        return get_ner_elmo_embedder(NER_ELMo_file[dataset_name])
    else:
        raise Exception(f'Unknown type: {e_type}')

def get_all_embedders(dataset_name: str, share_memory: Optional[bool] = False) -> List[CachedTextFieldEmbedder]:
    embedders = [
        get_cached_embedder(dataset_name, e_type) for e_type in ['bert', 'ner_elmo']
    ]

    if share_memory:
        list(map(lambda e: e.share_memory(), embedders))

    return embedders

def balance_dataset(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(neg_idx) <= len(pos_idx):
        return x, y

    negative_sample_size = min(len(neg_idx), len(pos_idx) * 2, 5000)

    smaller_negative: np.ndarray = np.random.choice(neg_idx, negative_sample_size)

    training_idxes = np.concatenate((pos_idx, smaller_negative))
    return x[training_idxes], y[training_idxes]
