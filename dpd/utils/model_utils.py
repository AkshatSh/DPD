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
from dpd.constants import CADEC_NER_ELMo, CADEC_BERT, CADEC_ELMo, ELMO_OPTIONS_FILE, ELMO_WEIGHT_FILE

def setup_embedder(text_field_embedder: BasicTextFieldEmbedder, save_file: SaveFile) -> CachedTextFieldEmbedder:

    word_embeddings = CachedTextFieldEmbedder(
        text_field_embedder=text_field_embedder,
    )

    word_embeddings.setup_cache(dataset_id=0)
    word_embeddings.setup_cache(dataset_id=1)

    word_embeddings.load(save_file=save_file)

    return word_embeddings

def get_elmo_embedder() -> CachedTextFieldEmbedder:
    elmo_embedder = ElmoTokenEmbedder(ELMO_OPTIONS_FILE, ELMO_WEIGHT_FILE)

def get_ner_elmo_embedder() -> CachedTextFieldEmbedder:
    elmo_embedder = NERElmoTokenEmbedder()
    word_embeddings = BasicTextFieldEmbedder(
        {"tokens": elmo_embedder},
    )

    return setup_embedder(
        text_field_embedder=word_embeddings,
        save_file=H5SaveFile(CADEC_NER_ELMo),
    )

def get_bert_embedder() -> CachedTextFieldEmbedder:
    bert_embedder = PretrainedBertEmbedder(
        pretrained_model="bert-base-uncased",
        top_layer_only=True, # conserve memory
    )

    word_embeddings = BasicTextFieldEmbedder(
        {"tokens": bert_embedder},
        allow_unmatched_keys=True,
    )

    return setup_embedder(
        text_field_embedder=word_embeddings,
        save_file=H5SaveFile(CADEC_BERT),
    )

def get_cached_embedder(e_type: str) -> CachedTextFieldEmbedder:
    if e_type == 'elmo':
        return get_elmo_embedder()
    elif e_type == 'bert':
        return get_bert_embedder()
    elif e_type == 'ner_elmo':
        return get_ner_elmo_embedder()
    else:
        raise Exception(f'Unknown type: {e_type}')

def get_all_embedders() -> List[CachedTextFieldEmbedder]:
    return [
        get_cached_embedder(e_type) for e_type in ['ner_elmo']
    ]