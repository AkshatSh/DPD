from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Iterator,
    Any,
)

import os
import sys
import argparse

import torch
import numpy as np
import allennlp
from allennlp.data import Vocabulary
import pickle


from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules import TextFieldEmbedder, TokenEmbedder
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data import Instance
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.token_indexers import PretrainedBertIndexer

import dpd
from dpd.dataset import BIODataset, BIODatasetReader
from dpd.utils import get_dataset_files, SaveFile, H5SaveFile
from dpd.models.embedder import NERElmoTokenEmbedder, CachedTextFieldEmbedder
from dpd.constants import (
    ELMO_OPTIONS_FILE,
    ELMO_WEIGHT_FILE,
    CADEC_ELMo,
    CADEC_BERT,
    CADEC_NER_ELMo,
    CONLL_BERT,
    CONLL_ELMo,
    CONLL_NER_ELMo,
)

def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Script to cache different embeddings for datasets')
    parser.add_argument('--embedder', type=str, default='ner_elmo', help='the embedder to cache')
    parser.add_argument('--dataset', type=str, default='cadec', help='the dataset to cache')
    parser.add_argument('--cuda', action='store_true', help='whether to use GPU if possible')
    return parser

def get_embedder_info(embedder_type: str) -> Tuple[TokenEmbedder, TokenIndexer, str, Dict[str, Any]]:
    embedder_type = embedder_type.lower()
    text_field_embedder_kwargs: Dict[str, Any] = {}
    if embedder_type == 'ner_elmo':
        return NERElmoTokenEmbedder(), ELMoTokenCharactersIndexer(), text_field_embedder_kwargs
    elif embedder_type == 'elmo':
        return ElmoTokenEmbedder(ELMO_OPTIONS_FILE, ELMO_WEIGHT_FILE), ELMoTokenCharactersIndexer(), text_field_embedder_kwargs
    elif embedder_type == 'bert':
        bert_embedder = PretrainedBertEmbedder(
                pretrained_model="bert-base-uncased",
                top_layer_only=True, # conserve memory
        )
        token_indexer = PretrainedBertIndexer(
            pretrained_model="bert-base-uncased",
            max_pieces=512, # max pieces allowed for positional embeddings
            do_lowercase=True,
            use_starting_offsets=True,
        )
        text_field_embedder_kwargs['allow_unmatched_keys'] = True
        return bert_embedder, token_indexer, text_field_embedder_kwargs
    else:
        raise Exception(f'Unknown embedder type: {embedder_type}')
    
def get_save_file(embedder_type: str, dataset_type: str) -> str:
    embedder_type = embedder_type.lower()
    dataset_type = dataset_type.lower()
    if dataset_type == 'cadec':
        if embedder_type == 'ner_elmo':
            return CADEC_NER_ELMo
        elif embedder_type == 'elmo':
            return CADEC_ELMo
        elif embedder_type == 'bert':
            return CADEC_BERT
        else:
            raise Exception(f'Unknown embedder type: {embedder_type}')
    elif dataset_type == 'conll':
        if embedder_type == 'ner_elmo':
            return CONLL_NER_ELMo
        elif embedder_type == 'elmo':
            return CONLL_ELMo
        elif embedder_type == 'bert':
            return CONLL_BERT
        else:
            raise Exception(f'Unknown embedder type: {embedder_type}')
    else:
        raise Exception(f'Unknown dataset type: {dataset_type}')

def main():
    args = get_args().parse_args()
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    train_file, valid_file, test_file = get_dataset_files(dataset=args.dataset)
    token_embedder, token_indexer, text_field_embedder_kwargs = get_embedder_info(args.embedder)

    train_bio = BIODataset(
        dataset_id=0,
        file_name=train_file,
    )
    train_bio.parse_file()

    train_reader = BIODatasetReader(
        bio_dataset=train_bio,
        token_indexers={
            'tokens': token_indexer,
        },
    )

    train_data = train_reader.read('temp.txt')

    valid_bio = BIODataset(
        dataset_id=1,
        file_name=valid_file,
    )
    valid_bio.parse_file()

    valid_reader = BIODatasetReader(
        bio_dataset=valid_bio,
        token_indexers={
            'tokens': token_indexer,
        },
    )

    valid_data = valid_reader.read('temp.txt')

    vocab = Vocabulary.from_instances(train_data + valid_data)
    embedder = BasicTextFieldEmbedder({"tokens": token_embedder}, **text_field_embedder_kwargs)
    cached_embedder = CachedTextFieldEmbedder(
        text_field_embedder=embedder,
    )

    if args.cuda:
        cuda_device = 0
        cached_embedder = cached_embedder.cuda(cuda_device)
    else:
        cuda_device = -1

    cached_embedder.cache(
        dataset_id=train_bio.dataset_id,
        dataset=train_data,
        vocab=vocab,
        cuda_device=cuda_device,
    )

    cached_embedder.cache(
        dataset_id=valid_bio.dataset_id,
        dataset=valid_data,
        vocab=vocab,
        cuda_device=cuda_device,
    )

    save_file_name = get_save_file(embedder_type=args.embedder, dataset_type=args.dataset)

    save_file = H5SaveFile(file_name=save_file_name)

    cached_embedder.save(save_file=save_file)
    save_file.close()


if __name__ == "__main__":
    main()