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

from allennlp.data import Instance

import dpd
from dpd.dataset import BIODataset, BIODatasetReader
from dpd.weak_supervision.feature_extractor import SpaCyFeatureExtractor
from dpd.utils import get_dataset_files, SaveFile, PickleSaveFile
from dpd.constants import (
    CADEC_SPACY,
    CONLL_SPACY,
)

def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Script to cache different linguistic features for the dataset')
    parser.add_argument('--feature_extractor', type=str, default='spacy', help='the feature extractor to use')
    parser.add_argument('--dataset', type=str, default='cadec', help='the dataset to cache')
    parser.add_argument('--cuda', action='store_true', help='whether to use GPU if possible')
    return parser
    
def get_save_file(feature_extractor_type: str, dataset_type: str) -> str:
    feature_extractor_type = feature_extractor_type.lower()
    dataset_type = dataset_type.lower()
    if dataset_type == 'cadec':
        if feature_extractor_type == 'spacy':
            return CADEC_SPACY
        else:
            raise Exception(f'Unknown feature_extractor type: {feature_extractor_type}')
    elif dataset_type == 'conll':
        if feature_extractor_type == 'spacy':
            return CONLL_SPACY
        else:
            raise Exception(f'Unknown feature_extractor type: {feature_extractor_type}')
    else:
        raise Exception(f'Unknown dataset type: {dataset_type}')

def main():
    args = get_args().parse_args()
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    train_file, valid_file, test_file = get_dataset_files(dataset=args.dataset)

    train_bio = BIODataset(
        dataset_id=0,
        file_name=train_file,
    )
    train_bio.parse_file()

    train_reader = BIODatasetReader(
        bio_dataset=train_bio,
    )

    train_data: Iterator[Instance] = train_reader.read('temp.txt')

    valid_bio = BIODataset(
        dataset_id=1,
        file_name=valid_file,
    )
    valid_bio.parse_file()

    valid_reader = BIODatasetReader(
        bio_dataset=valid_bio,
    )

    valid_data: Iterator[Instance] = valid_reader.read('temp.txt')
    vocab = Vocabulary.from_instances(train_data + valid_data)

    if args.cuda:
        cuda_device = 0
        cached_embedder = cached_embedder.cuda(cuda_device)
    else:
        cuda_device = -1

    feature_extractor = SpaCyFeatureExtractor()
    feature_extractor.cache(
        dataset_id=0,
        dataset=train_data,
        vocab=vocab,
    )
    feature_extractor.cache(
        dataset_id=1,
        dataset=valid_data,
        vocab=vocab,
    )

    save_file_name = get_save_file(feature_extractor_type=args.feature_extractor, dataset_type=args.dataset)

    save_file = PickleSaveFile(file_name=save_file_name)

    feature_extractor.save(save_file=save_file)
    save_file.close()


if __name__ == "__main__":
    main()