from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import os
import sys

import torch
import allennlp

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.data import DatasetReader
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.training.util import evaluate

from dpd.dataset import (
    ActiveBIODataset,
    BIODataset,
    BIODatasetReader,
    UnlabeledBIODataset,
)

from dpd.utils import (
    get_dataset_files,
    Logger,
    construct_f1_class_labels,
    PickleSaveFile,
)

from dpd.constants import (
    SPACY_file,
)

from dpd.weak_supervision.feature_extractor import SpaCyFeatureExtractor

from dpd.models import build_model
from dpd.models.embedder import CachedTextFieldEmbedder
from dpd.oracles import Oracle, GoldOracle
from dpd.heuristics import RandomHeuristic, ClusteringHeuristic
from dpd.weak_supervision import build_weak_data
from dpd.utils import get_all_embedders, log_train_metrics
from dpd.args import get_active_args

import logging

def construct_vocab(datasets: List[BIODataset]) -> Vocabulary:
    readers = [BIODatasetReader(
        bio_dataset=bio_dataset,
        token_indexers={
            'tokens': ELMoTokenCharactersIndexer(),
            'single_tokens': SingleIdTokenIndexer(), # including for future pipelines to use, one hot
        },
    ) for bio_dataset in datasets]

    allennlp_datasets = [r.read('tmp.txt') for r in readers]

    result = allennlp_datasets[0]
    for i in range(1, len(allennlp_datasets)):
        result += allennlp_datasets[i]

    vocab = Vocabulary.from_instances(result)

    return vocab

def main():
    args = get_active_args()
    args.add_argument('--model_path', type=str)
    args = args.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'

    train_file, valid_file, test_file = get_dataset_files(dataset=args.dataset)

    class_labels: List[str] = construct_f1_class_labels(args.binary_class)

    train_bio = BIODataset(
        dataset_id=0,
        file_name=train_file,
        binary_class=args.binary_class,
        dataset_name=args.dataset,
    )

    train_bio.parse_file()

    if args.test:
        print('using test set')
    valid_bio = BIODataset(
        dataset_id=1,
        file_name=valid_file if not args.test else test_file,
        binary_class=args.binary_class,
        dataset_name=args.dataset,
    )

    valid_bio.parse_file()

    vocab = construct_vocab([train_bio, valid_bio])

    model = build_model(
        model_type=args.model_type,
        vocab=vocab,
        hidden_dim=args.hidden_dim,
        class_labels=class_labels,
        cached=args.cached,
        dataset_name=args.dataset.lower(),
    )

    model.share_memory()
    with open(args.model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    # return
    dataset_reader = BIODatasetReader(
        bio_dataset=valid_bio,
        token_indexers={
            'tokens': ELMoTokenCharactersIndexer(),
            'single_tokens': SingleIdTokenIndexer(), # including for future pipelines to use, one hot
        },
    )

    instances = dataset_reader.read('temp.txt')

    iterator = BucketIterator(
        batch_size=1,
        sorting_keys=[("sentence", "num_tokens")],
    )

    iterator.index_with(vocab)

    if device == 'cuda':
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    metrics = evaluate(model, instances, iterator, cuda_device, "")

    print(metrics)

if __name__ == '__main__':
    main()
