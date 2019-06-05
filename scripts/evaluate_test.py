from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Any
)

import os
import sys
import pickle
from tqdm import tqdm

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
            # 'single_tokens': SingleIdTokenIndexer(), # including for future pipelines to use, one hot
        },
    ) for bio_dataset in datasets]

    allennlp_datasets = [r.read('tmp.txt') for r in readers]

    result = allennlp_datasets[0]
    for i in range(1, len(allennlp_datasets)):
        result += allennlp_datasets[i]

    vocab = Vocabulary.from_instances(result)

    return vocab

def simple_metrics(metrics: dict):
    def log_special_metrics(metric_name: str, metric_val: object) -> List[Tuple[str, int]]:
        if type(metric_val) == int or type(metric_val) == float:
            return [(metric_name, metric_val)]
        elif type(metric_val) == list:
            res = []
            for metric_val_item in metric_val:
                class_label = metric_val_item['class']
                for metric_n, metric_v in metric_val_item.items():
                    if metric_n == 'class':
                        # skip class names
                        continue
                    res.append(
                        (
                            f'{metric_name}_{class_label}_{metric_n}',
                            metric_v,
                        )
                    )
            return res
        else:
            logging.warning(f'Unknown metric type: {type(metric_val)} for ({metric_name}, {metric_val})')
            return []

    metric_list = []
    for metric, val in metrics.items():
        metric_name = metric
        set_name = 'train'
        if metric_name.startswith('_'):
            # ignore hidden
            metric_name = metric_name[1:]
        
        full_metric_name = f'{metric_name}'

        metric_list.extend(
            filter(
                lambda x: x is not None,
                log_special_metrics(
                    metric_name=full_metric_name,
                    metric_val=val,
                ),
            ),
        )

    return metric_list

def get_all_model_checkpoints(experiment_dir: str, experiment_name: str) -> List[Dict[str, Any]]: 
    res = []
    for trials in os.listdir(experiment_dir):
        trial_dir = os.path.join(experiment_dir)
        trial_num: int = int(trial_dir[-1])
        for files in os.listdir(trial_dir):
            for f in files:
                file_name = os.path.splitext(f)[0]
                model_prefix: str = "model_checkpoint_"
                if file_name.startswith(model_prefix):
                    dataset_size: int = int(file_name[len(model_prefix):])
                    res.append({
                        'experiment_name': experiment_name,
                        'trial': trial_num,
                        'dataset_size': dataset_size,
                        'full_path': os.path.join(trial_dir, f)
                    })
    return res

def evalaute_checkpoint(model, instances, iterator, cuda_device, model_path):
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    metrics = evaluate(model, instances, iterator, cuda_device, "")
    return simple_metrics(metrics)

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
        dataset_id=2,
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
    # return
    dataset_reader = BIODatasetReader(
        bio_dataset=valid_bio,
        token_indexers={
            'tokens': ELMoTokenCharactersIndexer(),
            # 'single_tokens': SingleIdTokenIndexer(), # including for future pipelines to use, one hot
        },
    )

    instances = dataset_reader.read('temp.txt')

    iterator = BucketIterator(
        batch_size=args.batch_size,
        sorting_keys=[("sentence", "num_tokens")],
    )

    iterator.index_with(vocab)

    if device == 'cuda':
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1


    checkpoint_info = get_all_model_checkpoints(args.model_path)
    results = []
    for ckpt in tqdm(checkpoint_info):
        dataset_size, trial, experiment_name, model_path = ckpt['dataset_size'], ckpt['tria'], ckpt['experiment_name'], ckpt['full_path']
        metrics = evalaute_checkpoint(model, instances, iterator, cuda_device, model_path)
        ckpt['metrics'] = metrics

    with open(os.path.join(model_path, f'results_test_{args.text}'), 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
