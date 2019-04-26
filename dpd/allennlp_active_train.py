from typing import (
    List,
    Tuple,
    Dict,
    Callable,
)

import os
import argparse
import logging

import torch
from torch import optim

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.data import DatasetReader
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer

from dpd.dataset import (
    ActiveBIODataset,
    BIODataset,
    BIODatasetReader,
    UnlabeledBIODataset,
)

from dpd.utils import (
    get_dataset_files,
    Logger,
)

from dpd.models import build_model
from dpd.oracles import Oracle, GoldOracle
from dpd.heuristics import RandomHeuristic
from dpd.weak_supervision import build_weak_data

ORACLE_SAMPLES = [10, 40, 50]

# type definitions

'''
EntryDataType:
 int (id)
 List[str] (input)
 List[str] (output)
 float (weight)
'''
EntryDataType = Dict[str, object]
DatasetType = List[EntryDataType]
MetricsType = Dict[str, object]

def train(
    model: Model,
    binary_class: str,
    train_data: DatasetType,
    valid_reader: DatasetReader,
    vocab: Vocabulary,
    optimizer_type: str,
    optimizer_learning_rate: float,
    optimizer_weight_decay: float,
    batch_size: int,
    patience: int,
    num_epochs: int,
    device: str,
) -> Tuple[Model, MetricsType]:
    train_reader = BIODatasetReader(
        ActiveBIODataset(train_data, dataset_id=0, binary_class=binary_class),
        token_indexers={
            'tokens': ELMoTokenCharactersIndexer(),
        },
    )

    train_dataset = train_reader.read('tmp.txt')
    valid_dataset = valid_reader.read('tmp.txt')

    cuda_device = -1

    if device == 'cuda':
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    optimizer = optim.SGD(
        model.parameters(),
        lr=optimizer_learning_rate,
        weight_decay=optimizer_weight_decay,
    )

    iterator = BucketIterator(
        batch_size=batch_size,
        sorting_keys=[("sentence", "num_tokens")],
    )

    iterator.index_with(vocab)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=valid_dataset,
        patience=patience,
        num_epochs=num_epochs,
        cuda_device=cuda_device,
    )
    metrics = trainer.train()

    return model, metrics

def log_train_metrics(
    logger: Logger,
    metrics: MetricsType,
    step: int,
    prefix='al'
):
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
        if metric.startswith('best_validation'):
            set_name = 'valid'
            metric_name = metric[len('best_validation_'):]
        elif metric.startswith('training'):
            set_name = 'train'
            metric_name = metric[len('training_'):]
        else:
            # ignore any other metric types
            continue
        
        if metric_name.startswith('_'):
            # ignore hidden
            metric_name = metric_name[1:]
        
        full_metric_name = f'{prefix}/{set_name}/{metric_name}'
        
        metric_list.extend(
            filter(
                lambda x: x is not None,
                log_special_metrics(
                    metric_name=full_metric_name,
                    metric_val=val,
                ),
            ),
        )

    for metric_name, metric_val in metric_list:
        logger.scalar_summary(tag=metric_name, value=metric_val, step=step)

def active_train_fine_tune_iteration(
    heuristic: RandomHeuristic,
    unlabeled_dataset: UnlabeledBIODataset,
    sample_size: int,
    labeled_indexes: List[int],
    oracle: Oracle,
    train_data: DatasetType,
    valid_reader: DatasetReader,
    vocab: Vocabulary,
    model: Model,
    optimizer_type: str,
    optimizer_learning_rate: float,
    optimizer_weight_decay: float,
    use_weak: bool,
    weak_weight: float,
    weak_function: str,
    batch_size: int,
    patience: int,
    num_epochs: int,
    device: str,
) -> Tuple[Model, Dict[str, object]]:
    # select new points from distribution
    distribution = heuristic.evaluate(unlabeled_dataset)
    new_points = []
    sample_size = min(sample_size, len(distribution) - 1)
    new_points = torch.multinomial(distribution, sample_size)
    new_points = new_points[:sample_size]

    # use new points to augment train_dataset
    # remove points from unlabaled corpus
    query = [
        (
            unlabeled_dataset[ind]['id'],
            unlabeled_dataset[ind]['input'],
        ) for ind in new_points
    ]

    labeled_indexes.extend(
        ind for (ind, _) in query
    )

    oracle_labels = [oracle.get_query(q) for q in query]
    train_data.extend(oracle_labels)

    # remove unlabeled data points from corpus
    [unlabeled_dataset.remove(q) for q in query]

    weak_data = []
    if use_weak:
        # builds a weak set to augment the training
        # set
        weak_data = build_weak_data(
            train_data,
            unlabeled_dataset,
            model,
            weight=weak_weight,
            function_type=weak_function,
        )

        model, _ = train(
            model=model,
            binary_class=unlabeled_dataset.binary_class,
            train_data=weak_data,
            valid_reader=valid_reader,
            vocab=vocab,
            optimizer_type=optimizer_type,
            optimizer_learning_rate=optimizer_learning_rate,
            optimizer_weight_decay=optimizer_weight_decay,
            batch_size=batch_size,
            patience=patience,
            num_epochs=num_epochs,
            device=device,
        )

    model, metrics = train(
        model=model,
        binary_class=unlabeled_dataset.binary_class,
        train_data=train_data,
        valid_reader=valid_reader,
        vocab=vocab,
        optimizer_type=optimizer_type,
        optimizer_learning_rate=optimizer_learning_rate,
        optimizer_weight_decay=optimizer_weight_decay,
        batch_size=batch_size,
        patience=patience,
        num_epochs=num_epochs,
        device=device,
    )

    return model, metrics

def active_train_iteration(
    heuristic: RandomHeuristic,
    unlabeled_dataset: UnlabeledBIODataset,
    sample_size: int,
    labeled_indexes: List[int],
    oracle: Oracle,
    train_data: DatasetType,
    valid_reader: DatasetReader,
    vocab: Vocabulary,
    model: Model,
    optimizer_type: str,
    optimizer_learning_rate: float,
    optimizer_weight_decay: float,
    use_weak: bool,
    weak_weight: float,
    weak_function: str,
    batch_size: int,
    patience: int,
    num_epochs: int,
    device: str,
) -> Tuple[Model, Dict[str, object]]:
    # select new points from distribution
    distribution = heuristic.evaluate(unlabeled_dataset)
    new_points = []
    sample_size = min(sample_size, len(distribution) - 1)
    new_points = torch.multinomial(distribution, sample_size)
    new_points = new_points[:sample_size]

    # use new points to augment train_dataset
    # remove points from unlabaled corpus
    query = [
        (
            unlabeled_dataset[ind]['id'],
            unlabeled_dataset[ind]['input'],
        ) for ind in new_points
    ]

    labeled_indexes.extend(
        ind for (ind, _) in query
    )

    oracle_labels = [oracle.get_query(q) for q in query]
    train_data.extend(oracle_labels)

    # remove unlabeled data points from corpus
    [unlabeled_dataset.remove(q) for q in query]

    weak_data = []
    if use_weak:
        # builds a weak set to augment the training
        # set
        weak_data = build_weak_data(
            train_data,
            unlabeled_dataset,
            model,
            weight=weak_weight,
            function_type=weak_function,
        )

    model, metrics = train(
        model=model,
        binary_class=unlabeled_dataset.binary_class,
        train_data=train_data + weak_data,
        valid_reader=valid_reader,
        vocab=vocab,
        optimizer_type=optimizer_type,
        optimizer_learning_rate=optimizer_learning_rate,
        optimizer_weight_decay=optimizer_weight_decay,
        batch_size=batch_size,
        patience=patience,
        num_epochs=num_epochs,
        device=device,
    )

    return model, metrics

def active_train(
    model: Model,
    unlabeled_dataset: UnlabeledBIODataset,
    valid_dataset: BIODataset,
    vocab: Vocabulary,
    oracle: Oracle,
    optimizer_type: str,
    optimizer_learning_rate: float,
    optimizer_weight_decay: float,
    use_weak: bool,
    weak_fine_tune: bool,
    weak_weight: float,
    weak_function: str,
    batch_size: int,
    patience: int,
    num_epochs: int,
    device: str,
    log_dir: str,
    model_name: str,
) -> Model:
    heuristic = RandomHeuristic()

    log_dir = os.path.join(log_dir, model_name)
    logger = Logger(logdir=log_dir)

    # keep track of all the ids that have been
    # labeled
    labeled_indexes: List[int] = []

    # the current training data that is being built up
    train_data: DatasetType = []

    valid_reader = BIODatasetReader(
        bio_dataset=valid_dataset,
        token_indexers={
            'tokens': ELMoTokenCharactersIndexer(),
        },
    )

    for i, sample_size in enumerate(ORACLE_SAMPLES):
        active_iteration_kwargs = dict(
            heuristic=heuristic,
            unlabeled_dataset=unlabeled_dataset,
            sample_size=sample_size,
            labeled_indexes=labeled_indexes,
            oracle=oracle,
            train_data=train_data,
            valid_reader=valid_reader,
            model=model,
            vocab=vocab,
            optimizer_type=optimizer_type,
            optimizer_learning_rate=optimizer_learning_rate,
            optimizer_weight_decay=optimizer_weight_decay,
            use_weak=use_weak,
            weak_weight=weak_weight,
            weak_function=weak_function,
            batch_size=batch_size,
            patience=patience,
            num_epochs=num_epochs,
            device=device,
        )

        if weak_fine_tune:
            model, metrics = active_train_fine_tune_iteration(**active_iteration_kwargs)
        else:
            model, metrics = active_train_iteration(**active_iteration_kwargs)

        log_train_metrics(logger, metrics, step=len(train_data))

        print(f'Finished experiment on training set size: {len(train_data)}')
    logger.flush()

def get_args() -> argparse.ArgumentParser:
    '''
    Create arg parse for active learning training containing options for
        optimizer
        hyper parameters
        model saving
        log directories
    '''
    parser = argparse.ArgumentParser(description='Build an active learning iterative pipeline')

    # Active Learning Pipeline Parameters
    parser.add_argument('--log_dir', type=str, default='logs/', help='the directory to log into')
    parser.add_argument('--model_name', type=str, default='active_learning_model', help='the name to give the model')

    # dataset parameters
    parser.add_argument('--dataset', type=str, default='CADEC', help='the dataset to use {CONLL, CADEC}')
    parser.add_argument('--binary_class', type=str, default='ADR', help='the binary class to use for the dataset')

    # hyper parameters
    parser.add_argument('--model_type', type=str, default='ELMo_bilstm_crf', help='the model type to use')
    parser.add_argument('--hidden_dim', type=int, default=512, help='the hidden dimensions for the model')

    # optimizer config
    parser.add_argument('--opt_type', type=str, default='SGD', help='the optimizer to use')
    parser.add_argument('--opt_lr', type=float, default=0.01, help='the learning rate for the optimizer')
    parser.add_argument('--opt_weight_decay', type=float, default=1e-4, help='weight decay for optimizer')

    # weak data config
    parser.add_argument('--use_weak', action='store_true', help='use the weak set during training')
    parser.add_argument('--use_weak_fine_tune', action='store_true', help='use the weak fine tuning approach')
    parser.add_argument('--weak_weight', type=float, default=1.0, help='the weight to give to the weak set during training')
    parser.add_argument('--weak_function', type=str, default='linear', help='the type of weak function to use')

    # training config
    parser.add_argument('--num_epochs', type=int, default=5, help='the number of epochs to run each iteration')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of training')
    parser.add_argument('--patience', type=int, default=5, help='patience parameter for training')

    # system config
    parser.add_argument('--cuda', action='store_true', help='use CUDA if available')

    # Parser data loader options
    return parser

def construct_f1_class_labels(class_label: str) -> List[str]:
    prefix = ['B', 'I']
    return [f'{p}-{class_label}' for p in prefix]

def construct_vocab(datasets: List[BIODataset]) -> Vocabulary:
    readers = [BIODatasetReader(
        bio_dataset=bio_dataset,
        token_indexers={
            'tokens': ELMoTokenCharactersIndexer(),
        },
    ) for bio_dataset in datasets]

    allennlp_datasets = [r.read('tmp.txt') for r in readers]

    result = allennlp_datasets[0]
    for i in range(1, len(allennlp_datasets)):
        result += allennlp_datasets[i]

    vocab = Vocabulary.from_instances(result)

    return vocab

def main():
    args = get_args().parse_args()

    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'

    train_file, valid_file = get_dataset_files(dataset=args.dataset)

    class_labels: List[str] = construct_f1_class_labels(args.binary_class)

    train_bio = BIODataset(
        dataset_id=0,
        file_name=train_file,
        binary_class=args.binary_class,
    )

    train_bio.parse_file()

    valid_bio = BIODataset(
        dataset_id=1,
        file_name=valid_file,
        binary_class=args.binary_class,
    )

    valid_bio.parse_file()

    vocab = construct_vocab([train_bio, valid_bio])

    unlabeled_corpus = UnlabeledBIODataset(
        dataset_id=train_bio.dataset_id,
        bio_data=train_bio,
    )

    model = build_model(
        model_type=args.model_type,
        vocab=vocab,
        hidden_dim=args.hidden_dim,
        class_labels=class_labels,
    )

    oracle = GoldOracle(train_bio)

    active_train(
        model=model,
        unlabeled_dataset=unlabeled_corpus,
        valid_dataset=valid_bio,
        vocab=vocab,
        oracle=oracle,
        optimizer_type=args.opt_type,
        optimizer_learning_rate=args.opt_lr,
        optimizer_weight_decay=args.opt_weight_decay,
        use_weak=args.use_weak,
        weak_fine_tune=args.use_weak_fine_tune,
        weak_weight=args.weak_weight,
        weak_function=args.weak_function,
        batch_size=args.batch_size,
        patience=args.patience,
        num_epochs=args.num_epochs,
        device=device,
        log_dir=args.log_dir,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()