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
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.data.token_indexers import PretrainedBertIndexer

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

# ORACLE_SAMPLES = [10, 40, 50, 400, 500]
ORACLE_SAMPLES = [10, 100, 400, 500]

logger = logging.getLogger(name=__name__)

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
        validation_metric='+f1-measure-overall',
    )
    metrics = trainer.train()

    return model, metrics

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
    cached_text_field_embedders: List[CachedTextFieldEmbedder],
    spacy_feature_extractor: SpaCyFeatureExtractor,
    optimizer_type: str,
    optimizer_learning_rate: float,
    optimizer_weight_decay: float,
    use_weak: bool,
    weak_weight: float,
    weak_function: List[str],
    weak_collator: str,
    sample_strategy: str,
    batch_size: int,
    patience: int,
    num_epochs: int,
    device: str,
    logger: Logger,
) -> Tuple[Model, Dict[str, object]]:
    # select new points from distribution
    distribution = heuristic.evaluate(unlabeled_dataset, sample_size)
    new_points = []
    sample_size = min(sample_size, len(distribution) - 1)
    if sample_strategy == 'sample':
        new_points = torch.multinomial(distribution, sample_size)
    elif sample_strategy == 'top_k':
        new_points = sorted(
            range(len(distribution)), 
            reverse=True,
            key=lambda ind: distribution[ind]
        )
    else:
        raise Exception(f'Unknown sampling strategry: {sample_strategy}')
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
            function_types=weak_function,
            collator_type=weak_collator,
            contextual_word_embeddings=cached_text_field_embedders,
            spacy_feature_extractor=spacy_feature_extractor,
            vocab=vocab,
        )

        model, weak_metrics = train(
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

        log_train_metrics(logger, weak_metrics, step=len(train_data), prefix='weak')

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
    cached_text_field_embedders: List[CachedTextFieldEmbedder],
    spacy_feature_extractor: SpaCyFeatureExtractor,
    optimizer_type: str,
    optimizer_learning_rate: float,
    optimizer_weight_decay: float,
    use_weak: bool,
    weak_weight: float,
    weak_function: List[str],
    weak_collator: str,
    sample_strategy: str,
    batch_size: int,
    patience: int,
    num_epochs: int,
    device: str,
    logger: Logger,
) -> Tuple[Model, Dict[str, object]]:
    # select new points from distribution
    # distribution contains score for each index
    distribution = heuristic.evaluate(unlabeled_dataset, sample_size)
    new_points = []

    # sample the sample size from the distribution
    sample_size = min(sample_size, len(distribution) - 1)
    if sample_strategy == 'sample':
        new_points = torch.multinomial(distribution, sample_size)
    elif sample_strategy == 'top_k':
        new_points = sorted(
            range(len(distribution)), 
            reverse=True,
            key=lambda ind: distribution[ind]
        )
    else:
        raise Exception(f'Unknown sampling strategry: {sample_strategy}')
    new_points = new_points[:sample_size]

    # new points now contains list of indexes in the unlabeled
    # corpus to annotate
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
            function_types=weak_function,
            collator_type=weak_collator,
            contextual_word_embeddings=cached_text_field_embedders,
            spacy_feature_extractor=spacy_feature_extractor,
            vocab=vocab,
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
    weak_function: List[str],
    weak_collator: str,
    sample_strategy: str,
    batch_size: int,
    patience: int,
    num_epochs: int,
    device: str,
    log_dir: str,
    model_name: str,
) -> Model:
    # heuristic =  ClusteringHeuristic(model.word_embeddings, unlabeled_dataset)
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

    cached_text_field_embedders: List[CachedTextFieldEmbedder] = get_all_embedders(unlabeled_dataset.dataset_name, share_memory=True)
    spacy_feature_extractor: SpaCyFeatureExtractor = SpaCyFeatureExtractor.setup(dataset_ids=[0, 1])
    spacy_feature_extractor.load(save_file=PickleSaveFile(SPACY_file[unlabeled_dataset.dataset_name]))

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
            cached_text_field_embedders=cached_text_field_embedders,
            spacy_feature_extractor=spacy_feature_extractor,
            vocab=vocab,
            optimizer_type=optimizer_type,
            optimizer_learning_rate=optimizer_learning_rate,
            optimizer_weight_decay=optimizer_weight_decay,
            use_weak=use_weak,
            weak_weight=weak_weight,
            weak_function=weak_function,
            weak_collator=weak_collator,
            sample_strategy=sample_strategy,
            batch_size=batch_size,
            patience=patience,
            num_epochs=num_epochs,
            device=device,
            logger=logger,
        )

        if weak_fine_tune:
            model, metrics = active_train_fine_tune_iteration(**active_iteration_kwargs)
        else:
            model, metrics = active_train_iteration(**active_iteration_kwargs)

        log_train_metrics(logger, metrics, step=len(train_data))

        print(f'Finished experiment on training set size: {len(train_data)}')
    logger.flush()

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
    args = get_active_args().parse_args()
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

    unlabeled_corpus = UnlabeledBIODataset(
        dataset_id=train_bio.dataset_id,
        bio_data=train_bio,
    )

    model = build_model(
        model_type=args.model_type,
        vocab=vocab,
        hidden_dim=args.hidden_dim,
        class_labels=class_labels,
        cached=args.cached,
        dataset_name=args.dataset.lower(),
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
        weak_collator=args.weak_collator,
        sample_strategy=args.sample_strategy,
        batch_size=args.batch_size,
        patience=args.patience,
        num_epochs=args.num_epochs,
        device=device,
        log_dir=args.log_dir,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()