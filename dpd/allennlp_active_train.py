from typing import (
    List,
    Tuple,
    Dict,
    Callable,
)

import argparse


from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary

from dpd.dataset.bio_dataset import (
    ActiveBIODataset,
    BIODataset,
    UnlabeledBIODataset,
)

from dpd.utils import (
    get_dataset_files,
)

from dpd.models import build_model
from dpd.dataset.bio_dataloader import BIODatasetReader
from dpd.oracles import GoldOracle

# type definitions

'''
EntryDataType:
 int (id)
 List[str] (input)
 List[str] (output)
 float (weight)
'''
EntryDataType = Tuple[int, List[str], List[str], float]
DatasetType = List[EntryDataType]
MetricsType = Dict[str, object]

def train(
    model: Model,
    train_data: DatasetType,
    test_data: DatasetType,
    vocab: Vocabulary,
    optimizer_type: str,
    optimizer_learning_rate: float,
    batch_size: int,
    patience: int,
    num_epochs: int,
    device: str,
) -> Tuple[Model, MetricsType]:
    train_reader = BIODatasetReader(ActiveBIODataset(train_data))
    test_reader = BIODatasetReader(ActiveBIODataset(test_data))

    train_dataset = train_reader.read('tmp.txt')
    test_dataset = valid_reader.read('tmp.txt')

    cuda_device = -1

    if device == 'cuda':
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    optimizer = optim.SGD(
        model.parameters(),
        lr=optimizer_learning_rate,
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
        validation_dataset=test_dataset,
        patience=patience,
        num_epochs=num_epochs,
        cuda_device=cuda_device,
    )
    metrics = trainer.train()

    # TODO(akshats): Log some metrics

    return model, metrics


def active_train(
    model: Model,
    unlabeled_dataset: UnlabeledBIODataset,
    valid_dataset: BIODataset,
    oracle: Oracle,
    optimizer_type: str,
    optimizer_learning_rate: float,
    optimizer_weight_decay: float,
    batch_size: int,
    patience: int,
    num_epochs: int,
    device: str,
) -> Model:

    oracle_samples = [1, 5, 10, 25, 50, 100, 200, 400, 400]
    for i, sample_size in enumerate(iteration_samples):
        # select new points from distribution
        # TODO: implement heuristics
        distribution = heuritic.evaluate(model, unlabeled_dataset, device)
        new_points = []
        sample_size = min(sample_size, len(distribution) - 1)
        new_points = torch.multinomial(distribution, sample_size)
        new_points = new_points[:sample_size]


        # use new points to augment train_dataset
        # remove points from unlabaled corpus
        query = [
            unlabeled_dataset.data[ind]
            for ind in new_points
        ]

        labeled_indexes.extend(
            ind for (ind, _) in query
        )

        # adds a weight
        outputs = [oracle.get_label(q) + (1.0,) for q in query]

        # move unlabeled points to labeled points
        [unlabeled_dataset.remove(q) for q in query]

        # TODO: build weak_set
        # TODO: train model
        model, metrics = train(
            model=model,
            train_data=train_data,
            test_data=test_data,
            vocab=vocab,
            optimizer_type=optimizer_type,
            optimizer_learning_rate=optimizer_learning_rate,
            optimizer_weight_decay=optimizer_weight_decay,
            batch_size=batch_size,
            patience=patience,
            num_epochs=num_epochs,
            device=device,
        )

        # gather some metrics
        f1_data, acc = utils.compute_f1_dataloader(model, test_data_loader, tag_vocab, device=device)
        f1_avg_valid = utils.compute_avg_f1(f1_data)

        # TODO(akshats): implement logger
        # log valid metics
        logger.scalar_summary("active valid f1", f1_avg_valid, len(train_data))
        logger.scalar_summary("active valid accuracy", acc, len(train_data))
        utils.log_metrics(logger, f1_data, "active valid", len(train_data))

        # TODO(akshats): use logger
        print(f'Finished experiment on training set size: {len(train_data)}')

def get_args() -> argparse.ArgumentParser:
    '''
    Create arg parse for active learning training containing options for
        optimizer
        hyper parameters
        model saving
        log directories
    '''
    parser = argparse.ArgumentParser(description='Build an active learning iterative pipeline')

    parser.add_argument('--dataset', type=str, default='CONLL', help='the dataset to use {CONLL, CADEC}')
    parser.add_argument('--binary_class', type=str, default=None, help='the binary class to use for the dataset')

    # hyper parameters
    parser.add_argument('--hidden_dim', type=int, default=512, help='the hidden dimensions for the model')

    # optimizer config
    parser.add_argument('--opt_type', type=str, default='SGD', help='the optimizer to use')
    parser.add_argument('--opt_lr', type=float, default=0.01, help='the learning rate for the optimizer')
    parser.add_argument('--opt_weight_decay', type=float, default=1e-4, help='weight decay for optimizer')

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

def main():
    args = get_args().parse_args()

    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'

    train_file: str, valid_file: str = get_dataset_files(dataset=args.dataset)

    class_labels: List[str] = construct_f1_class_labels(args.binary_class)

    train_bio = BIODataset(
        dataset_id=0,
        file_name=train_file,
        binary_class=args.binary_class,
    )

    valid_bio = BIODataset(
        dataset_id=1,
        file_name=valid_file,
        binary_class=args.binary_class,
    )

    unlabeled_corpus = UnlabeledBIODataset(
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
        oracle=oracle,
        optimizer_type=args.opt_type,
        optimizer_learning_rate=args.opt_lr,
        optimizer_weight_decay=args.opt_weight_decay,
        batch_size=args.batch_size,
        patience=args.patience,
        num_epochs=args.num_epochs,
        device=device,
    )


if __name__ == "__main__":
    main()