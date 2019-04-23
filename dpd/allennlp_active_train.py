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
from dpd.dataset.bio_dataloader import BIODatasetReader

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
) -> Model:
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

    return model


def active_train(
    model: Model,
    unlabeled_dataset: UnlabeledBIODataset,
    valid_dataset: BIODataset,
    optimizer_type: str,
    optimizer_learning_rate: float,
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

    # Parser data loader options
    return parser

def main():
    args = get_args().parse_args()


if __name__ == "__main__":
    main()