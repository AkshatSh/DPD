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
    pass

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