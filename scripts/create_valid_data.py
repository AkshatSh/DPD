from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import os
import sys

import dpd
from dpd.dataset import BIODataset
from dpd.constants import (
    CADEC_TRAIN,
    CADEC_VALID,
    CADEC_VALID_ORIGINAL,
    CADEC_TEST,
)

def serialize_split(data: List[object], file_name: str):
    with open(file_name, 'w') as f:
        for item in data:
            data_in = item['input']
            data_out = item['output']
            for d_in, d_out in zip(data_in, data_out):
                f.writelines(f'{d_in}\t{d_out}\n')
            f.writelines('\n')

def create_split(dataset: BIODataset, ratio: float):
    first_split = int(len(dataset) * ratio)
    first_data = dataset.data[:first_split]
    second_data = dataset.data[first_split:]
    return first_data, second_data

def main():
    dataset = BIODataset(dataset_id=0, file_name=CADEC_VALID_ORIGINAL)
    dataset.parse_file()
    valid_data, test_data = create_split(dataset, 0.5)
    serialize_split(valid_data, CADEC_VALID)
    serialize_split(test_data, CADEC_TEST)


if __name__ == "__main__":
    main()