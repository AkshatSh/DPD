from typing import (
    List,
    Tuple,
    Dict,
    Iterator,
    Optional,
)

from collections import Counter

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from tqdm import tqdm
import random

BIODataEntry = Dict[str, object]

class BIODataset(object):
    '''
    Given the file of a BIO encoded dataset, parses the all the files and
    creates a data list, where the list is tuples of (input, tagged) where input and tagged
    are two lists of the exact same elements, and tagged contains all the BIO tags for
    the input.

    Arguments:
        file_name: the name of the BIO encoded file
    '''
    def __init__(
        self,
        dataset_id: int,
        file_name: str,
        binary_class: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ):
        self.file_name = file_name
        self.data = []
        self.word_list = Counter()
        self.tags = Counter()
        self.binary_class = binary_class
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name.lower() if dataset_name is not None else None

    def __len__(self) -> int:
        return len(self.data)
    
    def _convert_binary(self, token: str) -> str:
        if self.binary_class is None or token == 'O':
            return token
        elif token[2:] != self.binary_class:
            return 'O'
        return token
    
    def __getitem__(self, index) -> BIODataEntry:
        return self.data[index]
    
    def __iter__(self) -> Iterator[BIODataEntry]:
        for item in self.data.__iter__():
            yield item

    def parse_file(self) -> None:
        with open(self.file_name) as f:
            currInput = []
            currOutput = []
            for _, line in enumerate(tqdm(f)):
                if len(line.strip()) == 0 or line.startswith('-DOCSTART'):
                    if len(currInput) > 0:
                        # marks the end of a sentence
                        self.data.append(
                            {
                                'id': len(self.data),
                                'input': currInput,
                                'output': currOutput,
                                'weight': 1.0,
                            }
                        )
                        currInput = []
                        currOutput = []
                else:
                    tokens = line.split()
                    # there has to be at least 2 things
                    assert len(tokens) >= 2

                    # seperates each line to 2 different things
                    # [word, tag]
                    # word, pos, sync_chunk, output = tokens
                    word, output = tokens[0], tokens[-1]
                    output = self._convert_binary(output)
                    self.word_list[word] += 1
                    currInput.append(word)

                    self.tags[output] += 1
                    
                    currOutput.append(output)

class ActiveBIODataset(BIODataset):
    def __init__(
        self,
        data: List[Dict[str, object]],
        dataset_id: int,
        binary_class: str,
        dataset_name: str,
    ):
        super().__init__(
            dataset_id=dataset_id,
            file_name='temp.txt',
            binary_class=binary_class,
            dataset_name=dataset_name,
        )
        self.dataset_id = dataset_id
        self.data = data

class UnlabeledBIODataset(BIODataset):
    def __init__(
        self,
        bio_data: BIODataset,
        dataset_id: int,
        shuffle: Optional[bool] = False
    ):
        super().__init__(
            dataset_id=dataset_id,
            file_name='temp.txt',
            binary_class=bio_data.binary_class,
            dataset_name=bio_data.dataset_name,
        )

        self.data = [
            {
                'input': data['input'],
                'id': data['id'],
                'weight': data['weight'],
            } for data in bio_data
        ]

        if shuffle:
            random.shuffle(self.data)
    
    def remove(self, query) -> None:
        s_id, s_in = query
        for i, item in enumerate(self.data):
            if item['id'] == s_id:
                assert s_in == item['input']
                del self.data[i]
                break