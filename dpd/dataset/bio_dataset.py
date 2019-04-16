from typing import (
    List,
    Tuple,
    Dict,
)

from collections import Counter

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from tqdm import tqdm

class BIODataset(object):
    '''
    Given the file of a BIO encoded dataset, parses the all the files and
    creates a data list, where the list is tuples of (input, tagged) where input and tagged
    are two lists of the exact same elements, and tagged contains all the BIO tags for
    the input.

    Arguments:
        file_name: the name of the BIO encoded file
    '''
    def __init__(self, file_name: str, binary_class: str = None):
        self.file_name = file_name
        self.data = []
        self.word_list = Counter()
        self.tags = Counter()
        self.binary_class = binary_class

    def __len__(self) -> int:
        return len(self.data)
    
    def _convert_binary(self, token: str) -> str:
        if self.binary_class is None or token == 'O':
            return token
        elif token[2:] != self.binary_class:
            return 'O'
        return token

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
                    assert len(tokens) > 2

                    # seperates each line to 2 different things
                    # [word, tag]
                    # word, pos, sync_chunk, output = tokens
                    word, output = tokens[0], tokens[-1]
                    output = self._convert_binary(output)
                    self.word_list[word] += 1
                    currInput.append(word)

                    self.tags[output] += 1
                    
                    currOutput.append(output)

class ActiveBIODataset(object):
    def __init__(
        self,
        data: List[Tuple[int, List[str], List[str], float]],
    ):
        self.data = [
            {
                'input': input,
                'output': output,
                'id': data_id,
                'weight': weight,
            } for data_id, input, output, weight in data
        ]
    
    def __len__(self):
        return len(self.data)

class UnlabeledBIODataset(object):
    def __init__(
        self,
        bio_data: BIODataset,
    ):
        self.data = [
            {
                'input': data['input'],
                'id': data['id'],
                'weight': data['weight'],
            } for data in bio_data
        ]
    
    def __len__(self):
        return len(self.data)
