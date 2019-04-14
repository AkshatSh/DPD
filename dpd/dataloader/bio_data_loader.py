from typing import (
    List,
    Tuple,
    Dict,
)

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
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.data = []
        self.word_list = []
        self.tags = []

    def __len__(self) -> int:
        return len(self.data)
    
    def parse_file(self) -> None:
        with open(self.file_name) as f:
            currInput = []
            currOutput = []
            for _, line in enumerate(tqdm(f)):
                if len(line.strip()) == 0:
                    # marks the end of a sentence
                    self.data.append(
                        {
                            'input': currInput,
                            'output': currOutput,
                        }
                    )
                    currInput = []
                    currOutput = []
                else:
                    tokens = line.split()
                    # seperates each line to 2 different things
                    # [word, tag]
                    word, output = tokens
                    self.word_list.append(word)
                    currInput.append(word)

                    if output not in self.tags:
                        self.tags.append(output)
                    
                    currOutput.append(output)